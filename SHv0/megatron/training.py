# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import time

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import set_global_variables
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron import mpu
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.model import ModelType
from megatron.optimizer import get_megatron_optimizer
from megatron.optimizer import gl_get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.learning_rates import AnnealingLR
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.schedules import get_forward_backward_func
from megatron.utils import report_memory

from megatron.utils import flops_calculator, throughput_calculator
from megatron.model import GPTModel, BertModel


# for use 1-v100 server to mock 8-v100 server
#torch.cuda.set_per_process_memory_fraction(1/8*1.1, 0)
torch.cuda.empty_cache() 

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_rank_0("[" + string + "] datetime: {} ".format(time_str))


# =======================
# === for GL version ====
# =======================
import asyncio
from collections import OrderedDict, deque
from megatron.arguments import parse_args
from megatron.utils import _is_cpu
from megatron.utils import gl_warmup_print
from megatron.utils import get_parameters_in_billions, _unwrap_model

#import ray
#import ray.workers.setup_worker
from concurrent.futures import ThreadPoolExecutor
ray_get = lambda futs: [fut.result() for fut in futs]

import random
import json

# ray.init()
# ray.init(num_gpus=8, num_cpus=86,
#        namespace='offloading', ignore_reinit_error=True,
#        _system_config={
#            'worker_lease_timeout_milliseconds': 0,
#            'max_io_workers': 86,
#            'object_timeout_milliseconds': 1, })
#ray.init(
#    ignore_reinit_error=True,
#    _system_config={
#        "max_io_workers": 8,  # More IO workers for local storage. Each IO worker tries using a different directories.
#        "object_spilling_config": json.dumps(
#            {
#              "type": "filesystem",
#              "params": {
#                # Each directory could mount at different devices.
#                "directory_path": [
#                  "/tmp/ray_ssd/spill_1",
#                  "/tmp/ray_ssd/spill_2",
#                  "/tmp/ray_ssd/spill_3",
#                  "/tmp/ray_ssd/spill_4",
#                  "/tmp/ray_ssd/spill_5",
#                  "/tmp/ray_ssd/spill_6",
#                  "/tmp/ray_ssd/spill_7",
#                  "/tmp/ray_ssd/spill_8",]}},
#        )
#    },
#)


def _free_cuda_tensor(tensor):
    if tensor.storage().size() > 0:
        assert tensor.storage_offset() == 0, "The tensor should have sole storage"
        tensor.storage().resize_(0)


# =====================================================
# === four kinds of hooks for forward and backward  ===
# =====================================================
def _append_loading_futs(_layer, futs, fwd=True):
    if fwd:
        if hasattr(_layer, "_gl_futs_loading_at_fwd"):
            _layer._gl_futs_loading_at_fwd += futs
        else:
            _layer._gl_futs_loading_at_fwd = futs
    else:
        if hasattr(_layer, "_gl_futs_loading_at_bwd"):
            _layer._gl_futs_loading_at_bwd += futs
        else:
            _layer._gl_futs_loading_at_bwd = futs


def _append_offloading_futs(_layer, futs, fwd=True):
    if fwd:
        if hasattr(_layer, "_gl_futs_offloading_at_fwd"):
            _layer._gl_futs_offloading_at_fwd += futs
        else:
            _layer._gl_futs_offloading_at_fwd = futs
    else:
        if hasattr(_layer, "_gl_futs_offloading_at_bwd"):
            _layer._gl_futs_offloading_at_bwd += futs
        else:
            _layer._gl_futs_offloading_at_bwd = futs


def _layer_waiting_futs(_layer, timers):
    timers("offloading-fwd-overhead").start()

    if hasattr(_layer, "_gl_futs_loading_at_fwd"):
        timers("offloading-fwd-2gpu-overhead").start()
        ray_get(_layer._gl_futs_loading_at_fwd)
        del _layer._gl_futs_loading_at_fwd
        timers("offloading-fwd-2gpu-overhead").stop()

    if hasattr(_layer, "_gl_futs_offloading_at_fwd"):
        timers("offloading-fwd-2cpu-overhead").start()
        ray_get(_layer._gl_futs_offloading_at_fwd)
        del _layer._gl_futs_offloading_at_fwd
        timers("offloading-fwd-2cpu-overhead").stop()

    timers("offloading-fwd-overhead").stop()

    timers("offloading-bwd-overhead").start()

    if hasattr(_layer, "_gl_futs_loading_at_bwd"):
        timers("offloading-bwd-2gpu-overhead").start()
        ray_get(_layer._gl_futs_loading_at_bwd)
        del _layer._gl_futs_loading_at_bwd
        timers("offloading-bwd-2gpu-overhead").stop()

    if hasattr(_layer, "_gl_futs_offloading_at_bwd"):
        timers("offloading-bwd-2cpu-overhead").start()
        ray_get(_layer._gl_futs_offloading_at_bwd)
        del _layer._gl_futs_offloading_at_bwd
        timers("offloading-bwd-2cpu-overhead").stop()

    # if _layer._gl_is_at_first_window:
    _layer._gl_save_for_backward.clear()

    timers("offloading-bwd-overhead").stop()


def _forward_pre_hook(module, input):
    if not torch._gl_is_forward_now:
        return

    timers = get_timers()

    _current_layer = module
    _handler = _current_layer._gl_handler
    _layers = _current_layer._gl_layers

    _layer_waiting_futs(_current_layer, timers)
    timers("offloading-func-call-overhead").start()
    gl_warmup_print(
        f"--- at _forward_pre_hook ...... layer={_current_layer._gl_layer_num}"
    )

    # hard code, assume that no accumulated gradients firstly.
    # todo. @gl
    # for debugging
    for param in _current_layer.parameters():
        param.grad = None

    _current_layer._gl_save_for_backward.clear()

    # to store save_for_backward tensor with torch.autograd.graph.saved_tensors_hooks
    torch._gl_current_layer = _current_layer

    """
    # todo, @gl. now resides at gpu all the time
    # the first window_size layers:
    if _current_layer._gl_is_at_first_window:
        _futs_main_grads = [
                _handler._layer_move_main_grads_to.remote(_current_layer._gl_layer_num, 'cpu')]
        _which_layer_to_cuda = _current_layer._gl_which_layer_to_cuda_pre_fwd
        _append_offloading_futs(_layers[_which_layer_to_cuda], _futs_main_grads)
    """

    # loading _which_layer_to_cuda_pre_fwd
    if (
        not _current_layer._gl_is_at_last_window
        and _current_layer._gl_which_layer_to_cuda_pre_fwd_required
    ):
        _which_layer_to_cuda = _current_layer._gl_which_layer_to_cuda_pre_fwd

        if torch._gl_in_warmup:
            s = time.time()

        futs = [_handler._layer_to_cuda_remote(_which_layer_to_cuda)]
        _append_loading_futs(_layers[_which_layer_to_cuda], futs)

        if torch._gl_in_warmup:
            e1 = time.time()
            ray_get(futs)
            e2 = time.time()

            info = f" \n\t layer{_current_layer._gl_layer_num}. _gl_cpu_version_items"
            for key, value in _current_layer._gl_cpu_version_items.items():
                info += f"\n\t {key}: {value.device}"

            gl_warmup_print(
                f"--debug-info = _forward_pre_hook: moving layer-{_which_layer_to_cuda} to cuda, ",
                f"triggered by {_current_layer._gl_layer_num}",
                "; \n\t async function call: ",
                e1 - s,
                "; \n\t synchronizing: ",
                e2 - s,
                info,
            )

    timers("offloading-func-call-overhead").stop()


def _forward_post_hook(module, input, output):
    if not torch._gl_is_forward_now:
        return

    timers = get_timers()
    timers("offloading-func-call-overhead").start()

    _current_layer = module
    _handler = _current_layer._gl_handler
    _layers = _current_layer._gl_layers

    gl_warmup_print(
        f"--- at _forward_post_hook ...... layer={_current_layer._gl_layer_num}"
    )
    # print(f'--- at _forward_post_hook ...... layer={_current_layer._gl_layer_num}')

    # for param in _current_layer.parameters():
    #     param.grad = None
    #     print("--debug: param.grad at fwd_post_hook:", param.grad)

    # for torch.autograd.graph.saved_tensors_hooks
    if _current_layer._gl_layer_num == _current_layer._gl_window_size - 1:
        del torch._gl_current_layer

    if (
        not _current_layer._gl_is_at_last_window
        and _current_layer._gl_which_layer_to_cpu_post_fwd_required
    ):
        _which_layer_to_cpu = _current_layer._gl_which_layer_to_cpu_post_fwd

        if torch._gl_in_warmup:
            s = time.time()

        futs = [_handler._layer_to_cpu_remote(_which_layer_to_cpu)]

        futs += [
            _handler._layer_move_save_for_backward_to_remote(
                _which_layer_to_cpu, "cpu", 'g2c', i
            )
            for i in range(len(_current_layer._gl_save_for_backward))
        ]

        _which_layer_to_cuda = _current_layer._gl_which_layer_to_cuda_pre_fwd
        _append_offloading_futs(_layers[_which_layer_to_cuda], futs)

        if torch._gl_in_warmup:
            e1 = time.time()
            ray_get(futs)
            e2 = time.time()

            info = f" \n\t layer{_current_layer._gl_layer_num}. _gl_cpu_version_items"
            for key, value in _current_layer._gl_cpu_version_items.items():
                info += f"\n\t {key}: {value.device}"

            gl_warmup_print(
                f"--debug-info = _forward_post_hook: ",
                f"moving layer-{_which_layer_to_cpu} to cpu, ",
                f"triggered by {_current_layer._gl_layer_num}",
                "; \n\t async function call: ",
                e1 - s,
                "; \n\t synchronizing: ",
                e2 - s,
                info,
            )

    timers("offloading-func-call-overhead").stop()


def _backward_pre_hook(module, grad_in, grad_out):
    if not torch._gl_is_backward_now:
        return

    timers = get_timers()

    _current_layer = module
    _handler = _current_layer._gl_handler
    _layers = _current_layer._gl_layers

    _layer_waiting_futs(_current_layer, timers)
    timers("offloading-func-call-overhead").start()
    gl_warmup_print(
        f"--- at _backward_pre_hook ...... layer={_current_layer._gl_layer_num}"
    )
    # print(f'--- at _backward_pre_hook ...... layer={_current_layer._gl_layer_num}')

    if (
        not _current_layer._gl_is_at_first_window
        and _current_layer._gl_which_layer_to_cuda_pre_bwd_required
    ):
        _which_layer_to_cuda = _current_layer._gl_which_layer_to_cuda_pre_bwd

        if torch._gl_in_warmup:
            s = time.time()

        futs = [_handler._layer_to_cuda_remote(_which_layer_to_cuda)]

        futs += [
            _handler._layer_move_save_for_backward_to_remote(
                _which_layer_to_cuda, _layers[_which_layer_to_cuda]._gl_cuda_device, 'c2g', i
            )
            for i in reversed(
                range(len(_layers[_which_layer_to_cuda]._gl_save_for_backward))
            )
        ]

        # for checkpoint
        checkpoint_chunk_size = _current_layer._gl_checkpoint_chunk_size
        first_layer_in_chunk = (
            _which_layer_to_cuda // checkpoint_chunk_size * checkpoint_chunk_size
        )
        last_layer_in_chunk = min(
            first_layer_in_chunk + checkpoint_chunk_size - 1, len(_layers) - 1
        )
        _append_loading_futs(_layers[last_layer_in_chunk], futs, fwd=False)

        """
        # todo. @gl. now resides at gpu all the time
        if _layers[_which_layer_to_cuda]._gl_is_at_first_window:
            # todo. @gl !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # if opt.step() at this end of the iteration.
            #    loading main_grads to cuda
            # else:
            #.   keeping main_grads at cpu_cache
            _futs_layer +=[
                    _handler._layer_move_main_grads_to.remote(_which_layer_to_cuda, 
                        _layers[_which_layer_to_cuda]._gl_cuda_device, )]
        """

        # !!! attention, please !!!
        # why +1 herein?
        #   unpacking save_for_backward tensors in ahead of backward_pre_hook
        _append_loading_futs(_layers[_which_layer_to_cuda + 1], futs, fwd=False)

        if torch._gl_in_warmup:
            e1 = time.time()
            ray_get(futs)
            e2 = time.time()

            info = f" \n\t layer{_current_layer._gl_layer_num}. _gl_cpu_version_items"
            for key, value in _current_layer._gl_cpu_version_items.items():
                info += f"\n\t {key}: {value.device}"

            gl_warmup_print(
                f"--debug-info = _backward_pre_hook: ",
                f"moving layer-{_which_layer_to_cuda} to cuda, ",
                f"triggered by {_current_layer._gl_layer_num}",
                "; \n\t async function call: ",
                e1 - s,
                "; \n\t synchronizing: ",
                e2 - s,
                info,
            )

    timers("offloading-func-call-overhead").stop()


# todo @gl.........................
#
# register the hook at the front of the layer.
#   for instance, the hook of layer-5 registered at layer-4
#   because no pure bwd hook api after the full execution of backward
#
#   registering into forward_pre_hook and forward_hook can deal with it,
#   but shows ....
def _backward_post_hook(module, grad_in, grad_out):
    if not torch._gl_is_backward_now:
        return

    timers = get_timers()
    timers("offloading-func-call-overhead").start()

    _current_layer = module
    _handler = _current_layer._gl_handler
    _layers = _current_layer._gl_layers
    # redirect to the behind of the layer for backward_post hook
    _current_layer = _layers[_current_layer._gl_layer_num + 1]

    gl_warmup_print(
        f"--- at _backward_post_hook ...... layer={_current_layer._gl_layer_num} \n"
    )

    # for _n, _p in _current_layer.named_parameters():
    #     if _p.grad is not None:
    #         print('before invoking cpu()', _n, _p.device, _p.grad.device)
    #     else:
    #         print('before invoking cpu()', _n, _p.device)

    # _current_layer.cpu()

    # for _n, _p in _current_layer.named_parameters():
    #     if _p.grad is not None:
    #         print('after', _n, _p.device, _p.grad.device)
    #     else:
    #         print('after',_n, _p.device)

    if (
        not _current_layer._gl_is_at_first_window
        and _current_layer._gl_which_layer_to_cpu_post_bwd_required
    ):
        _which_layer_to_cpu = _current_layer._gl_which_layer_to_cpu_post_bwd

        if torch._gl_in_warmup:
            s = time.time()

        futs = [
            _handler._layer_to_cpu_and_gather_grads_and_optimizer_update_remote(
                _which_layer_to_cpu
            )
        ]

        futs += [_handler._layer_reset_save_for_backward_remote(_which_layer_to_cpu)]

        _which_layer_to_cuda = _current_layer._gl_which_layer_to_cuda_pre_bwd
        _append_offloading_futs(_layers[_which_layer_to_cuda], futs, fwd=False)

        if torch._gl_in_warmup:
            e1 = time.time()
            ray_get(futs)
            e2 = time.time()

            info = f" \n\t layer{_current_layer._gl_layer_num}. _gl_cpu_version_items"
            for key, value in _current_layer._gl_cpu_version_items.items():
                info += f"\n\t {key}: {value.device}"

            gl_warmup_print(
                f"--debug-info = _backward_post_hook: ",
                f"moving & updating layer-{_which_layer_to_cpu} to cpu, ",
                f"triggered by {_current_layer._gl_layer_num}",
                "; \n\t async function call: ",
                e1 - s,
                "; \n\t synchronizing: ",
                e2 - s,
                info,
            )

    elif _current_layer._gl_is_at_first_window and _current_layer._gl_layer_num != 0:
        if torch._gl_in_warmup:
            s = time.time()

        # only offloading main_grads
        # todo, @gl
        # futs = [
        #         _handler._layer_gather_grads_and_optimizer_update_and_offloading_grads.remote(
        #             _current_layer._gl_layer_num)]
        futs = [
            _handler._layer_gather_grads_and_optimizer_update_remote(
                _current_layer._gl_layer_num
            )
        ]
        _append_offloading_futs(_current_layer, futs, fwd=False)

        if torch._gl_in_warmup:
            e1 = time.time()
            ray_get(futs)
            e2 = time.time()
            gl_warmup_print(
                f"--debug-info = _backward_post_hook: ",
                f"updating layer-{_current_layer._gl_layer_num}, ",
                f"triggered by {module._gl_layer_num}",
                "; \n\t async function call: ",
                e1 - s,
                "; \n\t synchronizing: ",
                e2 - s,
            )

    else:  # layer-0
        pass

    timers("offloading-func-call-overhead").stop()


# =====================================================
# ===                 hooks ending                  ===
# =====================================================

class MockTensor:
    def __init__(self, tensor, loc):
        self.device = tensor.device
        self.loc = loc

def _get_item_from_cpu_cache(_cpu_cache, key, value=None):
    if key in _cpu_cache and _cpu_cache[key] is not None:
        ## ----- nvme -----
        #if isinstance(_cpu_cache[key], MockTensor):
        #    #import offloading_utils
        #    #_mock_tensor = _cpu_cache[key]
        #    #_cpu_cache[key] = offloading_utils.load(_cpu_cache[key].loc)[0]
        #    #_cpu_cache[key].mock_tensor = _mock_tensor

        #    _cpu_cache[key] = ray.get(_cpu_cache[key].loc)
        ## ----------------
        return _cpu_cache[key]

    _cpu_cache[key] = torch.zeros(
        value.size(),
        dtype=value.dtype,
        layout=value.layout,
        device=torch.device("cpu"),
        pin_memory=True,  # (torch.cuda.is_available() and not value.is_sparse)
    ).to("cpu", non_blocking=True)
    return _cpu_cache[key]

def _move_item_to_nvme(_cpu_cache, key, value=None):
    if key in _cpu_cache and _cpu_cache[key] is not None:
        ## ----- nvme -----
        #if not isinstance(_cpu_cache[key], MockTensor): 
        #    #import offloading_utils
        #    if hasattr(_cpu_cache[key], 'mock_tensor'):
        #        #mock_tensor = _cpu_cache[key].mock_tensor
        #        #del _cpu_cache[key].mock_tensor
        #        #offloading_utils.save([_cpu_cache[key]], mock_tensor.loc)
        #        #_cpu_cache[key] = mock_tensor
        #        
        #        r = ray.put(_cpu_cache[key])
        #        mock_tensor = MockTensor(_cpu_cache[key], r)
        #        _cpu_cache[key] = mock_tensor
        #    else:
        #        #saved_name = '/tmp/' + str(id(_cpu_cache[key])) + str(random.random()) + '.pt'
        #        #mock_tensor = MockTensor(_cpu_cache[key], saved_name)
        #        #offloading_utils.save([_cpu_cache[key]], saved_name)
        #        #_cpu_cache[key] = mock_tensor
        #        
        #        r = ray.put(_cpu_cache[key])
        #        mock_tensor = MockTensor(_cpu_cache[key], r)
        #        _cpu_cache[key] = mock_tensor
        ## ----------------
        pass 

    return


#@ray.remote
class GL_PretrainHanlder:
    def __init__(
        self,
        args,
        train_valid_test_dataset_provider,
        model_provider,
        model_type,
        forward_step_func,
        extra_args_provider=None,
        args_defaults={},
    ):
        from megatron.initialize import initialize_megatron
        from megatron.training import setup_model_and_optimizer
        from megatron.training import train
        from megatron.training import save_checkpoint
        from megatron.training import evaluate_and_print_results
        from megatron import get_args, set_args
        from megatron import get_timers
        from megatron import print_rank_0
        from collections import OrderedDict, deque
        from megatron.utils import get_parameters_in_billions, _unwrap_model

        import time

        # setting log file, in case of unkown hangs in Alibaba PAI platform
        import sys, os
        if 'LOG_FILE' in os.environ:
            f = open(os.environ['LOG_FILE'], 'at')
            sys.stdout = f

        # setting save_for_backward wrapper
        from megatron.utils import saved_tensors_wrapper
        from megatron.utils import save_for_backward_wrapper

        # """
        # setting save_for_backward hooks
        # """
        # torch.autograd.function.FunctionCtx.save_for_backward = (
        #     save_for_backward_wrapper(
        #         torch.autograd.function.FunctionCtx.save_for_backward
        #     )
        # )
        # torch.autograd.function.FunctionCtx.saved_tensors = property(
        #     saved_tensors_wrapper(torch.autograd.function.FunctionCtx.saved_tensors)
        # )
        # """

        # redefine setting synchronize
        def _func():
            torch.cuda.current_stream().synchronize()

        torch.cuda.synchronize = _func

        # ? no influence
        # torch.__future__.set_overwrite_module_params_on_conversion(False)

        set_args(args)
        # Initalize and get arguments, timers, and Tensorboard writer.
        initialize_megatron(
            extra_args_provider=extra_args_provider,
            args_defaults=args_defaults,
            ignore_unknown_args=True,
        )

        assert (
            args.num_layers % args.activations_checkpoint_num_layers == 0
        ), "the number of layers should be divided by activations_checkpoint_num_layers"

        timers = get_timers()

        self.args = args
        self.num_layers = args.num_layers
        self.gl_window_size = args.gl_window_size
        self.gl_debug_print = args.gl_debug_print
        self.timers = timers

        self.executor = ThreadPoolExecutor(max_workers=args.gl_ray_max_concurrency)

        # Adjust the startup time so it reflects the largest value.
        # This will be closer to what scheduler will see (outside of
        # image ... launches.
        global _TRAIN_START_TIME
        start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
        torch.distributed.all_reduce(
            start_time_tensor, op=torch.distributed.ReduceOp.MIN
        )
        _TRAIN_START_TIME = start_time_tensor.item()
        print_rank_0(
            "time to initialize megatron (seconds): {:.3f}".format(
                time.time() - _TRAIN_START_TIME
            )
        )
        print_datetime("after megatron is initialized")

        timers("model-and-optimizer-setup").start()
        model, optimizer, lr_scheduler = setup_model_and_optimizer(
            model_provider, model_type
        )
        timers("model-and-optimizer-setup").stop()
        print_datetime(
            "after model, optimizer, and learning rate " "scheduler are built"
        )

        self.model, self.optimizer, self.lr_scheduler = model, optimizer, lr_scheduler
        self._language_model = None

        # Data stuff.
        timers("train/valid/test-data-iterators-setup").start()
        if args.virtual_pipeline_model_parallel_size is not None:
            all_data_iterators = [
                build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
                for _ in range(len(model))
            ]
            train_data_iterator = [
                data_iterators[0] for data_iterators in all_data_iterators
            ]
            valid_data_iterator = [
                data_iterators[1] for data_iterators in all_data_iterators
            ]
            test_data_iterator = [
                data_iterators[2] for data_iterators in all_data_iterators
            ]
        else:
            (
                train_data_iterator,
                valid_data_iterator,
                test_data_iterator,
            ) = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
        timers("train/valid/test-data-iterators-setup").stop()
        print_datetime("after dataloaders are built")

        self.train_data_iterator, self.valid_data_iterator, self.test_data_iterator = (
            train_data_iterator,
            valid_data_iterator,
            test_data_iterator,
        )

        self.forward_step_func = forward_step_func

        # Print setup timing.
        print_rank_0("done with setup ...")
        timers.log(
            ["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"]
        )
        print_rank_0("training ...")

    def process(self):
        args = self.args
        forward_step_func = self.forward_step_func
        model, optimizer, lr_scheduler = self.model, self.optimizer, self.lr_scheduler
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            self.train_data_iterator,
            self.valid_data_iterator,
            self.test_data_iterator,
        )

        iteration = 0

        def pack_for_bwk(tensor):
            packed = [tensor.device, tensor]
            timers = get_timers()
            timers("offloading-func-call-overhead").start()
            if hasattr(torch, "_gl_current_layer") and torch._gl_transformer_fwd:

                # avoid the tensors in the pooling layers are saved into the last transfromer layer
                # todo. @gl.
                #if len(torch._gl_current_layer._gl_save_for_backward) >= 4: 
                #    return packed

                torch._gl_current_layer._gl_save_for_backward.append(packed)
                print(f'layer = {torch._gl_current_layer._gl_layer_num}, {len(torch._gl_current_layer._gl_save_for_backward)}')

                gl_warmup_print(
                    f"---- pack save_for_backward - layer={torch._gl_current_layer._gl_layer_num}",
                    f";\n\t id_packed={id(packed)}; len={len(torch._gl_current_layer._gl_save_for_backward)};",
                    f";\n\t device={tensor.device}, size={tensor.size()}, id={id(tensor)}",
                )

            timers("offloading-func-call-overhead").stop()
            return packed

        def unpack_on_bwk(packed):
            device, tensor = packed
            # print(f'----unpack save_for_backward - id={id(packed)}, device={tensor.device}, {id(tensor)}')
            assert str(tensor.device) == str(
                device
            ), "---- error: unpack_on_bwk, should be loaded to GPU ---"
            return tensor

            # if str(tensor.device) != str(device):
            #    timers = get_timers()
            #    timers('offloading-bwd-overhead').start()
            #    timers('offloading-bwd-sfb-overhead').start()
            #
            #    print(f'---- error: unpack save_for_backward ',
            #            f';\n\t - id_packed={id(packed)}, device={tensor.device}, id={id(tensor)}', flush=True)
            #    cuda_tensor = tensor.to(device, non_blocking=True)

            #    timers('offloading-bwd-sfb-overhead').stop()
            #    timers('offloading-bwd-overhead').stop()
            #    return cuda_tensor
            # else:
            #    return tensor

        torch._gl_transformer_fwd = False
        #with torch.autograd.graph.saved_tensors_hooks(pack_for_bwk, unpack_on_bwk):
        #with torch.autograd.graph.save_on_cpu(pin_memory=True):
        if args.do_train and args.train_iters > 0:
            with torch.cuda.stream(torch.cuda.Stream()):
                iteration = train(
                    forward_step_func,
                    model,
                    optimizer,
                    lr_scheduler,
                    train_data_iterator,
                    valid_data_iterator,
                )
        print_datetime("after training is done")
        #import os
        #os._exit(0)
        #return 0

        # refine later
        if False and args.do_valid:
            prefix = "the end of training for val data"
            evaluate_and_print_results(
                prefix, forward_step_func, valid_data_iterator, model, iteration, False
            )

        if False and args.save and iteration != 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)

        if False and args.do_test:
            # Run on test data.
            prefix = "the end of training for test data"
            evaluate_and_print_results(
                prefix, forward_step_func, test_data_iterator, model, 0, True
            )

    def print_model(self):
        print(self.model)
        # print(self.model[0].module.module.language_model.encoder)

    # =====================================================
    # == basic actions for hooks of forward and backward ==
    # =====================================================
    def _layer_to_cuda_remote(self, *args, **kwargs):
        return self.executor.submit(self._layer_to_cuda, *args, **kwargs)

    def _layer_to_cpu_remote(self, *args, **kwargs):
        return self.executor.submit(self._layer_to_cpu, *args, **kwargs)

    def _layer_move_save_for_backward_to_remote(self, *args, **kwargs):
        return self.executor.submit(self._layer_move_save_for_backward_to, *args, **kwargs)

    def _layer_to_cpu_and_gather_grads_and_optimizer_update_remote(self, *args, **kwargs):
        return self.executor.submit(self._layer_to_cpu_and_gather_grads_and_optimizer_update, *args, **kwargs)

    def _layer_reset_save_for_backward_remote(self, *args, **kwargs):
        return self.executor.submit(self._layer_reset_save_for_backward, *args, **kwargs)

    def _layer_gather_grads_and_optimizer_update_remote(self, *args, **kwargs):
        return self.executor.submit(self._layer_gather_grads_and_optimizer_update, *args, **kwargs)


    def _model_to(self, device):
        _models = self.model if isinstance(self.model, list) else [self.model]

        _stream = torch.cuda.Stream()
        with torch.cuda.stream(_stream):
            if _is_cpu(device):
                for model_module in _models:
                    model_module.cpu()
            else:
                for model_module in _models:
                    model_module.cuda(device)

            _stream.synchronize()

    def _assert_language_model(self):
        _models = self.model if isinstance(self.model, list) else [self.model]

        if self._language_model is None:
            for name, module in _models[0].named_modules():
                if name.endswith("language_model"):
                    self._language_model = module
                    break

        assert (
            self._language_model is not None
        ), "--- _assert_language_model training.py --- language_model not found !"

    def _get_layer(self, layer_num):
        _model = self.model[0] if isinstance(self.model, list) else self.model

        self._assert_language_model()

        _transformer_layers = self._language_model.encoder.layers
        return _transformer_layers[layer_num]

    def _get_layers(self):
        self._assert_language_model()

        _transformer_layers = self._language_model.encoder.layers
        return _transformer_layers

    def _layer_to(self, layer_num, device, non_blocking=True):
        with torch.no_grad():
            _layer = self._get_layer(layer_num)

            _stream = torch.cuda.Stream()
            with torch.cuda.stream(_stream):
                if _is_cpu(device):
                    _layer.cpu(non_blocking=non_blocking)
                else:
                    _layer.cuda(device, non_blocking=non_blocking)

                _stream.synchronize()
        pass

    @torch.no_grad()
    def _layer_to_cuda(self, layer_num):
        _layer = self._get_layer(layer_num)

        _stream = torch.cuda.Stream()
        with torch.cuda.stream(_stream):
            _layer.cuda(torch.cuda.current_device(), non_blocking=True)

            _stream.synchronize()
        pass

    @torch.no_grad()
    def _layer_to_cpu(self, layer_num):
        _layer = self._get_layer(layer_num)

        _stream = torch.cuda.Stream()
        with torch.cuda.stream(_stream):
            _layer.cpu(non_blocking=True)

            _stream.synchronize()
        pass

    def _layer_to_cpu_and_gather_grads(self, layer_num):
        with torch.no_grad():
            _layer = self._get_layer(layer_num)

            _stream = torch.cuda.Stream()
            with torch.cuda.stream(_stream):
                _layer.cpu(non_blocking=True)
                _stream.synchronize()

                if _layer._gl_fp16:
                    for param in _layer.parameters():
                        if param is None or param.grad is None:
                            continue
                        if param.grad.data is not None:
                            param.main_grad.add_(param.grad.data)
                            # Now we can deallocate grad memory.
                            # _free_cuda_tensor(param.grad)
                            param.grad = None

                    _stream.synchronize()
        pass

    def _layer_reset_save_for_backward(self, layer_num):
        _layer = self._get_layer(layer_num)
        _cpu_cache = _layer._gl_save_for_backward_cpu_cache

        if len(_cpu_cache) > 0:
            # the layers have cpu cache
            for i, _packed in enumerate(_layer._gl_save_for_backward):
                if _packed[1] is None:
                    continue

                gl_warmup_print(
                    f"--- reset - save_for_backward: index={i}; layer={layer_num}",
                    f";\n\t id_packed={id(_packed)}; len(_cpu_cache)={len(_cpu_cache)}",
                    f";\n\t device={_packed[1].device}; size={_packed[1].size()}; id={id(_packed[1])}; \n",
                )

                # _packed.pop() # del _packed[1]
                # _packed.append(_cpu_cache[i] if i in _cpu_cache else None)
                _packed[1] = _cpu_cache[i] if i in _cpu_cache else None

        # the last last 'window_size' layers have no cpu_cache for save_for_backward tensors
        _layer._gl_save_for_backward.clear()

        pass

    def _layer_move_save_for_backward_to(
        self, layer_num, device, action, index, non_blocking=True
    ):
        with torch.no_grad():
            _layer = self._get_layer(layer_num)

            _stream = torch.cuda.Stream()
            with torch.cuda.stream(_stream):
                if action == 'g2c':
                    _save_for_backward = _layer._gl_save_for_backward

                    if len(_save_for_backward) == 0:
                        # no offloading actions for the first window_size layers
                        return

                    packed = _save_for_backward[index]

                    tensor_device, tensor = packed

                    if _is_cpu(tensor_device):
                        # only offloading cuda tensors to cpu.
                        # no actions for cpu tensors or None type
                        return

                    _cpu_cache = _layer._gl_save_for_backward_cpu_cache

                    if index not in _cpu_cache:
                        gl_warmup_print(
                            f"--- init cpu_cache for save_for_backward tensors ",
                            f"in layer-{_layer._gl_layer_num}",
                        )
                        _cpu_cache[index] = torch.empty(
                            tensor.size(),
                            dtype=tensor.dtype,
                            layout=tensor.layout,
                            device=torch.device("cpu"),
                            pin_memory=True,
                        )  # (torch.cuda.is_available() and not tensor.is_sparse))
                    else:
                        gl_warmup_print(
                            f"--- already allocated cpu_cache for ",
                            f"save_for_backward tensors in layer-{_layer._gl_layer_num}",
                        )
                        pass

                    # _free_cuda_tensor(packed[1])
                    _cpu_cache[index] = _get_item_from_cpu_cache(_cpu_cache, index).copy_(
                        tensor, non_blocking=True
                    )  # non_blocking=non_blocking
                    packed[1] = _cpu_cache[index]

                    # _free_cuda_tensor
                    _save_for_backward[index] = [tensor_device, None]

                    _move_item_to_nvme(_cpu_cache, index)
                    
                elif action == 'c2g':
                    _save_for_backward = _layer._gl_save_for_backward

                    if len(_save_for_backward) == 0:
                        # no offloading actions for the first window_size layers
                        return

                    packed = _save_for_backward[index]

                    tensor_device, tensor = packed

                    if not _is_cpu(tensor_device):
                        # only offloading cuda tensors to cpu.
                        # no actions for cpu tensors or None type
                        return
                 
                    _cpu_cache = _layer._gl_save_for_backward_cpu_cache
                    
                    packed[1] = _get_item_from_cpu_cache(_cpu_cache, index).to(
                        tensor_device, non_blocking=True
                    )  # non_blocking=non_blocking
                        
                    #packed[1] = _cpu_cache[index].to(
                    #    tensor_device, non_blocking=True
                    #)  # non_blocking=non_blocking
                else:
                    # sometimes, maybe an error
                    pass

                _stream.synchronize()
        pass

    def _layer_move_main_grads_to(self, layer_num, device, non_blocking=True):
        assert (
            False
        ), "todo. @gl. now, we put the main_grads of the first window layers at GPU side"
        pass

    def _layer_move_main_grads_to_cpu(self, layer_num, non_blocking=True):
        assert (
            False
        ), "todo. @gl. now, we put the main_grads of the first window layers at GPU side"
        pass

    def _layer_move_main_grads_to_cuda(self, layer_num, non_blocking=True):
        assert (
            False
        ), "todo. @gl. now, we put the main_grads of the first window layers at GPU side"
        pass

    def _layer_optimizer_update(self, layer_num, non_blocking=True):
        self.optimizer.layer_update(layer_num)

    # backward hook:
    # for the first window layers but no including layer-0
    def _layer_gather_grads_and_optimizer_update_and_offloading_grads(
        self, layer_num, non_blocking=True
    ):
        assert (
            False
        ), "todo. @gl. now, we put the main_grads of the first window layers at GPU side"
        _layer = self._get_layer(layer_num)

        _stream = torch.cuda.Stream()
        with torch.cuda.stream(_stream):
            if _layer._gl_fp16:
                for param in _layer.parameters():
                    if param is None or param.grad is None:
                        continue
                    if param.grad.data is not None:
                        param.main_grad.add_(param.grad.data)
                        # Now we can deallocate grad memory.
                        # _free_cuda_tensor(param.grad)
                        param.grad = None
                _stream.synchronize()

            self._layer_optimizer_update(layer_num)

            if _layer._gl_fp16:
                self._layer_move_main_grads_to_cpu(layer_num)
        pass

    # backward hook:
    def _layer_gather_grads_and_optimizer_update(self, layer_num, non_blocking=True):
        _layer = self._get_layer(layer_num)

        _stream = torch.cuda.Stream()
        with torch.cuda.stream(_stream):
            if _layer._gl_fp16:
                for param in _layer.parameters():
                    if param is None or param.grad is None:
                        continue
                    if param.grad.data is not None:
                        param.main_grad.add_(param.grad.data)
                        # Now we can deallocate grad memory.
                        # _free_cuda_tensor(param.grad)
                        param.grad = None
                _stream.synchronize()

            self._layer_optimizer_update(layer_num)
        pass

    # backward hook:
    # for the layers excluding the first window layers
    def _layer_to_cpu_and_gather_grads_and_optimizer_update(
        self, layer_num, non_blocking=True
    ):
        self._layer_to_cpu_and_gather_grads(layer_num)
        # print(f" ---------> {layer_num} _layer_to_cpu_and_gather_grads is done")
        if hasattr(torch, "_gl_is_last_batch"):
            self._layer_optimizer_update(layer_num)

    # =====================================================
    # ==              basic actions ending               ==
    # =====================================================

    def register_pretrain_handler_for_layers(self, handler):
        for layer_num in range(self.num_layers):
            self._get_layer(layer_num)._gl_handler = handler

    def register_gl_properties_for_layers(self):
        # register offloading order between different layers.
        _num_layers = self.num_layers
        _window_size = self.gl_window_size
        _checkpoint_chunk_size = self.args.activations_checkpoint_num_layers

        for _cur_layer_num in range(_num_layers):
            _layer = self._get_layer(_cur_layer_num)

            #  cuda device
            _layer._gl_cuda_device = torch.cuda.current_device()

            # layer number
            _layer._gl_layer_num = _cur_layer_num
            _layer._gl_window_size = _window_size
            _layer._gl_fp16 = self.args.fp16

            # activations-checkpoint-num-layers
            _layer._gl_checkpoint_chunk_size = _checkpoint_chunk_size

            # record layers
            _layer._gl_layers = self._get_layers()

            # save_for_backward
            _layer._gl_save_for_backward = deque()
            _layer._gl_save_for_backward_cpu_cache = OrderedDict()

            # candidate layer that should be offloaded to GPU
            #   before the forward process of current layer
            _which_layer_to_cuda_pre_fwd = min(
                _cur_layer_num + _window_size, _num_layers - 1
            )
            _layer._gl_which_layer_to_cuda_pre_fwd = _which_layer_to_cuda_pre_fwd

            _layer._gl_which_layer_to_cuda_pre_fwd_required = (
                _cur_layer_num < _num_layers - _window_size
            )

            # candidate layer that should be offloaded to CPU
            #   after the forward process of current layer
            _which_layer_to_cpu_post_fwd = min(
                _cur_layer_num, _num_layers - _window_size - 1
            )
            _layer._gl_which_layer_to_cpu_post_fwd = _which_layer_to_cpu_post_fwd

            _layer._gl_which_layer_to_cpu_post_fwd_required = (
                _cur_layer_num < _num_layers - _window_size
            )

            # candidate layer that should be offloaded to GPU
            #   before the backward process of current layer
            _which_layer_to_cuda_pre_bwd = max(_cur_layer_num - _window_size, 0)
            _layer._gl_which_layer_to_cuda_pre_bwd = _which_layer_to_cuda_pre_bwd

            _layer._gl_which_layer_to_cuda_pre_bwd_required = (
                _cur_layer_num >= _window_size
            )

            # candidate layer that should be offloaded to CPU
            #   after the backward process of current layer

            # _which_layer_to_cpu_pre_bwd = max((_cur_layer_num + 1) % _num_layers, _window_size) \
            #                         if _cur_layer_num != _num_layers - 1 else 0
            # _layer._gl_which_layer_to_cpu_pre_bwd = _which_layer_to_cpu_pre_bwd

            # _layer._gl_which_layer_to_cpu_pre_bwd_required = \
            #     (_cur_layer_num + 1) % _num_layers >= _window_size

            _which_layer_to_cpu_post_bwd = max(_cur_layer_num, _window_size)
            _layer._gl_which_layer_to_cpu_post_bwd = _which_layer_to_cpu_post_bwd

            _layer._gl_which_layer_to_cpu_post_bwd_required = (
                _cur_layer_num >= _window_size
            )

            _layer._gl_is_at_last_window = (
                not _layer._gl_which_layer_to_cuda_pre_fwd_required
            )
            _layer._gl_is_at_first_window = (
                not _layer._gl_which_layer_to_cpu_post_bwd_required
            )

            """
            Example: 12 layers and 4 window_size
            layer:          00 01 02 03 04 05 06 07 08 09 10 11
            cuda_pre_fwd:   04 05 06 07 08 09 10 11 11 11 11 11
                            T  T  T  T  T  T  T  T  F  F  F  F
            cpu_post_fwd:   00 01 02 03 04 05 06 07 07 07 07 07
                            T  T  T  T  T  T  T  T  F  F  F  F
            cuda_pre_bwd:   00 00 00 00 00 01 02 03 04 05 06 07
                            F  F  F  F  T  T  T  T  T  T  T  T
            cpu_pre_bwd:    04 04 04 04 05 06 07 08 09 10 11 00
                            F  F  F  T  T  T  T  T  T  T  T  F
            cpu_post_bwd:   04 04 04 04 04 05 06 07 08 09 10 11
                            F  F  F  F  T  T  T  T  T  T  T  T 
            """

            #print(
            #    f"--- layer={_cur_layer_num}",
            #    ";\n\t _is_at_first_window",
            #    _layer._gl_is_at_first_window,
            #    "; _is_at_last_window",
            #    _layer._gl_is_at_last_window,
            #    ";\n\t _to_cuda_pre_fwd:",
            #    _which_layer_to_cuda_pre_fwd,
            #    _layer._gl_which_layer_to_cuda_pre_fwd_required,
            #    ";\n\t _to_cpu_post_fwd:",
            #    _which_layer_to_cpu_post_fwd,
            #    _layer._gl_which_layer_to_cpu_post_fwd_required,
            #    ";\n\t _to_cuda_pre_bwd:",
            #    _which_layer_to_cuda_pre_bwd,
            #    _layer._gl_which_layer_to_cuda_pre_bwd_required,
            #    ";\n\t _to_cpu_post_bwd:",
            #    _which_layer_to_cpu_post_bwd,
            #    _layer._gl_which_layer_to_cpu_post_bwd_required,
            #)

    def register_hooks(self):
        _num_layers = self.num_layers

        # register forward and backward hooks
        for _layer_num in range(_num_layers):
            _layer = self._get_layer(_layer_num)

            # register forward_pre_hooks
            _layer.register_forward_pre_hook(_forward_pre_hook)

            # register forward_post_hooks
            _layer.register_forward_hook(_forward_post_hook)

            # register backward_pre_hooks
            if _layer_num != _num_layers - 1:
                _layer.register_full_backward_hook(
                    _backward_post_hook
                )  # warning!! for next layer
            _layer.register_full_backward_hook(_backward_pre_hook)

        pass

    # =====================================================
    # ===              register ending                  ===
    # =====================================================


def gl_pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    extra_args_provider=None,
    args_defaults={},
):

    set_global_variables(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        ignore_unknown_args=False,
    )

    args = get_args()
    timers = get_timers()

    print(f">-- rank={args.rank}; local_rank={args.local_rank};")
    pretrain_handler = GL_PretrainHanlder(
        args,
        train_valid_test_dataset_provider,
        model_provider,
        model_type,
        forward_step_func)


    pretrain_handler.register_pretrain_handler_for_layers(pretrain_handler)
    
    pretrain_handler.register_gl_properties_for_layers()
    pretrain_handler.register_hooks()

    pretrain_handler.process()
    return 0


def pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    extra_args_provider=None,
    args_defaults={},
):
    # ----------- for gl version ----------
    args = parse_args()
    if args.enable_gl:
        gl_pretrain(
            train_valid_test_dataset_provider,
            model_provider,
            model_type,
            forward_step_func,
            extra_args_provider=extra_args_provider,
            args_defaults=args_defaults,
        )
        return 0
    # -------------------------------------

    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        extra_args_provider=extra_args_provider, args_defaults=args_defaults
    )

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - _TRAIN_START_TIME
        )
    )
    print_datetime("after megatron is initialized")

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup").start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        model_provider, model_type
    )
    timers("model-and-optimizer-setup").stop()
    print_datetime("after model, optimizer, and learning rate " "scheduler are built")

    # Data stuff.
    timers("train/valid/test-data-iterators-setup").start()
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            for _ in range(len(model))
        ]
        train_data_iterator = [
            data_iterators[0] for data_iterators in all_data_iterators
        ]
        valid_data_iterator = [
            data_iterators[1] for data_iterators in all_data_iterators
        ]
        test_data_iterator = [
            data_iterators[2] for data_iterators in all_data_iterators
        ]
    else:
        (
            train_data_iterator,
            valid_data_iterator,
            test_data_iterator,
        ) = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
    timers("train/valid/test-data-iterators-setup").stop()
    print_datetime("after dataloaders are built")

    # Print setup timing.
    print_rank_0("done with setup ...")
    timers.log(["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"])
    print_rank_0("training ...")

    iteration = 0
    if args.do_train and args.train_iters > 0:
        iteration = train(
            forward_step_func,
            model,
            optimizer,
            lr_scheduler,
            train_data_iterator,
            valid_data_iterator,
        )
    print_datetime("after training is done")

    if False and args.do_valid:
        prefix = "the end of training for val data"
        evaluate_and_print_results(
            prefix, forward_step_func, valid_data_iterator, model, iteration, False
        )

    if False and args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler)

    if False and args.do_test:
        # Run on test data.
        prefix = "the end of training for test data"
        evaluate_and_print_results(
            prefix, forward_step_func, test_data_iterator, model, 0, True
        )


def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // args.global_batch_size
        args.train_iters = iterations

    print_rank_0("setting training iterations to {}".format(args.train_iters))


def get_model(
    model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True
):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if (
        mpu.get_pipeline_model_parallel_world_size() > 1
        and args.virtual_pipeline_model_parallel_size is not None
    ):
        assert (
            model_type != ModelType.encoder_and_decoder
        ), "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process, post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert (
                    args.pipeline_model_parallel_split_rank is not None
                ), "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder,
            )
        else:
            model = model_provider_func(
                pre_process=pre_process, post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) "
            "model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum(
                    [
                        sum([p.nelement() for p in model_module.parameters()])
                        for model_module in model
                    ]
                ),
            ),
            flush=True,
        )

    # GPU allocation.
    if args.enable_gl:

        def _is_leaf_module(module):
            return list(module.children()) == []

        for model_module in model:
            # todo @gl -------------------------------shit- must
            """
            File "/nas-alinlp/robot.sxy/AliDamoNLP/Megatron-LM/megatron/model/fused_layer_norm.py", line 41, in forward
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
            RuntimeError: expected scalar type Half but found Float
            """
            if args.fp16:
                model_module.half()

            model_module.cpu()

        cuda_device = torch.cuda.current_device()

        # _cpu_list = []
        # for i in range(args.gl_window_size, args.num_layers):
        #    _cpu_list.append(f'coder.layers.{i}')

        # for model_module in model:
        #    for name, module in model_module.named_modules():
        #        if _is_leaf_module(module):
        #    model_module.cuda(cuda_device)

        for model_module in model[1:]:
            model_module.cuda(cuda_device)

        assert isinstance(
            model, list
        ), "the model var should be a list consisting several torch Module"
        BGModel = model[0]

        if isinstance(BGModel, BertModel):
            BGModel.binary_head.cuda(cuda_device)
            BGModel.lm_head.cuda(cuda_device)
            BGModel.language_model.embedding.cuda(cuda_device)
            BGModel.language_model.pooler.cuda(cuda_device)
            BGModel.language_model.encoder.final_layernorm.cuda(cuda_device)
        elif isinstance(BGModel, GPTModel):
            BGModel.language_model.embedding.cuda(cuda_device)
            BGModel.language_model.encoder.final_layernorm.cuda(cuda_device)
        else:
            raise Exception("Please init model using GPT or Bert.")

        for i in range(args.num_layers):
            if i < args.gl_window_size:
                #print(f"      moving layer-{i} to cuda")
                BGModel.language_model.encoder.layers[i].cuda(cuda_device)
            else:
                #print(f"skipping layer-{i}     to cuda")
                BGModel.language_model.encoder.layers[i].cpu()

        for i in range(args.num_layers):
            _layer = BGModel.language_model.encoder.layers[i]

            info = f"layer {i}. parameters"
            for key, value in _layer.named_parameters():
                info += f"\n\t {key}: {value.device}"
            #print(info)

            info = f"layer {i}. _gl_cpu_version_items"
            for key, value in _layer._gl_cpu_version_items.items():
                info += f"\n\t {key}: {value.device}"
                assert str(value.device) == "cpu"
            #print(info)

            # for name, module in model_module.named_modules():
            #    # only move leaf module to CUDA
            #    if not _is_leaf_module(module):
            #        continue

            #    if 'coder.layers.' in name:
            #        _nl = name.split('.')
            #        layer_num = int(_nl[_nl.index('layers')+1])

            #        if layer_num >= args.gl_window_size:
            #            print(f'skipping {name} to CUDA')
            #            module.cpu()
            #            continue

            #    print(f'moving {name} to cuda')
            #    module.cuda(cuda_device)
    elif args.enable_l2l:
        def _forward_pre_hook(module, input):
            # print('--------- at pre fwd')   
            if hasattr(torch, '_gl_is_backward_now') and torch._gl_is_backward_now:
                # print('--------- at pre fwd ---- to cpu for bwd') 
                torch._current_layers.cpu()  
            module.cuda()

        def _forward_post_hook(module, input, output):
            # print('--------- at post fwd')
            if hasattr(torch, '_gl_is_backward_now') and torch._gl_is_backward_now:
                pass
            else:
                module.cpu()

        for model_module in model:
            for module_name, module in model_module.named_modules():
                if module_name == 'language_model.encoder.layers':                    
                    torch._current_layers = module
                    #register L2L offloading strategy
                    for child_name, child in module.named_children():
                        child.register_forward_pre_hook(_forward_pre_hook)
                        child.register_forward_hook(_forward_post_hook)
                        child.cpu()

                if module_name.startswith('language_model.encoder.layers'):
                    # skip moving to cuda
                    module.cpu()
                    pass
                else:
                    module.cuda(torch.cuda.current_device())
    else:
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    #    for name, param in model[0].named_parameters():
    #        print(name, param)

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if args.enable_gl:
        if args.gl_enable_ddp:
            if args.DDP_impl == "torch":
                i = torch.cuda.current_device()
                model = [
                    torchDDP(
                        model_module,
                        device_ids=[i],
                        output_device=i,
                        process_group=mpu.get_data_parallel_group(),
                    )
                    for model_module in model
                ]

            elif args.DDP_impl == "local":
                model = [
                    LocalDDP(
                        model_module,
                        args.accumulate_allreduce_grads_in_fp32,
                        args.use_contiguous_buffers_in_local_ddp,
                    )
                    for model_module in model
                ]

            else:
                raise NotImplementedError(
                    "Unknown DDP implementation specified: "
                    "{}. Exiting.".format(args.DDP_impl)
                )

    else:
        if wrap_with_ddp:
            if args.DDP_impl == "torch":
                i = torch.cuda.current_device()
                model = [
                    torchDDP(
                        model_module,
                        device_ids=[i],
                        output_device=i,
                        process_group=mpu.get_data_parallel_group(),
                    )
                    for model_module in model
                ]

            elif args.DDP_impl == "local":
                model = [
                    LocalDDP(
                        model_module,
                        args.accumulate_allreduce_grads_in_fp32,
                        args.use_contiguous_buffers_in_local_ddp,
                    )
                    for model_module in model
                ]

            else:
                raise NotImplementedError(
                    "Unknown DDP implementation specified: "
                    "{}. Exiting.".format(args.DDP_impl)
                )

    return model


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        decay_steps = args.lr_decay_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        decay_steps = args.lr_decay_samples
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_samples
    else:
        raise Exception("either train-iters or train-samples should be provided.")

    lr_scheduler = AnnealingLR(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler,
    )

    return lr_scheduler


def setup_model_and_optimizer(model_provider_func, model_type):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func, model_type)

    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    # unwrapped_model[0] is BertModel/GPTModel

    print(
        f"--- init model: rank={mpu.get_tensor_model_parallel_rank()} ",
        f" world-size={mpu.get_tensor_model_parallel_world_size()}---",
    )

    if mpu.get_data_parallel_rank() == 0:
        billion_params = get_parameters_in_billions(unwrapped_model[0])
        print(
            f" > number of parameters on pipeline model parallel rank {mpu.get_pipeline_model_parallel_rank()}, \
            tensor model parallel rank {mpu.get_tensor_model_parallel_rank()} \
            {round(billion_params, 3)} Billion",
            flush=True,
        )

    _param_count = lambda m: sum([_.numel() for _ in m.parameters()])
    _param_sum = _param_count(unwrapped_model[0])
    #for module_name, module in unwrapped_model[0].named_modules():
    #    print(
    #        f"{module_name}:  params_num={_param_count(module)}, ratio={round(_param_count(module)/_param_sum * 100, 4)}%"
    #    )
    #    for param_name, param in module.named_parameters():
    #        print(
    #            f"\t  --- param name={param_name}, size={param.size()}, numel={param.numel()}"
    #        )

    if args.enable_gl:
        optimizer = gl_get_megatron_optimizer(unwrapped_model)
        lr_scheduler = get_learning_rate_scheduler(optimizer.get_optimizer("default"))
    elif args.enable_l2l:
        optimizer = get_megatron_optimizer(unwrapped_model, cpu=True)
        lr_scheduler = get_learning_rate_scheduler(optimizer)
    else:
        optimizer = get_megatron_optimizer(unwrapped_model)
        lr_scheduler = get_learning_rate_scheduler(optimizer)

    if args.load is not None:
        timers = get_timers()
        # Extra barrier is added to make sure all ranks report the
        # max time.
        torch.distributed.barrier()
        timers("load-checkpoint").start()
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
        torch.distributed.barrier()
        timers("load-checkpoint").stop()
        timers.log(["load-checkpoint"])
    else:
        args.iteration = 0

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == "local"

    # get model without FP16 and/or TorchDDP wrappers
    if (
        args.iteration == 0
        and len(unwrapped_model) == 1
        and hasattr(unwrapped_model[0], "init_state_dict_from_bert")
    ):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, lr_scheduler


def train_step(forward_step_func, data_iterator, model, optimizer, lr_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    if (
        args.DDP_impl == "local"
        and args.use_contiguous_buffers_in_local_ddp
        and args.gl_enable_ddp
    ):
        for partition in model:
            partition.zero_grad_buffer()
    optimizer.zero_grad()

    forward_backward_func = get_forward_backward_func()

    losses_reduced = forward_backward_func(
        forward_step_func, data_iterator, model, optimizer, timers, forward_only=False
    )

    # Empty unused memory
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # All-reduce if needed.
    if args.DDP_impl == "local" and args.gl_enable_ddp:
        timers("backward-params-all-reduce").start()
        for model_module in model:
            model_module.allreduce_gradients()
        timers("backward-params-all-reduce").stop()

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    # This should only run for models that support pipelined model parallelism
    # (BERT and GPT-2).
    timers("backward-embedding-all-reduce").start()
    if (
        mpu.is_rank_in_embedding_group(ignore_virtual=True)
        and mpu.get_pipeline_model_parallel_world_size() > 1
    ):
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            unwrapped_model = model[0]
        elif mpu.is_pipeline_last_stage(ignore_virtual=True):
            unwrapped_model = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            unwrapped_model = model[0]
        unwrapped_model = unwrap_model(
            unwrapped_model, (torchDDP, LocalDDP, Float16Module)
        )

        if unwrapped_model.share_word_embeddings:
            word_embeddings_weight = unwrapped_model.word_embeddings_weight()
            if args.DDP_impl == "local" and args.gl_enable_ddp:
                grad = word_embeddings_weight.main_grad
            else:
                grad = word_embeddings_weight.grad
            torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())
    timers("backward-embedding-all-reduce").stop()

    # Update parameters.
    # if args.dprint:
    #    debug_info = "where is the paramaters? \n"
    #    model[0].cpu()
    #    for name, param in model[0].named_parameters():
    #        if param is None or param.grad is None:
    #            continue
    #        debug_info += f"\n\t | {name}: {param.device}, grads: {param.grad.device}"

    #    osd = optimizer.state_dict()
    #    #print(osd['optimizer'])
    #    #print(osd['grad_scaler'])
    #    #print(osd['fp32_from_fp16_params'])
    #    for tl in osd['fp32_from_fp16_params']:
    #        for t in tl:
    #            t = t.cpu()

    #    model[0].cuda()
    #    for name, param in model[0].named_parameters():
    #        if param is None or param.grad is None:
    #            continue
    #        debug_info += f"\n\t | {name}: {param.device}, grads: {param.grad.device}"

    #    debug_print(debug_info)
    #    print(optimizer.state_dict())

    #    exit()

    timers("optimizer").start()
    if args.enable_gl:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        # print(f'upading parameters ---- optimizer.step ---- {update_successful}, {grad_norm}, {num_zeros_in_grad}')
    else:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers("optimizer").stop()

    # Update learning rate.
    if update_successful:
        increment = (
            get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        )
        lr_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(
                losses_reduced_for_key
            )
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad

    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
    model,
):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = (
            total_loss_dict.get(advanced_iters_key, 0) + 1
        )
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = (
        total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    )
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = (
                total_loss_dict.get(key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
            )
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(
        got_nan
    )

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)

    add_to_logging("e2e-time")
    add_to_logging("forward-compute")
    add_to_logging("forward-recv")
    add_to_logging("forward-send")
    add_to_logging("forward-backward-send-forward-backward-recv")
    add_to_logging("backward-compute")
    add_to_logging("backward-recv")
    add_to_logging("backward-send")
    add_to_logging("backward-send-forward-recv")
    add_to_logging("backward-send-backward-recv")
    add_to_logging("backward-params-all-reduce")
    add_to_logging("backward-embedding-all-reduce")
    add_to_logging("optimizer-copy-to-main-grad")
    add_to_logging("optimizer-unscale-and-check-inf")
    add_to_logging("optimizer-clip-main-grad")
    add_to_logging("optimizer-copy-main-to-model-params")
    add_to_logging("optimizer")
    add_to_logging("batch-generator")

    if args.enable_gl:
        add_to_logging("offloading-func-call-overhead")
        add_to_logging("offloading-fwd-overhead")
        add_to_logging("offloading-bwd-overhead")
        add_to_logging("offloading-fwd-2gpu-overhead")
        add_to_logging("offloading-fwd-2cpu-overhead")
        add_to_logging("offloading-bwd-sfb-overhead")
        add_to_logging("offloading-bwd-2gpu-overhead")
        add_to_logging("offloading-bwd-2cpu-overhead")

    # Calculate batch size.
    batch_size = (
        args.micro_batch_size * args.data_parallel_size * get_num_microbatches()
    )

    total_iterations = (
        total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]
    )

    # Tensorboard values.
    if writer and (iteration % args.tensorboard_log_interval == 0) and is_last_rank():
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar("learning-rate", learning_rate, iteration)
            writer.add_scalar(
                "learning-rate vs samples", learning_rate, args.consumed_train_samples
            )
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar("batch-size", batch_size, iteration)
            writer.add_scalar(
                "batch-size vs samples", batch_size, args.consumed_train_samples
            )
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
            writer.add_scalar(
                key + " vs samples", loss_dict[key], args.consumed_train_samples
            )
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar("loss-scale", loss_scale, iteration)
            writer.add_scalar(
                "loss-scale vs samples", loss_scale, args.consumed_train_samples
            )
        if grad_norm is not None:
            writer.add_scalar("grad-norm", grad_norm, iteration)
            writer.add_scalar(
                "grad-norm vs samples", grad_norm, args.consumed_train_samples
            )
        if num_zeros_in_grad is not None:
            writer.add_scalar("num-zeros", num_zeros_in_grad, iteration)
            writer.add_scalar(
                "num-zeros vs samples", num_zeros_in_grad, args.consumed_train_samples
            )
        if params_norm is not None:
            writer.add_scalar("params-norm", params_norm, iteration)
            writer.add_scalar(
                "params-norm vs samples", params_norm, args.consumed_train_samples
            )
        if args.log_timers_to_tensorboard:
            timers.write(timers_to_log, writer, iteration, normalizer=total_iterations)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )

    if iteration % args.log_interval == 0:
        elapsed_time = timers("interval-time").elapsed()
        elapsed_time_per_iteration = elapsed_time / total_iterations
        if writer:
            if args.log_timers_to_tensorboard:
                writer.add_scalar(
                    "iteration-time", elapsed_time_per_iteration, iteration
                )
        log_string = " iteration {:8d}/{:8d} |".format(iteration, args.train_iters)
        #log_string += " consumed samples: {:12d} |".format(args.consumed_train_samples)
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(
            elapsed_time_per_iteration * 1000.0
        )
        log_string += " learning rate: {:.3E} |".format(learning_rate)
        log_string += " global batch size: {:5d} |".format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(
                    max(1, total_loss_dict[advanced_iters_key])
                )
                if avg > 0.0:
                    log_string += " {}: {:.6E} |".format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += " loss scale: {:.1f} |".format(loss_scale)
        if grad_norm is not None:
            log_string += " grad norm: {:.3f} |".format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += " num zeros: {:.1f} |".format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += " params norm: {:.3f} |".format(params_norm)
        log_string += " number of skipped iterations: {:3d} |".format(
            total_loss_dict[skipped_iters_key]
        )
        log_string += " number of nan iterations: {:3d} |".format(
            total_loss_dict[nan_iters_key]
        )
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)

        flops_calculator(model, args, elapsed_time_per_iteration)
        throughput_calculator(args, elapsed_time_per_iteration)

        if report_memory_flag and learning_rate > 0.0:
            # Report memory after optimizer state has been initialized.
            report_memory("(after {} iterations)".format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    torch.distributed.barrier()
    timers("save-checkpoint").start()
    save_checkpoint(iteration, model, optimizer, lr_scheduler)
    torch.distributed.barrier()
    timers("save-checkpoint").stop()
    timers.log(["save-checkpoint"])


def train(
    forward_step_func,
    model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    valid_data_iterator,
):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    timers("interval-time").start()
    print_datetime("before the start of training step")
    report_memory_flag = True
    while iteration < args.train_iters:
        if iteration + 1 % args.log_interval == 1:
            timers("e2e-time").reset()
            timers("e2e-time").start(cuda_sync=True)

        if args.enable_gl:
            # torch._gl_in_warmup = iteration // args.log_interval < 1
            torch._gl_in_warmup = iteration < 2

        update_num_microbatches(args.consumed_train_samples)
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step(
            forward_step_func, train_data_iterator, model, optimizer, lr_scheduler
        )

        # ------ gl version ------
        # torch.cuda.empty_cache()
        # exit()

        iteration += 1
        args.consumed_train_samples += (
            mpu.get_data_parallel_world_size()
            * args.micro_batch_size
            * get_num_microbatches()
        )

        if iteration + 1 % args.log_interval == 0:
            timers("e2e-time").stop(cuda_sync=True)

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        if args.enable_gl:
            _lr = optimizer.get_optimizer("default").param_groups[0]["lr"]
        else:
            _lr = optimizer.param_groups[0]["lr"]

        report_memory_flag = training_log(
            loss_dict,
            total_loss_dict,
            _lr,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
            model,
        )

        # gl......
        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            print_datetime("exiting program at iteration {}".format(iteration))
            return 0 #sys.exit()

        # Autoresume
        if args.adlr_autoresume and (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer, lr_scheduler)

        # Evaluation
        if False and args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            prefix = "iteration {}".format(iteration)
            evaluate_and_print_results(
                prefix, forward_step_func, valid_data_iterator, model, iteration, False
            )

        # Checkpointing
        saved_checkpoint = False
        if False and args.save and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler)
            saved_checkpoint = True

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor([train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler)
                print_datetime("exiting program after {} minutes".format(train_time))
                return 0 #sys.exit()

        # Exiting based on iterations
        if False and args.exit_interval and iteration % args.exit_interval == 0:
            if not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler)
            torch.distributed.barrier()
            print_datetime("exiting program at iteration {}".format(iteration))
            return 0 #sys.exit()

    return iteration


def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0("Evaluating iter {}/{}".format(iteration, args.eval_iters))

            forward_backward_func = get_forward_backward_func()
            loss_dicts = forward_backward_func(
                forward_step_func,
                data_iterator,
                model,
                optimizer=None,
                timers=None,
                forward_only=True,
            )

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = (
                            total_loss_dict.get(key, torch.cuda.FloatTensor([0.0]))
                            + loss_dict[key]
                        )

            args.consumed_valid_samples += (
                mpu.get_data_parallel_world_size()
                * args.micro_batch_size
                * get_num_microbatches()
            )
    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    return total_loss_dict


def evaluate_and_print_results(
    prefix, forward_step_func, data_iterator, model, iteration, verbose=False
):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    string = " validation loss at {} | ".format(prefix)
    for key in total_loss_dict:
        string += "{} value: {:.6E} | ".format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += "{} PPL: {:.6E} | ".format(key, ppl)
        if writer:
            writer.add_scalar(
                "{} validation".format(key), total_loss_dict[key].item(), iteration
            )
            writer.add_scalar(
                "{} validation vs samples".format(key),
                total_loss_dict[key].item(),
                args.consumed_train_samples,
            )
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar("{} validation ppl".format(key), ppl, iteration)
                writer.add_scalar(
                    "{} validation ppl vs samples".format(key),
                    ppl,
                    args.consumed_train_samples,
                )

    length = len(string) + 1
    print_rank_last("-" * length)
    print_rank_last(string)
    print_rank_last("-" * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_train_valid_test_data_iterators(build_train_valid_test_datasets_provider):
    """XXX"""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0("> building train, validation, and test datasets ...")

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert (
            args.train_samples is None
        ), "only backward compatiblity support for iteration-based training"
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (
                (args.iteration // args.eval_interval)
                * args.eval_iters
                * args.global_batch_size
            )

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_tensor_model_parallel_rank() == 0:

        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [
            train_samples,
            eval_iters * args.global_batch_size,
            test_iters * args.global_batch_size,
        ]
        print_rank_0(" > datasets target sizes (minimum size):")
        print_rank_0("    train:      {}".format(train_val_test_num_samples[0]))
        print_rank_0("    validation: {}".format(train_val_test_num_samples[1]))
        print_rank_0("    test:       {}".format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples
        )

        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples
        )
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, args.consumed_valid_samples
        )
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(
        flags,
        mpu.get_tensor_model_parallel_src_rank(),
        group=mpu.get_tensor_model_parallel_group(),
    )
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ["single", "cyclic"]

    if train_dataloader is not None:
        train_data_iterator = (
            iter(train_dataloader)
            if dl_type == "single"
            else iter(cyclic_iter(train_dataloader))
        )
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = (
            iter(valid_dataloader)
            if dl_type == "single"
            else iter(cyclic_iter(valid_dataloader))
        )
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = (
            iter(test_dataloader)
            if dl_type == "single"
            else iter(cyclic_iter(test_dataloader))
        )
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
