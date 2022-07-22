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

import sys, os
import pathlib
import subprocess

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD

from megatron import get_args
from megatron.model import LayerNorm

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .grad_scaler import CPU_DynamicGradScaler

from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
from .optimizer import CPU_Float16OptimizerWithFloat16Params, CPU_FP32Optimizer, CPUAdam

from megatron.utils import _get_cuda_bare_metal_version, _create_build_dir
from torch.utils import cpp_extension


def _get_params_for_weight_decay_optimization(modules):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, LayerNorm):
                no_weight_decay_params['params'].extend(
                    [p for p in list(module_._parameters.values())
                     if p is not None])
            else:
                weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                     if p is not None and n != 'bias'])
                no_weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                     if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params


def get_megatron_optimizer(
    model,
    is_cpu_version=False,
    is_for_weight_decay=True):

    args = get_args()

    # Base optimizer.
    if is_for_weight_decay:
        param_groups = _get_params_for_weight_decay_optimization(model)
    else:
        is_for_weight_decay = model


    if args.optimizer == 'adam':
        if (args.enable_l2l and is_cpu_version):
            optimizer = CPUAdam(
                param_groups,
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
            )
        else:
            optimizer = Adam(
                param_groups,
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps)

    elif args.optimizer == 'sgd':
        optimizer = SGD(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.sgd_momentum)

    else:
        raise Exception('{} optimizer is not supported.'.format(
            args.optimizer))

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == 'local':
        params_have_main_grad = True

    if args.fp16 or args.bf16:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None
        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)
        # Dynamic loss scale.
        else:
            if args.fp16:
                if is_cpu_version:
                    grad_scaler = CPU_DynamicGradScaler(
                        initial_scale=args.initial_loss_scale,
                        min_scale=args.min_loss_scale,
                        growth_factor=2.0,
                        backoff_factor=0.5,
                        growth_interval=args.loss_scale_window,
                        hysteresis=args.hysteresis,
                    )
                else:
                    grad_scaler = DynamicGradScaler(
                        initial_scale=args.initial_loss_scale,
                        min_scale=args.min_loss_scale,
                        growth_factor=2.0,
                        backoff_factor=0.5,
                        growth_interval=args.loss_scale_window,
                        hysteresis=args.hysteresis)

        # Megatron optimizer.
        if is_cpu_version:
            return CPU_Float16OptimizerWithFloat16Params(
                    optimizer,
                    args.clip_grad,
                    args.log_num_zeros_in_grad,
                    params_have_main_grad,
                    args.use_contiguous_buffers_in_local_ddp,
                    args.bf16,
                    grad_scaler)
        else:
            return Float16OptimizerWithFloat16Params(
                    optimizer,
                    args.clip_grad,
                    args.log_num_zeros_in_grad,
                    params_have_main_grad,
                    args.use_contiguous_buffers_in_local_ddp,
                    args.bf16,
                    grad_scaler)

    # FP32.
    if is_cpu_version:
        return CPU_FP32Optimizer(
                optimizer,
                args.clip_grad,
                args.log_num_zeros_in_grad,
                params_have_main_grad,
                args.use_contiguous_buffers_in_local_ddp)
    else:
        return FP32Optimizer(
                optimizer, args.clip_grad,
                args.log_num_zeros_in_grad,
                params_have_main_grad,
                args.use_contiguous_buffers_in_local_ddp)


# stronghold:
#   init optimizer for each layer
def strongh_get_megatron_optimizer(model):
    class MockOneOptimizer:
        def __init__(self, optimizers):
            self.optimizers = optimizers

        def step(self):
            return self.optimizers["default"].step()

        def zero_grad(self):
            return self.optimizers["default"].zero_grad()

        def scale_loss(self, tensor):
            return self.optimizers["default"].scale_loss(tensor)

        def get_loss_scale(self):
            return self.optimizers["default"].get_loss_scale()

        def reload_model_params(self):
            for opt in self.optimizers:
                opt.reload_model_params()

        def layer_update(self, layer_num):
            res = self.layer_step(layer_num)
            self.layer_zero_grad(layer_num)

        def layer_step(self, layer_num):
            layer_num = int(layer_num)
            if layer_num not in self.optimizers:
                print(
                    f"----error/warning: strongh_get_megatron_optimizer layer_step ",
                    f"-- optimizer {layer_num} is missing !!",
                )
                return
            # print(f"-- optimizer {layer_num} ------ is at step() ")
            return self.optimizers[layer_num].step()

        def layer_zero_grad(self, layer_num):
            layer_num = int(layer_num)
            if layer_num not in self.optimizers:
                print(
                    f"----error/warning: strongh_get_megatron_optimizer layer_zero_grad ",
                    f"-- optimizer {layer_num} is missing !!",
                )
                return
            return self.optimizers[layer_num].zero_grad()

        def get_optimizer(self, name):
            if name not in self.optimizers:
                print(
                    f"----error: strongh_get_megatron_optimizer -- optimizer {name} is missing !!"
                )
                exit()
            return self.optimizers[name]

        def __getattr__(self, name: str):
            if name not in self.__dict__:
                return self.optimizers["default"].__getattr__(name)
            return self.__dict__[name]

    class SH_CPU_OptimizerWrapper:
        def __init__(self, cpu_optimzer, start_index, end_index):
            self.cpu_optimzer = cpu_optimzer
            self.start_index = start_index
            self.end_index = end_index

        def step(self):
            self.cpu_optimzer.step(self.start_index, self.end_index)

        def zero_grad(self):
            self.cpu_optimzer.zero_grad(self.start_index, self.end_index)

    args = get_args()

    optimizers = {}

    cuda_default_opt_modules = []
    cpu_opt_modules = {}

    # @gl, hard code @todo
    for module in model[1:]:
        cuda_default_opt_modules.append(module)

    assert isinstance(model, list), "the model var should be a list consisting several torch Module"
    BGModel = model[0]

    if isinstance(BGModel, BertModel):
        cuda_default_opt_modules.append(BGModel.binary_head)
        cuda_default_opt_modules.append(BGModel.lm_head)
        cuda_default_opt_modules.append(BGModel.language_model.embedding)
        cuda_default_opt_modules.append(BGModel.language_model.pooler)
        cuda_default_opt_modules.append(BGModel.language_model.encoder.final_layernorm)
        cuda_default_opt_modules.append(BGModel.language_model.encoder.layers[0])
    elif isinstance(BGModel, GPTModel):
        cuda_default_opt_modules.append(BGModel.language_model.embedding)
        cuda_default_opt_modules.append(BGModel.language_model.encoder.final_layernorm)
        cuda_default_opt_modules.append(BGModel.language_model.encoder.layers[0])
    else:
        raise Exception("Please init model using GPT or Bert.")

    # layer-0 in the default optimizer
    with torch.no_grad():
        for _module in cuda_default_opt_modules:
            _module.cuda()
            _module.requires_grad_(requires_grad=True)
            for param_name, param in _module.named_parameters():
                if not param.is_leaf:
                    print('not a leaf tensor', param_name, param.grad_fn)
    optimizers["default"] = get_megatron_optimizer(cuda_default_opt_modules)

    _layers = BGModel.language_model.encoder.layers

    # layer-1 to layer-x in the window-size have individual cuda-version optimizer
    for i in range(1, args.gl_window_size):
        _layers[i].cuda()
        print(
            f"--- init optimizer for layer={i}; ",
            f" rank={mpu.get_tensor_model_parallel_rank()} world-size={mpu.get_tensor_model_parallel_world_size()}---",
        )
        for name, param in _layers[i].named_parameters():
            print(f".  ---- layer{i}; name={name}, size={param.size()}, numel={param.numel()}")
        optimizers[i] = get_megatron_optimizer([_layers[i]], cpu=False)

    if args.gl_window_size < len(_layers):
        # layer-?s outside of the window-size have a common cpu-version optimizer,
        # but support step() and zero_grad() seperately.
        cpu_optimzer = None
        all_param_groups = []
        for i in range(args.gl_window_size, len(_layers)):
            _layers[i].cpu()
            start_index = len(all_param_groups)

            param_groups = _get_params_for_weight_decay_optimization([_layers[i]])
            all_param_groups.extend(param_groups)

            end_index = len(all_param_groups)

            print(
                f"--- Appending param groups to CPU-version Adam for layer={i}.",
                f"start_index={start_index}, end_index={end_index}",
            )

            # if cpu_optimzer is None:
            #     cpu_optimzer = get_megatron_optimizer([_layers[i]], cpu=True)
            #     start_index = 0
            # else:
            #     start_index = len(cpu_optimzer.optimizer.param_groups)
            #     param_groups = _get_params_for_weight_decay_optimization([_layers[i]])
            #     cpu_optimzer.add_param_groups(param_groups, add_to_optimizer=True)

            # init cpu_optimizer as None
            optimizers[i] = SH_CPU_OptimizerWrapper(cpu_optimzer, start_index, end_index)

        # overwrite the cpu_optimizer
        if len(all_param_groups) > 0:
            cpu_optimzer = get_megatron_optimizer(all_param_groups, cpu=True, init_from_model=False)

        for i in range(args.gl_window_size, len(_layers)):
            optimizers[i].cpu_optimzer = cpu_optimzer

        # print("cpu optimizer for layers outside of the initial offloading window: ",
        #         cpu_optimzer.state_dict())

    """
    building an optimizer for each layer, which consumes too much CPU memory.
    (for example, 480layers requires 600G; but in CUDA version, 120layers plus optimizer only needs 32G)
    # for i in range(len(layer_modules)): # no layer-0 in layer_modules
    #    print(f'building optimizer for layer-{i+1}, cpu={i+1 >= args.gl_window_size}')
    #    optimizers[i+1] = get_megatron_optimizer([layer_modules[i]],
    #            cpu = i+1 >= args.gl_window_size)
     """

    return MockOneOptimizer(optimizers)


def load_utils(args):
    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, _ = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                "-O3",
                "-fopenmp",
                "-lpython3.9m",
                "-lboost_python-py39",
                "-I /usr/local/cuda/include/",
                "-lcurand",
                # "-mavx512f -mavx512cd -mavx512dq -mavx512bw -mavx512vl ",
            ],
            extra_cuda_cflags=[
                "-O3",
                "-gencode",
                "arch=compute_86,code=sm_86",
                "--use_fast_math",
                "-I /usr/local/cuda/include/",
                "-std=c++17" if sys.platform == "win32" else "-std=c++14",
                "--default-stream per-thread",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-lcurand",
            ]
            + extra_cuda_flags
            + cc_flag,
            verbose=(args.rank == 0),
        )

    extra_cuda_flags = ["-maxrregcount=50"]

    """
    CPU version of _amp_foreach_non_finite_check_and_unscale_
    """
    sources = [
        srcpath / "cpp/cpu_ops.cpp",
    ]
    optimizer_utils = _cpp_extention_load_helper(
        "optimizer_utils", sources, extra_cuda_flags
    )

    sources = [
        srcpath / "cpp/ds/custom_cuda_kernel.cu",
        srcpath / "cpp/ds/cpu_adam.cpp",
    ]
    ds_cpu_adam = _cpp_extention_load_helper(
        "ds_cpu_adam", sources, extra_cuda_flags
    )
