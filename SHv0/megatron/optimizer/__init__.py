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

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD

from megatron import get_args
from megatron.model import LayerNorm

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .grad_scaler import GL_CPU_DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
from .optimizer import GL_CPU_Float16OptimizerWithFloat16Params, GL_CPU_FP32Optimizer

import torch
import functools
import sys

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam as DeepSpeedCPUAdam
from megatron import mpu
from megatron.model import GPTModel, BertModel

def _get_params_for_weight_decay_optimization(modules):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, LayerNorm):
                no_weight_decay_params["params"].extend(
                    [p for p in list(module_._parameters.values()) if p is not None]
                )
            else:
                weight_decay_params["params"].extend(
                    [
                        p
                        for n, p in list(module_._parameters.items())
                        if p is not None and n != "bias"
                    ]
                )
                no_weight_decay_params["params"].extend(
                    [
                        p
                        for n, p in list(module_._parameters.items())
                        if p is not None and n == "bias"
                    ]
                )

    return weight_decay_params, no_weight_decay_params


class GL_DeepSpeedCPUAdam(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(
        self,
        model_params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        adamw_mode=True,
        fp32_optimizer_states=True,
    ):
        """Fast vectorized implementation of two variations of Adam optimizer on CPU:
        * Adam: A Method for Stochastic Optimization: (https://arxiv.org/abs/1412.6980);
        * AdamW: Fixing Weight Decay Regularization in Adam (https://arxiv.org/abs/1711.05101)
        DeepSpeed CPU Adam(W) provides between 5x to 7x speedup over torch.optim.adam(W).
        In order to apply this optimizer, the model requires to have its master parameter (in FP32)
        reside on the CPU memory.
        To train on a heterogeneous system, such as coordinating CPU and GPU, DeepSpeed offers
        the ZeRO-Offload technology which efficiently offloads the optimizer states into CPU memory,
        with minimal impact on training throughput. DeepSpeedCPUAdam plays an important role to minimize
        the overhead of the optimizer's latency on CPU. Please refer to ZeRO-Offload tutorial
        (https://www.deepspeed.ai/tutorials/zero-offload/) for more information on how to enable this technology.
        For calling step function, there are two options available: (1) update optimizer's states and (2) update
        optimizer's states and copy the parameters back to GPU at the same time. We have seen that the second
        option can bring 30% higher throughput than the doing the copy separately using option one.
        .. note::
                We recommend using our `config
                <https://www.deepspeed.ai/docs/config-json/#optimizer-parameters>`_
                to allow :meth:`deepspeed.initialize` to build this optimizer
                for you.
        Arguments:
            model_params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups.
            lr (float, optional): learning rate. (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square. (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability. (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                algorithm from the paper `On the Convergence of Adam and Beyond`_
                (default: False) NOT SUPPORTED in DeepSpeed CPUAdam!
            adamw_mode: select between Adam and AdamW implementations (default: AdamW)
            full_precision_optimizer_states: creates momementum and variance in full precision regardless of
                        the precision of the parameters (default: True)
        """

        default_args = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
            amsgrad=amsgrad,
        )
        super(GL_DeepSpeedCPUAdam, self).__init__(model_params, default_args)

        self.opt_id = DeepSpeedCPUAdam.optimizer_id
        DeepSpeedCPUAdam.optimizer_id = DeepSpeedCPUAdam.optimizer_id + 1
        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states

        import deepspeed_cpu_adam

        self.ds_opt_adam = deepspeed_cpu_adam

        self.ds_opt_adam.create_adam(
            self.opt_id,
            lr,
            betas[0],
            betas[1],
            eps,
            weight_decay,
            adamw_mode,
            False,
        )

    def __del__(self):
        # need to destroy the C++ object explicitly to avoid a memory leak when deepspeed.initialize
        # is used multiple times in the same process (notebook or pytest worker)
        if hasattr(self, 'ds_opt_adam'):
            self.ds_opt_adam.destroy_adam(self.opt_id)

    def __setstate__(self, state):
        super(DeepSpeedCPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, start_index, end_index, closure=None, fp16_param_groups=None):
        """Update the model parameters.
        .. note::
            This method will be called internally by ZeRO-Offload. DeepSpeed
            users should still use ``engine.step()`` as shown in the
            `Getting Started
            <https://www.deepspeed.ai/getting-started/#training>`_ guide.
        Args:
            closure (callable, optional): closure to compute the loss.
                Defaults to ``None``.
            fp16_param_groups: FP16 GPU parameters to update. Performing the
                copy here reduces communication time. Defaults to ``None``.
        Returns:
            loss: if ``closure`` is provided. Otherwise ``None``.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_id, group in enumerate(self.param_groups[start_index:end_index]):
            for param_id, p in enumerate(group["params"]):

                if p.grad is None:
                    continue

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # print(f'group {group_id} param {param_id} = {p.numel()}')
                    state["step"] = 0

                    # use full precision by default unless self.fp32_optimizer_states is off
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype

                    # gradient momentums
                    state["exp_avg"] = torch.zeros_like(
                        p.data, dtype=state_dtype, device="cpu"
                    )
                    # memory_format=torch.preserve_format)
                    # gradient variances
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, dtype=state_dtype, device="cpu"
                    )
                    # memory_format=torch.preserve_format)

                state["step"] += 1
                beta1, beta2 = group["betas"]

                if fp16_param_groups is not None:
                    self.ds_opt_adam.adam_update_copy(
                        self.opt_id,
                        state["step"],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["weight_decay"],
                        group["bias_correction"],
                        p.data,
                        p.grad.data,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        fp16_param_groups[group_id][param_id].data,
                    )
                else:
                    # print(f" ---------p.data.device = {p.data.device};  p.grad.data={p.grad.data.device}")
                    self.ds_opt_adam.adam_update(
                        self.opt_id,
                        state["step"],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["weight_decay"],
                        group["bias_correction"],
                        p.data,
                        p.grad.data,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                    )
        return loss


def get_megatron_optimizer(model, cpu=False, init_from_model=True):
    args = get_args()

    if init_from_model:
        # Base optimizer.
        param_groups = _get_params_for_weight_decay_optimization(model)
    else:
        param_groups = model

    if args.optimizer == "adam":
        if (args.enable_gl and cpu) or (args.enable_l2l and cpu):
            # from apex.optimizers import FusedAdam  GPU-only
            # optimizer = Adam(param_groups,
            #             lr=args.lr,
            #             weight_decay=args.weight_decay,
            #             betas=(args.adam_beta1, args.adam_beta2),
            #             eps=args.adam_eps)

            # deepspeed official version
            # optimizer = DeepSpeedCPUAdam(
            #     param_groups,
            #     lr=args.lr,
            #     weight_decay=args.weight_decay,
            #     betas=(args.adam_beta1, args.adam_beta2),
            #     eps=args.adam_eps,
            # )

            # remove GIL
            optimizer = GL_DeepSpeedCPUAdam(
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
                eps=args.adam_eps,
            )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.sgd_momentum,
        )
    else:
        raise Exception("{} optimizer is not supported.".format(args.optimizer))

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == "local":
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
                if cpu:
                    grad_scaler = GL_CPU_DynamicGradScaler(
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
                        hysteresis=args.hysteresis,
                    )

        # Megatron optimizer.
        if cpu:
            return GL_CPU_Float16OptimizerWithFloat16Params(
                optimizer,
                args.clip_grad,
                args.log_num_zeros_in_grad,
                params_have_main_grad,
                args.use_contiguous_buffers_in_local_ddp,
                args.bf16,
                grad_scaler,
            )
        else:
            return Float16OptimizerWithFloat16Params(
                optimizer,
                args.clip_grad,
                args.log_num_zeros_in_grad,
                params_have_main_grad,
                args.use_contiguous_buffers_in_local_ddp,
                args.bf16,
                grad_scaler,
            )

    # FP32.
    if cpu:
        return GL_CPU_FP32Optimizer(
            optimizer,
            args.clip_grad,
            args.log_num_zeros_in_grad,
            params_have_main_grad,
            args.use_contiguous_buffers_in_local_ddp,
        )
    else:
        return FP32Optimizer(
            optimizer,
            args.clip_grad,
            args.log_num_zeros_in_grad,
            params_have_main_grad,
            args.use_contiguous_buffers_in_local_ddp,
        )


# ------------------------- gl version -----------------------------------
# import ray
# @ray.remote
# def init_optimizer(optimizer):
#     for i in range(len(layer_modules)):
#         print(f'building optimizer for layer-{i+1}, cpu={i+1 >= args.gl_window_size}')
#         optimizers[i+1] = get_megatron_optimizer([layer_modules[i]],
#                 cpu = i+1 >= args.gl_window_size)


def gl_get_megatron_optimizer(model):
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
                    f"----error/warning: gl_get_megatron_optimizer layer_step ",
                    f"-- optimizer {layer_num} is missing !!",
                )
                return
            # print(f"-- optimizer {layer_num} ------ is at step() ")
            return self.optimizers[layer_num].step()

        def layer_zero_grad(self, layer_num):
            layer_num = int(layer_num)
            if layer_num not in self.optimizers:
                print(
                    f"----error/warning: gl_get_megatron_optimizer layer_zero_grad ",
                    f"-- optimizer {layer_num} is missing !!",
                )
                return
            return self.optimizers[layer_num].zero_grad()

        def get_optimizer(self, name):
            if name not in self.optimizers:
                print(
                    f"----error: gl_get_megatron_optimizer -- optimizer {name} is missing !!"
                )
                exit()
            return self.optimizers[name]

        def __getattr__(self, name: str):
            if name not in self.__dict__:
                return self.optimizers["default"].__getattr__(name)
            return self.__dict__[name]

    class GL_CPU_OptimizerWrapper:
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
        #print(
        #    f"--- init optimizer for layer={i}; ",
        #    f" rank={mpu.get_tensor_model_parallel_rank()} world-size={mpu.get_tensor_model_parallel_world_size()}---",
        #)
        #for name, param in _layers[i].named_parameters():
        #    print(f".  ---- layer{i}; name={name}, size={param.size()}, numel={param.numel()}")
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

            #print(
            #    f"--- Appending param groups to CPU-version Adam for layer={i}.",
            #    f"start_index={start_index}, end_index={end_index}",
            #)

            # if cpu_optimzer is None:
            #     cpu_optimzer = get_megatron_optimizer([_layers[i]], cpu=True)
            #     start_index = 0
            # else:
            #     start_index = len(cpu_optimzer.optimizer.param_groups)
            #     param_groups = _get_params_for_weight_decay_optimization([_layers[i]])
            #     cpu_optimzer.add_param_groups(param_groups, add_to_optimizer=True)
            
            # init cpu_optimizer as None
            optimizers[i] = GL_CPU_OptimizerWrapper(cpu_optimzer, start_index, end_index)

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


"""
CPU version of _amp_foreach_non_finite_check_and_unscale_
"""
import os
import pathlib
import subprocess

from torch.utils import cpp_extension


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
                "-lpython3.8m",
                "-lboost_python-py38",
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
    sources = [
        srcpath / "cpp/GL_CPU_Float16OptimizerWithFloat16Params.cpp",
    ]
    optimizer_utils = _cpp_extention_load_helper(
        "optimizer_utils", sources, extra_cuda_flags
    )

    sources = [
        srcpath / "cpp/offloading_utils.cpp",
    ]
    offloading_utils = _cpp_extention_load_helper(
        "offloading_utils", sources, extra_cuda_flags
    )

    sources = [
        srcpath / "cpp/deepspeed/custom_cuda_kernel.cu",
        srcpath / "cpp/deepspeed/cpu_adam.cpp",
    ]
    deepspeed_cpu_adam = _cpp_extention_load_helper(
        "deepspeed_cpu_adam", sources, extra_cuda_flags
    )


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
