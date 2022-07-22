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

"""Megatron optimizer."""

from abc import ABC
from abc import abstractmethod

import torch

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0

from .clip_grads import clip_grad_norm_fp32, count_zeros_fp32
from megatron.model.module import param_is_not_shared
from megatron.mpu.layers import param_is_not_tensor_parallel_duplicate
from torch._six import inf


def _zero_grad_group_helper(group, set_to_none):
    """Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer."""
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


def _multi_tensor_copy_this_to_that(this, that, overflow_buf=None):
    """Use multi-tensor-applier to copy values from one list to another.
    We don't have a blfoat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16."""
    if overflow_buf:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


class MegatronOptimizer(ABC):
    def __init__(
        self,
        optimizer,
        clip_grad,
        log_num_zeros_in_grad,
        params_have_main_grad,
        use_contiguous_buffers_in_local_ddp,
    ):

        """Input optimizer is the base optimizer for example Adam."""
        self.optimizer = optimizer
        assert self.optimizer, "no optimizer is provided."
        # Set gradient clipping and logging params.
        self.clip_grad = clip_grad
        self.log_num_zeros_in_grad = log_num_zeros_in_grad
        self.params_have_main_grad = params_have_main_grad
        self.use_contiguous_buffers_in_local_ddp = use_contiguous_buffers_in_local_ddp

        if self.use_contiguous_buffers_in_local_ddp:
            assert (
                self.params_have_main_grad
            ), "use of contiguous buffer requires that params have main grad"

    def get_parameters(self):
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                params.append(param)
        return params

    def clip_grad_norm(self, clip_grad):
        params = self.get_parameters()
        return clip_grad_norm_fp32(params, clip_grad)

    def count_zeros(self):
        params = self.get_parameters()
        return count_zeros_fp32(params)

    @abstractmethod
    def zero_grad(self, set_to_none=True):
        pass

    @abstractmethod
    def get_loss_scale(self):
        """The output should be a cuda tensor of size 1."""
        pass

    def scale_loss(self, loss):
        """Simple scaling."""
        return self.get_loss_scale() * loss

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reload_model_params(self):
        """Refreshes any internal state from the current model parameters.
        Call whenever the parameters are changed outside of the optimizer.
        For example, when we load a model from a checkpoint  without loading
        the optimizer, the model parameters are updated but for fp16 optimizer
        with main parameters, the main parameters need to also be updated."""
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    # Promote state so it can be retrieved or set via
    # "optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)


class Float16OptimizerWithFloat16Params(MegatronOptimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Arguments:
        optimizer: base optimizer such as Adam or SGD
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0
        log_num_zeros_in_grad: return number of zeros in the gradients.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a continuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
        bf16: if true, the model is running in bfloat16.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
    """

    def __init__(
        self,
        optimizer,
        clip_grad,
        log_num_zeros_in_grad,
        params_have_main_grad,
        use_contiguous_buffers_in_local_ddp,
        bf16,
        grad_scaler,
    ):

        super(Float16OptimizerWithFloat16Params, self).__init__(
            optimizer,
            clip_grad,
            log_num_zeros_in_grad,
            params_have_main_grad,
            use_contiguous_buffers_in_local_ddp,
        )

        self.bf16 = bf16
        self.grad_scaler = grad_scaler
        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            assert self.bf16, "fp16 expects a grad scaler."

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:
            self.found_inf = torch.cuda.FloatTensor([0.0])

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:
            self._scale_one = torch.cuda.FloatTensor([1.0])

        # ======================
        # main parameter stuff
        # ======================

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group["params"]):
                if param.requires_grad:

                    # float16 params:
                    if param.type() in [
                        "torch.cuda.HalfTensor",
                        "torch.cuda.BFloat16Tensor",
                    ]:
                        float16_params_this_group.append(param)
                        # Create a copy
                        main_param = param.detach().clone().float()
                        # Copy tensor model parallel attributes.
                        mpu.copy_tensor_model_parallel_attributes(main_param, param)
                        if hasattr(param, "shared"):
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group["params"][i] = main_param
                        fp32_from_float16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:
                            self.optimizer.state[main_param] = self.optimizer.state.pop(
                                param
                            )

                    # fp32 params.
                    elif param.type() == "torch.cuda.FloatTensor":
                        fp32_params_this_group.append(param)
                        param_group["params"][i] = param

                    else:
                        raise TypeError(
                            "Wrapped parameters must be one of "
                            "torch.cuda.FloatTensor,  "
                            "torch.cuda.HalfTensor, or "
                            "torch.cuda.BFloat16Tensor. "
                            "Received {}".format(param.type())
                        )

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)

    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(
            self.float16_groups, self.fp32_from_float16_groups
        ):
            for model_param, main_param in zip(model_group, main_group):
                if self.params_have_main_grad and hasattr(model_param, "main_grad"):
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None
                if (
                    self.params_have_main_grad
                    and not self.use_contiguous_buffers_in_local_ddp
                ):
                    model_param.main_grad = None

        # For fp32 grads, we need to reset the grads to main grad.
        if self.params_have_main_grad:
            for model_group in self.fp32_from_fp32_groups:
                for model_param in model_group:
                    model_param.grad = model_param.main_grad

                    # Safe to de-reference model's main_grad after copying.
                    # (If using contiguous buffers, main_grad's memory should
                    # persist and therefore should not be deallocated.)
                    if not self.use_contiguous_buffers_in_local_ddp:
                        model_param.main_grad = None

    def _unscale_main_grads_and_check_for_nan(self):
        main_grads = []
        # fp32 params fromm float16 ones.
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        # Reset found inf.
        self.found_inf.fill_(0.0)
        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale
        )
        # Update across all model parallel instances.
        torch.distributed.all_reduce(
            self.found_inf,
            op=torch.distributed.ReduceOp.MAX,
            group=mpu.get_model_parallel_group(),
        )

        # Check for nan.
        found_inf_flag = self.found_inf.item() > 0
        return found_inf_flag

    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(
            self.float16_groups, self.fp32_from_float16_groups
        ):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf
        )

    def _copy_model_params_to_main_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=model_data, that=main_data, overflow_buf=self._dummy_overflow_buf
        )

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    @torch.no_grad()
    def step(self):

        timers = get_timers()

        # Copy gradients from model params to main params.
        timers("optimizer-copy-to-main-grad").start()
        self._copy_model_grads_to_main_grads()
        timers("optimizer-copy-to-main-grad").stop()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            timers("optimizer-unscale-and-check-inf").start()
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            timers("optimizer-unscale-and-check-inf").stop()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                return False, None, None

        # Clip the main gradients.
        timers("optimizer-clip-main-grad").start()
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)
        timers("optimizer-clip-main-grad").stop()

        # count the zeros in the grads
        num_zeros_in_grad = self.count_zeros() if self.log_num_zeros_in_grad else None

        # Step the optimizer.
        self.optimizer.step()

        # Update params from main params.
        timers("optimizer-copy-main-to-model-params").start()
        self._copy_main_params_to_model_params()
        timers("optimizer-copy-main-to-model-params").stop()

        # Successful update.
        return True, grad_norm, num_zeros_in_grad

    def state_dict(self):
        state_dict = {}
        state_dict["optimizer"] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict["grad_scaler"] = self.grad_scaler.state_dict()
        state_dict["fp32_from_fp16_params"] = self.fp32_from_float16_groups
        return state_dict

    def load_state_dict(self, state_dict):
        # Optimizer.
        optimizer_key = "optimizer"
        if optimizer_key not in state_dict:
            optimizer_key = "optimizer_state_dict"
            print_rank_0(
                "***WARNING*** loading optimizer from " "an old checkpoint ..."
            )
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        # Grad scaler.
        if "grad_scaler" not in state_dict:
            print_rank_0(
                "***WARNING*** found an old checkpoint, will not "
                "load grad scaler ..."
            )
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
            else:
                print_rank_0(
                    "***WARNING*** fould the grad scaler in the "
                    "checkpoint but it is None in the class. "
                    "Skipping loading grad scaler ..."
                )

        # Copy data for the main params.
        fp32_from_float16_params_key = "fp32_from_fp16_params"
        if fp32_from_float16_params_key not in state_dict:
            fp32_from_float16_params_key = "fp32_from_fp16"
        for current_group, saved_group in zip(
            self.fp32_from_float16_groups, state_dict[fp32_from_float16_params_key]
        ):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)


class FP32Optimizer(MegatronOptimizer):
    def __init__(
        self,
        optimizer,
        clip_grad,
        log_num_zeros_in_grad,
        params_have_main_grad,
        use_contiguous_buffers_in_local_ddp,
    ):

        super(FP32Optimizer, self).__init__(
            optimizer,
            clip_grad,
            log_num_zeros_in_grad,
            params_have_main_grad,
            use_contiguous_buffers_in_local_ddp,
        )

        self._scale = torch.cuda.FloatTensor([1.0])

    def zero_grad(self, set_to_none=True):
        """Copied from torch.optim.optimizer"""
        for group in self.optimizer.param_groups:
            _zero_grad_group_helper(group["params"], set_to_none)

    def get_loss_scale(self):
        """FP32 optimizer does not do any scaling."""
        return self._scale

    @torch.no_grad()
    def step(self):
        """Clip gradients (if needed) and step the base optimizer.
        Always return successful since there is no overflow."""

        # Copy main_grads to grads.
        if self.params_have_main_grad:
            for param_group in self.optimizer.param_groups:
                for param in param_group["params"]:
                    if hasattr(param, 'main_grad'):
                        param.grad = param.main_grad

                    # Safe to de-reference model's main_grad after copying.
                    # (If using contiguous buffers, main_grad's memory should
                    # persist and therefore should not be deallocated.)
                    if not self.use_contiguous_buffers_in_local_ddp:
                        param.main_grad = None

        # Clip gradients.
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)

        # count the zeros in the grads
        num_zeros_in_grad = self.count_zeros() if self.log_num_zeros_in_grad else None

        # Update parameters.
        self.optimizer.step()

        # No overflow for FP32 optimizer.
        return True, grad_norm, num_zeros_in_grad

    def reload_model_params(self):
        pass

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class GL_CPU_Float16OptimizerWithFloat16Params(MegatronOptimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Arguments:
        optimizer: base optimizer such as Adam or SGD
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0
        log_num_zeros_in_grad: return number of zeros in the gradients.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a continuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
        bf16: if true, the model is running in bfloat16.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
    """

    def __init__(
        self,
        optimizer,
        clip_grad,
        log_num_zeros_in_grad,
        params_have_main_grad,
        use_contiguous_buffers_in_local_ddp,
        bf16,
        grad_scaler,
    ):

        super(GL_CPU_Float16OptimizerWithFloat16Params, self).__init__(
            optimizer,
            clip_grad,
            log_num_zeros_in_grad,
            params_have_main_grad,
            use_contiguous_buffers_in_local_ddp,
        )

        import optimizer_utils

        self.optimizer_utils = optimizer_utils

        self.bf16 = bf16
        self.grad_scaler = grad_scaler
        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            assert self.bf16, "fp16 expects a grad scaler."

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:
            self.found_inf = torch.FloatTensor([0.0])

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.IntTensor([0])

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:
            self._scale_one = torch.FloatTensor([1.0])

        # ======================
        # main parameter stuff
        # ======================

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []
        self.add_param_groups(self.optimizer.param_groups, add_to_optimizer=False)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def add_param_groups(self, param_groups, add_to_optimizer=False):
        # For all the groups in the original optimizer:
        for param_group in param_groups:
            if add_to_optimizer:
                self.optimizer.add_param_group(param_group)

            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group["params"]):
                if param.requires_grad:

                    # float16 params:
                    if param.type() in ["torch.HalfTensor", "torch.BFloat16Tensor"]:
                        float16_params_this_group.append(param)
                        # Create a copy
                        main_param = param.detach().clone().float()
                        # Copy tensor model parallel attributes.
                        mpu.copy_tensor_model_parallel_attributes(main_param, param)
                        if hasattr(param, "shared"):
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group["params"][i] = main_param
                        fp32_from_float16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:
                            self.optimizer.state[main_param] = self.optimizer.state.pop(
                                param
                            )

                    # fp32 params.
                    elif param.type() == "torch.FloatTensor":
                        fp32_params_this_group.append(param)
                        param_group["params"][i] = param

                    else:
                        raise TypeError(
                            "Wrapped parameters must be one of "
                            "torch.FloatTensor,  "
                            "torch.HalfTensor, or "
                            "torch.BFloat16Tensor. "
                            "Received {}".format(param.type())
                        )
            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

    def zero_grad(self, start_index=0, end_index=-1, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for group in self.float16_groups[start_index:end_index]:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_float16_groups[start_index:end_index]:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups[start_index:end_index]:
            _zero_grad_group_helper(group, set_to_none)

    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def _copy_model_grads_to_main_grads(self, start_index=0, end_index=-1):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(
            self.float16_groups[start_index:end_index],
            self.fp32_from_float16_groups[start_index:end_index],
        ):
            for model_param, main_param in zip(model_group, main_group):
                if self.params_have_main_grad and hasattr(model_param, "main_grad"):
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None
                if (
                    self.params_have_main_grad
                    and not self.use_contiguous_buffers_in_local_ddp
                ):
                    model_param.main_grad = None

        # For fp32 grads, we need to reset the grads to main grad.
        if self.params_have_main_grad:
            for model_group in self.fp32_from_fp32_groups[start_index:end_index]:
                for model_param in model_group:
                    model_param.grad = model_param.main_grad

                    # Safe to de-reference model's main_grad after copying.
                    # (If using contiguous buffers, main_grad's memory should
                    # persist and therefore should not be deallocated.)
                    if not self.use_contiguous_buffers_in_local_ddp:
                        model_param.main_grad = None

        # self.optimizer_utils._copy_model_grads_to_main_grads(
        #     self.float16_groups[start_index:end_index],
        #     self.fp32_from_float16_groups[start_index:end_index],
        #     self.fp32_from_fp32_groups[start_index:end_index],
        #     self.params_have_main_grad,
        #     self.use_contiguous_buffers_in_local_ddp,
        # )
        return

    def _unscale_main_grads_and_check_for_nan(self, start_index=0, end_index=-1):
        main_grads = []
        # fp32 params fromm float16 ones.
        for main_group in self.fp32_from_float16_groups[start_index:end_index]:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups[start_index:end_index]:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        # Reset found inf.
        self.found_inf.fill_(0.0)

        # Unscale and set found inf/nan

        # # GPU version only
        # torch._amp_foreach_non_finite_check_and_unscale_(
        #    main_grads, self.found_inf, self.grad_scaler.inv_scale)

        # # CPU and GPU version
        # assert (
        #     len(self.grad_scaler.inv_scale) == 1
        # ), " GL_CPU_Float16OptimizerWithFloat16Params, _unscale_main_grads_and_check_for_nan,  self.grad_scaler.inv_scale, not only includes one elements"
        # if self.grad_scaler.inv_scale.item() == 1.0:
        #     for grad in main_grads:
        #         if torch.isfinite(grad).all():
        #             self.found_inf.fill_(1.0)
        # else:
        #     for grad in main_grads:
        #         if torch.isfinite(grad).all():
        #             self.found_inf.fill_(1.0)
        #             continue
        #         grad *= self.grad_scaler.inv_scale

        # c++ version
        self.optimizer_utils.foreach_non_finite_check_and_unscale(
            main_grads, self.found_inf, self.grad_scaler.inv_scale
        )

        # Update across all model parallel instances.
        # done init cpu and gpu distributed mode seperately, madan
        torch.distributed.all_reduce(
            self.found_inf,
            op=torch.distributed.ReduceOp.MAX,
            group=mpu.get_model_parallel_group(),
        )

        # Check for nan.
        found_inf_flag = self.found_inf.item() > 0
        return found_inf_flag

    def _get_model_and_main_params_data_float16(self, start_index=0, end_index=-1):
        model_data = []
        main_data = []
        for model_group, main_group in zip(
            self.float16_groups[start_index:end_index],
            self.fp32_from_float16_groups[start_index:end_index],
        ):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_main_params_to_model_params(self, start_index=0, end_index=-1):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16(
            start_index, end_index
        )
        # _multi_tensor_copy_this_to_that(
        #     this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf
        # )
        self.optimizer_utils._multi_tensor_copy_this_to_that(main_data, model_data)

    def _copy_model_params_to_main_params(self, start_index=0, end_index=-1):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16(
            start_index, end_index
        )
        # _multi_tensor_copy_this_to_that(
        #     this=model_data, that=main_data, overflow_buf=self._dummy_overflow_buf
        # )
        self.optimizer_utils._multi_tensor_copy_this_to_that(model_data, main_data)

    def _clip_grad_norm_fp32(self, parameters, max_norm, norm_type=2):
        """Clips gradient norm of an iterable of parameters whose gradients
        are in fp32.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        # Filter parameters based on:
        #   - grad should not be none
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        grads = []
        grads_for_norm = []
        for param in parameters:
            grad_not_none = param.grad is not None
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none:
                grad = param.grad.detach()
            if grad_not_none:
                # Make sure the grads are in fp32
                assert param.grad.type() == "torch.FloatTensor"
                grads.append(grad)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)

        # Norm parameters.
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        total_norm = 0.0

        # Calculate norm.
        if norm_type == inf:
            total_norm = max(grad.abs().max() for grad in grads_for_norm)
            total_norm_cpu = torch.FloatTensor([float(total_norm)])
            # Take max across all model-parallel GPUs.
            torch.distributed.all_reduce(
                total_norm_cpu,
                op=torch.distributed.ReduceOp.MAX,
                group=mpu.get_model_parallel_group(),
            )
            total_norm = total_norm_cpu[0].item()

        else:
            if norm_type == 2.0:
                dummy_overflow_buf = torch.IntTensor([0])
                # Use apex's multi-tensor applier for efficiency reasons.
                # Multi-tensor applier takes a function and a list of list
                # and performs the operation on that list all in one kernel.
                grad_norm, _ = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False,  # no per-parameter norm
                )
                # Since we will be summing across data parallel groups,
                # we need the pow(norm-type).
                total_norm = grad_norm ** norm_type

            else:
                for grad in grads_for_norm:
                    grad_norm = torch.norm(grad, norm_type)
                    total_norm += grad_norm ** norm_type

            # Sum across all model-parallel GPUs.
            torch.distributed.all_reduce(
                total_norm,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_model_parallel_group(),
            )
            total_norm = total_norm.item() ** (1.0 / norm_type)

        # Scale.
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        if clip_coeff < 1.0:
            dummy_overflow_buf = torch.IntTensor([0])
            multi_tensor_applier(
                amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff
            )

        return total_norm

    def reload_model_params(self, start_index=0, end_index=-1):
        self._copy_model_params_to_main_params(start_index, end_index)

    @torch.no_grad()
    def step(self, start_index=0, end_index=-1):
        # print(
        #     f"---. GL_CPU_Float16OptimizerWithFloat16Params optimizer: start_index={start_index}, end_index={end_index}"
        # )
        timers = get_timers()

        # Copy gradients from model params to main params.
        timers("optimizer-copy-to-main-grad").start()
        self._copy_model_grads_to_main_grads(start_index, end_index)
        timers("optimizer-copy-to-main-grad").stop()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            timers("optimizer-unscale-and-check-inf").start()
            found_inf_flag = self._unscale_main_grads_and_check_for_nan(
                start_index, end_index
            )
            timers("optimizer-unscale-and-check-inf").stop()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                return False, None, None

        # Clip the main gradients.
        timers("optimizer-clip-main-grad").start()
        grad_norm = None
        # if self.clip_grad > 0.0:
        #     grad_norm = self._clip_grad_norm_fp32(self.get_parameters(), self.clip_grad)
        timers("optimizer-clip-main-grad").stop()

        # count the zeros in the grads
        num_zeros_in_grad = self.count_zeros() if self.log_num_zeros_in_grad else None

        # Step the optimizer. // cpu-version adam
        self.optimizer.step(start_index, end_index)

        # Update params from main params.
        timers("optimizer-copy-main-to-model-params").start()
        self._copy_main_params_to_model_params(start_index, end_index)
        timers("optimizer-copy-main-to-model-params").stop()

        # Successful update.
        return True, grad_norm, num_zeros_in_grad

    def state_dict(self):
        state_dict = {}
        state_dict["optimizer"] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict["grad_scaler"] = self.grad_scaler.state_dict()
        state_dict["fp32_from_fp16_params"] = self.fp32_from_float16_groups
        return state_dict

    def load_state_dict(self, state_dict):
        # Optimizer.
        optimizer_key = "optimizer"
        if optimizer_key not in state_dict:
            optimizer_key = "optimizer_state_dict"
            print_rank_0(
                "***WARNING*** loading optimizer from " "an old checkpoint ..."
            )
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        # Grad scaler.
        if "grad_scaler" not in state_dict:
            print_rank_0(
                "***WARNING*** found an old checkpoint, will not "
                "load grad scaler ..."
            )
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
            else:
                print_rank_0(
                    "***WARNING*** fould the grad scaler in the "
                    "checkpoint but it is None in the class. "
                    "Skipping loading grad scaler ..."
                )

        # Copy data for the main params.
        fp32_from_float16_params_key = "fp32_from_fp16_params"
        if fp32_from_float16_params_key not in state_dict:
            fp32_from_float16_params_key = "fp32_from_fp16"
        for current_group, saved_group in zip(
            self.fp32_from_float16_groups, state_dict[fp32_from_float16_params_key]
        ):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)


class GL_CPU_FP32Optimizer(MegatronOptimizer):
    def __init__(
        self,
        optimizer,
        clip_grad,
        log_num_zeros_in_grad,
        params_have_main_grad,
        use_contiguous_buffers_in_local_ddp,
    ):

        super(GL_CPU_FP32Optimizer, self).__init__(
            optimizer,
            clip_grad,
            log_num_zeros_in_grad,
            params_have_main_grad,
            use_contiguous_buffers_in_local_ddp,
        )

        self._scale = torch.FloatTensor([1.0])

    def zero_grad(self, start_index=0, end_index=-1, set_to_none=True):
        """Copied from torch.optim.optimizer"""
        for group in self.optimizer.param_groups[start_index:end_index]:
            _zero_grad_group_helper(group["params"], set_to_none)

    def get_loss_scale(self):
        """FP32 optimizer does not do any scaling."""
        return self._scale

    def add_param_groups(self, param_groups, add_to_optimizer=False):
        # For all the groups in the original optimizer:
        for param_group in param_groups:
            if add_to_optimizer:
                self.optimizer.add_param_group(param_group)

    @torch.no_grad()
    def step(self, start_index=0, end_index=-1):
        """Clip gradients (if needed) and step the base optimizer.
        Always return successful since there is no overflow."""

        # Copy main_grads to grads.
        if self.params_have_main_grad:
            # print(
            #     "------------",
            #     start_index,
            #     end_index,
            #     self.optimizer.param_groups[start_index:end_index],
            # )
            for param_group in self.optimizer.param_groups[start_index:end_index]:
                for param in param_group["params"]:
                    if hasattr(param, 'main_grad'):
                        param.grad = param.main_grad.to(param.device)

                    # Safe to de-reference model's main_grad after copying.
                    # (If using contiguous buffers, main_grad's memory should
                    # persist and therefore should not be deallocated.)
                    if not self.use_contiguous_buffers_in_local_ddp:
                        param.main_grad = None

        # # Clip gradients.
        grad_norm = None
        # if self.clip_grad > 0.0:
        #     grad_norm = self.clip_grad_norm(self.clip_grad)

        # count the zeros in the grads
        num_zeros_in_grad = self.count_zeros() if self.log_num_zeros_in_grad else None

        # Update parameters.
        # print("optimizer index: ", start_index, end_index)
        self.optimizer.step(start_index, end_index)

        # No overflow for FP32 optimizer.
        return True, grad_norm, num_zeros_in_grad

    def reload_model_params(self):
        pass

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    # https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    # https://pytorch.org/cppdocs/api/file_torch_csrc_api_include_torch_nn_utils_clip_grad.h.html#file-torch-csrc-api-include-torch-nn-utils-clip-grad-h
    def clip_grad_norm(self, clip_grad):
        def _clip_grad_norm_fp32(parameters, max_norm, norm_type=2):
            """Clips gradient norm of an iterable of parameters whose gradients
            are in fp32.

            This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
            added functionality to handle model parallel parameters. Note that
            the gradients are modified in place.

            Arguments:
                parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                    single Tensor that will have gradients normalized
                max_norm (float or int): max norm of the gradients
                norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                    infinity norm.

            Returns:
                Total norm of the parameters (viewed as a single vector).
            """

            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]

            # Filter parameters based on:
            #   - grad should not be none
            #   - parameter should not be shared
            #   - should not be a replica due to tensor model parallelism
            grads = []
            grads_for_norm = []
            for param in parameters:
                grad_not_none = param.grad is not None
                is_not_shared = param_is_not_shared(param)
                is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
                if grad_not_none:
                    grad = param.grad.detach()
                if grad_not_none:
                    # Make sure the grads are in fp32
                    assert param.grad.type() == "torch.FloatTensor"
                    grads.append(grad)
                if grad_not_none and is_not_shared and is_not_tp_duplicate:
                    grads_for_norm.append(grad)

            # Norm parameters.
            max_norm = float(max_norm)
            norm_type = float(norm_type)
            total_norm = 0.0

            # Calculate norm.
            if norm_type == inf:
                total_norm = max(grad.abs().max() for grad in grads_for_norm)
                # Take max across all model-parallel GPUs.
                torch.distributed.all_reduce(
                    total_norm,
                    op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_model_parallel_group(),
                )
                total_norm = total_norm.item()

            else:
                if norm_type == 2.0:
                    dummy_overflow_buf = torch.IntTensor([0])
                    # Use apex's multi-tensor applier for efficiency reasons.
                    # Multi-tensor applier takes a function and a list of list
                    # and performs the operation on that list all in one kernel.

                    # # ????? todo. @gl -----------
                    # grad_norm, _ = multi_tensor_applier(
                    #     amp_C.multi_tensor_l2norm,
                    #     dummy_overflow_buf,
                    #     [grads_for_norm],
                    #     False,  # no per-parameter norm
                    # )

                    # # Since we will be summing across data parallel groups,
                    # # we need the pow(norm-type).
                    # total_norm = grad_norm ** norm_type

                    total_norm = torch.norm(
                        torch.stack(
                            [
                                torch.norm(p.grad.detach(), norm_type)
                                for p in grads_for_norm
                            ]
                        ),
                        norm_type,
                    )
                else:
                    for grad in grads_for_norm:
                        grad_norm = torch.norm(grad, norm_type)
                        total_norm += grad_norm ** norm_type

                # Sum across all model-parallel GPUs.
                torch.distributed.all_reduce(
                    total_norm,
                    op=torch.distributed.ReduceOp.SUM,
                    group=mpu.get_model_parallel_group(),
                )
                total_norm = total_norm.item() ** (1.0 / norm_type)

            # Scale.
            clip_coeff = max_norm / (total_norm + 1.0e-6)

            # if clip_coeff < 1.0:
            #     dummy_overflow_buf = torch.IntTensor([0])
            #     # ????? todo. @gl -----------
            #     multi_tensor_applier(
            #         amp_C.multi_tensor_scale,
            #         dummy_overflow_buf,
            #         [grads, grads],
            #         clip_coeff,
            #     )

            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            for p in grads_for_norm:
                p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))

            return total_norm

        params = self.get_parameters()
        return _clip_grad_norm_fp32(params, clip_grad)
