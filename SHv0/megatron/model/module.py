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

"""Megatron Module"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from megatron import get_args
from megatron import mpu


_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)

# ===== GL version ========
from collections import OrderedDict
from torch import Tensor
from typing import (
    Union,
    Tuple,
    Any,
    Callable,
    Iterator,
    Set,
    Optional,
    overload,
    TypeVar,
    Mapping,
    Dict,
    List,
)
from megatron import get_args

# ===========================


def param_is_not_shared(param):
    return not hasattr(param, "shared") or not param.shared


class _MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, share_word_embeddings=True):
        super(_MegatronModule, self).__init__()
        self.share_word_embeddings = share_word_embeddings

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)

    def word_embeddings_weight(self):
        if (
            not mpu.is_pipeline_last_stage(ignore_virtual=True)
            or mpu.get_pipeline_model_parallel_world_size() == 1
        ):
            return self.language_model.embedding.word_embeddings.weight
        else:
            if not self.share_word_embeddings:
                raise Exception(
                    "word_embeddings_weight() called for last "
                    "stage, but share_word_embeddings is false"
                )
            return self.word_embeddings.weight

    def initialize_word_embeddings(self, init_method_normal):
        args = get_args()
        if not self.share_word_embeddings:
            raise Exception(
                "initialize_word_embeddings() was called but "
                "share_word_embeddings is false"
            )

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. Nothing to do if we aren't
        # using pipeline parallelism.
        if args.pipeline_model_parallel_size == 1:
            return

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if mpu.is_pipeline_last_stage():
            assert not mpu.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = "word_embeddings_for_head"
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = mpu.VocabParallelEmbedding(
                args.padded_vocab_size,
                args.hidden_size,
                init_method=init_method_normal(args.init_method_std),
            )
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Zero out initial weights for decoder embedding.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if (
            not mpu.is_pipeline_first_stage(ignore_virtual=True)
            and not mpu.is_pipeline_last_stage(ignore_virtual=True)
            and mpu.is_rank_in_embedding_group()
        ):
            self.language_model.embedding.zero_parameters()

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if mpu.is_rank_in_embedding_group():
                torch.distributed.all_reduce(
                    self.word_embeddings_weight().data, group=mpu.get_embedding_group()
                )
                # All-reduce other embeddings as well as necessary. The last stage
                # does not have these other embeddings, so just create placeholder
                # tensors of the right shape with all zeros.
                # NOTE: We don't currently support T5 with the interleaved schedule.
                if args.pipeline_model_parallel_split_rank is not None:
                    # TODO: Support tokentype embedding.
                    dimensions = (args.max_position_embeddings, args.hidden_size)
                    if mpu.is_pipeline_last_stage(ignore_virtual=True):
                        position_embeddings = torch.nn.Embedding(*dimensions).cuda()
                        position_embeddings.weight.data.fill_(0)
                    else:
                        self.language_model.embedding.cuda()
                        position_embeddings = (
                            self.language_model.embedding.position_embeddings
                        )
                    torch.distributed.all_reduce(
                        position_embeddings.weight.data, group=mpu.get_embedding_group()
                    )
        else:
            print(
                "WARNING! Distributed processes aren't initialized, so "
                "word embeddings in the last layer are not initialized. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


class GL_MegatronModule(_MegatronModule):
    def __init__(self, share_word_embeddings=True):
        super(GL_MegatronModule, self).__init__(share_word_embeddings)
        self._gl_ = OrderedDict()

    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        if "_gl_" in self.__dict__ and name.startswith("_gl_"):
            _gl_ = self.__dict__["_gl_"]
            if name in _gl_:
                return _gl_[name]
            else:
                raise AttributeError(
                    "'{}' object has no attribute '{}'".format(
                        type(self).__name__, name
                    )
                )
        return _MegatronModule.__getattr__(self, name)

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        if "_gl_" in self.__dict__ and name.startswith("_gl_"):
            remove_from(self._gl_)
            self._gl_[name] = value
        else:
            _MegatronModule.__setattr__(self, name, value)

    def __delattr__(self, name):
        if "_gl_" in self.__dict__ and name.startswith("_gl_"):
            _gl_ = self.__dict__["_gl_"]
            if name in _gl_:
                del self._gl_[name]
        else:
            _MegatronModule.__delattr__(self, name)

    # remove prefix '__debug' while debugging
    def _debug_apply(self, fn):
        # ------------ start debug info ------------
        print(f"-----debug GL_MegatronModule '_apply' function \n")
        for name, module in self.named_children():
            print(name)
        exit()
        # ------------ above debug info ------------

        for module in self.children():
            module._apply(fn)

        def compute_should_use_set_data(tensor, tensor_applied):
            if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                # If the new tensor has compatible tensor type as the existing tensor,
                # the current behavior is to change the tensor in-place using `.data =`,
                # and the future behavior is to overwrite the existing tensor. However,
                # changing the current behavior is a BC-breaking change, and we want it
                # to happen in future releases. So for now we introduce the
                # `torch.__future__.get_overwrite_module_params_on_conversion()`
                # global flag to let the user control whether they want the future
                # behavior of overwriting the existing tensor or not.
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False

        for key, param in self._parameters.items():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't want to
                # track autograd history of `param_applied`, so we have to use
                # `with torch.no_grad():`
                with torch.no_grad():
                    param_applied = fn(param)
                should_use_set_data = compute_should_use_set_data(param, param_applied)
                if should_use_set_data:
                    param.data = param_applied
                else:
                    assert isinstance(param, Parameter)
                    assert param.is_leaf
                    self._parameters[key] = Parameter(
                        param_applied, param.requires_grad
                    )

                if param.grad is not None:
                    with torch.no_grad():
                        grad_applied = fn(param.grad)
                    should_use_set_data = compute_should_use_set_data(
                        param.grad, grad_applied
                    )
                    if should_use_set_data:
                        param.grad.data = grad_applied
                    else:
                        assert param.grad.is_leaf
                        self._parameters[key].grad = grad_applied.requires_grad_(
                            param.grad.requires_grad
                        )

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self


args = get_args()
if args.enable_gl:
    MegatronModule = GL_MegatronModule
else:
    MegatronModule = _MegatronModule


class Float16Module(MegatronModule):
    def __init__(self, module, args):
        super(Float16Module, self).__init__()

        if args.fp16:
            self.add_module("module", module.half())

            def float16_convertor(val):
                return val.half()

        elif args.bf16:
            self.add_module("module", module.bfloat16())

            def float16_convertor(val):
                return val.bfloat16()

        else:
            raise Exception("should not be here")

        self.float16_convertor = float16_convertor

    def forward(self, *inputs, **kwargs):
        if mpu.is_pipeline_first_stage():
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if mpu.is_pipeline_last_stage():
            outputs = float16_to_fp32(outputs)
        return outputs

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ):
        return self.module.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)
