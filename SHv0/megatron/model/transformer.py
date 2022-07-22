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

"""Transformer."""
import math
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import mpu

# from .module import MegatronModule

from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType
from megatron.model import LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu


# =======================
# === for GL version ====
# =======================
import time
from collections import OrderedDict, deque
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
from megatron import get_timers

args = get_args()
if args.enable_gl:
    from .module import GL_MegatronModule as MegatronModule  # todo @gl
else:
    from .module import _MegatronModule as MegatronModule

import random
#import offloading_utils
#import ray

torch._C._jit_set_profiling_executor(False)


""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
        )

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
        )

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel + bias_parallel
            )

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_number,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
    ):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = args.params_dtype

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(projection_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size
        )

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method,
            )
        else:
            assert attention_type == AttnType.cross_attn
            self.query = mpu.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method,
            )

            self.key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method,
            )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16,
            self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
        )

        # Inference key-value memory
        self.inference_key_memory = None
        self.inference_value_memory = None
        self.inference_current_sequence_len = 0

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device(),
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
    ):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if set_inference_key_value_memory:
            assert inference_max_sequence_len and inference_max_sequence_len > 0
            self.inference_key_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1)
            )
            self.inference_value_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1)
            )
            self.inference_current_sequence_len = 0
        # Some consistency check.
        if inference_max_sequence_len:
            assert self.inference_current_sequence_len < self.inference_key_memory.size(
                0
            )
            assert inference_max_sequence_len == self.inference_key_memory.size(0)
        # This is added for safety. In case inference_max_sequence_len
        # is not provided, make sure there is no potential memory left
        # from previous inference.
        if not inference_max_sequence_len:
            self.inference_key_memory = None
            self.inference_value_memory = None

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
                mixed_x_layer, 3
            )
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer, value_layer) = mpu.split_tensor_along_last_dim(
                mixed_kv_layer, 2
            )

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        # ===================================================
        # Adjust key, value, and attention mask for inference
        # ===================================================

        if inference_max_sequence_len:
            # Adjust the range variables.
            start = self.inference_current_sequence_len
            self.inference_current_sequence_len += key_layer.size(0)
            end = self.inference_current_sequence_len
            # Copy key and values.
            self.inference_key_memory[start:end, ...] = key_layer
            self.inference_value_memory[start:end, ...] = value_layer
            key_layer = self.inference_key_memory[:end, ...]
            value_layer = self.inference_value_memory[:end, ...]
            # Adjust attention mask
            attention_mask = attention_mask[..., start:end, :end]

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor

    # @gl. todo.
    # print('=====--+++++')
    # torch==1.9.0 is ok, but 1.10.0 fails to run the native megatron-lm source code
    # JIT cannot deal with input tensor without concrete number of dimensions

    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_number,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.padding,
    ):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm = (
            args.apply_residual_connection_post_layernorm
        )

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(args.hidden_size, eps=args.layernorm_epsilon)

        # Self attention.
        self.self_attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type,
        )
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size, eps=args.layernorm_epsilon
        )

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn,
            )
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size, eps=args.layernorm_epsilon
            )

        # MLP
        self.mlp = ParallelMLP(init_method, output_layer_init_method)

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        enc_dec_attn_mask=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
    ):
        # hidden_states: [b, s, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = self.self_attention(
            layernorm_output,
            attention_mask,
            set_inference_key_value_memory=set_inference_key_value_memory,
            inference_max_sequence_len=inference_max_sequence_len,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # jit scripting for a nn.module (with dropout) is not
        # trigerring the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout,
            )

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = self.inter_attention(
                layernorm_output, enc_dec_attn_mask, encoder_output=encoder_output
            )
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            # re-enable torch grad to enable fused optimization.
            with torch.enable_grad():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout,
                )

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            output = bias_dropout_add_func(
                mlp_output, mlp_bias.expand_as(residual), residual, self.hidden_dropout
            )

        return output


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.padding,
        pre_process=True,
        post_process=True,
    ):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None

        # Store activation checkpoiting flag.
        self.activations_checkpoint_method = args.activations_checkpoint_method
        self.activations_checkpoint_num_layers = args.activations_checkpoint_num_layers
        self.distribute_checkpointed_activations = (
            args.distribute_checkpointed_activations
        )

        # Number of layers.
        self.num_layers = mpu.get_num_layers(
            args, args.model_type == ModelType.encoder_and_decoder
        )

        # Transformer layers.
        # avoid too much cuda allocation
        _cuda_cache_queue = deque(maxlen=min(args.gl_window_size * 2, args.num_layers))

        def build_layer(layer_number):
            # ====================
            # =-----gl version ---
            if args.enable_gl:
                layer = GL_ParallelTransformerLayer(
                    init_method,
                    output_layer_init_method,
                    layer_number,
                    layer_type=layer_type,
                    self_attn_mask_type=self_attn_mask_type,
                    _gl_cuda_cache_queue=_cuda_cache_queue,
                )
                return layer
            # ---------------------
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
            )
            # ========ending=======

        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, (
                "num_layers_per_stage must be divisible by "
                "virtual_pipeline_model_parallel_size"
            )
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = (
                self.num_layers // args.virtual_pipeline_model_parallel_size
            )
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size
            ) + (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1 + offset) for i in range(self.num_layers)]
        )

        # ============================================================
        # -----gl version ---
        if args.enable_gl:
            device = torch.cuda.current_device()
            _max_num_units = min(args.gl_window_size * 2, args.num_layers)

            for i in range(_max_num_units):
                unit = OrderedDict()
                _layer = self.layers[0]  # every layer is of the same structure

                for module_name, module in _layer.named_modules():
                    for param_name, param in module._parameters.items():
                        _pkey = module_name + param_name

                        if param is not None:
                            unit[_pkey] = (
                                _layer._gl_cpu_version_items[_pkey]
                                .to(device)
                                .requires_grad_(param.requires_grad)
                                .cuda()
                            )

                        _gkey = _pkey + ".grad"
                        if param.grad is not None:
                            unit[_gkey] = (
                                _layer._gl_cpu_version_items[_gkey]
                                .to(device)
                                .requires_grad_(param.grad.requires_grad)
                                .cuda()
                            )
                        else:
                            unit[_gkey] = _layer._gl_cpu_version_items[_gkey].to(device)

                    for buffer_name, buffer in module._buffers.items():
                        if buffer is not None:
                            _bkey = "buffer_" + module_name + buffer_name
                            unit[_bkey] = _layer._gl_cpu_version_items[_bkey].to(device)

                _layer._gl_cuda_cache_queue.append(unit)

            #print(
            #    f"\t --> _gl_cuda_cache_queue, len={len(_layer._gl_cuda_cache_queue)}, _max_num_units={_max_num_units}"
            #)
            #print(f"\t --> each unit in _gl_cuda_cache_queue:")
            #for key, t in unit.items():
            #    print(f"\t \t --> {key}: {t.size()}, {t.device}")
            ## ============================ending===========================

        if self.post_process:
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                args.hidden_size, eps=args.layernorm_epsilon
            )

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(
        self, hidden_states, attention_mask, encoder_output, enc_dec_attn_mask
    ):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*inputs):
                # print(f'------ custom_forward ------- {start} - {end}')
                x_ = inputs[0]
                attention_mask = inputs[1]
                encoder_output = inputs[2]
                enc_dec_attn_mask = inputs[3]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask, encoder_output, enc_dec_attn_mask)

                    """
                    # moved to unified autograd.Function.save_for_backward
                    ## @gl. for offloading save_for_backward tensors
                    #for t in inputs:
                    #    if t is None:
                    #        layer._gl_save_for_backward.append(['cpu', None])
                    #    else:
                    #        layer._gl_save_for_backward.append([t.device, t])
                    """

                return x_

            return custom_forward

        def distribute_checkpointed_activations_helper(layer_number):
            """Distribute checkpointed activations across the tensor model
            Parallel ranks if the `distribute-checkpointed-activations
            is on and either of the following conditions is met:
              - it is not the first layer in the in the pipeline stage.
                The first layer is used in the pipeline parallelism
                and changing its shape throws error in the backward pass.
              - we are at the first pipline stage so the input tensor is
                not used in pipeline parallelism. Note that no pipeline
                parallelism is a special case of this.
            """
            not_first_layer_in_pipeline_stage = layer_number > 0
            is_first_pipeline_stage = mpu.get_pipeline_model_parallel_rank() == 0
            return self.distribute_checkpointed_activations and (
                not_first_layer_in_pipeline_stage or is_first_pipeline_stage
            )

        if self.activations_checkpoint_method == "uniform":
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers:
                hidden_states = mpu.checkpoint(
                    custom(l, l + self.activations_checkpoint_num_layers),
                    distribute_checkpointed_activations_helper(l),
                    hidden_states,
                    attention_mask,
                    encoder_output,
                    enc_dec_attn_mask,
                )
                l += self.activations_checkpoint_num_layers
        elif self.activations_checkpoint_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers):
                if l < self.activations_checkpoint_num_layers:
                    hidden_states = mpu.checkpoint(
                        custom(l, l + 1),
                        distribute_checkpointed_activations_helper(l),
                        hidden_states,
                        attention_mask,
                        encoder_output,
                        enc_dec_attn_mask,
                    )
                else:
                    hidden_states = custom(l, l + 1)(
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask
                    )
        else:
            raise ValueError("Invalid activation checkpoint method.")

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        enc_dec_attn_mask=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
    ):

        # Checks.
        if inference_max_sequence_len:
            assert (
                self.activations_checkpoint_method is None
            ), "inference does not work with activation checkpointing"

        if self.pre_process:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            # If the input flag for fp32 residual connection is set, convert for float.
            if self.fp32_residual_connection:
                hidden_states = hidden_states.transpose(0, 1).contiguous().float()
            # Otherwise, leave it as is.
            else:
                hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        if encoder_output is not None:
            encoder_output = encoder_output.transpose(0, 1).contiguous()

        def pack_for_bwk(tensor):
            packed = [tensor.device, tensor]
            timers = get_timers()
            timers("offloading-func-call-overhead").start()
            if hasattr(torch, "_gl_current_layer"):
                torch._gl_current_layer._gl_save_for_backward.append(packed)
                #print(f'layer = {torch._gl_current_layer._gl_layer_num}, {len(torch._gl_current_layer._gl_save_for_backward)}')

            timers("offloading-func-call-overhead").stop()
            return packed

        def unpack_on_bwk(packed):
            device, tensor = packed
            assert str(tensor.device) == str(
                device
            ), "---- error: unpack_on_bwk, should be loaded to GPU ---"
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack_for_bwk, unpack_on_bwk):
            if self.activations_checkpoint_method is not None:
                hidden_states = self._checkpointed_forward(
                    hidden_states, attention_mask, encoder_output, enc_dec_attn_mask
                )
            else:
                for index in range(self.num_layers):
                    layer = self._get_layer(index)
                    hidden_states = layer(
                        hidden_states,
                        attention_mask,
                        encoder_output=encoder_output,
                        enc_dec_attn_mask=enc_dec_attn_mask,
                        set_inference_key_value_memory=set_inference_key_value_memory,
                        inference_max_sequence_len=inference_max_sequence_len,
                    )

        # Final layer norm.
        if self.post_process:
            # Reverting data format change [s b h] --> [b s h].
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            output = self.final_layernorm(hidden_states)
        else:
            output = hidden_states

        return output


# -------------------- gl version -----------------------------------

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


class GL_ParallelTransformerLayer(ParallelTransformerLayer):
    def __init__(self, *args, **kwargs):
        assert (
            "_gl_cuda_cache_queue" in kwargs
        ), "Please initialize GL_ParallelTransformerLayer using _cuda_cache_queue"
        self._gl_cuda_cache_queue = kwargs.pop("_gl_cuda_cache_queue")

        super(GL_ParallelTransformerLayer, self).__init__(*args, **kwargs)

        # storage parameters, gradients and buffers in cpu-side
        self._gl_cpu_version_items = OrderedDict()

        # init cpu_version storage
        for module_name, module in self.named_modules():
            for param_name, param in module._parameters.items():
                if param is not None:
                    _pkey = module_name + param_name

                    self._gl_cpu_version_items[_pkey] = torch.empty(
                        param.size(),
                        dtype=param.dtype,
                        layout=param.layout,
                        device=torch.device("cpu"),
                        pin_memory=True,  # (torch.cuda.is_available() and not tensor.is_sparse),
                    ).copy_(param, non_blocking=False)
                    param.data = self._gl_cpu_version_items[_pkey]

                    _gkey = _pkey + ".grad"
                    if param.grad is not None:
                        self._gl_cpu_version_items[_gkey] = (
                            torch.empty(
                                param.grad.size(),
                                dtype=param.grad.dtype,
                                layout=param.grad.layout,
                                device=torch.device("cpu"),
                                pin_memory=True,  # (torch.cuda.is_available() and not tensor.is_sparse)
                            )
                            .copy_(
                                param.grad,
                                non_blocking=False,
                            )
                            .requires_grad_(param.grad.requires_grad)
                        )
                        param.grad.data = self._gl_cpu_version_items[_gkey]
                    else:
                        self._gl_cpu_version_items[_gkey] = torch.empty(
                            param.size(),
                            dtype=param.dtype,
                            layout=param.layout,
                            device=torch.device("cpu"),
                            pin_memory=True,  # (torch.cuda.is_available() and not tensor.is_sparse)
                        ).copy_(torch.zeros_like(param), non_blocking=False)

                        param.grad = self._gl_cpu_version_items[_gkey]

            for buffer_name, buffer in module._buffers.items():
                if buffer is not None:
                    _bkey = "buffer_" + module_name + buffer_name

                    self._gl_cpu_version_items[_bkey] = torch.empty(
                        buffer.size(),
                        dtype=buffer.dtype,
                        layout=buffer.layout,
                        device=torch.device("cpu"),
                        pin_memory=True,  # (torch.cuda.is_available() and not tensor.is_sparse)
                    ).copy_(buffer, non_blocking=False)
        return

    def compute_should_use_set_data(self, tensor, tensor_applied):
        # if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
        #     # If the new tensor has compatible tensor type as the existing tensor,
        #     # the current behavior is to change the tensor in-place using `.data =`,
        #     # and the future behavior is to overwrite the existing tensor. However,
        #     # changing the current behavior is a BC-breaking change, and we want it
        #     # to happen in future releases. So for now we introduce the
        #     # `torch.__future__.get_overwrite_module_params_on_conversion()`
        #     # global flag to let the user control whether they want the future
        #     # behavior of overwriting the existing tensor or not.
        #     return not torch.__future__.get_overwrite_module_params_on_conversion()
        # else:
        #     return False

        """
        If return False, would be error on "grad of a different type"
        todo. @gl
        """
        return True

    @torch.no_grad()
    def cpu(self, non_blocking=True):
        # already cached in cpu-side
        if not hasattr(self, "_gl_cuda_cache_unit"):
            return

        _cuda_cache_unit = self._gl_cuda_cache_unit

        def _apply_cpu(module_name, module, cpu_version_items):
            for param_name, param in module._parameters.items():
                if param is None:
                    continue
                if param.grad is None and str(param.device) == "cpu":
                    continue
                if (
                    param.grad is not None
                    and str(param.grad.device) == "cpu"
                    and str(param.device) == "cpu"
                ):
                    continue

                _pkey = module_name + param_name

                param_applied = (
                    _get_item_from_cpu_cache(self._gl_cpu_version_items, _pkey, param)
                    .copy_(param, non_blocking=non_blocking)
                    .requires_grad_(param.requires_grad)
                )
                _move_item_to_nvme(self._gl_cpu_version_items, _pkey)

                if param.grad is None:
                    param_applied.grad = None

                    use_shallow_copy = self.compute_should_use_set_data(
                        param, param_applied
                    )

                    if use_shallow_copy:
                        _cuda_cache_unit[_pkey] = param.data
                        param.data = param_applied
                        param.grad = None
                    else:
                        assert param.is_leaf
                        _cuda_cache_unit[_pkey] = module._parameters[param_name]
                        module._parameters[param_name] = param_applied.requires_grad_(
                            param.requires_grad
                        )

                else:
                    _gkey = _pkey + ".grad"

                    grad_applied = (
                        _get_item_from_cpu_cache(
                            self._gl_cpu_version_items, _gkey, param.grad
                        )
                        .copy_(param.grad)  # todo? @gl copy_ or add_
                        .requires_grad_(param.grad.requires_grad)
                    )

                    # print("grad_applied", grad_applied.device)
                    # grad_applied = grad_applied.cpu()
                    # print("grad_applied", grad_applied.device)

                    use_shallow_copy = self.compute_should_use_set_data(
                        param.grad, grad_applied
                    )

                    # param_applied.grad = grad_applied

                    if use_shallow_copy:
                        _cuda_cache_unit[_pkey] = param.data
                        _cuda_cache_unit[_gkey] = param.grad.data

                        param_applied.grad = grad_applied
                        param.data = param_applied
                        # param.grad.data = grad_applied

                    else:
                        assert param.is_leaf
                        _cuda_cache_unit[_pkey] = module._parameters[param_name]
                        _cuda_cache_unit[_gkey] = module._parameters[param_name].grad

                        module._parameters[param_name] = param_applied.requires_grad_(
                            param.requires_grad
                        )
                        module._parameters[
                            param_name
                        ].grad = grad_applied.requires_grad_(param.grad.requires_grad)

                    _move_item_to_nvme(self._gl_cpu_version_items, _gkey)

            for buffer_name, buffer in module._buffers.items():
                if buffer is None:
                    continue

                if str(buffer.device) == "cpu":
                    continue

                _bkey = "buffer_" + module_name + buffer_name

                _cuda_cache_unit[_bkey] = module._buffers[buffer_name]

                module._buffers[buffer_name] = _get_item_from_cpu_cache(
                    self._gl_cpu_version_items, _bkey, buffer
                ).copy_(buffer, non_blocking=non_blocking)

                _move_item_to_nvme(self._gl_cpu_version_items, _bkey)

        for module_name, module in self.named_modules():
            _apply_cpu(module_name, module, self._gl_cpu_version_items)

        # for key, value in self._gl_cpu_version_items.items():
        #     print(f" -- debug: _gl_cpu_version_items after apply_cpu: {key}, {value.device}")

        # return cuda_cache as a unit
        self._gl_cuda_cache_queue.append(_cuda_cache_unit)

        # remove the cuda cache from itself
        del self._gl_cuda_cache_unit

        pass

    @torch.no_grad()
    def cuda(self, device=None, non_blocking=True):
        # already cached in cuda-side
        if hasattr(self, "_gl_cuda_cache_unit"):
            return

        # current cuda device
        device = torch.cuda.current_device() if device is None else device

        # pick up an allocated cuda_cuda_unit from cuda_cache_queue
        _cuda_cache_unit = self._gl_cuda_cache_queue.pop()
        self._gl_cuda_cache_unit = _cuda_cache_unit

        # for key, value in self._gl_cpu_version_items.items():
        #     print(f" -- debug: _gl_cpu_version_items before: {key}, {value.device}")

        def _apply_cuda(module_name, module, cpu_version_items):
            for param_name, param in module._parameters.items():
                if param is None:
                    continue

                _pkey = module_name + param_name

                if _pkey not in _cuda_cache_unit or _cuda_cache_unit[_pkey] is None:
                    print(f" --> _apply_cuda: failed to reuse cuda tensor: {_pkey}")

                    param_applied = _get_item_from_cpu_cache(
                        cpu_version_items, _pkey).to(
                            device,
                            non_blocking=non_blocking,
                            copy=True,
                        )
                else:
                    param_applied = _cuda_cache_unit[_pkey].copy_(
                        _get_item_from_cpu_cache(
                            cpu_version_items, _pkey), non_blocking=non_blocking
                    )
                    assert (
                        str(param_applied.device) != "cpu"
                    ), "the tensor should be on cuda device after invoking _apply_cuda()"

                param_applied = param_applied.requires_grad_(param.requires_grad)

                if param.grad is None:
                    param_applied.grad = None

                    use_shallow_copy = self.compute_should_use_set_data(
                        param, param_applied
                    )
                    if use_shallow_copy:
                        param.data = param_applied
                        param.grad = None
                    else:
                        assert param.is_leaf
                        module._parameters[param_name] = param_applied.requires_grad_(
                            True
                        )

                else:
                    _gkey = _pkey + ".grad"

                    if _gkey not in _cuda_cache_unit or _cuda_cache_unit[_gkey] is None:
                        print(f" --> _apply_cuda: failed to reuse cuda tensor: {_gkey}")

                        grad_applied = _get_item_from_cpu_cache(
                            cpu_version_items, _gkey).to(
                                device,
                                non_blocking=non_blocking,
                                copy=True,
                            )
                    else:
                        # print(f"before cpu_version_items {_gkey},", cpu_version_items[_gkey].device,
                        #     id(cpu_version_items[_gkey]) == id(self._gl_cpu_version_items[_gkey]))

                        grad_applied = _cuda_cache_unit[_gkey].copy_(
                            _get_item_from_cpu_cache(
                                cpu_version_items, _gkey), non_blocking=non_blocking
                        )

                        # print(f"after {_gkey},", cpu_version_items[_gkey].device,
                        #     id(cpu_version_items[_gkey]) == id(self._gl_cpu_version_items[_gkey]))

                        assert (
                            str(grad_applied.device) != "cpu"
                        ), "the gradient should be on cuda device after invoking _apply_cuda()"

                        # param_applied.grad = grad_applied

                        use_shallow_copy = self.compute_should_use_set_data(
                            param, param_applied
                        )
                        if use_shallow_copy:
                            param_applied.grad = grad_applied
                            param.data = param_applied

                            # the big buggy error: it would make cpu_verion grads be located at CUDA side
                            # why. how to address it?
                            # todo. @gl.
                            # param.grad.data = grad_applied
                        else:
                            assert param.is_leaf
                            module._parameters[
                                param_name
                            ] = param_applied.requires_grad_(param.requires_grad)
                            module._parameters[
                                param_name
                            ].grad = grad_applied.requires_grad_(
                                param.grad.requires_grad
                            )

            for buffer_name, buffer in module._buffers.items():
                if buffer is None:
                    continue

                _bkey = "buffer_" + module_name + buffer_name

                if _bkey not in _cuda_cache_unit or _cuda_cache_unit[_bkey] is None:
                    print(
                        f" --> _apply_cuda: failed to reuse cuda tensor for buffers: {_bkey}"
                    )

                    module._buffers[buffer_name] = _get_item_from_cpu_cache(
                        cpu_version_items, _bkey).to(
                            device,
                            non_blocking=non_blocking,
                            copy=True,
                        )
                else:
                    module._buffers[buffer_name] = _cuda_cache_unit[_bkey].copy_(
                        _get_item_from_cpu_cache(
                            cpu_version_items, _bkey), non_blocking=non_blocking
                    )
                    assert (
                        str(module._buffers[buffer_name].device) != "cpu"
                    ), "the buffers should be on cuda device after invoking _apply_cuda()"

        for module_name, module in self.named_modules():
            _apply_cuda(module_name, module, self._gl_cpu_version_items)

        # for key, value in self._gl_cpu_version_items.items():
        #     print(f" -- debug: _gl_cpu_version_items after apply_cuda: {key}, {value.device}")

        pass

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        memo = set()
        for name, module in self._modules.items():
            if "_gl_" in name:
                continue

            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self

            for name, module in self._modules.items():
                if module is None:
                    continue
                if "_gl_" in name:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    if "_gl_" in m[0]:
                        continue
                    else:
                        yield m


# -------------------------------------------------------------------------------
