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

from abc import ABC
from abc import abstractmethod

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron import get_args
from megatron import mpu
from .module import MegatronModule



class MemoryBuffer:

    def __init__(self, numel, dtype, device=None):
        self.numel = numel
        self.dtype = dtype
        
        # @gl 
        if device is None:
            device = torch.cuda.current_device()

        self.data = torch.zeros(self.numel,
                                dtype=self.dtype,
                                device=device,
                                requires_grad=False)


    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()


    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor



class DistributedDataParallelBase(MegatronModule, ABC):
    """Abstract class for DDP."""

    def __init__(self, module):
        super(DistributedDataParallelBase, self).__init__()
        # Keep a pointer to the model.
        self.module = module


    @abstractmethod
    def allreduce_gradients(self):
        pass


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix,
                                                          keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)



class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to storre and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """

    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers):

        super(DistributedDataParallel, self).__init__(module)

        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        
        # -------------------
        # ---- GL version ---
        args = get_args()
        if args.enable_gl:
            assert self.use_contiguous_buffers, "Please use_contiguous_buffers"

            self._grad_buffers_cuda = None
            self._grad_buffers_cpu = None

            if self.use_contiguous_buffers:
                self._grad_buffers_cuda = {}
                self._grad_buffers_cpu = {}

            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            
            _cpu_list = [f'coder.layers.{i}'
                for i in range(args.gl_window_size, args.num_layers)]
            
            _if_name_in_cpu_list = lambda _cpu_list, name: sum([n in name for n in _cpu_list]) > 0

            # First calculate total number of elements per type.
            type_num_elements_cpu = {}
            type_num_elements_cuda = {}

            for name, param in self.module.named_parameters():
                if _if_name_in_cpu_list(_cpu_list, name):
                    if param.requires_grad:
                        dtype = _get_buffer_type(param)
                        type_num_elements_cpu[dtype] = \
                                type_num_elements_cpu.get(dtype, 0) \
                                    + param.data.nelement()
                else:
                    if param.requires_grad:
                        dtype = _get_buffer_type(param)
                        type_num_elements_cuda[dtype] = \
                                type_num_elements_cuda.get(dtype, 0) \
                                    + param.data.nelement()

            # Allocate the buffer.
            for dtype, num_elements in type_num_elements_cpu.items():
                self._grad_buffers_cpu[dtype] = \
                        MemoryBuffer(num_elements, dtype, device=torch.device('cpu'))

            for dtype, num_elements in type_num_elements_cuda.items():
                self._grad_buffers_cuda[dtype] = \
                        MemoryBuffer(num_elements, dtype)

           
            print(f'---- DDP in model/distributed.py --- ', 
                    f';\n\t allocate cpu_buffer={self._grad_buffers_cpu}; ',
                    f';\n\t cuda_buffer={self._grad_buffers_cuda}')

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for name, param in self.module.named_parameters():
                if _if_name_in_cpu_list(_cpu_list, name):
                    print(f'---- CPU cached      --- {name}')
                    if param.requires_grad:
                        dtype = _get_buffer_type(param)
                        type_num_elements_cpu[dtype] -= param.data.nelement()
                        param.main_grad = self._grad_buffers_cpu[dtype].get(
                            param.data.shape, type_num_elements_cpu[dtype])
                else:
                    print(f'----      GPU cached --- {name}')
                    if param.requires_grad:
                        dtype = _get_buffer_type(param)
                        type_num_elements_cuda[dtype] -= param.data.nelement()
                        param.main_grad = self._grad_buffers_cuda[dtype].get(
                            param.data.shape, type_num_elements_cuda[dtype])

            """
            # moved accumalation action after offloading finished
            # in training.py backward_pre_hook, except the fisrt layer (layer-0) 

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)
            """
            
            # todo. @gl.
            """
            now, the main_grads of first window layers stay in gpu all the time.
            to refine it, only layer-0 resides at GPU, and other layers in the first window layers offloads main_grads
            """
            self.grad_accs = []
            for name, param in self.module.named_parameters():
                if param.requires_grad and not _if_name_in_cpu_list(_cpu_list, name):
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)

            return

        # ---- default version ---
        
        self._grad_buffers = None

        if self.use_contiguous_buffers:
            self._grad_buffers = {}

            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                               + param.data.nelement()

            # Allocate the buffer.
            for dtype, num_elements in type_num_elements.items():
                self._grad_buffers[dtype] = MemoryBuffer(num_elements, dtype)

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()
                    param.main_grad = self._grad_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype])

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)
        
        # --------END--------
        # -------------------


    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad.data is not None:
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook


    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        
        # -------------------
        # ---- GL version ---
        args = get_args()
        if args.enable_gl:
            assert self._grad_buffers_cpu is not None or \
                    self._grad_buffers_cuda is not None, \
                    'cpu and cuda buffers are not initialized.'
            
            for _, buffer_ in self._grad_buffers_cpu.items():
                buffer_.zero()
            for _, buffer_ in self._grad_buffers_cuda.items():
                buffer_.zero()
            
            return
        
        # ---- default version ---
            
        assert self._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero()

        # --------END--------
        # -------------------


    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks."""
        # -------------------
        # ---- GL version ---

        args = get_args()
        if args.enable_gl:
            if self._grad_buffers_cpu is None and \
                    self._grad_buffers_cuda is None:
                print('---Error - model/distributed.py --- ',
                        'allreduce_gradients --- ',
                        'please use biffer to store grads now, no bucket support ---')
                exit()

            """
            todo. @gl. support torch.distributed can use gloo and nccl
            if self._grad_buffers_cpu is not None:
                for _, buffer_ in self._grad_buffers_cpu.items():
                    buffer_.data /= mpu.get_data_parallel_world_size()
                    torch.distributed.all_reduce(
                        buffer_.data, group=mpu.get_data_parallel_group())
            """

            if self._grad_buffers_cuda is not None:
                for _, buffer_ in self._grad_buffers_cuda.items():
                    buffer_.data /= mpu.get_data_parallel_world_size()
                    torch.distributed.all_reduce(
                        buffer_.data, group=mpu.get_data_parallel_group())
            return

        # ---- default version ---

        # If we have buffers, simply reduce the data in the buffer.
        if self._grad_buffers is not None:
            for _, buffer_ in self._grad_buffers.items():
                buffer_.data /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(
                    buffer_.data, group=mpu.get_data_parallel_group())
        else:
            # Otherwise, bucketize and all-reduce
            buckets = {}
            # Pack the buckets.
            for param in self.module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
                    param.main_grad = param.grad

            # For each bucket, all-reduce and copy all-reduced grads.
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                coalesced /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(
                    coalesced, group=mpu.get_data_parallel_group())
                for buf, synced in zip(grads, _unflatten_dense_tensors(
                        coalesced, grads)):
                    buf.copy_(synced)
        
        # --------END--------
        # -------------------
