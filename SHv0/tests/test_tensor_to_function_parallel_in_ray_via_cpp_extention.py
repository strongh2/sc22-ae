import os
import pathlib
import subprocess
import time

import ray
import torch
from torch.utils import cpp_extension
from collections import OrderedDict

# ray.init(num_cpus=2, num_gpus=1)


def load_utils():
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
            ],
            extra_cuda_cflags=[
                "-O3",
                "-gencode",
                "arch=compute_70,code=sm_70",
                "--use_fast_math",
            ]
            + extra_cuda_flags
            + cc_flag,
            verbose=True,
        )

    extra_cuda_flags = ["-maxrregcount=50"]
    sources = [
        srcpath / "cpp/tensor_to_cpu_or_cuda.cpp",
    ]
    test_utils = _cpp_extention_load_helper("test_utils", sources, extra_cuda_flags)


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


# -------------------------


futs = {}


def _forward_pre_hook(module, input):
    fut = module._handler._layer_to_cuda.remote(module._moving_layer_num)
    if module._moving_layer_num in futs:
        futs[module._moving_layer_num].append(fut)
    else:
        futs[module._moving_layer_num] = [fut]

    # if module._layer_num in futs:
    #     ray.get(futs[module._layer_num])


def _forward_post_hook(module, input, output):
    fut = module._handler._layer_to_cpu.remote(module._layer_num)

    if module._layer_num in futs:
        futs[module._layer_num].append(fut)
    else:
        futs[module._layer_num] = [fut]


@ray.remote
class Handler:
    def __init__(self):
        load_utils()
        import test_utils
        import torch_tensorrt

        self.utils = test_utils

        self.num_layers = 96
        self.windows_size = 1

        self.gpu_cache_units = []

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=12288, nhead=96)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )
        self.layers = self.transformer_encoder.layers

        self.transformer_encoder_tensorrt = torch_tensorrt.compile(
            self.transformer_encoder,
            inputs=[
                torch.rand(2048, 2, 12288),
                torch_tensorrt.Input(shape=(2048, 2, 12288)),
            ],
        )

        for i in range(self.num_layers):
            layer = self.layers[i]
            layer._layer_num = i
            layer._moving_layer_num = min(self.num_layers - 1, i + self.windows_size)

        for m in self.transformer_encoder.modules():
            if m._parameters is not None and len(m._parameters) > 0:
                _cpu_parameters = OrderedDict()
                for k, v in m._parameters.items():
                    if v is not None:
                        _cpu_parameters[k] = torch.empty(
                            v.size(),
                            dtype=v.dtype,
                            layout=v.layout,
                            device=torch.device("cpu"),
                            pin_memory=True,
                        )
                        _cpu_parameters[k].copy_(v)
                    else:
                        _cpu_parameters[k] = None
                m._cpu_parameters = _cpu_parameters
                del m._parameters

        for i in range(self.num_layers - self.windows_size):
            layer = self.layers[i]
            layer.register_forward_pre_hook(_forward_pre_hook)
        for i in range(self.num_layers):
            layer = self.layers[i]
            layer.register_forward_hook(_forward_post_hook)

    def _init_cache(self):
        self.gpu_cache_units = []

        for i in range(2 * self.windows_size + 1):
            params = {}
            for module_name, m in self.layers[0].named_modules():
                if (
                    hasattr(m, "_cpu_parameters")
                    and m._cpu_parameters is not None
                    and len(m._cpu_parameters) > 0
                ):
                    for k, v in m._cpu_parameters.items():
                        if v is not None:
                            params[module_name + k] = torch.empty(
                                v.size(),
                                dtype=v.dtype,
                                layout=v.layout,
                                device=torch.device("cuda:0"),
                            )
                        else:
                            params[module_name + k] = None

            self.gpu_cache_units.append(params)

    def _register_handler(self, handler):
        for i in range(self.num_layers):
            layer = self.layers[i]
            layer._handler = handler

    def _layer_to_cuda(self, layer_num):
        _layer = self.layers[layer_num]

        params = self.gpu_cache_units.pop()
        for module_name, m in _layer.named_modules():
            if (
                hasattr(m, "_cpu_parameters")
                and m._cpu_parameters is not None
                and len(m._cpu_parameters) > 0
            ):
                m._parameters = {
                    k: params[module_name + k].copy_(v, non_blocking=True)
                    if v is not None
                    else None
                    for k, v in m._cpu_parameters.items()
                }

        # for m in _layer.modules():
        #     if hasattr(m, "_cpu_parameters"):
        #         m._parameters = {
        #             k: v.to("cuda", non_blocking=True)
        #             for k, v in m._cpu_parameters.items()
        #         }

        # _p, _b = OrderedDict(), OrderedDict()
        # for k, v in _layer.named_modules():
        #     if v._parameters is not None and len(v._parameters) > 0:
        #         _p[k] = v._parameters
        #     # _b[k] = v._buffers

        # _stream = torch.cuda.Stream()
        # with torch.cuda.stream(_stream):
        #     _pp = self.utils.tensor_to_cuda(_p)
        #     # _bb = self.utils.tensor_to_cuda(_b)

        # for k, v in _layer.named_modules():
        #     if v._parameters is not None and len(v._parameters) > 0:
        #         v._parameters = _pp[k]
        #     # v._buffers = _bb[k]

    def _layer_to_cpu(self, layer_num):
        _layer = self.layers[layer_num]

        params = {}
        for module_name, m in _layer.named_modules():
            if (
                hasattr(m, "_parameters")
                and m._parameters is not None
                and len(m._parameters) > 0
            ):
                for k, v in m._parameters.items():
                    params[module_name + k] = v
                del m._parameters

        self.gpu_cache_units.append(params)

        # for m in _layer.modules():
        #     if hasattr(m, "_cpu_parameters"):
        #         del m._parameters

        # _p, _b = OrderedDict(), OrderedDict()
        # for k, v in _layer.named_modules():
        #     if v._parameters is not None and len(v._parameters) > 0:
        #         _p[k] = v._parameters
        #     # _b[k] = v._buffers

        # _stream = torch.cuda.Stream()
        # with torch.cuda.stream(_stream):
        #     _pp = self.utils.tensor_to_cpu(_p)
        #     # _bb = self.utils.tensor_to_cpu(_b)

        # for k, v in _layer.named_modules():
        #     if v._parameters is not None and len(v._parameters) > 0:
        #         v._parameters = _pp[k]
        #     # v._buffers = _bb[k]

    def to_cuda(self):
        for i in range(self.windows_size):
            _layer = self.layers[i]

            params = self.gpu_cache_units.pop()

            for module_name, m in _layer.named_modules():
                if (
                    hasattr(m, "_cpu_parameters")
                    and m._cpu_parameters is not None
                    and len(m._cpu_parameters) > 0
                ):
                    m._parameters = {
                        k: params[module_name + k].copy_(v, non_blocking=True)
                        if v is not None
                        else None
                        for k, v in m._cpu_parameters.items()
                    }

            # for m in _layer.modules():
            #     if hasattr(m, "_cpu_parameters"):
            #         m._parameters = {
            #             k: v.to("cuda", non_blocking=True)
            #             for k, v in m._cpu_parameters.items()
            #         }
        for i in range(self.windows_size, self.num_layers):
            _layer = self.layers[i]
            for m in _layer.modules():
                if hasattr(m, "_parameters"):
                    del m._parameters

    def fwd(self, x):
        return self.transformer_encoder(x)

    def train1(self, input):
        self.transformer_encoder.cpu()
        input = input.cpu()
        start_time = time.time()

        with torch.no_grad():
            output = self.fwd(input)

        end_time = time.time()
        print("train1: ", end_time - start_time)
        return output

    def train2(self, input):
        self.transformer_encoder.cuda()
        input = input.cuda()

        start_time = time.time()

        with torch.no_grad():
            output = self.fwd(input)

        end_time = time.time()
        print("train2: ", end_time - start_time)
        return output

    def train3(self, input):
        self._init_cache()
        self.to_cuda()
        input = input.cuda()

        start_time = time.time()

        with torch.no_grad():
            output = self.fwd(input)

        end_time = time.time()
        print("train3: ", end_time - start_time)

        return output


handler = Handler.options(num_gpus=1, num_cpus=20, max_concurrency=20).remote()

ray.get(handler._register_handler.remote(handler))

#  sequence-len, batch-size, features
input = torch.rand(2048, 2, 12288)

# ray.get([handler.train1.remote(input)])
# ray.get([handler.train1.remote(input)])

# ray.get([handler.train2.remote(input)])
# ray.get([handler.train2.remote(input)])
# ray.get([handler.train2.remote(input)])
# ray.get([handler.train2.remote(input)])
# ray.get([handler.train2.remote(input)])
# ray.get([handler.train2.remote(input)])
# exit()

ray.get([handler.train3.remote(input)])
for _, f in futs:
    ray.get(f)
ray.get([handler.train3.remote(input)])
for _, f in futs:
    ray.get(f)
ray.get([handler.train3.remote(input)])
for _, f in futs:
    ray.get(f)
ray.get([handler.train3.remote(input)])
for _, f in futs:
    ray.get(f)
ray.get([handler.train3.remote(input)])
for _, f in futs:
    ray.get(f)
ray.get([handler.train3.remote(input)])
for _, f in futs:
    ray.get(f)

"""
conclusion:

1. add py::call_guard<py::gil_scoped_release> to PYBIND11_MODULE
2. (openmp in cpp source code + ray's max_concurrency=1) is better than (no openmp + ray's max_concurrency=10)
"""
