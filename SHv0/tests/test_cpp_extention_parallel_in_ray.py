import os
import pathlib
import subprocess
import time

import ray
import torch
from torch.utils import cpp_extension

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
        srcpath / "cpp/foreach_non_finite_check_and_unscale.cpp",
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


@ray.remote
class Handler:
    def __init__(self, tensors):
        load_utils()
        import test_utils

        self.utils = test_utils
        self.tensors = tensors

    def change_shared_tensor(self, id):
        # print(f"{id} pre  ---. {self.tensors}")
        self.utils.foreach_non_finite_check_and_unscale(
            self.tensors, torch.FloatTensor([0.0]), torch.FloatTensor([1.5])
        )
        # print(f"{id} post ---. {self.tensors}")


tensors = [torch.randn([1024, 2048]) for i in range(100)]
# print(tensors)  # A

handler = Handler.options(num_gpus=1, num_cpus=2, max_concurrency=2).remote(tensors)

start_time = time.time()

# [ray.get(handler.change_shared_tensor.remote(i)) for i in range(10)]
futs = [handler.change_shared_tensor.remote(i) for i in range(10)]
ray.get(futs)

end_time = time.time()

print(end_time - start_time)  # A

"""
conclusion:

1. add py::call_guard<py::gil_scoped_release> to PYBIND11_MODULE
2. (openmp in cpp source code + ray's max_concurrency=1) is better than (no openmp + ray's max_concurrency=10)
"""
