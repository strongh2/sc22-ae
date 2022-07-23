import sys, os
import pathlib
import subprocess

from torch.utils import cpp_extension

from megatron.utils import _get_cuda_bare_metal_version, _create_build_dir
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


    sources = [
        srcpath / "cpp/utils.cpp",
    ]
    # offloading_utils = _cpp_extention_load_helper(
    #     "offloading_utils", sources, extra_cuda_flags
    # )
