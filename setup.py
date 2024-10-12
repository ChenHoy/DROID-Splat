from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

cxx_compiler_flags = []
if os.name == "nt":
    cxx_compiler_flags.append("/wd4624")

setup(
    name="droid_backends",
    ext_modules=[
        CUDAExtension(
            "droid_backends",
            include_dirs=[os.path.join(ROOT, "thirdparty/eigen")],
            sources=[
                "src/lib/droid.cpp",
                "src/lib/droid_kernels.cu",
                "src/lib/correlation_kernels.cu",
                "src/lib/altcorr_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

setup(
    name="lietorch",
    version="0.2",
    description="Lie Groups for PyTorch",
    packages=["lietorch"],
    package_dir={"": "thirdparty/lietorch"},
    ext_modules=[
        CUDAExtension(
            "lietorch_backends",
            include_dirs=[
                os.path.join(ROOT, "thirdparty/lietorch/lietorch/include"),
                os.path.join(ROOT, "thirdparty/eigen"),
            ],
            sources=[
                "thirdparty/lietorch/lietorch/src/lietorch.cpp",
                "thirdparty/lietorch/lietorch/src/lietorch_gpu.cu",
                "thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": [
                    "-O2",
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
                "thirdparty/simple-knn/spatial.cu",
                "thirdparty/simple-knn/simple_knn.cu",
                "thirdparty/simple-knn/ext.cpp",
            ],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

setup(
    name="diff_gaussian_rasterization",
    packages=["diff_gaussian_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "thirdparty/diff-gaussian-rasterization-w-pose/cuda_rasterizer/rasterizer_impl.cu",
                "thirdparty/diff-gaussian-rasterization-w-pose/cuda_rasterizer/forward.cu",
                "thirdparty/diff-gaussian-rasterization-w-pose/cuda_rasterizer/backward.cu",
                "thirdparty/diff-gaussian-rasterization-w-pose/rasterize_points.cu",
                "thirdparty/diff-gaussian-rasterization-w-pose/ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-I"
                    + os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "thirdparty/diff-gaussian-rasterization-w-pose/third_party/glm/",
                    )
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
