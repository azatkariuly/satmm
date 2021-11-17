from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='satmm_cuda',
    ext_modules=[
        CUDAExtension('satmm_cuda', [
            'satmm.cpp',
            'satmm_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })