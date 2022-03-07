import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


if __name__ == '__main__':
    setup(
        name='SmartCenterPoint',
        version='0.1.0',
        description='SmartCenterPoint is a general codebase for 3D object detection from lidar point cloud',
        install_requires=[
            'numpy==1.22.2',
            'llvmlite==0.36.0',
            'numba==0.53.0',
            'tensorboardX==2.4.1',
            'easydict==1.9',
            'pyyaml==5.4.1',
            'scikit-image==0.19.1',
            'SharedArray==3.2.1',
            'spconv-cu113==2.1.21',
            'torch==1.10.2+cu113',
        ],

        author='Gavin Xu',
        author_email='Gavin.Xu@smart.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='src.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='src.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ]
            ),
        ],
    )
