import os
import setuptools
import sys

import numpy as np

def read(fname):
  with open(os.path.join(os.path.dirname(__file__), fname), 'rt') as f:
    return f.read()

def requirements():
  with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'rt') as f:
    return f.readlines()

extra_compile_args = []
if sys.platform == 'win32':
  extra_compile_args += [
    '/std:c++11', '/O2'
  ]
else:
  extra_compile_args += [
    '-std=c++11', '-O3'
  ]

if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

setuptools.setup(
  name="zfpc",
  version="0.0.1",
  setup_requires=['pbr', 'numpy'],
  install_requires=['numpy'],
  python_requires="~=3.7", # >= 3.7 < 4.0
  ext_modules=[
    setuptools.Extension(
      'zfpc',
      sources=[ 'zfpc.pyx' ],
      language='c++',
      include_dirs=[ np.get_include() ],
      extra_compile_args=extra_compile_args,
    )
  ],
  author="William Silversmith",
  author_email="ws9@princeton.edu",
  packages=setuptools.find_packages(),
  package_data={
    'zfpc': [
      'LICENSE',
    ],
  },
  description="zfp container for optimal compression of 1D-4D arrays by representing correlated dimensions as separate streams.",
  long_description=read('README.md'),
  long_description_content_type="text/markdown",
  license = "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
  keywords = "compression zfp volumetric-data numpy image-processing 2d 3d 4d",
  url = "https://github.com/seung-lab/zfpc/",
  classifiers=[
    "Intended Audience :: Developers",
    "Development Status :: 2 - Pre-Alpha",
    "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows :: Windows 10",
  ],  
)


