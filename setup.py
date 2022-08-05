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

setuptools.setup(
  name="zfpc",
  version="0.1.1",
  setup_requires=['numpy'],
  install_requires=['numpy'],
  python_requires=">=3.7,<4.0", # >= 3.7 < 4.0
  author="William Silversmith",
  author_email="ws9@princeton.edu",
  packages=setuptools.find_packages(),
  package_data={
    'zfpc': [
      'LICENSE',
    ],
  },
  description="zfp container (zfpc) for optimal compression of 1D-4D arrays by representing correlated dimensions as separate zfp streams.",
  long_description=read('README.md'),
  long_description_content_type="text/markdown",
  license = "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
  keywords = "compression zfp volumetric-data numpy image-processing 2d 3d 4d",
  url = "https://github.com/seung-lab/zfpc/",
  classifiers=[
    "Intended Audience :: Developers",
    "Development Status :: 3 - Alpha",
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


