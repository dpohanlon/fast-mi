# setup.py
import os
from skbuild import setup
from setuptools import find_packages

setup(
    name="fast_mutual_information",
    version="0.1.0",
    description="Fast pairwise mutual information computation.",
    author="Daniel O'Hanlon",
    author_email="dpohanlon@gmail.com",
    license="MIT",
    packages=['fast_mutual_information'],
    install_requires=[
        "numpy>=1.18.0",
        "pybind11>=2.6",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
    ],
    python_requires='>=3.9,<4.0',
    # cmake_args=["-DCMAKE_CXX_STANDARD=11"],
)
