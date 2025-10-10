# setup.py
import os
from skbuild import setup
from setuptools import find_packages

setup(
    name="fast_mutual_information",
    version="0.1.1",
    description="Fast pairwise mutual information computation.",
    author="Daniel O'Hanlon, Sergio Garcia-Busto",
    author_email="dpohanlon@gmail.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.18.0",
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
    python_requires=">=3.9,<4.0",
    # cmake_args=["-DCMAKE_CXX_STANDARD=11"],
)
