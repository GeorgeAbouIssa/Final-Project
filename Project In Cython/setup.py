from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os

# Current Date and Time (UTC): 2025-05-06 06:24:13
# Current User's Login: GeorgeAbouIssa

# First compile the forward declarations
extensions = [
    Extension(
        "forward_declarations",
        ["forward_declarations.pxd"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
]

# Then compile the main modules
extensions += [
    Extension(
        "MovementPhases",
        ["MovementPhases.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        "ObstacleHandler",
        ["ObstacleHandler.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        "DisconnectedGoal",
        ["DisconnectedGoal.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        "ConnectedMatterAgent",
        ["ConnectedMatterAgent.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
    Extension(
        "Controller",
        ["Controller.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
]

setup(
    name="ConnectedMatterProject",
    version="1.0.0",
    author="GeorgeAbouIssa",
    description="Cython implementation of connected matter agent simulation",
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
        'initializedcheck': False,
        'cdivision': True,
    }),
    install_requires=[
        'numpy',
        'matplotlib',
        'cython',
    ],
)