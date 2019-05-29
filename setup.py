from distutils.core import setup
from Cython.Build import cythonize

import numpy

setup(ext_modules=cythonize("information_coefficient.pyx"),
      include_dirs=[numpy.get_include()])