from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("sq_dist", ['sq_dist.pyx'])]

setup(
    name = 'pointwise distance computation',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
    )
