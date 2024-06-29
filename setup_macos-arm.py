from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy
import platform as plt
import sys
import pathlib

# Remove previous build artifacts
os.system('rm -f pyMilne.*.so pyMilne.*.cpp')
p = pathlib.Path(sys.executable)
root_dir = str(pathlib.Path(*p.parts[0:-2]))

# Get the conda prefix
conda_prefix = os.environ.get("CONDA_PREFIX")

# Determine platform-specific settings
if plt.system() == 'Darwin':
    root_dir = '/opt/local/'  # using this one if macports are installed
    CC = f'{conda_prefix}/bin/clang'
    CXX = f'{conda_prefix}/bin/clang++'
    link_opts = ["-stdlib=libc++", "-bundle", "-undefined",
                 "dynamic_lookup", "-fopenmp", "-lgomp"]
    # Adjust flags for ARM architecture
    comp_flags = ['-Ofast', '-g0', '-fstrict-aliasing', '-mtune=native', '-std=c++20', '-fPIC',
                  '-fopenmp', '-I./src', "-DNPY_NO_DEPRECATED_API", '-mprefer-vector-width=256', '-DNDEBUG',
                  '-pedantic', '-Wall', f'-I{conda_prefix}/include/eigen3']
else:
    # root_dir = '/usr/'
    CC = f'{conda_prefix}/bin/gcc'
    CXX = f'{conda_prefix}/bin/g++'
    link_opts = ["-shared", "-fopenmp"]
    comp_flags = ['-Ofast', '-g0', '-fstrict-aliasing', '-mtune=native', '-std=c++20', '-fPIC',
                  '-fopenmp', '-I./src', "-DNPY_NO_DEPRECATED_API", '-mprefer-vector-width=256', '-DNDEBUG',
                  '-pedantic', '-Wall', f'-I{conda_prefix}/include/eigen3']

# Set environment variables for the compilers
os.environ["CC"] = CC
os.environ["CXX"] = CXX

# Print debug information
print("Conda Prefix:", conda_prefix)
print("Compiler:", os.environ["CC"])
print("C++ Compiler:", os.environ["CXX"])

# Define the extension module
extension = Extension("pyMilne",
                      sources=["pyMilne.pyx", "src/wrapper_tools_spatially_coupled.cpp", "src/lm_sc.cpp",
                               "src/spatially_coupled_helper.cpp"],
                      include_dirs=[
                          "./", numpy.get_include(), './eigen3', f'{root_dir}/include/',
                          f'{conda_prefix}/include/eigen3'],
                      language="c++",
                      extra_compile_args=comp_flags,
                      extra_link_args=link_opts,
                      library_dirs=['./', "/usr/lib/"],
                      libraries=['fftw3'])

extension.cython_directives = {'language_level': "3"}

# Setup the package
setup(
    name='pyMilne',
    version='3.0',
    author='J. de la Cruz Rodriguez (ISP-SU 2018 - 2023)',
    ext_modules=[extension],
    cmdclass={'build_ext': build_ext}
)
