# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-10-02 13:03:55
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-03 12:55:06
from distutils.core import setup
from Cython.Build import cythonize
import os
from shutil import copyfile

# Cython files that will be compiled and main
dirname = os.path.dirname(os.path.abspath(__file__))+'/'
pyx_files = [dirname+"*.pyx"]
py_files = [dirname+"main.py", dirname+"utils.py"]
release_folder = dirname+"release/"
# Compile cython files
# annotate=True # enables generation html file
setup(ext_modules=cythonize(pyx_files))
# Remove .c files and build directory after setup the .so files
files = os.listdir(os.path.abspath(os.getcwd()))
# .c and .cpython ... .so files
c_cpython_files = [file for file in files if ".c" in file]
cpython_files = [file for file in files if ".cpython" in file]  # .so files
# Move compiled files to release folder.Also utils, main
if not os.path.isdir(release_folder):
    os.makedirs(release_folder)
# copy .so and .py files
for file in cpython_files+py_files:
    filename = os.path.basename(file)
    release_filename = release_folder+filename
    # If both are the same file, catch exception.
    # Updates release file. Needed due to error on update process
    try:
        copyfile(file, release_filename)
    except:
        os.remove(release_filename)
        copyfile(file, release_filename)
        pass
# remove .so and .c files in main folder
for file in c_cpython_files:
    os.remove(os.path.abspath(file))
    print("file:", file, "has been removed")
