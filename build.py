import os
import sys

# Declare variables
verbose = ""
use_ocl = " "
build_type = ' -DCMAKE_BUILD_TYPE="Debug" ' 
test = 0
extra_defs = ''

if len(sys.argv) > 1 :
    argc = 1
    if sys.argv[argc] == "clean" :
        os.system("rm -rf obj")
        os.system("rm -rf bin")
        exit(0)
    while argc < len(sys.argv) :
        command = sys.argv[argc]
        if command == "extraflags":
            argc+=1
            extra_flags = sys.argv[argc]
        argc+=1
        if command == "verbose":
            verbose = " VERBOSE=1 "
        if command == "noocl":
            use_ocl = ' -DDONT_USE_OPENCL="Yes" '
        if command == "debugfindocl":
            extra_defs = extra_defs + ' -DDEBUG_FINDOPENCL="Yes" '


if not os.path.exists('obj'):
    os.mkdir('obj')
    
if not os.path.exists('bin'):
    os.mkdir('bin')
    
os.chdir('obj')

cmake_command = 'cmake .. ' + build_type + use_ocl + extra_defs + ' '
make_command = 'make ' + verbose
install_command = 'make install '

if not os.system(cmake_command):
    if not os.system(make_command):
        if not os.system(install_command):
            sys.exit(0)

sys.exit(1)
