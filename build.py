import os
import sys

# Declare variables
verbose = ""
use_ocl = " "
build_type = ' -DCMAKE_BUILD_TYPE="Debug" ' 
test = 0

if sys.argv[1] == "clean" :
    os.rmdir("obj")
    exit(0)

if len(sys.argv) > 1 :
    argc = 1
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


if not os.path.exists('obj'):
    os.mkdir('obj')
    
os.chdir('obj')

cmake_command = 'cmake .. ' + use_simd + ' '
make_command = 'make ' + verbose
test_command = 'src/Unit\\ Testing/Core/core_test'

if not os.system(cmake_command):
    if not os.system(make_command):
       sys.exit(0)

sys.exit(1)