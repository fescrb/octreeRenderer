# Look for OpenCL.
#
#
#
# This script will define the following variables:
#  
#  OpenCL_FOUND         - True iff OpenCL is found.
#  OpenCL_INCLUDE_DIR   - The location of the OpenCL headers.
#  OpenCL_LIBRARIES     - A list containing the path of the OpenCL libraries.

find_library(OpenCL_LIBRARIES OpenCL PATHS ENV LD_LIBRARY_PATH)

find_path(OpenCL_INCLUDE_DIR CL/cl.h PATHS $ENV{AMDAPPSDKROOT}/include)
