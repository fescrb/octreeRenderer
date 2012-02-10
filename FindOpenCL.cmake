# Look for OpenCL.
#
#
#
# This script will define the following variables:
#  
#  OpenCL_FOUND         - True iff OpenCL is found.
#  OpenCL_INCLUDE_DIR   - The location of the OpenCL headers.
#  OpenCL_LIBRARIES     - A list containing the path of the OpenCL libraries.

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    find_library(OpenCL_LIBRARIES OpenCL PATHS ENV LD_LIBRARY_PATH)

    find_path(OpenCL_INCLUDE_DIR CL/cl.h PATHS $ENV{AMDAPPSDKROOT}/include ENV CPLUS_INCLUDE_PATH)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    find_library(OpenCL_LIBRARIES OpenCL)

    find_path(OpenCL_INCLUDE_DIR OpenCL/cl.h)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

if(${OpenCL_LIBRARIES} MATCHES "OpenCL_LIBRARIES-NOTFOUND" OR ${OpenCL_INCLUDE_DIR} MATCHES "OpenCL_INCLUDE_DIR-NOTFOUND")
     set(OpenCL_FOUND "No")
else(${OpenCL_LIBRARIES} MATCHES "OpenCL_LIBRARIES-NOTFOUND" OR ${OpenCL_INCLUDE_DIR} MATCHES "OpenCL_INCLUDE_DIR-NOTFOUND")
     set(OpenCL_FOUND "Yes")
endif(${OpenCL_LIBRARIES} MATCHES "OpenCL_LIBRARIES-NOTFOUND" OR ${OpenCL_INCLUDE_DIR} MATCHES "OpenCL_INCLUDE_DIR-NOTFOUND")

if(${DEBUG_FINDOPENCL} MATCHES "Yes")

     message("OpenCL_LIBRARIES " ${OpenCL_LIBRARIES})

     message("OpenCL_INCLUDE_DIR " ${OpenCL_INCLUDE_DIR})

endif(${DEBUG_FINDOPENCL} MATCHES "Yes")
