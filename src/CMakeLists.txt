include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add OpenCL subdirectory only if we can.
if(USE_OPENCL)
     include_directories(${OpenCL_INCLUDE_DIR})
     include_directories(OpenCL)
     add_subdirectory(OpenCL)
endif(USE_OPENCL)

include(CMakeSources.cmake)

add_executable(octreeRenderer ${sources})
target_link_libraries(octreeRenderer ${Device_LIBRARIES})
