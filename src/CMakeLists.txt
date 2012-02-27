include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# This is added every time.
include_directories(Common)
add_subdirectory(Common)
include_directories(Math)
add_subdirectory(Math)
include_directories(Octree)
add_subdirectory(Octree)
include_directories(Base)
add_subdirectory(Base)

# Add OpenCL subdirectory only if we can.
if(USE_OPENCL)
     include_directories(${OpenCL_INCLUDE_DIR})
     include_directories(OpenCL)
     add_subdirectory(OpenCL)
endif(USE_OPENCL)

if(USE_OPENMP_IMPLEMENTATION)
     include_directories(OpenMP)
     add_subdirectory(OpenMP)
endif(USE_OPENMP_IMPLEMENTATION)

if(USE_SERIAL_IMPLEMENTATION)
     include_directories(Serial)
     add_subdirectory(Serial)
endif(USE_SERIAL_IMPLEMENTATION)

include(CMakeSources.cmake)

add_executable(octreeRenderer ${sources})
target_link_libraries(octreeRenderer ${Device_LIBRARIES})

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(octreeRenderer rt)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

install(TARGETS octreeRenderer RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION bin)

install(FILES ${shaders} DESTINATION bin/shaders)
