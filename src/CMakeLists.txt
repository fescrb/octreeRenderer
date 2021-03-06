include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# This is added every time.
include_directories(Common)
add_subdirectory(Common)
include_directories(Graphics)
add_subdirectory(Graphics)
include_directories(Math)
add_subdirectory(Math)
include_directories(Octree)
add_subdirectory(Octree)
include_directories(IO)
add_subdirectory(IO)
include_directories(Base)
add_subdirectory(Base)

# Add OpenCL subdirectory only if we can.
if(USE_OPENCL)
     include_directories(${OpenCL_INCLUDE_DIR})
     include_directories(OpenCL)
     add_subdirectory(OpenCL)
endif(USE_OPENCL)

if(USE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(CUDA)
    add_subdirectory(CUDA)
endif(USE_CUDA)

if(USE_OPENMP_IMPLEMENTATION)
     include_directories(OpenMP)
     add_subdirectory(OpenMP)
endif(USE_OPENMP_IMPLEMENTATION)

if(USE_SERIAL_IMPLEMENTATION)
     include_directories(Serial)
     add_subdirectory(Serial)
endif(USE_SERIAL_IMPLEMENTATION)

include(CMakeSources.cmake)

if(USE_CUDA)
    cuda_add_library(Octree ${sources})
else(USE_CUDA)
    add_library(Octree ${sources})
endif(USE_CUDA)

add_executable(octreeRenderer octreeRenderer.cpp)
target_link_libraries(octreeRenderer Octree ${Device_LIBRARIES})

add_executable(geometryToOctree geometryToOctree.cpp)
target_link_libraries(geometryToOctree Octree ${Device_LIBRARIES})

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(octreeRenderer rt)
    target_link_libraries(geometryToOctree rt)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

install(TARGETS octreeRenderer RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION bin)
install(TARGETS geometryToOctree RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION bin)

install(FILES ${shaders} DESTINATION bin/shaders)
