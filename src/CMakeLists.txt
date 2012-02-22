include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# This is added every time.
include_directories(Common)
add_subdirectory(Common)
include_directories(Math)
add_subdirectory(Math)
include_directories(Octree)
add_subdirectory(Octree)

# Add OpenCL subdirectory only if we can.
if(USE_OPENCL)
     include_directories(${OpenCL_INCLUDE_DIR})
     include_directories(OpenCL)
     add_subdirectory(OpenCL)
endif(USE_OPENCL)

if(USE_TEST_IMPLEMENTATION)
     include_directories(Test)
     add_subdirectory(Test)
endif(USE_TEST_IMPLEMENTATION)

include(CMakeSources.cmake)

add_executable(octreeRenderer ${sources})
target_link_libraries(octreeRenderer ${Device_LIBRARIES})

install(TARGETS octreeRenderer RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION bin)

install(FILES ${shaders} DESTINATION bin/shaders)
