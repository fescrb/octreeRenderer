include(CMakeSources.cmake)

include_directories(${OpenCL_INCLUDE_DIR})

add_executable(octreeRenderer ${sources})
target_link_libraries(octreeRenderer ${OpenCL_LIBRARIES})
