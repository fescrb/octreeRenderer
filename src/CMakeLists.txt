include_directories(.)

include_directories(${OpenCL_INCLUDE_DIR})
include_directories(OpenCL)
add_subdirectory(OpenCL)

include(CMakeSources.cmake)

add_executable(octreeRenderer ${sources})
target_link_libraries(octreeRenderer ${OpenCL_LIBRARIES})
