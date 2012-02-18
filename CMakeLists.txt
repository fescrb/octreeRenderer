project(octreeRenderer)

# Check command line options for usage of OpenCL.
set(USE_OPENCL "Yes")
if(DONT_USE_OPENCL)
     set(USE_OPENCL "No")
endif(DONT_USE_OPENCL)

cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

# If we are allowed to use OpenCL, try to look for it.
if(USE_OPENCL)
     find_package(OpenCL)
	  
     if(OpenCL_FOUND)
	  set(USE_OPENCL "Yes")

	  add_definitions(-DUSE_OPENCL)
	  set(Device_LIBRARIES ${Device_LIBRARIES} ${OpenCL_LIBRARIES}) 
     else(OpenCL_FOUND)
	  set(USE_OPENCL "No")
	  message("ERROR: OpenCL not found")
     endif(OpenCL_FOUND)
endif(USE_OPENCL)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

message("libs " ${OpenGL_LIBRARIES})

set(Device_LIBRARIES ${Device_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}) 
include_directories(${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    add_definitions(-D_LINUX)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions(-D_OSX)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

if(NOT USE_OPENCL)
	set(USE_TEST_IMPLEMENTATION "Yes")
	
	message("MESSAGE: Using Host as device")
	
	add_definitions(-DUSE_HOST_CPU)
endif(NOT USE_OPENCL)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
	set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

add_subdirectory(src)
