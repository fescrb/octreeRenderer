project(octreeRenderer)

# Check command line options for usage of OpenCL.
set(USE_OPENCL "Yes")
if(DONT_USE_OPENCL)
     set(USE_OPENCL "No")
endif(DONT_USE_OPENCL)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	if(DONT_USE_OPENMP)
		set(USE_OPENMP "Yes")
    endif(DONT_USE_OPENMP)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

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

if(USE_OPENMP)

	find_package(OpenMP)

	if(OPENMP_FOUND)
		message("Message: Using OpenMP")
		set(USE_OPENMP "Yes")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
		# We need to compile the serial implementation to use OpenMP
		set(USE_SERIAL_IMPLEMENTATION "Yes")
		set(USE_OPENMP_IMPLEMENTATION "Yes")
		add_definitions(-DUSE_OPENMP)
	else(OPENMP_FOUND)
		message("Error: OpenMP not found")
		set(USE_OPENMP "No")
	endif(OPENMP_FOUND)

endif(USE_OPENMP)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
set(Device_LIBRARIES ${Device_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}) 
include_directories(${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})
# To allow usage of glext extensions
add_definitions(-DGL_GLEXT_PROTOTYPES)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    add_definitions(-D_LINUX)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions(-D_OSX)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

if(NOT USE_OPENCL AND NOT USE_OPENMP)
	set(USE_SERIAL_IMPLEMENTATION "Yes")
	
	message("MESSAGE: Using serial implementation")
	
	add_definitions(-DUSE_SERIAL)
endif(NOT USE_OPENCL AND NOT USE_OPENMP)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
	set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

add_subdirectory(src)
