#include "OpenCLContext.h"
#include "OpenCLUtils.h"

#include <stdio.h>

OpenCLContext::OpenCLContext() {
	cl_uint num_of_platforms = 0;
	cl_platform_id plat;
	// We get the number of platforms.
	cl_int err = clGetPlatformIDs(0, NULL, &num_of_platforms);

	if(clIsError(err)){
		printf("Error: %s.\n", clErrorToCString(err)); return;
	}

	m_PlatformIDs = (cl_platform_id*) malloc(sizeof(cl_platform_id)*num_of_platforms + 1);

	err = clGetPlatformIDs(num_of_platforms, m_PlatformIDs, NULL);
}

OpenCLContext::~OpenCLContext() {

}
