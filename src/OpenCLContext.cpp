#include "OpenCLContext.h"
#include "OpenCLUtils.h"

#include <stdio.h>

OpenCLContext::OpenCLContext() {
	cl_uint num_of_platforms = 0;
	cl_platform_id plat;
	// We get the number of platforms.
	cl_int err = clGetPlatformIDs(1, &plat, &num_of_platforms);

	printf("There are %d platforms, with error %s.\n", num_of_platforms, clErrorToCString(err));
}

OpenCLContext::~OpenCLContext() {

}
