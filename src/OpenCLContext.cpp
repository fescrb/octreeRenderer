#include "OpenCLContext.h"

#include <stdio.h>

OpenCLContext::OpenCLContext() {
	cl_uint num_of_platforms = 0;
	// We get the number of platforms.
	clGetPlatformIDs(0, NULL, &num_of_platforms);

	printf("There are %d platforms.\n", num_of_platforms);
}

OpenCLContext::~OpenCLContext() {

}
