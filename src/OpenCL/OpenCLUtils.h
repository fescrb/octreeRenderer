#ifndef _OPENCL_UTILS_H
#define _OPENCL_UTILS_H

#include <CL/cl.h>
#include <stdio.h>

const char* clErrorToCString(cl_int error_code);

inline bool clIsError(cl_int error_code) {
	return error_code != CL_SUCCESS;
}

inline void clPrintError(cl_int error_code) {
	printf("Error: %s.\n", clErrorToCString(error_code));
}

#endif //_OPENCL_UTILS_H
