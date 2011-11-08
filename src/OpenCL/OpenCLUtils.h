#ifndef _OPENCL_UTILS_H
#define _OPENCL_UTILS_H

#include <CL/cl.h>

const char* clErrorToCString(cl_int error_code);

inline bool clIsError(cl_int error_code) {
	return error_code != CL_SUCCESS;
}

#endif //_OPENCL_UTILS_H
