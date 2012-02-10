#ifndef _OPENCL_UTILS_H
#define _OPENCL_UTILS_H

#include <stdio.h>

#ifdef _LINUX
    #include <CL/cl.h>
#endif //_LINUX

#ifdef _OSX
    #include <OpenCL/cl.h>
#endif //_OSX

const char* clErrorToCString(cl_int error_code);

inline bool clIsError(cl_int error_code) {
	return error_code != CL_SUCCESS;
}

inline void clPrintError(cl_int error_code) {
	printf("Error: %s.\n", clErrorToCString(error_code));
}

const char* clDeviceTypeToCString(cl_device_type device_type);

#endif //_OPENCL_UTILS_H
