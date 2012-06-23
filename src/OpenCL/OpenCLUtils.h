#ifndef _OPENCL_UTILS_H
#define _OPENCL_UTILS_H

#include <stdio.h>

#include "CLIncludes.h"

#include "DebugUtils.h"

// This macro is only available in OCL 1.2,
// we must define it to avoid compiler errors
#ifndef CL_MEM_COPY_HOST_WRITE_ONLY
    #define CL_MEM_COPY_HOST_WRITE_ONLY 0
#endif

const char* clErrorToCString(cl_int error_code);

inline bool clIsError(cl_int error_code) {
	return error_code != CL_SUCCESS;
}

inline bool clIsBuildError(cl_int error_code) {
    return error_code == CL_BUILD_PROGRAM_FAILURE;
}

inline void clPrintError(cl_int error_code) {
	printf("Error: %s.\n", clErrorToCString(error_code));
	printf("In :");
	printStackTrace(4,1);
}

const char* clDeviceTypeToCString(cl_device_type device_type);

const char* clProgramBuildStatusToCString(cl_build_status build_status);

const char* clGetChannelOrderString(cl_channel_order channel_order);

const char* clGetImageChannelTypeString(cl_channel_type channel_type);

#endif //_OPENCL_UTILS_H
