#include "OpenCLUtils.h"

const char* errorToCString(cl_int error_number) {
	switch(error_number) {
		case CL_SUCCESS:
			return "CL_SUCCESS";
		case CL_DEVICE_NOT_FOUND:
			return "CL_DEVICE_NOT_FOUND";
		default:
			return "Error number invalid or unknown.";
	}
}
