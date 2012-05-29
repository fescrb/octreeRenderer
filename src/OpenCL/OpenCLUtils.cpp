#include "OpenCLUtils.h"

#include "OpenCLExtra.h"

const char* clErrorToCString(cl_int error_number) {
	switch(error_number) {
		case CL_SUCCESS:
			return "CL_SUCCESS";
		case CL_DEVICE_NOT_FOUND:
			return "CL_DEVICE_NOT_FOUND";
		case CL_DEVICE_NOT_AVAILABLE:
			return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE:
			return "CL_COMPILER_NOT_AVAILABLE";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES:
			return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY:
			return "CL_OUT_OF_HOST_MEMORY";
		case CL_PROFILING_INFO_NOT_AVAILABLE:
			return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case CL_MEM_COPY_OVERLAP:
			return "CL_MEM_COPY_OVERLAP";
		case CL_IMAGE_FORMAT_MISMATCH:
			return "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:
			return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE:
			return "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE:
			return "CL_MAP_FAILURE";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_INVALID_VALUE:
			return "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE:
			return "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM:
			return "CL_INVALID_PLATFORM";
		case CL_INVALID_DEVICE:
			return "CL_INVALID_DEVICE";
		case CL_INVALID_CONTEXT:
			return "CL_INVALID_CONTEXT";
		case CL_INVALID_QUEUE_PROPERTIES:
			return "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE:
			return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR:
			return "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT:
			return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case CL_INVALID_IMAGE_SIZE:
			return "CL_INVALID_IMAGE_SIZE";
		case CL_INVALID_SAMPLER:
			return "CL_INVALID_SAMPLER";
		case CL_INVALID_BINARY:
			return "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS:
			return "CL_INVALID_BUILD_OPTIONS";
		case CL_INVALID_PROGRAM:
			return "CL_INVALID_PROGRAM";
		case CL_INVALID_PROGRAM_EXECUTABLE:
			return "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME:
			return "CL_INVALID_KERNEL_NAME";
		case CL_INVALID_KERNEL_DEFINITION:
			return "CL_INVALID_KERNEL_DEFINITION";
		case CL_INVALID_KERNEL:
			return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX:
			return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE:
			return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE:
			return "CL_INVALID_ARG_SIZE";
		case CL_INVALID_KERNEL_ARGS:
			return "CL_INVALID_KERNEL_ARGS";
		case CL_INVALID_WORK_DIMENSION:
			return "CL_INVALID_WORK_DIMENSION";
		case CL_INVALID_WORK_GROUP_SIZE:
			return "CL_INVALID_WORK_GROUP_SIZE";
		case CL_INVALID_WORK_ITEM_SIZE:
			return "CL_INVALID_WORK_ITEM_SIZE";
		case CL_INVALID_GLOBAL_OFFSET:
			return "CL_INVALID_GLOBAL_OFFSET";
		case CL_INVALID_EVENT_WAIT_LIST:
			return "CL_INVALID_EVENT_WAIT_LIST";
		case CL_INVALID_EVENT:
			return "CL_INVALID_EVENT";
		case CL_INVALID_OPERATION:
			return "CL_INVALID_OPERATION";
		case CL_INVALID_GL_OBJECT:
			return "CL_INVALID_GL_OBJECT";
		case CL_INVALID_BUFFER_SIZE:
			return "CL_INVALID_BUFFER_SIZE";
		case CL_INVALID_MIP_LEVEL:
			return "CL_INVALID_MIP_LEVEL";
		case CL_INVALID_GLOBAL_WORK_SIZE:
			return "CL_INVALID_GLOBAL_WORK_SIZE";
		case CL_INVALID_PROPERTY:
			return "CL_INVALID_PROPERTY";
		case CL_NO_ICD_FOUND:
			return "CL_NO_ICD_FOUND: OpenCL has found no .icd files.";
		default:
			return "Error number invalid or unknown";
	}
}

const char* clDeviceTypeToCString(cl_device_type device_type){
	switch(device_type){
		case CL_DEVICE_TYPE_GPU:
			return "GPU";
		case CL_DEVICE_TYPE_CPU:
			return "CPU";
		case CL_DEVICE_TYPE_ACCELERATOR:
			return "Accelerator";
		case CL_DEVICE_TYPE_DEFAULT:
			return "Default";
		default:
			return "Device type invalid or unknown";
	}
}

const char* clProgramBuildStatusToCString(cl_build_status build_status) {
	switch(build_status){
		case CL_BUILD_NONE:
			return "CL_BUILD_NONE";
		case CL_BUILD_ERROR:
			return "CL_BUILD_ERROR";
		case CL_BUILD_SUCCESS:
			return "CL_BUILD_SUCCESS";
		case CL_BUILD_IN_PROGRESS:
			return "CL_BUILD_IN_PROGRESS";
		default:
			return "Build status invalid or unknown";
	}
}

const char* clGetChannelOrderString(cl_channel_order channel_order) {
    switch(channel_order){
        case CL_R:
            return "CL_R";
        case CL_Rx:
            return "CL_Rx";
        case CL_A:
            return "CL_A";
        case CL_INTENSITY:
            return "CL_INTENSITY";
        case CL_LUMINANCE:
            return "CL_LUMINANCE";
        case CL_RG:
            return "CL_RG";
        case CL_RGx:
            return "CL_RGx";
        case CL_RA:
            return "CL_RA";
        case CL_RGB:
            return "CL_RGB";
        case CL_RGBx:
            return "CL_RGBx";
        case CL_RGBA:
            return "CL_RGBA";
        case CL_ARGB:
            return "CL_ARGB";
        case CL_BGRA:
            return "CL_BGRA";
        default:
            return "Channel order invalid or unknown";
    }
}

const char* clGetImageChannelTypeString(cl_channel_type channel_type) {
    switch(channel_type){
        case CL_SNORM_INT8:
            return "CL_SNORM_INT8";
        case CL_SNORM_INT16:
            return "CL_SNORM_INT16";
        case CL_UNORM_INT8:
            return "CL_UNORM_INT8";
        case CL_UNORM_INT16:
            return "CL_UNORM_INT16";
        case CL_UNORM_SHORT_565:
            return "CL_UNORM_SHORT_565";
        case CL_UNORM_SHORT_555:
            return "CL_UNORM_SHORT_555";
        case CL_UNORM_INT_101010:
            return "CL_UNORM_INT_101010";
        case CL_SIGNED_INT8:
            return "CL_SIGNED_INT8";
        case CL_SIGNED_INT16:
            return "CL_SIGNED_INT16";
        case CL_SIGNED_INT32:
            return "CL_SIGNED_INT32";
        case CL_UNSIGNED_INT8:
            return "CL_UNSIGNED_INT8";
        case CL_UNSIGNED_INT16:
            return "CL_UNSIGNED_INT16";
        case CL_UNSIGNED_INT32:
            return "CL_UNSIGNED_INT32";
        case CL_HALF_FLOAT:
            return "CL_HALF_FLOAT";
        case CL_FLOAT:
            return "CL_FLOAT";
        default:
            return "Channel order invalid or unknown";
    }
}