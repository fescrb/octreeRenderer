#include "CUDAUtils.h"

bool cudaIsError(cudaError_t value){
    return value != cudaSuccess;
}

const char* cudaErrorToString(cudaError_t error_code) {
    switch(error_code) {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";
        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        default:
            return "Error Unknown";
    }
}

const char* cudaComputeModeToString(int compute_mode) {
    switch(compute_mode) {
        case cudaComputeModeDefault:
            return "cudaComputeModeDefault";
        case cudaComputeModeExclusive:
            return "cudaComputeModeExclusive";
        case cudaComputeModeProhibited:
            return "cudaComputeModeProhibited";
        case cudaComputeModeExclusiveProcess:
            return "cudaComputeModeExclusiveProcess";
        default:
            return "Compute Mode Unknown";
    }
}
