#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

#include "DebugUtils.h"

bool cudaIsError(cudaError_t value);

const char* cudaErrorToString(cudaError_t error_code);

const char* cudaComputeModeToString(int compute_mode);

inline void cudaPrintError(cudaError_t error_code) {
    printf("Error: %s.\n", cudaErrorToString(error_code));
    printf("In :");
    printStackTrace(4,1);
}

#endif //_CUDA_UTILS_H