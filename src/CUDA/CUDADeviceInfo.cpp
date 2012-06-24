#include "CUDADeviceInfo.h"

#include "CUDAUtils.h"

#include <cstdio>

CUDADeviceInfo::CUDADeviceInfo(int device_index) {
    cudaError_t error = cudaGetDeviceProperties(&m_properties, device_index);
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
}

char* CUDADeviceInfo::getName() {
    m_properties.name;
}

int CUDADeviceInfo::getMultiprocessorCount() {
    return m_properties.multiProcessorCount;
}

void CUDADeviceInfo::printInfo() {
    printf("\n");
    printf("Name:                               %s\n",m_properties.name);
    printf("Total Global Memory:                %d\n",m_properties.totalGlobalMem);
    printf("Shared Memory per Block:            %d\n",m_properties.sharedMemPerBlock);
    printf("Registers per Block:                %d\n",m_properties.regsPerBlock);
    printf("Warp Size:                          %d\n",m_properties.warpSize);
    printf("Max Threads per Block:              %d\n",m_properties.maxThreadsPerBlock);
    printf("Max Threads Dimensions of Block:    [%d,%d,%d]\n",m_properties.maxThreadsDim[0],m_properties.maxThreadsDim[1],m_properties.maxThreadsDim[2]);
    printf("Max Threads Dimensions of Grid:     [%d,%d,%d]\n",m_properties.maxGridSize[0],m_properties.maxGridSize[1],m_properties.maxGridSize[2]);
    printf("Multiprocessor Count:               %d\n",m_properties.multiProcessorCount);
    printf("Max Memory Pitch:                   %d\n",m_properties.memPitch);
    printf("Total Constant Memory:              %d\n",m_properties.totalConstMem);
    printf("Compute Capabilities Version:       %d.%d\n", m_properties.major,m_properties.minor);
    printf("Texture Alignment Requirement:      %d\n",m_properties.textureAlignment);
    printf("Memcpy-Kernel Overlap:              %s\n",m_properties.deviceOverlap ? "Yes" : "No");
    printf("Device Compute Mode:                %s\n",cudaComputeModeToString(m_properties.computeMode));
    printf("Max 1D Texture Size:                %d\n", m_properties.maxTexture1D);
    printf("Max 2D Texture Dimensions:          [%d,%d]\n", m_properties.maxTexture2D[0], m_properties.maxTexture2D[1]);
    printf("Max 3D Texture Dimensions:          [%d,%d,%d]\n", m_properties.maxTexture3D[0], m_properties.maxTexture3D[1], m_properties.maxTexture3D[2]);
    //printf("Max 22 Texture Array Size:          [[%d,%d],%d]\n",m_properties.maxTexture2DArray[0], m_properties.maxTexture2DArray[1], m_properties.maxTexture2DArray[2]);
    printf("Concurrent Kernels:                 %s\n", m_properties.concurrentKernels ? "Yes" : "No");
    printf("\n");
}

