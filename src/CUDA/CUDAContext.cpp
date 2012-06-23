#include "CUDAContext.h"

#include "CUDAIncludes.h"
#include "CUDAUtils.h"
#include "CUDADevice.h"

CUDAContext::CUDAContext() {
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
    
    printf("CUDA Context: %d devices\n",count);
    
    m_pCUDADevice = new CUDADevice(0);
}

Device* CUDAContext::getDevice(int index) {
    return m_pCUDADevice;
}

unsigned int CUDAContext::getNumDevices(){
    return 1;
}


std::vector< Device* > CUDAContext::getDeviceList() {
    std::vector< Device* > list = std::vector< Device* >(0);
    list.push_back(m_pCUDADevice);
    
    return list;
}

void CUDAContext::printDeviceInfo() {
    m_pCUDADevice->printInfo();
}



