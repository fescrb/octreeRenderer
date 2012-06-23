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
    
    for(int i = 0; i < count; i++) {
        m_vCUDADevices.push_back(new CUDADevice(i));
    }
}

Device* CUDAContext::getDevice(int index) {
    return m_vCUDADevices[index];
}

unsigned int CUDAContext::getNumDevices(){
    return m_vCUDADevices.size();
}


std::vector< Device* > CUDAContext::getDeviceList() {
    std::vector< Device* > list = std::vector< Device* >(0);
    
    for(int i = 0; i < m_vCUDADevices.size(); i++) {
        list.push_back(m_vCUDADevices[i]);
    }
    
    return list;
}

void CUDAContext::printDeviceInfo() {
    for(int i = 0; i < m_vCUDADevices.size(); i++) {
        m_vCUDADevices[i]->printInfo();
    }
}



