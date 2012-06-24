#ifndef _CUDA_DEVICE_INFO_H
#define _CUDA_DEVICE_INFO_H

#include "DeviceInfo.h"

#include "CUDAIncludes.h"

class CUDADeviceInfo
:   public DeviceInfo {
    public:
                         CUDADeviceInfo(int device_index);
        
        void             printInfo();
        
        char            *getName();
        
        int              getMultiprocessorCount();
        
    private:
        cudaDeviceProp   m_properties;
};

#endif //_CUDA_DEVICE_INFO_H