#ifndef _CUDA_CONTEXT_H
#define _CUDA_CONTEXT_H

#include "Context.h"

class CUDADevice;

class CUDAContext
:   public Context {
    public:
                                     CUDAContext();
                     
        void                         printDeviceInfo();

        unsigned int                 getNumDevices();
        Device                      *getDevice(int index);
        
        std::vector<Device*>         getDeviceList();
        
    private:
        std::vector<CUDADevice*>     m_vCUDADevices;
};

#endif //_CUDA_CONTEXT_H