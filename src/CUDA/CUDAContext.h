#ifndef _CUDA_CONTEXT_H
#define _CUDA_CONTEXT_H

#include "Context.h"

class CUDAContext
:   public Context {
    public:
                             CUDAContext();
                     
        void                 printDeviceInfo();

        unsigned int         getNumDevices();
        Device              *getDevice(int index);
        
        std::vector<Device*> getDeviceList();
};

#endif //_CUDA_CONTEXT_H