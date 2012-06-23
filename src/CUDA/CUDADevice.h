#ifndef _CUDA_DEVICE_H
#define _CUDA_DEVICE_H

#include "Device.h"

class CUDADevice
: public Device {
    public:
                             CUDADevice(int device_index);
        virtual             ~CUDADevice();
        
        void                 printInfo();

        /**
         * We clear the framebuffer if we needen't generate it
         * @param The dimensions of the required framebuffer.
         */
        void                 makeFrameBuffer(vector::int2 size);
        void                 sendData(Bin bin);
        void                 sendHeader(Bin bin);
        void                 setRenderInfo(renderinfo *info);
        void                 advanceTask(int index);
        void                 renderTask(int index);
        void                 calculateCostsForTask(int index);
        
        framebuffer_window   getFrameBuffer();
        unsigned char       *getFrame();
        unsigned int        *getCosts();
        
        bool                 isCPU();
        
    private:
        char                *m_pOctree;
        char                *m_pHeader;
};

#endif //_CUDA_DEVICE_H