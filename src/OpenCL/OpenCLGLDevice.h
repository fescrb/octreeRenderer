#ifndef _OPENCL_GL_DEVICE_H
#define _OPENCL_GL_DEVICE_H

#include "OpenCLDevice.h"

class OpenCLGLDevice
:   public OpenCLDevice {
    public:
                             OpenCLGLDevice(cl_device_id device_id, cl_context context);
                    
        void                 makeFrameBuffer(int2 size);
        framebuffer_window   getFrameBuffer();
};

#endif //_OPENCL_GL_DEVICE_H