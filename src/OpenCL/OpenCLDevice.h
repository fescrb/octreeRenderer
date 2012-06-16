#ifndef _OPENCL_DEVICE_H
#define _OPENCL_DEVICE_H

#include "Device.h"

#include "CLIncludes.h"

#include "HighResTimer.h"

class OpenCLProgram;

class OpenCLDevice:
	public Device{

	public:
                                     OpenCLDevice(cl_device_id device_id, cl_context context);
        virtual                     ~OpenCLDevice();

        void                         printInfo();
        virtual void                 makeFrameBuffer(int2 size);
        void                         sendData(Bin bin);
        void                         sendHeader(Bin bin);
        void                         setRenderInfo(renderinfo *info);
        void                         advanceTask(int index);
        void                         renderTask(int index);
        void                         calculateCostsForTask(int index);
        virtual framebuffer_window   getFrameBuffer();
        unsigned char               *getFrame();
        unsigned int                *getCosts();
        
        void                         renderEnd();

        cl_context                   getOpenCLContext();
        cl_device_id                 getOpenCLDeviceID();

        void                         onRenderingFinished();

    protected:

        cl_device_id                 m_DeviceID;
        cl_context                   m_context;
        cl_command_queue             m_commandQueue;
        cl_mem                       m_memory;
        cl_mem                       m_header;

        OpenCLProgram               *m_pProgram;
        cl_kernel                    m_rayTraceKernel;
        cl_kernel                    m_rayBundleTraceKernel;
        cl_kernel                    m_clearBufferKernel;
        cl_kernel                    m_calculateCostsKernel;
        cl_kernel                    m_clearCostsKernel;

        GLuint                       m_texture;

        cl_event                     m_eventRenderingFinished;
        cl_event                     m_eventFrameBufferRead;

        cl_mem                       m_frameBuff;
        int2                         m_frameBufferResolution;
        
        cl_mem                       m_depthBuff;
        cl_mem                       m_iterationsBuff;
        cl_mem                       m_octreeDepthBuff;
        cl_mem                       m_windowCosts;
};

#endif //_OPENCL_DEVICE_H
