#ifndef _OPENCL_DEVICE_H
#define _OPENCL_DEVICE_H

#include "Device.h"

#ifdef _LINUX
    #include <CL/cl.h>
#endif //_LINUX

#ifdef _OSX
    #include <OpenCL/cl.h>
#endif //_OSX

#include "HighResTimer.h"

class OpenCLProgram;

class OpenCLDevice:
	public Device{

	public:
							 OpenCLDevice(cl_device_id device_id, cl_context context);
		virtual 			~OpenCLDevice();

		void	 			 printInfo();
		void    			 makeFrameBuffer(int2 size);
        void                 sendData(Bin bin);
        void                 sendHeader(Bin bin);
		void                 renderTask(int index, renderinfo *info);
		framebuffer_window   getFrameBuffer();
		unsigned char       *getFrame();

		cl_context			 getOpenCLContext();
		cl_device_id		 getOpenCLDeviceID();

		void 				 onRenderingFinished();

        high_res_timer       getRenderTime();
        high_res_timer       getBufferToTextureTime();

	private:

		cl_device_id 		 m_DeviceID;
        cl_context           m_context;
        cl_command_queue     m_commandQueue;
        cl_mem               m_memory;
        cl_mem               m_header;

		OpenCLProgram 		*m_pProgram;
        cl_kernel            m_rayTraceKernel;
        cl_kernel            m_clearFrameBuffKernel;
        cl_kernel            m_clearDepthBuffKernel;

        GLuint               m_texture;

		cl_event             m_eventRenderingFinished;
		cl_event             m_eventFrameBufferRead;

        cl_mem               m_frameBuff;
        int2                 m_frameBufferResolution;
        
        cl_mem               m_depthBuff;
        cl_mem               m_iterationsBuff;
        cl_mem               m_octreeDepthBuff;

        high_res_timer       m_renderStart;
        high_res_timer       m_renderEnd;
        high_res_timer       m_transferStart;
        high_res_timer       m_transferEnd;
};

#endif //_OPENCL_DEVICE_H
