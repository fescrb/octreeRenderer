#ifndef _OPENCL_DEVICE_H
#define _OPENCL_DEVICE_H

#include "Device.h"

#ifdef _LINUX
    #include <CL/cl.h>
#endif //_LINUX

#ifdef _OSX
    #include <OpenCL/cl.h>
#endif //_OSX

class OpenCLDevice:
	public Device{

	public:
							 OpenCLDevice(cl_device_id device_id, cl_context context);
		virtual 			~OpenCLDevice();

		void	 			 printInfo();
		void    			 makeFrameBuffer(int2 size);
		void 			 	 sendData(OctreeSegment* segment);
		void				 render(int2 start, int2 size, renderinfo *info);
		GLuint   			 getFrameBuffer();
		char    			*getFrame();
        
	private:

		cl_device_id 		 m_DeviceID;
        cl_context           m_context;
        cl_command_queue     m_commandQueue;
        cl_mem               m_memory;
        
        cl_program           m_rayTracingProgram;
        cl_kernel            m_rayTraceKernel;
        
        GLuint               m_texture;
        
        cl_mem               m_frameBuff;
        int2                 m_frameBufferResolution;
};

#endif //_OPENCL_DEVICE_H
