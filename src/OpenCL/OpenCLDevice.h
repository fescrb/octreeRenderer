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
		void 			 	 sendData(OctreeSegment* segment);
		void				 render(int2 start, int2 size, RenderInfo *info);
		char    			*getFrame();
        
	private:

		cl_device_id 		 m_DeviceID;
        cl_context           m_context;
        cl_command_queue     m_commandQueue;
        cl_mem               m_memory;
};

#endif //_OPENCL_DEVICE_H
