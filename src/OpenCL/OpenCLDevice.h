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
        
        void                 initializeCommandQueue();

		void	 			 printInfo();
		void 			 	 sendData(char* data);
		void				 render(RenderInfo &info);
		char    			*getFrame();
	private:

		cl_device_id 		 m_DeviceID;
        cl_command_queue     m_commandQueue;
};

#endif //_OPENCL_DEVICE_H
