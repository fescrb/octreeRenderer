#ifndef _OPENCL_DEVICE_H
#define _OPENCL_DEVICE_H

#include "Device.h"
#include <CL/cl.h>

class OpenCLDevice:
	public Device{

	public:
							 OpenCLDevice(cl_device_id device_id);
		virtual 			~OpenCLDevice();

		void	 			 printInfo();
		void 			 	 sendData(char* data);
		void				 render(RenderInfo &info);
		char    			*getFrame();
	private:

		cl_device_id 		 m_DeviceID;
};

#endif //_OPENCL_DEVICE_H
