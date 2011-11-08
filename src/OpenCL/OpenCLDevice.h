#ifndef _OPENCL_DEVICE_H
#define _OPENCL_DEVICE_H

#include "Device.h"
#include <CL/cl.h>

class OpenCLDevice:
	public Device{

	public:
							 OpenCLDevice(cl_device_id);
		virtual 			~OpenCLDevice();

	private:


};

#endif //_OPENCL_DEVICE_H
