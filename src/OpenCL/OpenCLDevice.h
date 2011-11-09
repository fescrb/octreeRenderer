#ifndef _OPENCL_DEVICE_H
#define _OPENCL_DEVICE_H

#include "Device.h"
#include <CL/cl.h>
#include "OpenCLDeviceInfo.h"

class OpenCLDevice:
	public Device{

	public:
							 OpenCLDevice(cl_device_id device_id);
		virtual 			~OpenCLDevice();

		void	 			 printInfo();
	private:

		cl_device_id 		 m_DeviceID;

		OpenCLDeviceInfo 	*m_pDeviceInfo;
};

#endif //_OPENCL_DEVICE_H
