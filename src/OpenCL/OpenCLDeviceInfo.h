#ifndef _OPENCL_DEVICE_INFO_H
#define _OPENCL_DEVICE_INFO_H

#include "DeviceInfo.h"

#ifdef _LINUX
    #include <CL/cl.h>
#endif //_LINUX

#ifdef _OSX
    #include <OpenCL/cl.h>
#endif //_OSX

class OpenCLDeviceInfo
:	public DeviceInfo {
	public:
					 	 OpenCLDeviceInfo(cl_device_id device);
		virtual 		~OpenCLDeviceInfo();

		void	 	 	 printInfo();

	private:
		char			*m_sDeviceName;

		char			*m_sDeviceVendorString;
		char			*m_sOpenCLVersionString;

		cl_device_type	 m_deviceType;
		unsigned int 	 m_maxComputeUnits;
		unsigned int 	 m_maxComputeUnitFrequency;
		unsigned long	 m_globalMemorySize;
};

#endif //_OPENCL_DEVICE_INFO_H
