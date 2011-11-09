#ifndef _OPENCL_DEVICE_INFO_H
#define _OPENCL_DEVICE_INFO_H

#include <CL/cl.h>

class OpenCLDeviceInfo {
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
