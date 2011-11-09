#ifndef _OPENCL_DEVICE_INFO_H
#define _OPENCL_DEVICE_INFO_H

#include <CL/cl.h>

class OpenCLDeviceInfo {
	public:
					 OpenCLDeviceInfo(cl_device_id device);
		virtual 	~OpenCLDeviceInfo();

	private:
		char		*m_sDeviceName;

		unsigned int m_maxComputeUnits;
		unsigned int m_maxComputeUnitFrequency;
};

#endif //_OPENCL_DEVICE_INFO_H
