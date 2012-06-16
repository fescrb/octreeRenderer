#ifndef _OPENCL_DEVICE_INFO_H
#define _OPENCL_DEVICE_INFO_H

#include "DeviceInfo.h"

#include "CLIncludes.h"

class OpenCLDeviceInfo
:	public DeviceInfo {
	public:
					 	 OpenCLDeviceInfo(cl_device_id device, cl_context context);
		virtual 		~OpenCLDeviceInfo();

		void	 	 	 printInfo();
        
        char            *getName();

	private:
		char			*m_sDeviceName;

		char			*m_sDeviceVendorString;
		char			*m_sOpenCLVersionString;

		cl_device_type	 m_deviceType;
		unsigned int 	 m_maxComputeUnits;
		unsigned int 	 m_maxComputeUnitFrequency;
        unsigned int     m_maxWorkGroupSize;
        size_t           m_workSizesDimensions;
        size_t          *m_maxWorkGroupSizes;
		unsigned long	 m_globalMemorySize;
        unsigned long    m_localMemorySize;
        cl_image_format *m_image_formats;
        cl_uint          m_format_count;
};

#endif //_OPENCL_DEVICE_INFO_H
