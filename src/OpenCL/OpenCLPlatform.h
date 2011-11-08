#ifndef _OPENCL_PLATFORM_H
#define _OPENCL_PLATFORM_H

#include <CL/cl.h>

class OpenCLDevice;

class OpenCLPlatform {

	public:
								 OpenCLPlatform(cl_platform_id platform_id);
		virtual 				~OpenCLPlatform();
	private:
		cl_platform_id 			 m_PlatformID;

		unsigned int	 		 m_numberOfDevices;
		OpenCLDevice 		   **m_apDevices;
};

#endif //_OPENCL_PLATFORM_H
