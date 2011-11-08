#ifndef _OPENCL_PLATFORM_H
#define _OPENCL_PLATFORM_H

#include <CL/cl.h>

class OpenCLPlatform {

	public:
								 OpenCLPlatform(cl_platform_id);
		virtual 				~OpenCLPlatform();
	private:
		cl_platform_id 			 m_PlatformID;
};

#endif //_OPENCL_PLATFORM_H
