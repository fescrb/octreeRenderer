#ifndef _OPENCL_CONTEXT_H
#define _OPENCL_CONTEXT_H

#include "Context.h"
#include "OpenCLPlatform.h"

#include <CL/cl.h>

class OpenCLContext :
	public Context {

	public:
							 OpenCLContext();
		virtual 			~OpenCLContext();

		void				 printDeviceInfo();
		unsigned int		 getNumDevices();
		Device				*getDevice(int index);

	private:
		unsigned int		 m_numberOfPlatforms;
		OpenCLPlatform		*m_aPlatforms;

};

#endif //_OPENCL_CONTEXT_H
