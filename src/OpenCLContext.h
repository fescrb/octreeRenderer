#ifndef _OPENCL_CONTEXT_H
#define _OPENCL_CONTEXT_H

#include "Context.h"

#include <CL/cl.h>

class OpenCLContext :
	public Context {

	public:
							 OpenCLContext();
		virtual 			~OpenCLContext();

	private:
		cl_platform_id		 m_platform_id;

};

#endif //_OPENCL_CONTEXT_H
