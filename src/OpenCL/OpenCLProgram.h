#ifndef _OPENCL_PROGRAM_H
#define _OPENCL_PROGRAM_H

#ifdef _LINUX
    #include <CL/cl.h>
#endif //_LINUX

#ifdef _OSX
    #include <OpenCL/cl.h>
#endif //_OSX

class OpenCLDevice;

class OpenCLProgram {
	public:
		explicit				 OpenCLProgram(OpenCLDevice* device, const char* sourceFilename);
		
		cl_kernel				 getOpenCLKernel(const char* kernel_name);
		
	private:
		cl_program               m_program;
		
		OpenCLDevice			*m_pDevice;
};

#endif //_OPENCL_PROGRAM_H