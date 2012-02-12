#ifndef _OPENCL_CONTEXT_H
#define _OPENCL_CONTEXT_H

#include <vector>

#ifdef _LINUX
    #include <CL/cl.h>
#endif //_LINUX

#ifdef _OSX
    #include <OpenCL/cl.h>
#endif //_OSX

#include "Context.h"

class OpenCLPlatform;
class OpenCLDevice;

class OpenCLContext :
	public Context {

	public:
		explicit                         OpenCLContext();
		virtual                         ~OpenCLContext();

		void                             printDeviceInfo();
		unsigned int                     getNumDevices();
		Device                          *getDevice(int index);

	private:
        unsigned int                     getNumPlatforms();
        
        std::vector<OpenCLDevice*>       getDeviceList();
        
        std::vector<OpenCLPlatform*>     m_vpPlatforms;

};

#endif //_OPENCL_CONTEXT_H
