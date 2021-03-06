#ifndef _OPENCL_CONTEXT_H
#define _OPENCL_CONTEXT_H

#include <vector>

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

		std::vector<Device*>    		 getDeviceList();
		
	private:
        unsigned int                     getNumPlatforms();
        
        std::vector<OpenCLPlatform*>     m_vpPlatforms;

};

#endif //_OPENCL_CONTEXT_H
