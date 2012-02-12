#ifndef _OPENCL_PLATFORM_H
#define _OPENCL_PLATFORM_H

#include <vector>

#ifdef _LINUX
    #include <CL/cl.h>
#endif //_LINUX

#ifdef _OSX
    #include <OpenCL/cl.h>
#endif //_OSX

class OpenCLContext;
class OpenCLDevice;

class OpenCLPlatform {

	public:
                                     OpenCLPlatform(cl_platform_id platform_id, OpenCLContext* context);
		virtual                     ~OpenCLPlatform();

		inline unsigned int          getNumDevices(){
			return m_vpDevices.size();
		}

		OpenCLDevice*                getDevice(const unsigned int dev){
			return m_vpDevices[dev];
		}
    
        void                         initializeCommandQueues();
	private:
		cl_platform_id               m_PlatformID;

        std::vector<OpenCLDevice*> 	 m_vpDevices;
};

#endif //_OPENCL_PLATFORM_H
