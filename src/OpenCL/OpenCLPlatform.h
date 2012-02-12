#ifndef _OPENCL_PLATFORM_H
#define _OPENCL_PLATFORM_H

#include <vector>

#ifdef _LINUX
    #include <CL/cl.h>
#endif //_LINUX

#ifdef _OSX
    #include <OpenCL/cl.h>
#endif //_OSX

class OpenCLPlatformInfo;
class OpenCLDevice;

class OpenCLPlatform {

	public:
                                     OpenCLPlatform(cl_platform_id platform_id);
		virtual                     ~OpenCLPlatform();

		inline unsigned int          getNumDevices(){
			return m_vpDevices.size();
		}
    
        std::vector<OpenCLDevice*>   getDeviceList();
    
        void                         printInfo();
	private:
		cl_platform_id               m_PlatformID;

        std::vector<OpenCLDevice*> 	 m_vpDevices;
    
        OpenCLPlatformInfo          *m_pPlatformInfo;
};

#endif //_OPENCL_PLATFORM_H
