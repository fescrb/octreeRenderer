#ifndef _OPENCL_PLATFORM_H
#define _OPENCL_PLATFORM_H

#include <vector>

#include "CLIncludes.h"

class OpenCLPlatformInfo;
class OpenCLDevice;

class OpenCLPlatform {

	public:
                                     OpenCLPlatform(cl_platform_id platform_id);
		virtual                     ~OpenCLPlatform();

		inline unsigned int          getNumDevices(){
			return m_vpDevices.size();
		}
		
		OpenCLPlatformInfo*          getInfo();
    
        std::vector<OpenCLDevice*>   getDeviceList();
    
        void                         printInfo();
	private:
		cl_platform_id               m_PlatformID;

        std::vector<OpenCLDevice*> 	 m_vpDevices;
    
        OpenCLPlatformInfo          *m_pPlatformInfo;
};

#endif //_OPENCL_PLATFORM_H
