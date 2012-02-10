#ifndef _OPENCL_PLATFORM_H
#define _OPENCL_PLATFORM_H

#ifdef _LINUX
    #include <CL/cl.h>
#endif //_LINUX

#ifdef _OSX
    #include <OpenCL/cl.h>
#endif //_OSX

class OpenCLDevice;

class OpenCLPlatform {

	public:
								 OpenCLPlatform(cl_platform_id platform_id);
		virtual 				~OpenCLPlatform();

		inline unsigned int		 getNumDevices(){
			return m_numberOfDevices;
		}

		OpenCLDevice*			 getDevice(const unsigned int dev){
			return m_apDevices[dev];
		}
	private:
		cl_platform_id 			 m_PlatformID;

		unsigned int	 		 m_numberOfDevices;
		OpenCLDevice 		   **m_apDevices;
};

#endif //_OPENCL_PLATFORM_H
