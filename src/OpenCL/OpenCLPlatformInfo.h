#ifndef _OPENCL_PLATFORM_INFO_H
#define _OPENCL_PLATFORM_INFO_H

#ifdef _LINUX
    #include <CL/cl.h>
#endif //_LINUX

#ifdef _OSX
    #include <OpenCL/cl.h>
#endif //_OSX

class OpenCLPlatformInfo {
    public:
        explicit             OpenCLPlatformInfo(cl_platform_id platform_id);
                            ~OpenCLPlatformInfo();
    
        void                 printInfo();
        
        char*                getName();
        bool                 getAllowsOpenGLSharing();
    
    private:
        char                *m_sPlatformProfile;
        char                *m_sPlatformVersion;
        char                *m_sPlatformName;
        char                *m_sPlatformVendor;
        char                *m_sPlatformExtensions;
        
        bool                 m_openGLSharing;
};

#endif //_OPENCL_PLATFORM_INFO_H
