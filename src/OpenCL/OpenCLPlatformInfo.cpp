#include "OpenCLPlatformInfo.h"

#include <cstdlib>
#include <cstdio>

OpenCLPlatformInfo::OpenCLPlatformInfo(cl_platform_id platform_id)
:   m_sPlatformProfile(0),
    m_sPlatformVersion(0),
    m_sPlatformName(0),
    m_sPlatformVendor(0),
    m_sPlatformExtensions(0) {
    
    size_t size = 0;
    clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, size, NULL, &size);
    
    m_sPlatformProfile = (char*) malloc (size+1);
    clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, size, m_sPlatformProfile, NULL);
    
    size = 0;
    clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, size, NULL, &size);
    
    m_sPlatformVersion = (char*) malloc (size+1);
    clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, size, m_sPlatformVersion, NULL);
    
    size = 0;
    clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, size, NULL, &size);
    
    m_sPlatformName = (char*) malloc (size+1);
    clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, size, m_sPlatformName, NULL);
    
    size = 0;
    clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, size, NULL, &size);
    
    m_sPlatformVendor = (char*) malloc (size+1);
    clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, size, m_sPlatformVendor, NULL);
    
    size = 0;
    clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, size, NULL, &size);
    
    m_sPlatformExtensions = (char*) malloc (size+1);
    clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, size, m_sPlatformExtensions, NULL);
}

OpenCLPlatformInfo::~OpenCLPlatformInfo() {
    if (m_sPlatformProfile) 
        free(m_sPlatformProfile);
    if (m_sPlatformVersion)
        free(m_sPlatformVersion);
    if (m_sPlatformName) 
        free(m_sPlatformName);
    if (m_sPlatformVendor)
        free(m_sPlatformVendor);
    if (m_sPlatformExtensions) 
        free(m_sPlatformExtensions);
}

void OpenCLPlatformInfo::printInfo() {
    printf("\n");
	printf("Platform Name:                  %s\n", m_sPlatformName);
	printf("Platform Vendor:                %s\n", m_sPlatformVendor);
	printf("Platform Profile:               %s\n", m_sPlatformProfile);
	printf("Platform Version:               %s\n", m_sPlatformVersion);
	printf("Platform Extensions:            %s\n", m_sPlatformExtensions);
}

char* OpenCLPlatformInfo::getName() {
    return m_sPlatformName;
}