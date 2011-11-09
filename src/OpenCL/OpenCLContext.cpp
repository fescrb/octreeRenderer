#include "OpenCLContext.h"
#include "OpenCLUtils.h"

// To print info, might need to be removed later.
#include "OpenCLDevice.h"

OpenCLContext::OpenCLContext() {
	cl_uint num_of_platforms = 0;

	// We get the number of platforms.
	cl_int err = clGetPlatformIDs(0, NULL, &num_of_platforms);

	// If error, we print to stdout and leave.
	if(clIsError(err)){
		clPrintError(err); return;
	}

	m_numberOfPlatforms = num_of_platforms;

	// We allocate memory for the list of platforms and retrieve it.
	m_aPlatforms = (OpenCLPlatform*) malloc(sizeof(OpenCLPlatform)*m_numberOfPlatforms + 1);
	cl_platform_id *platform_ids = (cl_platform_id*) malloc(sizeof(cl_platform_id)*m_numberOfPlatforms + 1);

	err = clGetPlatformIDs(num_of_platforms, platform_ids, NULL);

	if(clIsError(err)){
		clPrintError(err); return;
	}

	// We initialize the platforms.
	for(int i = 0; i < m_numberOfPlatforms; i++) {
		m_aPlatforms[i] = OpenCLPlatform(platform_ids[i]);
	}
}

OpenCLContext::~OpenCLContext() {

}

void OpenCLContext::printDeviceInfo(){
	for(int i = 0; i < m_numberOfPlatforms; i++) {
		OpenCLPlatform platform = m_aPlatforms[i];
		for(int j = 0; j < platform.getNumberOfDevices(); j++) {
			platform.getDevice(j)->printInfo();
		}
	}
}
