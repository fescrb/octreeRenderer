#include "OpenCLPlatform.h"
#include "OpenCLPlatformInfo.h"
#include "OpenCLDevice.h"

#include "OpenCLUtils.h"

OpenCLPlatform::OpenCLPlatform(cl_platform_id platform_id)
:	m_PlatformID(platform_id),
    m_pPlatformInfo(new OpenCLPlatformInfo(platform_id)){
	cl_uint device_num;

	// Get the number of devices for this platform.
	cl_int err = clGetDeviceIDs(m_PlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &device_num);

	// If error, we print to stdout and leave.
	if(clIsError(err)){
		clPrintError(err); return;
	}

	// We allocate the space for the devices.
	cl_device_id *aDevice_ids = (cl_device_id*) malloc(sizeof(cl_device_id) * device_num + 1);

	// Now we get the list of devices.
	err = clGetDeviceIDs(m_PlatformID, CL_DEVICE_TYPE_ALL, device_num, aDevice_ids, NULL);

	if(clIsError(err)){
		clPrintError(err); return;
	}
    
	for(int i = 0; i < device_num; i++) {
		// Create the context.
        // TODO: fix properties
        //cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM, platform_id, 0};
        
        // Perhaps should have callback specified?
        cl_context context = clCreateContext(0, 1, &aDevice_ids[i], NULL, NULL, &err);
        
        if(clIsError(err)){
            clPrintError(err); return;
        }
        
        m_vpDevices.push_back(new OpenCLDevice(aDevice_ids[i], context));
	}
}

OpenCLPlatform::~OpenCLPlatform(){

}

std::vector<OpenCLDevice*> OpenCLPlatform::getDeviceList() {
    return m_vpDevices;
}

void OpenCLPlatform::printInfo() {
    getInfo()->printInfo();
    for(int i = 0; i < getNumDevices(); i++) {
        m_vpDevices[i]->printInfo();
    }
}

OpenCLPlatformInfo* OpenCLPlatform::getInfo() {
    return m_pPlatformInfo;
}