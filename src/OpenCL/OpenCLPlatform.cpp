#include "OpenCLPlatform.h"
#include "OpenCLDevice.h"

#include "OpenCLUtils.h"

OpenCLPlatform::OpenCLPlatform(cl_platform_id platform_id, OpenCLContext* context)
:	m_PlatformID(platform_id){
	cl_uint device_num;

	// Get the number of devices for this platform.
	cl_int err = clGetDeviceIDs(m_PlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &device_num);

	// If error, we print to stdout and leave.
	if(clIsError(err)){
		clPrintError(err); return;
	}

	// We allocate the space for the devices.
	m_vpDevices.resize(device_num);
	cl_device_id *aDevice_ids = (cl_device_id*) malloc(sizeof(cl_device_id) * device_num + 1);

	// Now we get the list of devices.
	err = clGetDeviceIDs(m_PlatformID, CL_DEVICE_TYPE_ALL, device_num, aDevice_ids, NULL);

	if(clIsError(err)){
		clPrintError(err); return;
	}

	for(int i = 0; i < device_num; i++) {
		m_vpDevices.push_back(new OpenCLDevice(aDevice_ids[i], context));
	}
}

OpenCLPlatform::~OpenCLPlatform(){

}

void OpenCLPlatform::initializeCommandQueues() {
    for(int i = 0; i < m_vpDevices.size(); i++) {
		m_vpDevices[i]->initializeCommandQueue();
	}
}

std::vector<OpenCLDevice*> OpenCLPlatform::getDeviceList() {
    return m_vpDevices;
}