#include "OpenCLDevice.h"

#include "OpenCLDeviceInfo.h"
#include "OpenCLUtils.h"

OpenCLDevice::OpenCLDevice(cl_device_id device_id, cl_context context)
:	m_DeviceID(device_id),
    m_context(context){
	m_pDeviceInfo = new OpenCLDeviceInfo(device_id);
    
    cl_int err = 0;
    
    // Perhaps profiling should be enabled?
    m_commandQueue = clCreateCommandQueue(context, device_id, 0, &err);
    
    if(clIsError(err)){
		clPrintError(err); return;
	}
}


OpenCLDevice::~OpenCLDevice(){

}

void OpenCLDevice::printInfo() {
	m_pDeviceInfo->printInfo();
}

void OpenCLDevice::sendData(char* data) {

}

void OpenCLDevice::render(RenderInfo &info) {

}

char* OpenCLDevice::getFrame() {

}
