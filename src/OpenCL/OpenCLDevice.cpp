#include "OpenCLDevice.h"

#include "OpenCLDeviceInfo.h"

OpenCLDevice::OpenCLDevice(cl_device_id device_id)
:	m_DeviceID(device_id){
	m_pDeviceInfo = new OpenCLDeviceInfo(device_id);
    
    //m_commandQueue = clCreateCommandQueue();
}


OpenCLDevice::~OpenCLDevice(){

}

void OpenCLDevice::initializeCommandQueue() {
    
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
