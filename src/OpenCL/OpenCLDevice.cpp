#include "OpenCLDevice.h"

#include "OpenCLDeviceInfo.h"

OpenCLDevice::OpenCLDevice(cl_device_id device_id)
:	m_DeviceID(device_id){
	m_pDeviceInfo = new OpenCLDeviceInfo(device_id);
}


OpenCLDevice::~OpenCLDevice(){

}

void OpenCLDevice::printInfo() {
	m_pDeviceInfo->printInfo();
}
