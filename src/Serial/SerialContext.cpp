#include "SerialContext.h"

#include "SerialDevice.h"

SerialContext::SerialContext() 
:	m_hostCPU(new SerialDevice()){
	
}

void SerialContext::printDeviceInfo() {
	m_hostCPU->printInfo();
}

unsigned int SerialContext::getNumDevices() {
	return 1;
}
Device* SerialContext::getDevice(int index) {
	return m_hostCPU;
}

std::vector<Device*> SerialContext::getDeviceList() {
	std::vector<Device*> devList;
	devList.push_back(m_hostCPU);
	return devList;
}