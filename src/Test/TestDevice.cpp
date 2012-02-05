#include "TestDevice.h"

#include "TestDeviceInfo.h"

TestDevice::TestDevice() {
	m_pDeviceInfo =new TestDeviceInfo();
}

TestDevice::~TestDevice() {
	
}

void TestDevice::printInfo() {
	m_pDeviceInfo->printInfo();
}
