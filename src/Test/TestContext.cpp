#include "TestContext.h"

#include "TestDevice.h"

TestContext::TestContext() 
:	m_hostCPU(new TestDevice()){
	
}

void TestContext::printDeviceInfo() {
	m_hostCPU->printInfo();
}

unsigned int TestContext::getNumDevices() {
	return 1;
}
Device* TestContext::getDevice(int index) {
	return m_hostCPU;
}
