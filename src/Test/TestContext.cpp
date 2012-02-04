#include "TestContext.h"

#include "TestDevice.h"

TestContext::TestContext() 
:	m_hostCPU(new TestDevice()){
	
}

void TestContext::printDeviceInfo() {
	m_hostCPU->printInfo();
}
