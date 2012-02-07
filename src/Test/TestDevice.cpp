#include "TestDevice.h"

#include "TestDeviceInfo.h"

#include <cstdlib>

TestDevice::TestDevice()
:	m_pOctreeData(0),
 	m_pFrame(0) {
	m_pDeviceInfo = new TestDeviceInfo();
}

TestDevice::~TestDevice() {
	if(m_pFrame)
		free(m_pFrame);
}

void TestDevice::printInfo() {
	m_pDeviceInfo->printInfo();
}

void TestDevice::sendData(char* data) {
	m_pOctreeData = data;
}

void TestDevice::render(RenderInfo &info) {
	if(!m_pFrame) {
		int buffSize = 3*info.resolution[0]*info.resolution[1];
		m_pFrame = (char*) malloc(buffSize);

		char* tmpPtr = m_pFrame;

		// Set all to 0.
		while(tmpPtr != m_pFrame+buffSize) {
			tmpPtr[0] = 0;
			tmpPtr++;
		}
	}

	for(int y = 0; y < info.resolution[1]; y++) {
		for(int x = 0; x < info.resolution[0]; x++) {
			// Ray setup.

			// Traversal.
		}
	}
}

char* TestDevice::getFrame() {
	return m_pFrame;
}
