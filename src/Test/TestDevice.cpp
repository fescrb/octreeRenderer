#include "TestDevice.h"

#include "TestDeviceInfo.h"

#include <cstdlib>
#include <cmath>

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

	//TODO (BIG) do in vector math
	float plane_step = (tan(info.fov)*info.eyePlaneDist) / (float) info.resolution[0];
	float plane_start[2] = { info.eyePos[0] - (plane_step * ((float)info.resolution[0]/2.0f)),
							 info.eyePos[1] - (plane_step * ((float)info.resolution[1]/2.0f))};

	float tx0 = -256.0f;
	float ty0 = -256.0f;
	float tz0 = -256.0f;
	float tx1 =  256.0f;
	float ty1 =  256.0f;
	float tz1 =  256.0f;

	for(int y = 0; y < info.resolution[1]; y++) {
		for(int x = 0; x < info.resolution[0]; x++) {
			// Ray setup.
			float v[3] = { plane_start[0] + (plane_step*x),
						   plane_start[1] + (plane_step * y),
						   info.eyePos[2] + info.eyePlaneDist};
			float t = 0;

			// Traversal.
			// TODO make false
			bool collission = true;
			while(!collission) {

			}
		}
	}
}

char* TestDevice::getFrame() {
	return m_pFrame;
}
