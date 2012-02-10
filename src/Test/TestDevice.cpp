#include "TestDevice.h"

#include "TestDeviceInfo.h"

#include <cstdlib>
#include <cmath>

// Remove later
#include <cstdio>

float min(float3 vector) {
	float minimum = vector[0] < vector[1] ? vector[0] : vector[1];
	return minimum < vector[2] ? minimum : vector[2];
}

float max(float3 vector) {
	float maximum = vector[0] > vector[1] ? vector[0] : vector[1];
	return maximum > vector[2] ? maximum : vector[2];
}

char* getAttributes(char* node) {
	short* addr_short = (short*)node;
	
	return node + (addr_short[1] * 4);
}

bool noChildren(char* node) {
	return !node[0];
}

bool nodeHasChildAt(float3 rayPos, float3 nodeCentre, char* node) {
	float3 flagVector = rayPos - nodeCentre;
	int flag = 0;
	for(int i = 0; i < 3; i++)
		if(flagVector[3] >= 0)
			flag | (1 << i);
	
	return node[0] & (1 << flag);  
}

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

struct Stack {
	float3 far_corner, node_centre;
	float t_min, t_max;
};

void TestDevice::render(RenderInfo &info) {
	if(!m_pFrame) {
		int buffSize = 3*info.resolution[0]*info.resolution[1];
		m_frameBufferResolution[0] = info.resolution[0];
		m_frameBufferResolution[1] = info.resolution[1];
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
	
	float half_size = 256.0f;

	for(int y = 0; y < info.resolution[1]; y++) {
		for(int x = 0; x < info.resolution[0]; x++) {
			// Ray setup.
			float3 o(plane_start[0] + (plane_step*x),
					 plane_start[1] + (plane_step * y),
					 info.eyePos[2] + info.eyePlaneDist);
					 
			float3 d(info.viewDir); // As this is a parallel projection.
			float t = 0;
			
			float3 corner_far(d[0] >= 0 ? half_size : -half_size,
							  d[1] >= 0 ? half_size : -half_size,
							  d[2] >= 0 ? half_size : -half_size);
							  
			float3 corner_close(corner_far.neg());

			float3 vt0 = (corner_close - o) / d;
			float3 vt1 = (corner_far - o) / d;

			float t_min = max(vt0);
			float t_max = min(vt1);
			
			// If we are out 
			if(t < t_min)
				t = t_min;
			
			char* curr_address = m_pOctreeData;
			float3 voxelCentre(0.0f, 0.0f, 0.0f);
			bool collission = false;	
			int curr_index = 0;
			
			// We are out of the volume and we will never get to it.
			
			if(t > t_max) {
				collission = true;
				curr_address = 0; // Set to null.
			}

			// Traversal.
			while(!collission) {
				collission = true;
				
				if(noChildren(curr_address)){
					collission = true;
				} else {
					//bool node_has
					
					vt0 = (corner_close - o) / d;
					vt1 = (corner_far - o) / d;
					
					float t_min = max(vt0);
					float t_max = min(vt1);
				}
			}
			
			// If there was a collission.
			if(curr_address) {
				char* attributes = getAttributes(curr_address);
				setFramePixel(x, y, attributes[0], attributes[1], attributes[2]);
			}
		}
	}
}

char* TestDevice::getFrame() {
	return m_pFrame;
}

void TestDevice::setFramePixel(int x, int y, char red, char green, char blue) {
	char* pixelPtr = &m_pFrame[(y*m_frameBufferResolution[0]*3)+(x*3)];
	
	pixelPtr[0] = red;
	pixelPtr[1] = green;
	pixelPtr[2] = blue;
}
