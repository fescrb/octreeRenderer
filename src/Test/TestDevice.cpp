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

char makeXYZFlag(float3 rayPos, float3 nodeCentre) {
	float3 flagVector = rayPos - nodeCentre;
	char flag = 0;
	for(int i = 0; i < 3; i++)
		if(flagVector[i] >= 0.0f)
			flag | (1 << i);
		
	return flag;
}

char makeChildFlag(float3 rayPos, float3 nodeCentre) {
	printf("x %f y %f z %f flag %d\n", rayPos[0], rayPos[1], rayPos[2], makeXYZFlag(rayPos, nodeCentre));
	return 0  | (1 << makeXYZFlag(rayPos, nodeCentre));
}

bool nodeHasChildAt(float3 rayPos, float3 nodeCentre, char* node) {
	printf("node %d flag %d\n", node[0], makeChildFlag(rayPos,nodeCentre));
	return node[0] & makeChildFlag(rayPos,nodeCentre);  
}

char* getChild(float3 rayPos, float3 nodeCentre, char* node) {
	int counter = 0;
	bool found = false;
	char flag = makeChildFlag(rayPos, nodeCentre);
	for(int i = 0; i <= flag; i++) 
		if(node[0] & (1 << i))
			counter++;
	return node + (counter*4);
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
	char* address;
	float3 far_corner, node_centre;
	float t_min, t_max;
};

// Returns new index.
int push(Stack* stack, int index, char* node, float3 far_corner, float3 node_centre, int t_min, int t_max){
	stack[index].address = node;
	stack[index].far_corner = far_corner;
	stack[index].node_centre = node_centre;
	stack[index].t_min = t_min;
	stack[index].t_max = t_max;
	return index+1;
}

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

			float t_min = max((corner_close - o) / d);
			float t_max = min((corner_far - o) / d);
			
			// If we are out 
			if(t < t_min)
				t = t_min;
			
			char* curr_address = m_pOctreeData;
			float3 voxelCentre(0.0f, 0.0f, 0.0f);
			bool collission = false;	
			int curr_index = 0;
			Stack stack[info.maxOctreeDepth - 1];
			
			// We are out of the volume and we will never get to it.
			
			if(t > t_max) {
				collission = true;
				curr_address = 0; // Set to null.
			}

			// Traversal.
			while(!collission) {
				
				if(noChildren(curr_address)){
					collission = true;
				} else {
					// If we are inside the node
					if(t_min <= t && t < t_max) {
						float3 rayPos = o + (d * t);
						
						char xyz_flag = makeXYZFlag(rayPos, voxelCentre);
						float nodeHalfSize = fabs((corner_far-voxelCentre)[0])/2.0f;
						
						float3 tmpNodeCentre( xyz_flag & 1 ? voxelCentre[0] + nodeHalfSize : voxelCentre[0] - nodeHalfSize,
											  xyz_flag & 2 ? voxelCentre[1] + nodeHalfSize : voxelCentre[1] - nodeHalfSize,
											  xyz_flag & 4 ? voxelCentre[2] + nodeHalfSize : voxelCentre[2] - nodeHalfSize);
						
						float3 tmp_corner_far(d[0] >= 0 ? tmpNodeCentre[0] + nodeHalfSize : tmpNodeCentre[0] - nodeHalfSize,
											  d[1] >= 0 ? tmpNodeCentre[1] + nodeHalfSize : tmpNodeCentre[1] - nodeHalfSize,
											  d[2] >= 0 ? tmpNodeCentre[2] + nodeHalfSize : tmpNodeCentre[2] - nodeHalfSize);
						
						float tmp_max = min((tmp_corner_far - rayPos) / d);
						
						if(nodeHasChildAt(rayPos, voxelCentre, curr_address)) {
							// If the voxel we are at is not empty, go down.
							
							curr_index = push(stack, curr_index, curr_address, corner_far, voxelCentre, t_min, t_max);
							
							corner_far = tmp_corner_far;
							voxelCentre = tmpNodeCentre;
							corner_close =  float3(d[0] >= 0 ? voxelCentre[0] - nodeHalfSize : voxelCentre[0] + nodeHalfSize,
												   d[1] >= 0 ? voxelCentre[1] - nodeHalfSize : voxelCentre[1] + nodeHalfSize,
												   d[2] >= 0 ? voxelCentre[2] - nodeHalfSize : voxelCentre[2] + nodeHalfSize);
							t_max = tmp_max;
							t_min = max((corner_close - rayPos) / d);
							
							curr_address = getChild(rayPos,voxelCentre,curr_address);
							
						} else {
							// If the child is empty, we step the ray.
							t = tmp_max;
						}
					} else {
						// We are outside the node. Pop the stack
						curr_index--;
						if(curr_index>=0) {
							// Pop that stack!
							corner_far = stack[curr_index].far_corner;
						} else {
							// We are outside the volume.
							curr_address = 0;
							collission = true;
						}
					}
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
