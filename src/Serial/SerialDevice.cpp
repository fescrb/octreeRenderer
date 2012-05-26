#include "SerialDevice.h"

#include "SerialDeviceInfo.h"
#include "OctreeSegment.h"

#include "RenderInfo.h"
#include "MathUtil.h"

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
	int* addr_int = (int*)node;
	return node + ((addr_int[1] & ~(255 << 24)) * 4) +4;
}

bool noChildren(char* node) {
    return !node[7];
}

char makeXYZFlag(float3 rayPos, float3 nodeCentre) {
	float3 flagVector = rayPos - nodeCentre;
	char flag = 0;
	for(int i = 0; i < 3; i++)
		if(flagVector[i] >= 0.0f)
			flag |= (1 << i);
	
	return flag;
}

bool nodeHasChildAt(float3 rayPos, float3 nodeCentre, char* node) {
    return node[7] & (1 << makeXYZFlag(rayPos, nodeCentre));  
}

char* getChild(float3 rayPos, float3 nodeCentre, char* node) {
	char xyz_flag = makeXYZFlag(rayPos, nodeCentre);
    int *node_int = (int*)node;
    int pos = (node_int[0] >> (xyz_flag * 3)) & 0b111;
    node_int+=(pos+2);
    node+=(pos+2)*4;
    return node + (node_int[0]*4);
}

SerialDevice::SerialDevice()
:	m_pOctreeData(0),
 	m_pFrame(0),
    m_texture(0){
	m_pDeviceInfo = new SerialDeviceInfo();
}

SerialDevice::~SerialDevice() {
	if(m_pFrame)
		free(m_pFrame);
}

void SerialDevice::printInfo() {
	m_pDeviceInfo->printInfo();
}

void SerialDevice::makeFrameBuffer(int2 size) {
    // Generate frame buffer if non-existant or not the same size;
    if (size != m_frameBufferResolution) {
        if(m_pFrame)
            free(m_pFrame);
        m_pFrame = (char*)malloc(3*size[0]*size[1]);
        m_frameBufferResolution = size;
    }
    
    // Clear.
    int i = 0;
    int bufferSize = 3*m_frameBufferResolution[0]*m_frameBufferResolution[1];
    while ( i < bufferSize) {
        m_pFrame[i]=0;
        i++;
    }
}

void SerialDevice::sendData(Bin bin) {
	m_pOctreeData = bin.getDataPointer();
}

void SerialDevice::sendHeader(Bin bin) {
    m_pHeader = bin.getDataPointer();
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

void SerialDevice::traceRay(int x, int y, renderinfo* info) {
    float half_size = 256.0f;
    
    // Ray setup.
    float3 o(info->viewPortStart + (info->viewStep * (x)) + (info->up * (y)));
    float3 d(o-info->eyePos); //Perspective projection now.
    normalize(d);
    float t = 0.0f;
    
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
    Stack stack[((int*)m_pHeader)[0] - 1];
    
    // We are out of the volume and we will never get to it.
    
    if(t > t_max) {
        collission = true;
        curr_address = 0; // Set to null.
    }
    
    float3 rayPos = o;
    
    // Traversal.
    while(!collission) {
        
        if(noChildren(curr_address)){
            collission = true;
        } else {
            // If we are inside the node
            if(t_min <= t && t < t_max) {
                rayPos = o + (d * t);
                
                char xyz_flag = makeXYZFlag(rayPos, voxelCentre);
                float nodeHalfSize = fabs((corner_far-voxelCentre)[0])/2.0f;
                
                float3 tmpNodeCentre( xyz_flag & 1 ? voxelCentre[0] + nodeHalfSize : voxelCentre[0] - nodeHalfSize,
                                      xyz_flag & 2 ? voxelCentre[1] + nodeHalfSize : voxelCentre[1] - nodeHalfSize,
                                      xyz_flag & 4 ? voxelCentre[2] + nodeHalfSize : voxelCentre[2] - nodeHalfSize);
                
                float3 tmp_corner_far(d[0] >= 0 ? tmpNodeCentre[0] + nodeHalfSize : tmpNodeCentre[0] - nodeHalfSize,
                                      d[1] >= 0 ? tmpNodeCentre[1] + nodeHalfSize : tmpNodeCentre[1] - nodeHalfSize,
                                      d[2] >= 0 ? tmpNodeCentre[2] + nodeHalfSize : tmpNodeCentre[2] - nodeHalfSize);
                
                float tmp_max = min((tmp_corner_far - o) / d);
                
                if(nodeHasChildAt(rayPos, voxelCentre, curr_address)) {
                    // If the voxel we are at is not empty, go down.
                    curr_index = push(stack, curr_index, curr_address, corner_far, voxelCentre, t_min, t_max);
                    
                    curr_address = getChild(rayPos,voxelCentre,curr_address);
                    
                    corner_far = tmp_corner_far;
                    voxelCentre = tmpNodeCentre;
                    corner_close =  float3(d[0] >= 0 ? voxelCentre[0] - nodeHalfSize : voxelCentre[0] + nodeHalfSize,
                                           d[1] >= 0 ? voxelCentre[1] - nodeHalfSize : voxelCentre[1] + nodeHalfSize,
                                           d[2] >= 0 ? voxelCentre[2] - nodeHalfSize : voxelCentre[2] + nodeHalfSize);
                    t_max = tmp_max;
                    t_min = max((corner_close - rayPos) / d);
                    
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
        
        unsigned char red = attributes[0];
        unsigned char green = attributes[1];
        unsigned char blue = attributes[2];
        
        // If attributes contains a normal
        if(((int*)m_pHeader)[1] > 4) {
            //Fixed direction light coming from (1, 1, 1);
            float4 direction_towards_light = normalize(float4(info->lightPos-rayPos, 0.0f));
            float4 normal = float4(fixed_point_8bit_to_float(attributes[4]),
                                   fixed_point_8bit_to_float(attributes[5]),
                                   fixed_point_8bit_to_float(attributes[6]),
                                   fixed_point_8bit_to_float(attributes[7]));
            // K_diff is always 1, for now
            float diffuse_coefficient = dot(direction_towards_light,normal);
            if(diffuse_coefficient<0)
                diffuse_coefficient*=-1.0f;
            /*printf("rayPos %f %f %f dir_to_light %f %f %f %f, normal %f %f %f %f diff %f\n",
                   rayPos[0],
                   rayPos[1],
                   rayPos[2],
                   direction_towards_light[0],
                   direction_towards_light[1],
                   direction_towards_light[2],
                   direction_towards_light[3],
                   normal[0],
                   normal[1],
                   normal[2],
                   normal[3],
                   diffuse_coefficient);*/
            red*=diffuse_coefficient;
            green*=diffuse_coefficient;
            blue*=diffuse_coefficient;
            
        }
        
        setFramePixel(x, y, red, green, blue);
    }
}

void SerialDevice::renderTask(int index, renderinfo *info) {	
    m_renderStart.reset();
 
    //printf("",m_pHeader[1]);
    
    rect window = m_tasks[index];
    int2 start = window.getOrigin();
    int2 size = window.getSize();
    
	int2 end = start+size;

	for(int y = start[1]; y < end[1]; y++) {
		for(int x = start[0]; x < end[0]; x++) {
			traceRay(start[0]+x, start[1]+y, info);
		}
	}
    
    m_renderEnd.reset();
}

GLuint SerialDevice::getFrameBuffer() {
    m_transferStart.reset();
    if (!m_texture) {
        glGenTextures(1, &m_texture);
    
		glBindTexture(GL_TEXTURE_2D, m_texture);
		
		glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
		
		glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
		
		glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
		glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	} else 
		 glBindTexture(GL_TEXTURE_2D, m_texture);
    
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGB,
                 m_frameBufferResolution[0],
                 m_frameBufferResolution[1],
                 0,
                 GL_RGB,
                 GL_UNSIGNED_BYTE,
                 m_pFrame);
    m_transferEnd.reset();
    return m_texture;
}

char* SerialDevice::getFrame() {
	return m_pFrame;
}

void SerialDevice::setFramePixel(int x, int y, char red, char green, char blue) {
	char* pixelPtr = &m_pFrame[(y*m_frameBufferResolution[0]*3)+(x*3)];
	
	pixelPtr[0] = red;
	pixelPtr[1] = green;
	pixelPtr[2] = blue;
}

high_res_timer SerialDevice::getRenderTime() {
    return m_renderEnd - m_renderStart;
}

high_res_timer SerialDevice::getBufferToTextureTime() {
    return m_transferEnd - m_transferStart;
}