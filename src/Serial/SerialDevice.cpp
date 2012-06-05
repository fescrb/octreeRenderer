#include "SerialDevice.h"

#include "SerialDeviceInfo.h"

#include "Octree.h"

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

float min(float4 vector) {
    float minimum = vector[0] < vector[1] ? vector[0] : vector[1];
    minimum = minimum < vector[2] ? minimum : vector[2];
    return minimum < vector[3] ? minimum : vector[3];
}


float max(float3 vector) {
	float maximum = vector[0] > vector[1] ? vector[0] : vector[1];
	return maximum > vector[2] ? maximum : vector[2];
}

char* getAttributes(char* node) {
	return node + ((node[2] >> 4) * 4);
}

uchar3 getColours(char* attr) {
    unsigned short rgb_565 = ((unsigned short*)attr)[0];
    uchar3 colour = uchar3();

    colour.setX((rgb_565 >> 8) & ~7);
    colour.setY(((unsigned char)(rgb_565 >> 3)) & ~3);
    colour.setZ((rgb_565 & 31) << 3);

    //printf("rgb565 %d red %d green %d blue %d\n", rgb_565, colour[0], colour[1], colour[2]);

    return colour;
}

float4 getNormal(char* attr) {
    unsigned short normals_short = ((unsigned short*)attr)[1];
    float4 normal;

    char x = normals_short >> 8;
    normal.setX(fixed_point_8bit_to_float(x));
    char y = 0 | (normals_short & 254);
    normal.setY(fixed_point_8bit_to_float(y));

    float z = sqrt(1 - (normal.getX()*normal.getX()) - (normal.getY()*normal.getY()));
    if(normals_short & 1)
        z*=-1.0f;

    normal.setZ(z);
    normal.setW(0.0f);

    //printf("short %d x %f y %f z %f\n", normals_short, normal[0], normal[1], normal[2]);

    return normal;
}

bool noChildren(char* node) {
    return !node[3];
}

char makeXYZFlag(float3 rayPos, float3 nodeCentre, float3 direction ) {
	float3 flagVector = rayPos - nodeCentre;
	char flag = 0;
	for(int i = 0; i < 3; i++)
        if(flagVector[i] > F32_EPSILON)
            flag |= (1 << i);
        else if (flagVector[i] <= F32_EPSILON && flagVector[i] >= -F32_EPSILON)
            if(direction[i] >= 0.0f)
                flag |= (1 << i);

	return flag;
}

char makeXYZFlag(float3 t_centre_vector, float t, float3 direction) {
    char flag = 0;

    for(int i = 0; i < 3; i++) {
        if( t >= t_centre_vector[i] ) {
            if(direction[i] >= 0.0f)
                flag |= (1 << i);
        } else {
            if(direction[i] < 0.0f)
                flag |= (1 << i);
        }
    }

    return flag;
}

bool nodeHasChildAt(char* node, char xyz_flag) {
    return node[3] & (1 << xyz_flag);
}

char* getChild(char* node, char xyz_flag) {
    int *node_int = (int*)node;
    int pos = 0;//(node_int[0] >> (xyz_flag * 3)) & 7;
    switch(xyz_flag) {
        case 1:
            pos = node_int[0] & 1;
            break;
        case 2:
            pos = (node_int[0] >> 1) & 3;
            break;
        case 3:
            pos = (node_int[0] >> 3) & 3;
            break;
        case 4:
            pos = (node_int[0] >> 5) & 7;
            break;
        case 5:
            pos = (node_int[0] >> 8) & 7;
            break;
        case 6:
            pos = (node_int[0] >> 11) & 7;
            break;
        case 7:
            pos = (node_int[0] >> 14) & 7;
            break;
        default:
            break;
    }
    printf("addr %p flag %d int %d pos %d\n", node, xyz_flag ,node_int[0], pos);
    node_int+=(pos+1);
    node+=(pos+1)*4;
    return node + (node_int[0]*4);
}

SerialDevice::SerialDevice()
:	Device(),
    m_pOctreeData(0),
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
int push(Stack* stack, int index, char* node, float3 far_corner, float3 node_centre, float t_min, float t_max){
	stack[index].address = node;
	stack[index].far_corner = far_corner;
	stack[index].node_centre = node_centre;
	stack[index].t_min = t_min;
	stack[index].t_max = t_max;
	return index+1;
}

void SerialDevice::makeFrameBuffer(int2 size) {
    Device::makeFrameBuffer(size);
    m_renderStart.reset();
}

void SerialDevice::traceRayBundle(int x, int y, int width, renderinfo* info) {
    float3 o(info->eyePos);
    float3 from_centre_to_start = -(info->viewStep/2.0f) - (info->up/2.0f);
    float3 d_lower_left((info->viewPortStart + (info->viewStep * (x*width)) + (info->up * (y*width)))-o);
    d_lower_left+=from_centre_to_start;
    float3 d_upper_left((info->viewPortStart + (info->viewStep * (x*width)) + (info->up * ((y+1)*width)))-o);
    d_upper_left+=from_centre_to_start;
    float3 d_lower_right((info->viewPortStart + (info->viewStep * ((x+1)*width)) + (info->up * (y*width)))-o);
    d_lower_right+=from_centre_to_start;
    float3 d_upper_right((info->viewPortStart + (info->viewStep * ((x+1)*width)) + (info->up * ((y+1)*width)))-o);
    d_upper_right+=from_centre_to_start;

    float3 d((info->viewPortStart + (info->viewStep * (((x)*width)+(width/2))) + (info->up * (((y)*width)+(width/2))))-o);
    d+=from_centre_to_start;

    float max_mag = max(float4(mag(d_lower_left), mag(d_upper_left), mag(d_lower_right), mag(d_upper_right)));

    float t = 0.0f;
    float t_prev = t;

    float half_size = OCTREE_ROOT_HALF_SIZE;

    float3 corner_far(d[0] >= 0 ? half_size : -half_size,
                      d[1] >= 0 ? half_size : -half_size,
                      d[2] >= 0 ? half_size : -half_size);

    float pixel_half_size = info->pixel_half_size * width;

    float3 corner_close(corner_far.neg());

    float t_min = max((corner_close - o) / d);
    float t_max = min((corner_far - o) / d);
    float t_out = t_max;

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
        if(t >= t_out) {
            collission = true;
            curr_address = 0;
        } else if(noChildren(curr_address)){
            collission = true;
        } else {
            // If we are inside the node
            if(t_min <= t && t < t_max) {
                // We check if all rays fit
                if(max((corner_close - o) / d_lower_left) >= min((corner_far - o) / d_lower_left))
                    collission = true;

                if(max((corner_close - o) / d_lower_right) >= min((corner_far - o) / d_lower_right))
                    collission = true;

                if(max((corner_close - o) / d_upper_left) >= min((corner_far - o) / d_upper_left))
                    collission = true;

                if(max((corner_close - o) / d_upper_right) >= min((corner_far - o) / d_upper_right))
                    collission = true;

                if(collission) {
                    //t = t_prev;
                    break;
                }

                rayPos = o + (d * t);

                float3 t_centre_vector = (voxelCentre - o) / d;

                char xyz_flag = makeXYZFlag(t_centre_vector, t, d);
                float nodeHalfSize = fabs(fabs(corner_far[0])-fabs(voxelCentre[0]))/2.0f;

                float3 tmpNodeCentre( xyz_flag & 1 ? voxelCentre[0] + nodeHalfSize : voxelCentre[0] - nodeHalfSize,
                                      xyz_flag & 2 ? voxelCentre[1] + nodeHalfSize : voxelCentre[1] - nodeHalfSize,
                                      xyz_flag & 4 ? voxelCentre[2] + nodeHalfSize : voxelCentre[2] - nodeHalfSize);

                float3 tmp_corner_far(d[0] >= 0 ? tmpNodeCentre[0] + nodeHalfSize : tmpNodeCentre[0] - nodeHalfSize,
                                      d[1] >= 0 ? tmpNodeCentre[1] + nodeHalfSize : tmpNodeCentre[1] - nodeHalfSize,
                                      d[2] >= 0 ? tmpNodeCentre[2] + nodeHalfSize : tmpNodeCentre[2] - nodeHalfSize);

                float tmp_max = min((tmp_corner_far - o) / d);

                if(nodeHasChildAt(curr_address, xyz_flag)) {
                    // If the voxel we are at is not empty, go down.

                    // We check for LOD.
                    if(nodeHalfSize < pixel_half_size*t) {
                        collission = true;
                        break;
                    }

                    curr_index = push(stack, curr_index, curr_address, corner_far, voxelCentre, t_min, t_max);

                    curr_address = getChild(curr_address, xyz_flag);

                    corner_far = tmp_corner_far;
                    voxelCentre = tmpNodeCentre;
                    corner_close =  float3(d[0] >= 0 ? voxelCentre[0] - nodeHalfSize : voxelCentre[0] + nodeHalfSize,
                                           d[1] >= 0 ? voxelCentre[1] - nodeHalfSize : voxelCentre[1] + nodeHalfSize,
                                           d[2] >= 0 ? voxelCentre[2] - nodeHalfSize : voxelCentre[2] + nodeHalfSize);
                    t_max = tmp_max;
                    t_min = max((corner_close - o) / d);

                } else {
                    // If the child is empty, we step the ray.
                    t_prev = t;//mag(corner_close-o)/max_mag;
                    t = tmp_max;
                }
            } else {
                // We are outside the node. Pop the stack
                curr_index--;
                if(curr_index>=0) {
                    // Pop that stack!
                    corner_far = stack[curr_index].far_corner;
                    curr_address = stack[curr_index].address;
                    voxelCentre = stack[curr_index].node_centre;
                    float nodeHalfSize = fabs(fabs(corner_far[0])-fabs(voxelCentre[0]));
                    corner_close =  float3(d[0] >= 0 ? voxelCentre[0] - nodeHalfSize : voxelCentre[0] + nodeHalfSize,
                                           d[1] >= 0 ? voxelCentre[1] - nodeHalfSize : voxelCentre[1] + nodeHalfSize,
                                           d[2] >= 0 ? voxelCentre[2] - nodeHalfSize : voxelCentre[2] + nodeHalfSize);
                    t_max = stack[curr_index].t_max;
                    t_min = stack[curr_index].t_min;
                } else {
                    // We are outside the volume.
                    curr_address = 0;
                    collission = true;
                }
            }
        }
    }

    for(int i = (x*width); i < (x+1)*width; i++) {
        for(int j = (y*width); j < (y+1)*width; j++) {
            //printf("%d %d buffer val %f\n", i, j, t);
            setDepthBufferValue(i, j, t_prev);
        }
    }
}

void SerialDevice::traceRay(int x, int y, renderinfo* info) {
    float half_size = OCTREE_ROOT_HALF_SIZE;
    char depth_in_octree = 0;
    short it = 0;

    float pixel_half_size = info->pixel_half_size;

    //if(x == 292 && y == 322)
        //printf("yep\n");

    // Ray setup.
    float3 o(info->viewPortStart + (info->viewStep * (x)) + (info->up * (y)));
    float3 d(o-info->eyePos); //Perspective projection now.
    o = info->eyePos;
    //d = normalize(d);
    float t = getDepthBufferValue(x,y);

    float3 corner_far(d[0] >= 0 ? half_size : -half_size,
                      d[1] >= 0 ? half_size : -half_size,
                      d[2] >= 0 ? half_size : -half_size);

    float3 corner_close(corner_far.neg());

    float t_min = max((corner_close - o) / d);
    float t_max = min((corner_far - o) / d);
    float t_out = t_max;

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
        ++it;

        if(t >= t_out) {
            collission = true;
            curr_address = 0;
        } else if(noChildren(curr_address)){
            collission = true;
        } else {
            // If we are inside the node
            if(t_min <= t && t < t_max) {
                rayPos = o + (d * t);

                float3 t_centre_vector = (voxelCentre - o) / d;

                char xyz_flag = makeXYZFlag(t_centre_vector, t, d);
                float nodeHalfSize = fabs((corner_far-voxelCentre)[0])/2.0f;

                float3 tmpNodeCentre( xyz_flag & 1 ? voxelCentre[0] + nodeHalfSize : voxelCentre[0] - nodeHalfSize,
                                      xyz_flag & 2 ? voxelCentre[1] + nodeHalfSize : voxelCentre[1] - nodeHalfSize,
                                      xyz_flag & 4 ? voxelCentre[2] + nodeHalfSize : voxelCentre[2] - nodeHalfSize);

                float3 tmp_corner_far(d[0] >= 0 ? tmpNodeCentre[0] + nodeHalfSize : tmpNodeCentre[0] - nodeHalfSize,
                                      d[1] >= 0 ? tmpNodeCentre[1] + nodeHalfSize : tmpNodeCentre[1] - nodeHalfSize,
                                      d[2] >= 0 ? tmpNodeCentre[2] + nodeHalfSize : tmpNodeCentre[2] - nodeHalfSize);

                float tmp_max = min((tmp_corner_far - o) / d);

                if(nodeHasChildAt(curr_address, xyz_flag)) {
                    // If the voxel we are at is not empty, go down.

                    // We check for LOD.
                    if(nodeHalfSize < pixel_half_size*t) {
                        collission = true;
                        break;
                    }

                    curr_index = push(stack, curr_index, curr_address, corner_far, voxelCentre, t_min, t_max);

                    curr_address = getChild(curr_address, xyz_flag);

                    corner_far = tmp_corner_far;
                    voxelCentre = tmpNodeCentre;
                    corner_close =  float3(d[0] >= 0 ? voxelCentre[0] - nodeHalfSize : voxelCentre[0] + nodeHalfSize,
                                           d[1] >= 0 ? voxelCentre[1] - nodeHalfSize : voxelCentre[1] + nodeHalfSize,
                                           d[2] >= 0 ? voxelCentre[2] - nodeHalfSize : voxelCentre[2] + nodeHalfSize);
                    t_max = tmp_max;
                    t_min = max((corner_close - o) / d);

                    ++depth_in_octree;

                } else {
                    // If the child is empty, we step the ray.
                    t = tmp_max;
                }
            } else {
                // We are outside the node. Pop the stack
                curr_index--;
                if(curr_index>=0) {
                    --depth_in_octree;
                    // Pop that stack!
                    corner_far = stack[curr_index].far_corner;
                    curr_address = stack[curr_index].address;
                    voxelCentre = stack[curr_index].node_centre;
                    t_max = stack[curr_index].t_max;
                    t_min = stack[curr_index].t_min;
                } else {
                    // We are outside the volume.
                    curr_address = 0;
                    collission = true;
                }
            }
        }
    }

    float ambient = 0.2f;

    // If there was a collission.
    if(curr_address) {
        printf("finished one\n");
        char* attributes = getAttributes(curr_address);

        uchar3 colour = getColours(attributes);

        unsigned char red = colour.getX();
        unsigned char green = colour.getY();
        unsigned char blue = colour.getZ();

        //Fixed direction light coming from (1, 1, 1);
        float4 direction_towards_light = normalize(float4(info->lightPos-rayPos, 0.0f));
        float4 normal = getNormal(attributes);
        // K_diff is always 1, for now
        float diffuse_coefficient = dot(direction_towards_light,normal);
        if(diffuse_coefficient<0)
            diffuse_coefficient=0.0f;
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
        red=(red*diffuse_coefficient*(1.0f-ambient))+(red*ambient);
        green=(green*diffuse_coefficient*(1.0f-ambient))+(green*ambient);
        blue=(blue*diffuse_coefficient*(1.0f-ambient))+(blue*ambient);

        setFramePixel(x, y, red, green, blue);
    }
    setInfoPixels(x, y, getDepthBufferValue(x,y)/10.0f/*fabs(dot(rayPos, info->viewDir))/(OCTREE_ROOT_HALF_SIZE*2.0f)*/, it, depth_in_octree);
}

void SerialDevice::renderTask(int index, renderinfo *info) {
    //printf("",m_pHeader[1]);

    rect window = m_tasks[index];
    int2 start = window.getOrigin();
    int2 size = window.getSize();
    int2 end = start+size;

    for(int y = start[1]; y < end[1]/8; y++) {
        for(int x = start[0]; x < end[0]/8; x++) {
            traceRayBundle(x, y, 8, info);
            //printf("done %d %d\n", x, y);
        }
    }

    for(int y = start[1]; y < end[1]; y++) {
        for(int x = start[0]; x < end[0]; x++) {
            traceRay(x, y, info);
            //printf("done %d %d\n", x, y);
        }
    }
}

framebuffer_window SerialDevice::getFrameBuffer() {
    m_renderEnd.reset();
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

    //Octree Depth
    /*glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_LUMINANCE,
                 getTotalTaskWindow().getWidth(),
                 getTotalTaskWindow().getHeight(),
                 0,
                 GL_LUMINANCE,
                 GL_UNSIGNED_BYTE,
                 m_pOctreeDepth);*/
    //Iterations
    /*glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_LUMINANCE,
                 getTotalTaskWindow().getWidth(),
                 getTotalTaskWindow().getHeight(),
                 0,
                 GL_LUMINANCE,
                 GL_UNSIGNED_BYTE,
                 m_pIterations);*/
    if(m_renderMode == COLOUR) {
        glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGB,
                 getTotalTaskWindow().getWidth(),
                 getTotalTaskWindow().getHeight(),
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 m_pFrame);
    } else if (m_renderMode == DEPTH) {
        glTexImage2D(GL_TEXTURE_2D,
                    0,
                    GL_LUMINANCE,
                    getTotalTaskWindow().getWidth(),
                    getTotalTaskWindow().getHeight(),
                    0,
                    GL_LUMINANCE,
                    GL_FLOAT,
                    m_pDepthBuffer);
    }
    m_transferEnd.reset();

    framebuffer_window fb_window;
    fb_window.window = getTotalTaskWindow();
    fb_window.texture = m_texture;

    return fb_window;
}

unsigned char* SerialDevice::getFrame() {
	return m_pFrame;
}

void SerialDevice::setFramePixel(int x, int y, unsigned char red, unsigned char green, unsigned char blue) {
	unsigned char* pixelPtr = &m_pFrame[((y - getTotalTaskWindow().getY() )*getTotalTaskWindow().getWidth()*4)+((x-getTotalTaskWindow().getX())*4)];

	pixelPtr[0] = red;
	pixelPtr[1] = green;
	pixelPtr[2] = blue;
    pixelPtr[3] = 255;
}

void SerialDevice::setInfoPixels(int x, int y, float depth, unsigned char iterations, unsigned char depth_in_octree) {
    //printf("depth is %f\n",depth);
    int index = (getTotalTaskWindow().getWidth()*y) + x;
    char color_per_level = 255/((int*)m_pHeader)[0];
    m_pDepthBuffer[index] = depth;
    m_pIterations[index] = iterations;
    m_pOctreeDepth[index] = depth_in_octree*color_per_level;
}

float SerialDevice::getDepthBufferValue(int x, int y) {
   int index = (getTotalTaskWindow().getWidth()*y) + x;
   return m_pDepthBuffer[index];
}

void SerialDevice::setDepthBufferValue(int x, int y, float value) {
    int index = (getTotalTaskWindow().getWidth()*y) + x;
    m_pDepthBuffer[index] = value;
}

high_res_timer SerialDevice::getRenderTime() {
    return m_renderEnd - m_renderStart;
}

high_res_timer SerialDevice::getBufferToTextureTime() {
    return m_transferEnd - m_transferStart;
}
