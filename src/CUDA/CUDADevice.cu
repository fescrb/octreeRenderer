#include "CUDADevice.h"

#include "CUDAIncludes.h"
#include "CUDAUtils.h"
#include "CUDADeviceInfo.h"
#include "CUDARenderInfo.h"

#include "CUDAMath.cuh"

#include "SizeMacros.h"

#define STACK_SIZE 10

/*
 * Device code
 */

inline __device__ char* get_attributes(char* node) {
    return node + ((((unsigned char*)node)[2] >> 4) * 4);
}

inline __device__ bool no_children(const uint node_header) {
    return !(node_header>>24);
}

inline __device__ char makeXYZFlag(const float3 t_centre_vector, const float t, const float3 direction) {
    char flag = 0;

    if( t >= t_centre_vector.x ) {
        if(direction.x >= 0.0f)
            flag |= 1; 
    } else {
        if(direction.x < 0.0f)
            flag |= 1; 
    }

    if( t >= t_centre_vector.y ) {
        if(direction.y >= 0.0f)
            flag |= 2; 
    } else {
        if(direction.y < 0.0f)
            flag |= 2; 
    }

    if( t >= t_centre_vector.z ) {
        if(direction.z >= 0.0f)
            flag |= 4; 
    } else {
        if(direction.z < 0.0f)
            flag |= 4; 
    }

    return flag;
}

inline __device__ bool node_has_child_at(const uint node_header, char xyz_flag) {
    return (node_header>>24) & (1 << xyz_flag);  
}

inline __device__ char* get_child(const uint node_header, char* node, char xyz_flag) {
    int *node_int = (int*)node;
    int pos = 0;//(node_int[0] >> (xyz_flag * 3)) & 0b111;
    switch(xyz_flag) {
    case 1:
        pos = node_header & 1;
        break;
    case 2:
        pos = (node_header >> 1) & 3;
        break;
    case 3:
        pos = (node_header >> 3) & 3;
        break;
    case 4:
        pos = (node_header >> 5) & 7;
        break;
    case 5:
        pos = (node_header >> 8) & 7;
        break;
    case 6:
        pos = (node_header >> 11) & 7;
        break;
    case 7:
        pos = (node_header >> 14) & 7;
        break;
    default:
        break;
    }
    int diff = 0;
    if(node[2] & 2) {
        unsigned short *node_short = (unsigned short*)node;
        node_short+=(pos+2);
        diff = node_short[0];
        pos /= 2;
    } else {
        node_int+=(pos+1);
        diff = node_int[0];
    }
    node+=(pos+1)*4;
    return node + (diff*4);
}

inline __device__ uchar3 getColours(char* attr) {
    unsigned short rgb_565 = ((short*)attr)[0];
    uchar3 colour;
    
    colour.x = (rgb_565 >> 8) & ~7;
    colour.y = ((unsigned char)(rgb_565 >> 3)) & ~3;
    colour.z = (rgb_565 & 31) << 3;
    
    return colour;
}

inline __device__ float3 getNormal(char* attr) {
    unsigned short normals_short = ((short*)attr)[1];
    float3 normal;
    
    char x = normals_short >> 8;
    normal.x = fixed_point_8bit_to_float(x);
    char y = 0 | (normals_short & 254);
    normal.y = fixed_point_8bit_to_float(y);
    
    float z = sqrt(1 - (normal.x*normal.x) - (normal.y*normal.y));
    if(normals_short & 1)
        z*=-1.0f;
    
    normal.z = z;
    
    return normal;
}

struct collision {
    char *node;
    float t;
    short it;
};

struct stack{
    char* address;
    float3 node_centre;
    float t_max;
};

inline __device__ int push(struct stack* short_stack, int curr_index, char* curr_address, float3 voxelCentre, float t_max) {
    if(curr_index >= STACK_SIZE) {
        curr_index = STACK_SIZE - 1;
        for(int i = 1; i < STACK_SIZE; i++) 
            short_stack[i-1]=short_stack[i];
    }
    
    short_stack[curr_index].address = curr_address;
    short_stack[curr_index].node_centre = voxelCentre;
    short_stack[curr_index].t_max = t_max;
    curr_index++;
    return curr_index;
}

__device__ collision find_collision(char *octree, const float3 o, const float3 d, float t, const float pixel_half_size){
    unsigned short it = 1;
    unsigned char depth_in_octree = 0;
    
    float half_size = OCTREE_ROOT_HALF_SIZE;
    
    float3 corner_far_step = make_float3(d.x >= 0.0f ? half_size : -half_size,
                                         d.y >= 0.0f ? half_size : -half_size,
                                         d.z >= 0.0f ? half_size : -half_size);
    
    float3 corner_far = corner_far_step;
    
    float3 corner_close = make_float3(-corner_far.x,-corner_far.y,-corner_far.z);
    
    float t_min = max_component((corner_close - o)/d);
    float t_max = min_component((corner_far - o)/d);
    const float t_out = t_max;
    
    /* Move in if we are out */
    if (t < t_min)
        t = t_min;
    
    char *curr_address = octree;
    uint* curr_address_uint = (uint*)curr_address;
    uint node_header = curr_address_uint[0];
    
    float3 voxelCentre = make_float3(0.0f, 0.0f, 0.0f);
    bool collision = false;
    
    int curr_index = 0;
    struct stack short_stack[STACK_SIZE];
    
    while(!collision) {
        it++;
        if(t >= t_out) {
            collision = true;
            curr_address = 0;
        } else if (no_children(node_header)) {
            collision = true;
        } else {
            if(t < t_max) {
                float3 t_centre_vector = (voxelCentre - o) / d;
                
                char xyz_flag = makeXYZFlag(t_centre_vector, t, d);
                float nodeHalfSize = half_size/2.0f;
                float3 tmp_corner_far_step = corner_far_step/2.0f;
                
                float3 tmpNodeCentre = make_float3(xyz_flag & 1 ? voxelCentre.x + nodeHalfSize : voxelCentre.x - nodeHalfSize,
                                                   xyz_flag & 2 ? voxelCentre.y + nodeHalfSize : voxelCentre.y - nodeHalfSize,
                                                   xyz_flag & 4 ? voxelCentre.z + nodeHalfSize : voxelCentre.z - nodeHalfSize);
                
                float3 tmp_node_far = tmpNodeCentre+tmp_corner_far_step;
                
                float tmp_max = min_component((tmp_node_far-o)/d);
                
                if(node_has_child_at(node_header, xyz_flag)) {
                    // Check for LOD
                    if(nodeHalfSize < pixel_half_size*t) {
                        collision = true;
                        break;
                    }
                    
                    curr_index = push(short_stack, curr_index, curr_address, voxelCentre, t_max);
                    
                    curr_address = get_child(node_header, curr_address, xyz_flag);
                    curr_address_uint = (uint*)curr_address;
                    node_header = curr_address_uint[0];
                    
                    voxelCentre = tmpNodeCentre;
                    t_max = tmp_max;
                    
                    depth_in_octree++;
                    half_size = nodeHalfSize;
                    corner_far_step = tmp_corner_far_step;
                } else {
                    t = tmp_max;
                }
            } else {
                /* We are outside the node. Pop the stack */
                curr_index--;
                if(curr_index>=0) {
                    /* Pop that stack! */
                    voxelCentre = short_stack[curr_index].node_centre;
                    curr_address = short_stack[curr_index].address;
                    curr_address_uint = (uint*)curr_address;
                    node_header = curr_address_uint[0];
                    t_max = short_stack[curr_index].t_max;
                    half_size*=2.0f;
                    corner_far_step=corner_far_step*2.0f;
                    depth_in_octree--;
                } else {
                    /* Since we are using a short stack, we restart from the root node. */
                    curr_index = 0;
                    curr_address = octree;
                    curr_address_uint = (uint*)curr_address;
                    node_header = curr_address_uint[0];
                    half_size = OCTREE_ROOT_HALF_SIZE;
                    //collission = true;
                    corner_far_step=corner_far_step*2.0f;
                    corner_far = (float3)(corner_far_step);
                    t_max = min_component((corner_far - o) / d);
                    depth_in_octree = 0;
                }
            }
        }
    }
    
    struct collision col;
    col.node = curr_address;
    col.t = t;
    col.it = it;
    return col;
}

__global__ void ray_trace(cuda_render_info* render_info,char* header, char* octree, char* framebuffer, short* it_buffer, const int x_start) {
    int x = threadIdx.x + (blockDim.x*blockIdx.x); 
    int y = threadIdx.y + (blockDim.y*blockIdx.y);
    
    float3 o = render_info->viewPortStart + (render_info->viewStep * (x_start + x)) + (render_info->up * y);
    float3 d = o-render_info->eyePos;
    o = render_info->eyePos;
    
    float t_start = 0.0f;
    
    struct collision col = find_collision(octree, o, d, t_start, render_info->pixel_half_size);
    
    float3 ray_pos = o + (d * col.t);
    float ambient = 0.2f;
    
    // Change later?
    int row_stride = gridDim.x;
    
    int index = ( x + (y * row_stride));
    it_buffer[index] = col.it;
    
    if(col.node) {
        char* attributes = get_attributes(col.node);
        
        uchar3 colour = getColours(attributes);
        
        unsigned char red = colour.x;
        unsigned char green = colour.y;
        unsigned char blue = colour.z;
        
        float3 direction_towards_light = normalize(render_info->lightPos - ray_pos);
        float3 normal = getNormal(attributes);
        
        float diffuse_coefficient = dot(direction_towards_light,normal);
        if(diffuse_coefficient<0)
            diffuse_coefficient=0.0f;
        
        red=(red*diffuse_coefficient*(1.0f-ambient))+(red*ambient);
        green=(green*diffuse_coefficient*(1.0f-ambient))+(green*ambient);
        blue=(blue*diffuse_coefficient*(1.0f-ambient))+(blue*ambient);
        
        index*=3;
        
        framebuffer[index + 0] = red;
        framebuffer[index + 1] = green;
        framebuffer[index + 2] = blue;
    }
}

__global__ void clear_framebuffer(char* framebuffer) {
    const int x = threadIdx.x + (blockDim.x*blockIdx.x); 
    const int y = threadIdx.y + (blockDim.y*blockIdx.y);
    
    const int index = ( x + (y * gridDim.x) ) * 3;
    
    framebuffer[index + 0] = 0;
    framebuffer[index + 1] = 0;
    framebuffer[index + 2] = 0;
}

__global__ void clear_itbuffer(short* it_buffer) {
    const int x = threadIdx.x + (blockDim.x*blockIdx.x); 
    const int y = threadIdx.y + (blockDim.y*blockIdx.y);
    
    const int index = ( x + (y * gridDim.x) );
    
    it_buffer[index] = 0;
}

__global__ void clear_costbuffer(unsigned int* cost_buffer) {
    const int x = threadIdx.x + (blockDim.x*blockIdx.x);
    
    cost_buffer[x] = 0;
}

__global__ void calculate_costs(short *it_buffer, uint* cost_buffer, const int height, const uint x_start) {
    __shared__ uint local_costs[RAY_BUNDLE_WINDOW_SIZE];
    
    const int x = threadIdx.x + (blockDim.x*blockIdx.x);
    
    uint val = 0;
    for(int y = 0; y < height; y++)
        val += (it_buffer[x + (y*gridDim.x)]);
    
    local_costs[threadIdx.x] = val;
    
    __syncthreads();
    
    if(threadIdx.x==0) {
        for(int i = 1; i < RAY_BUNDLE_WINDOW_SIZE; i++) {
            val+= local_costs[i];
        }
        cost_buffer[(x_start+x)/RAY_BUNDLE_WINDOW_SIZE] = val;
    }
    
}

/*
 * Host code
 */

CUDADevice::CUDADevice(int device_index)
:   Device(false),
    m_device_index(device_index),
    m_pDevFramebuffer(0),
    m_pItBuffer(0) {
    m_pDeviceInfo = new CUDADeviceInfo(device_index);
    
    cudaSetDevice(m_device_index);
    
    cudaError_t error = cudaMalloc(&m_dev_render_info, sizeof(cuda_render_info));
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
}

CUDADevice::~CUDADevice() {

}

void CUDADevice::printInfo() {
    m_pDeviceInfo->printInfo();
}

void CUDADevice::sendData(Bin bin){
    cudaSetDevice(m_device_index);
    
    cudaError_t error = cudaMalloc(&m_pOctree, bin.getSize());
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
    error = cudaMemcpy(m_pOctree, bin.getDataPointer(), bin.getSize(), cudaMemcpyHostToDevice);
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
}

void CUDADevice::sendHeader(Bin bin) {
    cudaSetDevice(m_device_index);
    
    cudaError_t error = cudaMalloc(&m_pHeader, bin.getSize());
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
    error = cudaMemcpy(m_pHeader, bin.getDataPointer(), bin.getSize(), cudaMemcpyHostToDevice);
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
}

void CUDADevice::makeFrameBuffer(vector::int2 size){
    cudaSetDevice(m_device_index);
    if(m_frameBufferResolution != size) {
        if(m_pDevFramebuffer) {
            cudaFree(m_pDevFramebuffer);
            m_pDevFramebuffer = 0;
            cudaFree(m_pItBuffer);
            m_pItBuffer = 0;
            cudaFree(m_pCostBuffer);
            m_pCostBuffer = 0;
        }
        cudaMalloc(&m_pDevFramebuffer, size.getX()*size.getY()*3);
        cudaMalloc(&m_pItBuffer, size.getX()*size.getY()*sizeof(short));
        cudaMalloc(&m_pCostBuffer, (size.getX()/RAY_BUNDLE_WINDOW_SIZE)*sizeof(uint));
    }
    Device::makeFrameBuffer(size);
    
    dim3 threads(size.getX(), size.getY());
    clear_framebuffer<<<threads,1>>>(m_pDevFramebuffer);
    clear_itbuffer<<<threads,1>>>(m_pItBuffer);
    clear_costbuffer<<<size.getX()/RAY_BUNDLE_WINDOW_SIZE,1>>>(m_pCostBuffer);
}

void CUDADevice::setRenderInfo(renderinfo* info) {
    if(m_tasks.size()==0)
        return;
    
    cudaSetDevice(m_device_index);
    cuda_render_info render_info = cudaConvert(info);
    
    cudaError_t error = cudaMemcpy(m_dev_render_info, &render_info, sizeof(cuda_render_info), cudaMemcpyHostToDevice);
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
}

void CUDADevice::advanceTask(int index) {

}

void CUDADevice::renderTask(int index) {
    if(m_tasks.size()==0)
        return;
    
    cudaSetDevice(m_device_index);
    dim3 threads(m_tasks[index].getWidth(), m_tasks[index].getHeight());
    
    rect window = m_tasks[index];
    printf("device %p task %d start %d %d size %d %d\n", this, index, window.getX(), window.getY(), window.getWidth(), window.getHeight());
    
    ray_trace<<<threads,1>>>(m_dev_render_info, m_pHeader, m_pOctree, m_pDevFramebuffer, m_pItBuffer, m_tasks[index].getX());
}

void CUDADevice::calculateCostsForTask(int index) {
    if(m_tasks.size()==0)
        return;
    
    cudaSetDevice(m_device_index);
    dim3 blocks(m_tasks[index].getWidth()/RAY_BUNDLE_WINDOW_SIZE);
    dim3 threads(RAY_BUNDLE_WINDOW_SIZE);
    calculate_costs<<<blocks,threads>>>(m_pItBuffer, m_pCostBuffer, m_tasks[index].getHeight(), m_tasks[index].getX());
}

void CUDADevice::renderEnd() {
    cudaSetDevice(m_device_index);
    cudaDeviceSynchronize();
    
    Device::renderEnd();
}

framebuffer_window CUDADevice::getFrameBuffer() {
    framebuffer_window window;
    
    getFrame();
    
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
                 getTotalTaskWindow().getWidth(),
                 getTotalTaskWindow().getHeight(),
                 0,
                 GL_RGB,
                 GL_UNSIGNED_BYTE,
                 m_pFrame);
    
    window.window = getTotalTaskWindow();
    window.texture = m_texture;
    
    m_transferEnd.reset();
    
    return window;
}

unsigned char* CUDADevice::getFrame() {
    m_transferStart.reset();
    cudaSetDevice(m_device_index);
    cudaError_t error = cudaMemcpy( m_pFrame, m_pDevFramebuffer, getTotalTaskWindow().getWidth()*getTotalTaskWindow().getHeight()*3, cudaMemcpyDeviceToHost);
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
    
    return m_pFrame;
}

unsigned int* CUDADevice::getCosts() {
    cudaSetDevice(m_device_index);
    cudaError_t error = cudaMemcpy( m_pCosts, m_pCostBuffer, m_frameBufferResolution.getX()/RAY_BUNDLE_WINDOW_SIZE*sizeof(uint), cudaMemcpyDeviceToHost);
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
    
    if(getTotalTaskWindow().getWidth() ) {
        printf("this %p ", this);
        for(int x = 0; x < (m_frameBufferResolution[0]/RAY_BUNDLE_WINDOW_SIZE); x++) {
            printf("%d ", m_pCosts[x]);
        }
        printf("\n");
    }
    
    return m_pCosts;
}

bool CUDADevice::isCPU(){
    return false;
}
