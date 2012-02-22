#include "OpenCLDevice.h"

#include "OpenCLDeviceInfo.h"
#include "OpenCLUtils.h"

#include "OctreeSegment.h"

#include "SourceFile.h"
#include "SourceFileManager.h"

OpenCLDevice::OpenCLDevice(cl_device_id device_id, cl_context context)
:	m_DeviceID(device_id),
    m_context(context),
    m_frameBufferResolution(int2(0)),
    m_texture(0) {
	m_pDeviceInfo = new OpenCLDeviceInfo(device_id);
    
    cl_int err = 0;
    
    // Perhaps profiling should be enabled?
    m_commandQueue = clCreateCommandQueue(context, device_id, 0, &err);
    
    if(clIsError(err)){
		clPrintError(err); return;
	}
    
    // Create octree memory in the object, the host will only write, not read. And the device will only read.
    // We make it 512 bytes only for now.
    m_memory = clCreateBuffer(context, CL_MEM_COPY_HOST_WRITE_ONLY | CL_MEM_READ_ONLY, 512, NULL, &err);
    
    if(clIsError(err)){
        clPrintError(err); return;
    }
}


OpenCLDevice::~OpenCLDevice(){

}

void OpenCLDevice::printInfo() {
	m_pDeviceInfo->printInfo();
}

void OpenCLDevice::makeFrameBuffer(int2 size) {
    cl_int error;
    if(size != m_frameBufferResolution && !m_frameBufferResolution[0]) {
        error = clReleaseMemObject(m_frameBuff);
        if(clIsError(error)){
            clPrintError(error);
        }
    }
	m_frameBuff = clCreateBuffer ( m_context, CL_MEM_WRITE_ONLY, size[1]*size[0]*3, NULL, &error);
    if(clIsError(error)){
        clPrintError(error);
    }
    m_frameBufferResolution = size;
}

void OpenCLDevice::sendData(OctreeSegment* segment) {
    clEnqueueWriteBuffer(m_commandQueue, m_memory, CL_FALSE, 0, segment->getSize(), (void*)segment->getData(), 0, NULL, NULL);
}

void OpenCLDevice::render(int2 start, int2 size, renderinfo *info) {

}

GLuint OpenCLDevice::getFrameBuffer() {
	if (!m_texture) {
        glGenTextures(1, &m_texture);
    }
    
    int size = m_frameBufferResolution[0]*m_frameBufferResolution[1]*3;
    char* frameBuffer = (char*) malloc(size);
    
    cl_int error;
    error = clEnqueueReadBuffer ( m_commandQueue, m_frameBuff, GL_FALSE, 0, size, (void*) frameBuffer, 0, NULL, NULL);
    if(clIsError(error)){
        clPrintError(error);
    }
    
    error = clFinish(m_commandQueue);
    if(clIsError(error)){
        clPrintError(error);
    }
    
    glBindTexture(GL_TEXTURE_2D, m_texture);
    
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGB,
                 m_frameBufferResolution[0],
                 m_frameBufferResolution[1],
                 0,
                 GL_RGB,
                 GL_UNSIGNED_BYTE,
                 frameBuffer);
    
    return m_texture;
}

char* OpenCLDevice::getFrame() {

}

cl_context OpenCLDevice::getOpenCLContext() {
	return m_context;
}


cl_device_id OpenCLDevice::getOpenCLDeviceID() {
	return m_DeviceID;
}
