#include "OpenCLDevice.h"

#include "OpenCLDeviceInfo.h"
#include "OpenCLProgram.h"
#include "OpenCLUtils.h"
#include "OpenCLRenderInfo.h"

#include "OctreeSegment.h"

#include "SourceFile.h"
#include "SourceFileManager.h"

void CL_CALLBACK staticOnRenderingFinished(cl_event event, cl_int event_command_exec_status, void *user_data){
	((OpenCLDevice*)user_data)->onRenderingFinished();
}

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
    
    m_pProgram = new OpenCLProgram(this, "RayTracing.cl");
	
	m_rayTraceKernel = m_pProgram->getOpenCLKernel("ray_trace");
}

OpenCLDevice::~OpenCLDevice(){

}

void OpenCLDevice::printInfo() {
	m_pDeviceInfo->printInfo();
}

void OpenCLDevice::makeFrameBuffer(int2 size) {
    if(size != m_frameBufferResolution) {
		cl_int error;
		if(m_frameBufferResolution[0]) {
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
		error = clSetKernelArg( m_rayTraceKernel, 3, sizeof(cl_mem), &m_frameBuff);
		if(clIsError(error)){
			clPrintError(error); exit(1);
		}
    }
}

void OpenCLDevice::sendData(OctreeSegment* segment) {
    clEnqueueWriteBuffer(m_commandQueue, m_memory, CL_FALSE, 0, segment->getSize(), (void*)segment->getData(), 0, NULL, NULL);
}

void OpenCLDevice::render(int2 start, int2 size, renderinfo *info) {
    m_renderStart.reset();
    
	cl_int error = clSetKernelArg( m_rayTraceKernel, 0, sizeof(cl_mem), &m_memory);
 	if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    
    cl_renderinfo cl_info= convert(*info);
    error = clSetKernelArg( m_rayTraceKernel, 1, sizeof(cl_renderinfo), &cl_info);
 	if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    int frameBufferWidth = m_frameBufferResolution[0];
	error = clSetKernelArg( m_rayTraceKernel, 2, sizeof(cl_int), &frameBufferWidth);
 	if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    
    size_t offset[2] = {start[0], start[1]};
    size_t dimensions[2] = {size[0], size[1]};
    error = clEnqueueNDRangeKernel( m_commandQueue, m_rayTraceKernel, 2, offset, dimensions, NULL, 0, NULL, &m_eventRenderingFinished);
	if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    error = clSetEventCallback( m_eventRenderingFinished,
								CL_COMPLETE ,
								staticOnRenderingFinished,
								(void*)this);
	if(clIsError(error)){
		clPrintError(error);
	}
}

GLuint OpenCLDevice::getFrameBuffer() {
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
    char* frameBuffer = getFrame();
    
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
	
	free(frameBuffer);
    
    m_transferEnd.reset();
    
    return m_texture;
}

char* OpenCLDevice::getFrame() {
	int size = m_frameBufferResolution[0]*m_frameBufferResolution[1]*3;
	char* frameBuffer = (char*) malloc(size+1);
    
    cl_int error;
    error = clEnqueueReadBuffer ( m_commandQueue, m_frameBuff, GL_FALSE, 0, size, (void*) frameBuffer, 1, &m_eventRenderingFinished, &m_eventFrameBufferRead);
    if(clIsError(error)){
        clPrintError(error);
    }
    cl_event events[2] = {m_eventRenderingFinished, m_eventFrameBufferRead};
    error = clWaitForEvents(2, events);
	if(clIsError(error)){
        clPrintError(error);
    }
    return frameBuffer;
}

void OpenCLDevice::onRenderingFinished() {
	m_renderEnd.reset();
    m_transferStart.reset();
}

high_res_timer OpenCLDevice::getRenderTime() {
    return m_renderEnd - m_renderStart;
}

high_res_timer OpenCLDevice::getBufferToTextureTime() {
    return m_transferEnd - m_transferStart;
}

cl_context OpenCLDevice::getOpenCLContext() {
	return m_context;
}


cl_device_id OpenCLDevice::getOpenCLDeviceID() {
	return m_DeviceID;
}
