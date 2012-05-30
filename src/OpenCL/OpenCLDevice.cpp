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
:   Device(),
    m_DeviceID(device_id),
    m_context(context),
    m_frameBufferResolution(int2(0)),
    m_texture(0) {
	m_pDeviceInfo = new OpenCLDeviceInfo(device_id, context);

    cl_int err = 0;

    // Perhaps profiling should be enabled?
    m_commandQueue = clCreateCommandQueue(context, device_id, 0, &err);

    if(clIsError(err)){
		clPrintError(err); return;
	}

    // Create octree memory in the object, the host will only write, not read. And the device will only read.
    // We make it 64MB for now
    m_memory = clCreateBuffer(context, CL_MEM_COPY_HOST_WRITE_ONLY | CL_MEM_READ_ONLY, 64*1024*1024, NULL, &err);

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
    Device::makeFrameBuffer(size);
    if(size != m_frameBufferResolution) {
        cl_int error;
        if(m_frameBufferResolution[0]) {
            error = clReleaseMemObject(m_frameBuff);
            if(clIsError(error)){
                clPrintError(error);
            }
            error = clReleaseMemObject(m_depthBuff);
            if(clIsError(error)){
                clPrintError(error);
            }
            error = clReleaseMemObject(m_iterationsBuff);
            if(clIsError(error)){
                clPrintError(error);
            }
            error = clReleaseMemObject(m_octreeDepthBuff);
            if(clIsError(error)){
                clPrintError(error);
            }
        }
        // Image format
        cl_image_format image_format = {CL_RGBA, CL_UNSIGNED_INT8};
        // Only needed in OpenCL 1.2
        //cl_image_desc image_descriptor = { /*width*/size[0],
        //                                   /*height*/size[1],
        //                                   /*depth*/ 1,
        //                                   /*image array size*/ 1,
        //                                   /*row pitch*/ 0,
        //                                   /*slice pitch*/ 0,
        //                                   /*mip level*/ 0, /*num samples*/ 0,
        //                                   /*buffer*/NULL};
        //clCreateBuffer ( m_context, CL_MEM_WRITE_ONLY, size[1]*size[0]*3, NULL, &error);
        m_frameBuff = clCreateImage2D ( m_context, CL_MEM_WRITE_ONLY, &image_format, size[0], size[1], 0, NULL, &error);
        if(clIsError(error)){
            clPrintError(error);
        }
        /*image_format.image_channel_order = CL_R;
        image_format.image_channel_order = CL_FLOAT;
        m_depthBuff = clCreateImage2D ( m_context, CL_MEM_READ_WRITE, &image_format, size[0], size[1], 0, NULL, &error);
        if(clIsError(error)){
            clPrintError(error);
        }
        image_format.image_channel_order = CL_UNSIGNED_INT8;
        m_iterationsBuff = clCreateImage2D ( m_context, CL_MEM_WRITE_ONLY, &image_format, size[0], size[1], 0, NULL, &error);
        if(clIsError(error)){
            clPrintError(error);
        }
        m_octreeDepthBuff = clCreateImage2D ( m_context, CL_MEM_WRITE_ONLY, &image_format, size[0], size[1], 0, NULL, &error);
        if(clIsError(error)){
            clPrintError(error);
        }*/
        m_frameBufferResolution = size;
        error = clSetKernelArg( m_rayTraceKernel, 4, sizeof(cl_mem), &m_frameBuff);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
        size_t origin[3] = {0, 0, 0}; 
        size_t region[3] = {size[0], size[1], 1}; 
        error = clEnqueueWriteImage(m_commandQueue, m_frameBuff, CL_FALSE, origin, region, region[0]*4, 0, m_pFrame, 0, NULL, NULL);
        if(clIsError(error)){
            clPrintError(error);
        }
    }
}

void OpenCLDevice::sendData(Bin bin){
    clEnqueueWriteBuffer(m_commandQueue, m_memory, CL_FALSE, 0, bin.getSize(), (void*)bin.getDataPointer(), 0, NULL, NULL);
}

void OpenCLDevice::sendHeader(Bin bin) {
    cl_int err = 0;

    // We create memory for the header.
    m_header = clCreateBuffer(m_context, CL_MEM_COPY_HOST_WRITE_ONLY | CL_MEM_READ_ONLY, bin.getSize(), NULL, &err);

    if(clIsError(err)){
        clPrintError(err); return;
    }

    clEnqueueWriteBuffer(m_commandQueue, m_header, CL_FALSE, 0, bin.getSize(), (void*)bin.getDataPointer(), 0, NULL, NULL);
}

void OpenCLDevice::renderTask(int index, renderinfo *info) {
    m_renderStart.reset();

    rect window = m_tasks[index];

    printf("start %d %d size %d %d\n", window.getX(), window.getY(), window.getWidth(), window.getHeight());

	cl_int error = clSetKernelArg( m_rayTraceKernel, 0, sizeof(cl_mem), &m_memory);
 	if(clIsError(error)){
        clPrintError(error); exit(1);
    }

    error = clSetKernelArg( m_rayTraceKernel, 1, sizeof(cl_mem), &m_header);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }

    cl_renderinfo cl_info= convert(*info);
    error = clSetKernelArg( m_rayTraceKernel, 2, sizeof(cl_renderinfo), &cl_info);
 	if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    cl_int2 origin = { window.getX(), window.getY()};
	error = clSetKernelArg( m_rayTraceKernel, 3, sizeof(cl_int2), &origin);
 	if(clIsError(error)){
        clPrintError(error); exit(1);
    }

    size_t offset[2] = {window.getOrigin()[0], window.getOrigin()[1]};
    size_t dimensions[2] = {window.getSize()[0], window.getSize()[1]};
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

framebuffer_window OpenCLDevice::getFrameBuffer() {
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
                 getTotalTaskWindow().getWidth(),
                 getTotalTaskWindow().getHeight(),
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 frameBuffer);

    m_transferEnd.reset();

    framebuffer_window fb_window;
    fb_window.window = getTotalTaskWindow();
    fb_window.texture = m_texture;

    return fb_window;
}

char* OpenCLDevice::getFrame() {
    size_t origin[3] = {getTotalTaskWindow().getX(), getTotalTaskWindow().getY(), 0};
    size_t region[3] = {getTotalTaskWindow().getWidth(), getTotalTaskWindow().getHeight(), 1};
    int size = getTotalTaskWindow().getWidth()*getTotalTaskWindow().getHeight()*4;

    cl_int error;
    //error = clEnqueueReadBuffer ( m_commandQueue, m_frameBuff, GL_FALSE, 0, size, (void*) m_pFrame, 1, &m_eventRenderingFinished, &m_eventFrameBufferRead);
    error = clEnqueueReadImage ( m_commandQueue, m_frameBuff, GL_FALSE, origin, region, region[0]*4, 0, m_pFrame, 1, &m_eventRenderingFinished, &m_eventFrameBufferRead);
    if(clIsError(error)){
        clPrintError(error);
    }
    cl_event events[2] = {m_eventRenderingFinished, m_eventFrameBufferRead};
    error = clWaitForEvents(2, events);
	if(clIsError(error)){
        clPrintError(error);
    }
    return m_pFrame;
}

void OpenCLDevice::onRenderingFinished() {
    printf("render end\n");
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
