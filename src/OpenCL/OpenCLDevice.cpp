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
    // We make it 128MB for now
    m_memory = clCreateBuffer(context, CL_MEM_COPY_HOST_WRITE_ONLY | CL_MEM_READ_ONLY, 128*1024*1024, NULL, &err);

    if(clIsError(err)){
        clPrintError(err); return;
    }

    m_pProgram = new OpenCLProgram(this, "RayTracing.cl");

	m_rayTraceKernel = m_pProgram->getOpenCLKernel("ray_trace");
    m_rayBundleTraceKernel = m_pProgram->getOpenCLKernel("trace_bundle");
    m_clearFrameBuffKernel = m_pProgram->getOpenCLKernel("clear_framebuffer");
    m_clearDepthBuffKernel = m_pProgram->getOpenCLKernel("clear_depthbuffer");
}

OpenCLDevice::~OpenCLDevice(){

}

void OpenCLDevice::printInfo() {
	m_pDeviceInfo->printInfo();
}

void OpenCLDevice::makeFrameBuffer(int2 size) {
    cl_int error;
    Device::makeFrameBuffer(size);
    if(size != m_frameBufferResolution) {
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
        cl_image_format single_channel_image_format = {CL_INTENSITY, CL_FLOAT};
        m_depthBuff = clCreateImage2D ( m_context, CL_MEM_READ_WRITE, &single_channel_image_format, size[0], size[1], 0, NULL, &error);
        if(clIsError(error)){
            clPrintError(error);
        }
        /*image_format.image_channel_order = CL_UNSIGNED_INT8;
        m_iterationsBuff = clCreateImage2D ( m_context, CL_MEM_WRITE_ONLY, &image_format, size[0], size[1], 0, NULL, &error);
        if(clIsError(error)){
            clPrintError(error);
        }
        m_octreeDepthBuff = clCreateImage2D ( m_context, CL_MEM_WRITE_ONLY, &image_format, size[0], size[1], 0, NULL, &error);
        if(clIsError(error)){
            clPrintError(error);
        }*/
        m_frameBufferResolution = size;
        error = clSetKernelArg( m_rayTraceKernel, 3, sizeof(cl_mem), &m_frameBuff);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
        error = clSetKernelArg( m_rayTraceKernel, 4, sizeof(cl_mem), &m_depthBuff);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
        error = clSetKernelArg( m_rayBundleTraceKernel, 4, sizeof(cl_mem), &m_depthBuff);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
        error = clSetKernelArg( m_clearFrameBuffKernel, 0, sizeof(cl_mem), &m_frameBuff);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
        
        error = clSetKernelArg( m_clearDepthBuffKernel, 0, sizeof(cl_mem), &m_depthBuff);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
        size_t origin[2] = {0, 0}; 
        size_t region[2] = {size[0], size[1]}; 
        error = clEnqueueNDRangeKernel( m_commandQueue, m_clearDepthBuffKernel, 2, origin, region, NULL, 0, NULL, NULL);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
        
        /*error = clEnqueueWriteImage(m_commandQueue, m_frameBuff, CL_FALSE, origin, region, region[0]*4, 0, m_pFrame, 0, NULL, NULL);
        if(clIsError(error)){
            clPrintError(error);
        }*/
    }
    size_t origin[2] = {0, 0}; 
    size_t region[2] = {size[0], size[1]}; 
    error = clEnqueueNDRangeKernel( m_commandQueue, m_clearFrameBuffKernel, 2, origin, region, NULL, 0, NULL, NULL);
    if(clIsError(error)){
        clPrintError(error); exit(1);
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
    if(window.getWidth() == 0 || window.getHeight() == 0)
        return;
    
    rect bundle_window = rect(window.getOrigin()/RAY_BUNDLE_WINDOW_SIZE, window.getSize()/RAY_BUNDLE_WINDOW_SIZE);
    
    cl_int bundle_window_size = RAY_BUNDLE_WINDOW_SIZE;
    
    cl_renderinfo cl_info= convert(*info);

    //printf("device %p task %d start %d %d size %d %d\n", this, index, window.getX(), window.getY(), window.getWidth(), window.getHeight());

    /*
     * We first trace the ray bundles 
     */
    
    cl_int error = clSetKernelArg( m_rayBundleTraceKernel, 0, sizeof(cl_mem), &m_memory);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }

    error = clSetKernelArg( m_rayBundleTraceKernel, 1, sizeof(cl_mem), &m_header);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    
    error = clSetKernelArg( m_rayBundleTraceKernel, 2, sizeof(cl_renderinfo), &cl_info);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    
    error = clSetKernelArg( m_rayBundleTraceKernel, 3, sizeof(cl_int), &bundle_window_size);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    
    size_t bundle_offset[2] = {bundle_window.getOrigin()[0], bundle_window.getOrigin()[1]};
    size_t bundle_dimensions[2] = {bundle_window.getSize()[0], bundle_window.getSize()[1]};
    error = clEnqueueNDRangeKernel( m_commandQueue, m_rayBundleTraceKernel, 2, bundle_offset, bundle_dimensions, NULL, 0, NULL, NULL);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    
    /*
     * We now trace the rays
     */ 
    
	error = clSetKernelArg( m_rayTraceKernel, 0, sizeof(cl_mem), &m_memory);
 	if(clIsError(error)){
        clPrintError(error); exit(1);
    }

    error = clSetKernelArg( m_rayTraceKernel, 1, sizeof(cl_mem), &m_header);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }

    error = clSetKernelArg( m_rayTraceKernel, 2, sizeof(cl_renderinfo), &cl_info);
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
    framebuffer_window fb_window;
    fb_window.window = getTotalTaskWindow();
    fb_window.texture = 0;
    
    if(fb_window.window.getWidth() == 0 || fb_window.window.getHeight() == 0)
        return fb_window;
    
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
    unsigned char* frameBuffer = getFrame();

    glBindTexture(GL_TEXTURE_2D, m_texture);

    if(m_renderMode == COLOUR) {
        glTexImage2D(GL_TEXTURE_2D,
                    0,
                    GL_RGB,
                    getTotalTaskWindow().getWidth(),
                    getTotalTaskWindow().getHeight(),
                    0,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    frameBuffer);
    
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

    fb_window.texture = m_texture;

    return fb_window;
}

unsigned char* OpenCLDevice::getFrame() {
    size_t origin[3] = {getTotalTaskWindow().getX(), getTotalTaskWindow().getY(), 0};
    size_t region[3] = {getTotalTaskWindow().getWidth(), getTotalTaskWindow().getHeight(), 1};
    int size = getTotalTaskWindow().getWidth()*getTotalTaskWindow().getHeight()*4;

    cl_int error;
    
    //printf("device %d origin %d %d region %d %d\n", this, origin[0], origin[1], region[0], region[1]);
    
    // Read the depth buffer, not always necessary
    error = clEnqueueReadImage( m_commandQueue, m_depthBuff, GL_FALSE, origin, region, 0, 0, m_pDepthBuffer, 1, &m_eventRenderingFinished, NULL);
    if(clIsError(error)){
        clPrintError(error);
    }
    
    //error = clEnqueueReadBuffer ( m_commandQueue, m_frameBuff, GL_FALSE, 0, size, (void*) m_pFrame, 1, &m_eventRenderingFinished, &m_eventFrameBufferRead);
    error = clEnqueueReadImage ( m_commandQueue, m_frameBuff, GL_FALSE, origin, region, 0, 0, m_pFrame, 1, &m_eventRenderingFinished, &m_eventFrameBufferRead);
    if(clIsError(error)){
        clPrintError(error);
    }
    
    cl_event events[2] = {m_eventRenderingFinished, m_eventFrameBufferRead};
    error = clWaitForEvents(2, events);
	if(clIsError(error)){
        clPrintError(error);
    }
    
    /*printf("--------------\n");
    for(int y = 0; y < getTotalTaskWindow().getHeight(); y++) 
        for(int x = 0; x < getTotalTaskWindow().getWidth(); x++) {     
            int index = (x*4)+(y*4*getTotalTaskWindow().getWidth());
            if(m_pFrame[index] > 244){
                printf("x %d y %d\n", x, y); 
                printf("depthbuffer value %f\n", m_pDepthBuffer[(x)+(y*getTotalTaskWindow().getWidth())]);
            }
        }
    printf("--------------\n");*/
    
    return m_pFrame;
}

void OpenCLDevice::onRenderingFinished() {
    //printf("render end\n");
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
