#include "OpenCLDevice.h"

#include "OpenCLDeviceInfo.h"
#include "OpenCLProgram.h"
#include "OpenCLUtils.h"
#include "OpenCLRenderInfo.h"

#include "OctreeSegment.h"

#include "SourceFile.h"
#include "SourceFileManager.h"

#include "SizeMacros.h"

void CL_CALLBACK staticOnRenderingFinished(cl_event event, cl_int event_command_exec_status, void *user_data){
	((OpenCLDevice*)user_data)->onRenderingFinished();
}

OpenCLDevice::OpenCLDevice(cl_device_id device_id, cl_context context)
:   Device(false),
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
    // We make it 512MB for now
    m_memory = clCreateBuffer(context, CL_MEM_COPY_HOST_WRITE_ONLY | CL_MEM_READ_ONLY, 512*1024*1024, NULL, &err);

    if(clIsError(err)){
        clPrintError(err); return;
    }

    m_pProgram = new OpenCLProgram(this, "RayTracing.cl");

	m_rayTraceKernel = m_pProgram->getOpenCLKernel("ray_trace");
    m_rayBundleTraceKernel = m_pProgram->getOpenCLKernel("trace_bundle");
    m_clearBufferKernel = m_pProgram->getOpenCLKernel("clear_buffer");
    m_calculateCostsKernel = m_pProgram->getOpenCLKernel("calculate_costs");
    m_clearCostsKernel = m_pProgram->getOpenCLKernel("clear_uintbuffer");
    
    cl_int bundle_window_size = RAY_BUNDLE_WINDOW_SIZE;
    
    err = clSetKernelArg( m_rayBundleTraceKernel, 3, sizeof(cl_int), &bundle_window_size);
    if(clIsError(err)){
        clPrintError(err); exit(1);
    }
}

OpenCLDevice::~OpenCLDevice(){

}

void OpenCLDevice::printInfo() {
	m_pDeviceInfo->printInfo();
}

void OpenCLDevice::makeFrameBuffer(int2 size) {
    cl_int error;
    Device::makeFrameBuffer(size);
    
    size_t origin[2] = {0, 0};
    size_t region[2] = {size[0], size[1]};
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
            error = clReleaseMemObject(m_windowCosts);
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
        cl_image_format it_image_format = {CL_INTENSITY, CL_FLOAT};
        m_iterationsBuff = clCreateImage2D ( m_context, CL_MEM_READ_WRITE, &it_image_format, size[0], size[1], 0, NULL, &error);
        if(clIsError(error)){
            clPrintError(error);
        }
        /*m_octreeDepthBuff = clCreateImage2D ( m_context, CL_MEM_WRITE_ONLY, &image_format, size[0], size[1], 0, NULL, &error);
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
        error = clSetKernelArg( m_rayTraceKernel, 5, sizeof(cl_mem), &m_iterationsBuff);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
      
        error = clSetKernelArg( m_clearBufferKernel, 0, sizeof(cl_mem), &m_depthBuff);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
        
        error = clEnqueueNDRangeKernel( m_commandQueue, m_clearBufferKernel, 2, origin, region, NULL, 0, NULL, NULL);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
        
        error = clSetKernelArg( m_clearBufferKernel, 0, sizeof(cl_mem), &m_frameBuff);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }

        // FOR COST CREATION
        error = clSetKernelArg( m_calculateCostsKernel, 0, sizeof(cl_mem), &m_iterationsBuff);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
        m_windowCosts = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, sizeof(cl_uint)*(size[0]/RAY_BUNDLE_WINDOW_SIZE), NULL, &error);
        error = clSetKernelArg( m_calculateCostsKernel, 1, sizeof(cl_mem), &m_windowCosts);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
        //error = clSetKernelArg( m_calculateCostsKernel, 2, sizeof(cl_uint)*(WINDOW_SIZE)*(WINDOW_SIZE), NULL);
        //if(clIsError(error)){
        //    clPrintError(error); exit(1);
        //}
        error = clSetKernelArg( m_clearCostsKernel, 0, sizeof(cl_mem), &m_windowCosts);
        if(clIsError(error)){
            clPrintError(error); exit(1);
        }
        
        /*error = clEnqueueWriteImage(m_commandQueue, m_frameBuff, CL_FALSE, origin, region, region[0]*4, 0, m_pFrame, 0, NULL, NULL);
        if(clIsError(error)){
            clPrintError(error);
        }*/
    }
    error = clEnqueueNDRangeKernel( m_commandQueue, m_clearBufferKernel, 2, origin, region, NULL, 0, NULL, NULL);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    size_t clear_origin[1] = {0};
    size_t clear_region[1] = {(size[0]/RAY_BUNDLE_WINDOW_SIZE)};
    error = clEnqueueNDRangeKernel( m_commandQueue, m_clearCostsKernel, 1, clear_origin, clear_region, NULL, 0, NULL, NULL);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
}

void OpenCLDevice::sendData(Bin bin){
    cl_int error = clEnqueueWriteBuffer(m_commandQueue, m_memory, CL_FALSE, 0, bin.getSize(), (void*)bin.getDataPointer(), 0, NULL, NULL);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    
    error = clSetKernelArg( m_rayTraceKernel, 0, sizeof(cl_mem), &m_memory);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    
    error = clSetKernelArg( m_rayBundleTraceKernel, 0, sizeof(cl_mem), &m_memory);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
}

void OpenCLDevice::sendHeader(Bin bin) {
    cl_int err = 0;

    // We create memory for the header.
    m_header = clCreateBuffer(m_context, CL_MEM_READ_ONLY, bin.getSize(), NULL, &err);

    if(clIsError(err)){
        clPrintError(err); return;
    }

    err= clEnqueueWriteBuffer(m_commandQueue, m_header, CL_FALSE, 0, bin.getSize(), (void*)bin.getDataPointer(), 0, NULL, NULL);
    if(clIsError(err)){
        clPrintError(err); return;
    }
    
    err = clSetKernelArg( m_rayTraceKernel, 1, sizeof(cl_mem), &m_header);
    if(clIsError(err)){
        clPrintError(err); exit(1);
    }
    
    err = clSetKernelArg( m_rayBundleTraceKernel, 1, sizeof(cl_mem), &m_header);
    if(clIsError(err)){
        clPrintError(err); exit(1);
    }
}

void OpenCLDevice::setRenderInfo(renderinfo *info) {
    cl_renderinfo cl_info= convert(*info);
    
    cl_int error = clSetKernelArg( m_rayBundleTraceKernel, 2, sizeof(cl_renderinfo), &cl_info);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }

    error = clSetKernelArg( m_rayTraceKernel, 2, sizeof(cl_renderinfo), &cl_info);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
}

void OpenCLDevice::advanceTask(int index) {
    rect window = m_tasks[index];
    if(window.getWidth() == 0 || window.getHeight() == 0)
        return;
    
    rect bundle_window = rect(window.getOrigin()/RAY_BUNDLE_WINDOW_SIZE, window.getSize()/RAY_BUNDLE_WINDOW_SIZE);

    /*
     * We first trace the ray bundles
     */


    size_t bundle_offset[2] = {bundle_window.getOrigin()[0], bundle_window.getOrigin()[1]};
    size_t bundle_dimensions[2] = {bundle_window.getSize()[0], bundle_window.getSize()[1]};
    cl_int error = clEnqueueNDRangeKernel( m_commandQueue, m_rayBundleTraceKernel, 2, bundle_offset, bundle_dimensions, NULL, 0, NULL, NULL);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
}

void OpenCLDevice::renderTask(int index) {
    rect window = m_tasks[index];
    if(window.getWidth() == 0 || window.getHeight() == 0)
        return;

    //printf("device %p task %d start %d %d size %d %d\n", this, index, window.getX(), window.getY(), window.getWidth(), window.getHeight());

    /*
     * We now trace the rays
     */

    size_t offset[2] = {window.getOrigin()[0], window.getOrigin()[1]};
    size_t dimensions[2] = {window.getSize()[0], window.getSize()[1]};
    cl_int error = clEnqueueNDRangeKernel( m_commandQueue, m_rayTraceKernel, 2, offset, dimensions, NULL, 0, NULL, &m_eventRenderingFinished);
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

void OpenCLDevice::calculateCostsForTask(int index) {
    rect window = m_tasks[index];
    if(window.getWidth() == 0 || window.getHeight() == 0)
        return;
    
    cl_int error = clSetKernelArg(m_calculateCostsKernel, 0, sizeof(cl_mem), &m_iterationsBuff);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
    

    size_t offset[1] = {window.getOrigin()[0]};
    size_t dimensions[1] = {window.getSize()[0]};
    size_t work_group[1] = {RAY_BUNDLE_WINDOW_SIZE};
    printf("offset %d  dimensions %d  work_group %d \n", offset[0], dimensions[0], work_group[0]);
    error = clEnqueueNDRangeKernel(m_commandQueue, m_calculateCostsKernel, 1, offset, dimensions, work_group, 0, NULL, NULL);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
}

void OpenCLDevice::renderEnd() {
    clFinish(m_commandQueue);
    Device::renderEnd();
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

    error = clWaitForEvents(1, &m_eventRenderingFinished);
    if(clIsError(error)){
        clPrintError(error);
    }
    
    m_transferStart.reset();
    //printf("device %d origin %d %d region %d %d\n", this, origin[0], origin[1], region[0], region[1]);

    // Read the depth buffer, not always necessary
    error = clEnqueueReadImage( m_commandQueue, m_depthBuff, CL_FALSE, origin, region, 0, 0, m_pDepthBuffer, 1, &m_eventRenderingFinished, NULL);
    if(clIsError(error)){
        clPrintError(error);
    }

    //error = clEnqueueReadBuffer ( m_commandQueue, m_frameBuff, GL_FALSE, 0, size, (void*) m_pFrame, 1, &m_eventRenderingFinished, &m_eventFrameBufferRead);
    error = clEnqueueReadImage ( m_commandQueue, m_frameBuff, GL_FALSE, origin, region, 0, 0, m_pFrame, 1, &m_eventRenderingFinished, &m_eventFrameBufferRead);
    if(clIsError(error)){
        clPrintError(error);
    }
    
    error = clWaitForEvents(1, &m_eventFrameBufferRead);
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

unsigned int* OpenCLDevice::getCosts() {
    clEnqueueReadBuffer(m_commandQueue, 
                        m_windowCosts, 
                        CL_FALSE, 
                        0, 
                        sizeof(cl_uint)*(m_frameBufferResolution[0]/RAY_BUNDLE_WINDOW_SIZE),
                        m_pCosts,
                        0,
                        NULL,
                        NULL);
    
    clFinish(m_commandQueue);
    
    /*if(getTotalTaskWindow().getWidth() ) {
        printf("this %p ", this);
        for(int x = 0; x < (m_frameBufferResolution[0]/RAY_BUNDLE_WINDOW_SIZE); x++) {
            printf("%d ", m_pCosts[x]);
        }
        printf("\n");
    }*/

    
    return m_pCosts;
}

void OpenCLDevice::onRenderingFinished() {
    //printf("render end\n");
    
}

cl_context OpenCLDevice::getOpenCLContext() {
	return m_context;
}


cl_device_id OpenCLDevice::getOpenCLDeviceID() {
	return m_DeviceID;
}
