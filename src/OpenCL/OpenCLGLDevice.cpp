#include "OpenCLGLDevice.h"

#include "OpenCLUtils.h"

OpenCLGLDevice::OpenCLGLDevice(cl_device_id device_id, cl_context context)
:   OpenCLDevice(device_id, context){
    
}
                    
void OpenCLGLDevice::makeFrameBuffer(int2 size) {
    Device::makeFrameBuffer(size);
    if (!m_texture) {
        glGenTextures(1, &m_texture);

        glBindTexture(GL_TEXTURE_2D, m_texture);

        glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
        
        glTexImage2D(GL_TEXTURE_2D,
                    0,
                    GL_RGB,
                    size.getX(),
                    size.getY(),
                    0,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    m_pFrame);
    }
    glGenTextures(1, 0);
    cl_int error;
    
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
        m_frameBuff = clCreateFromGLTexture2D(m_context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, m_texture, &error);
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
    
    error = clEnqueueAcquireGLObjects(m_commandQueue, 1, &m_frameBuff, 0, NULL, NULL);

    
    size_t origin[2] = {0, 0};
    size_t region[2] = {size[0], size[1]};
    error = clEnqueueNDRangeKernel( m_commandQueue, m_clearFrameBuffKernel, 2, origin, region, NULL, 0, NULL, NULL);
    if(clIsError(error)){
        clPrintError(error); exit(1);
    }
}

framebuffer_window OpenCLGLDevice::getFrameBuffer() {
    
}