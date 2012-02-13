#include "OpenCLDevice.h"

#include "OpenCLDeviceInfo.h"
#include "OpenCLUtils.h"

OpenCLDevice::OpenCLDevice(cl_device_id device_id, cl_context context)
:	m_DeviceID(device_id),
    m_context(context){
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

void OpenCLDevice::sendData(char* data, size_t size) {
    clEnqueueWriteBuffer(m_commandQueue, m_memory, CL_FALSE, 0, size, (void*)data, NULL, 0, NULL);
}

void OpenCLDevice::render(RenderInfo &info) {

}

char* OpenCLDevice::getFrame() {

}
