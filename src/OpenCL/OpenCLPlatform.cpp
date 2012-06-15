#include "OpenCLPlatform.h"
#include "OpenCLPlatformInfo.h"
#include "OpenCLDevice.h"

#include "OpenCLUtils.h"

#include "Graphics.h"

OpenCLPlatform::OpenCLPlatform(cl_platform_id platform_id)
:	m_PlatformID(platform_id),
    m_pPlatformInfo(new OpenCLPlatformInfo(platform_id)){
	cl_uint device_num;

	// Get the number of devices for this platform.
	cl_int err = clGetDeviceIDs(m_PlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &device_num);

	// If error, we print to stdout and leave.
	if(clIsError(err)){
		clPrintError(err); return;
	}

	// We allocate the space for the devices.
	cl_device_id *aDevice_ids = (cl_device_id*) malloc(sizeof(cl_device_id) * device_num + 1);

	// Now we get the list of devices.
	err = clGetDeviceIDs(m_PlatformID, CL_DEVICE_TYPE_ALL, device_num, aDevice_ids, NULL);

	if(clIsError(err)){
		clPrintError(err); return;
	}
    
	for(int i = 0; i < device_num; i++) {
		// Create the context.
        // TODO: fix properties
        cl_context context;
        //cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM, platform_id, 0};
        if(m_pPlatformInfo->getAllowsOpenGLSharing()) {
            printf("%s allows OpenGL sharing\n", m_pPlatformInfo->getName());
            cl_context_properties properties[7] = {CL_GL_CONTEXT_KHR,  (cl_context_properties)glXGetCurrentContext(),
                                                   CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
                                                   CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id,
                                                   0};
            context = clCreateContext(properties, 1, &aDevice_ids[i], NULL, NULL, &err);
        } else {
            printf("%s doesn't allow OpenGL sharing\n", m_pPlatformInfo->getName());
            context = clCreateContext(0, 1, &aDevice_ids[i], NULL, NULL, &err);
        }
        
        if(clIsError(err)){
            clPrintError(err); return;
        }
        
        m_vpDevices.push_back(new OpenCLDevice(aDevice_ids[i], context));
        //m_vpDevices.insert(m_vpDevices.begin(), new OpenCLDevice(aDevice_ids[i], context));
	}
}

OpenCLPlatform::~OpenCLPlatform(){

}

std::vector<OpenCLDevice*> OpenCLPlatform::getDeviceList() {
    return m_vpDevices;
}

void OpenCLPlatform::printInfo() {
    getInfo()->printInfo();
    for(int i = 0; i < getNumDevices(); i++) {
        m_vpDevices[i]->printInfo();
    }
}

OpenCLPlatformInfo* OpenCLPlatform::getInfo() {
    return m_pPlatformInfo;
}