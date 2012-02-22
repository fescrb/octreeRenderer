#include "OpenCLProgram.h"

#include "OpenCLDevice.h"
#include "OpenCLUtils.h"

#include "SourceFileManager.h"
#include "SourceFile.h"

OpenCLProgram::OpenCLProgram(OpenCLDevice* device, const char* sourceFilename)
:	m_pDevice(device) {
	const char* source = SourceFileManager::getSource(sourceFilename)->getSource();
    
	int err;
    m_program = clCreateProgramWithSource(device->getOpenCLContext(), 1, &source, NULL, &err);
	
	if(clIsError(err)){
        clPrintError(err); return;
    }
    
    char *options = (char*) malloc (512);
    
    sprintf(options, " -D _OCL -I %s ", SourceFileManager::getDefaultInstance()->getShaderLocation());
    
	cl_device_id device_id = device->getOpenCLDeviceID();
	
    err = clBuildProgram( m_program, 1, &device_id, options, NULL, NULL);
    
    if(clIsError(err) && !clIsBuildError(err)){
        clPrintError(err); return;
    }
    
    char log[1024];
    cl_build_status build_status;
    err = clGetProgramBuildInfo( m_program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
    
    err = clGetProgramBuildInfo( m_program, device_id, CL_PROGRAM_BUILD_LOG, 1024, log, NULL);
    printf("Device %s Build:\nStatus: %s\nLog:\n%s\n", clProgramBuildStatusToCString(build_status),device->getName(), log);
}

cl_kernel OpenCLProgram::getOpenCLKernel(const char* kernel_name) {
	int err;
	cl_kernel kernel = clCreateKernel( m_program, kernel_name, &err);
    
    if(clIsError(err)) {
        clPrintError(err);
    }
    
    return kernel;
}