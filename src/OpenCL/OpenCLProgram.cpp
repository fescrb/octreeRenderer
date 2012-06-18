#include "OpenCLProgram.h"

#include "OpenCLDevice.h"
#include "OpenCLUtils.h"

#include "SourceFileManager.h"
#include "SourceFile.h"

OpenCLProgram::OpenCLProgram(OpenCLDevice* device, const char* sourceFilename)
:	m_pDevice(device) {
	SourceFile *sourceFile = SourceFileManager::getSource(sourceFilename);
	const char** source = sourceFile->getSource();
    
	int err;
	std::vector<size_t> lineLengths = sourceFile->getLineLength(); 
    m_program = clCreateProgramWithSource(device->getOpenCLContext(), sourceFile->getNumLines(), source, &lineLengths[0], &err);
	
	if(clIsError(err)){
        clPrintError(err); return;
    }
    
    char *options = (char*) malloc (512);
    
    sprintf(options, " -cl-fast-relaxed-math -D _OCL -I %s ", SourceFileManager::getDefaultInstance()->getShaderLocation());
    
	cl_device_id device_id = device->getOpenCLDeviceID();
	
    err = clBuildProgram( m_program, 1, &device_id, options, NULL, NULL);
    
    if(clIsError(err) && !clIsBuildError(err)){
        clPrintError(err); return;
    }
    
    cl_build_status build_status;
    err = clGetProgramBuildInfo( m_program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
    
	size_t size;
	err = clGetProgramBuildInfo( m_program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
	char *log = (char*) malloc (size+1);
    err = clGetProgramBuildInfo( m_program, device_id, CL_PROGRAM_BUILD_LOG, size, log, NULL);
    printf("Device %s Build:\nStatus: %s\nLog:\n%s\n", device->getName(), clProgramBuildStatusToCString(build_status), log);
	
	/* Not available in ocl 1.1...
	 * err = clGetProgramInfo(	m_program, CL_PROGRAM_NUM_KERNELS, 1, &m_numKernels, NULL);

	if(!m_numKernels) {
		printf("Error! Program %s has no kernels.\n", sourceFilename);
	}*/
}

cl_kernel OpenCLProgram::getOpenCLKernel(const char* kernel_name) {
	int err;
	cl_kernel kernel = clCreateKernel( m_program, kernel_name, &err);
    
    if(clIsError(err)) {
        clPrintError(err);
    }
    
    return kernel;
}