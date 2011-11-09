#include "OpenCLDeviceInfo.h"

#include "OpenCLUtils.h"

#include <stdio.h>

OpenCLDeviceInfo::OpenCLDeviceInfo(cl_device_id device){
	// To get the device name, first we get the length.
	size_t stringSize;
	cl_int err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &stringSize);

	// If error, we print to stdout and leave.
	if(clIsError(err)){
		clPrintError(err); return;
	}

	// We now allocate memory for it and get it.
	m_sDeviceName = (char*) malloc(stringSize+1);
	err = clGetDeviceInfo(device, CL_DEVICE_NAME, stringSize, m_sDeviceName, NULL);

	// Find the number of compute units. We know the size of an integer, so we don't ask for it.
	cl_uint maxComputeUnits;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);
	m_maxComputeUnits = maxComputeUnits;

	if(clIsError(err)){
		clPrintError(err); return;
	}

	// Find the frequency of the compute units. Same as before, we needn't query the size.
	cl_uint maxClockFrequency;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &maxClockFrequency, NULL);
	m_maxComputeUnitFrequency = maxClockFrequency;

	if(clIsError(err)){
		clPrintError(err); return;
	}
}

OpenCLDeviceInfo::~OpenCLDeviceInfo(){

}

void OpenCLDeviceInfo::printInfo(){
	printf("\nDevice Name:                    %s\n", m_sDeviceName);
	printf("\nMaximum Compute Units:          %d\n", m_maxComputeUnits);
	printf("\nMaximum Compute Unit Frequency: %d\n", m_maxComputeUnitFrequency);
}
