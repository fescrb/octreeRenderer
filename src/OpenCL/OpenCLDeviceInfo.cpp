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

	if(clIsError(err)){
		clPrintError(err); return;
	}

	// As with device name, we need to get size of both vendor string and version string.
	err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, NULL, &stringSize);
	if(clIsError(err)){
		clPrintError(err); return;
	}

	m_sDeviceVendorString = (char*) malloc(stringSize+1);
	err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, stringSize, m_sDeviceVendorString, NULL);
	if(clIsError(err)){
		clPrintError(err); return;
	}

	// Now device version.
	err = clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &stringSize);
	if(clIsError(err)){
		clPrintError(err); return;
	}

	m_sOpenCLVersionString = (char*) malloc(stringSize+1);
	err = clGetDeviceInfo(device, CL_DEVICE_VERSION, stringSize, m_sOpenCLVersionString, NULL);
	if(clIsError(err)){
		clPrintError(err); return;
	}

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

	// We now query the device type. No need to check size.
	err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &m_deviceType, NULL);

	if(clIsError(err)){
		clPrintError(err); return;
	}
}

OpenCLDeviceInfo::~OpenCLDeviceInfo(){

}

void OpenCLDeviceInfo::printInfo(){
	printf("\n");
	printf("Device Name:                    %s\n", m_sDeviceName);
	printf("Vendor Name:                    %s\n", m_sDeviceVendorString);
	printf("OpenCL Version:                 %s\n", m_sOpenCLVersionString);
	printf("Device Type:                    %s\n", clDeviceTypeToCString(m_deviceType));
	printf("Maximum Compute Units:          %d\n", m_maxComputeUnits);
	printf("Maximum Compute Unit Frequency: %d\n", m_maxComputeUnitFrequency);
}
