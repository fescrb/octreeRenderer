#include "OpenCLDeviceInfo.h"

#include "OpenCLUtils.h"

#include <cstdio>
#include <cstdlib>

OpenCLDeviceInfo::OpenCLDeviceInfo(cl_device_id device, cl_context context){
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

	if(clIsError(err)){
		clPrintError(err); return;
	}
	m_maxComputeUnits = maxComputeUnits;

	// Find the frequency of the compute units. Same as before, we needn't query the size.
	cl_uint maxClockFrequency;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &maxClockFrequency, NULL);

	if(clIsError(err)){
		clPrintError(err); return;
	}
	m_maxComputeUnitFrequency = maxClockFrequency;
    
    size_t maxWorkGroupSize;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);

    if(clIsError(err)){
        clPrintError(err); return;
    }
    m_maxWorkGroupSize = maxWorkGroupSize;
    
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &m_workSizesDimensions);
    if(clIsError(err)){
        clPrintError(err); return;
    }
    m_maxWorkGroupSizes = (size_t*)malloc(m_workSizesDimensions);
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, m_workSizesDimensions, m_maxWorkGroupSizes, NULL);
    m_workSizesDimensions/=sizeof(size_t);
    if(clIsError(err)){
        clPrintError(err); return;
    }

	// We now query the device type. No need to check size.
	err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &m_deviceType, NULL);

	if(clIsError(err)){
		clPrintError(err); return;
	}

	cl_ulong globalMemSize;
	err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMemSize, NULL);

	if(clIsError(err)){
		clPrintError(err); return;
	}
	m_globalMemorySize = globalMemSize;
    
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &m_localMemorySize, NULL);

    if(clIsError(err)){
        clPrintError(err); return;
    }
    
    err = clGetSupportedImageFormats(context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 0, NULL, &m_format_count);
    if(clIsError(err)){
        clPrintError(err); return;
    }
    m_image_formats = (cl_image_format*) malloc (sizeof(cl_image_format)*m_format_count + 1);
    err = clGetSupportedImageFormats(context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 10, m_image_formats, NULL);
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
    printf("Maximum Work Group Size:        %d\n", m_maxWorkGroupSize);
    printf("Maximum Work Group Sizes:       %d", m_maxWorkGroupSizes[0]);
    for(int i = 1; i < m_workSizesDimensions; i++) {
        printf(", %d", m_maxWorkGroupSizes[i]);
    }
    printf("\n");
	printf("Global Memory Size:             %lu bytes\n", m_globalMemorySize);
    printf("Local Memory Size:              %lu bytes\n", m_localMemorySize);
    printf("2D Image formats allowed:       ");
    for(int i = 0; i < m_format_count; i++) {
        printf("%s,%s\n                                ", 
               clGetChannelOrderString(m_image_formats[i].image_channel_order),
               clGetImageChannelTypeString(m_image_formats[i].image_channel_data_type));
    }
    printf("\n");
}

char* OpenCLDeviceInfo::getName() {
    return m_sDeviceName;
}
