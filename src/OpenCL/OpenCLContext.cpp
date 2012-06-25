#include "OpenCLContext.h"
#include "OpenCLUtils.h"

// To print info, might need to be removed later.
#include "OpenCLPlatform.h"
#include "OpenCLDevice.h"

#ifdef _LINUX
#include "OpenCLPlatformInfo.h"
#include <string>
#endif

OpenCLContext::OpenCLContext() {
	cl_uint num_of_platforms = 0;

	// We get the number of platforms.
	cl_int err = clGetPlatformIDs(0, NULL, &num_of_platforms);

	// If error, we print to stdout and leave.
	if(clIsError(err)){
		clPrintError(err); return;
	}

	// We allocate memory for the list of platforms and retrieve it.
	cl_platform_id *platform_ids = (cl_platform_id*) malloc(sizeof(cl_platform_id)*num_of_platforms + 1);

	err = clGetPlatformIDs(num_of_platforms, platform_ids, NULL);

	if(clIsError(err)){
		clPrintError(err); return;
	}

	// We initialize the platforms.
	for(int i = 0; i < num_of_platforms; i++) {
        m_vpPlatforms.push_back(new OpenCLPlatform(platform_ids[i]));
        
        /*#ifdef _LINUX
            std::string name(m_vpPlatforms[m_vpPlatforms.size()-1]->getInfo()->getName());
            if(name.find("Intel")!=name.npos) {
                delete m_vpPlatforms[i];
                m_vpPlatforms.pop_back();
            }
        #endif*/
        
        std::string name(m_vpPlatforms[m_vpPlatforms.size()-1]->getInfo()->getName());
        if(name.find("NVIDIA")!=name.npos) {
            delete m_vpPlatforms[i];
            m_vpPlatforms.pop_back();
        }
	}
}

OpenCLContext::~OpenCLContext() {

}

void OpenCLContext::printDeviceInfo(){
	for(int i = 0; i < getNumPlatforms(); i++) {
		m_vpPlatforms[i]->printInfo();
	}
}

unsigned int OpenCLContext::getNumDevices() {
	unsigned int count = 0;
	for(int i = 0; i < getNumPlatforms(); i++) {
		count += m_vpPlatforms[i]->getNumDevices();
	}
	return count;
}

Device* OpenCLContext::getDevice(int index) {
    return getDeviceList()[index];
}

unsigned int OpenCLContext::getNumPlatforms() {
    return m_vpPlatforms.size();
}

std::vector<Device*> OpenCLContext::getDeviceList() {
    std::vector<Device*> ret;
    for (int i = 0; i < m_vpPlatforms.size(); i++) {
        std::vector<OpenCLDevice*> dev = m_vpPlatforms[i]->getDeviceList();
		for(int j = 0; j < dev.size(); j++) {
			ret.push_back(dev[j]);
		}
    }
    return ret;
}
