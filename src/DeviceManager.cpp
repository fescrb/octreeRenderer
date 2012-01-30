#include "DeviceManager.h"

#ifdef USE_OPENCL
	#include "OpenCL/OpenCLContext.h"
#endif

DeviceManager::DeviceManager(){
	#ifdef USE_OPENCL
		m_vContext.push_back(new OpenCLContext());
	#endif
}

DeviceManager::~DeviceManager(){
	while(m_vContext.size()){
		delete m_vContext[m_vContext.size()-1];
		m_vContext.pop_back();
	}
}

void DeviceManager::printDeviceInfo() {
	for(int i = 0; i < m_vContext.size(); i++) {
		m_vContext[i]->printDeviceInfo();
	}
}
