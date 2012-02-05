#include "DeviceManager.h"

#ifdef USE_OPENCL
	#include "OpenCLContext.h"
#endif

#ifdef USE_HOST_CPU
	#include "TestContext.h"
#endif

DeviceManager::DeviceManager(){
	#ifdef USE_OPENCL
		m_vContext.push_back(new OpenCLContext());
	#endif
	
	#ifdef USE_HOST_CPU
		m_vContext.push_back(new TestContext());
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
