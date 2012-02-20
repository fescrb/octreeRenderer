#include "DeviceManager.h"

#include "DataManager.h"
#include "RenderInfo.h"

#include "Device.h"

#ifdef USE_OPENCL
	#include "OpenCLContext.h"
#endif

#ifdef USE_HOST_CPU
	#include "TestContext.h"
#endif

DeviceManager::DeviceManager(DataManager *dataManager)
:	m_pDataManager(dataManager){
}

void DeviceManager::detectDevices() {
	#ifdef USE_OPENCL
		m_vContext.push_back(new OpenCLContext());
	#endif
	
	#ifdef USE_HOST_CPU
		m_vContext.push_back(new TestContext());
	#endif
	
	printDeviceInfo();
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

int	DeviceManager::getNumDevices(){
	unsigned int count = 0;
	for(int i = 0; i < m_vContext.size(); i++) {
		count += m_vContext[i]->getNumDevices();
	}
	return count;
}

Device* DeviceManager::getDevice(int index) {
	int start = -1;
	for(int i = 0; i < m_vContext.size(); i++) {
		int end = m_vContext[i]->getNumDevices() + start;
		if(start < index && index <= end)
			return m_vContext[i]->getDevice((index-1) - start);
		start = end;
	}

	// Index out of range.
	return 0;
}

std::vector<GLuint>	DeviceManager::renderFrame(RenderInfo *info, int2 resolution) {
	int devices = getNumDevices();
	
	std::vector<GLuint> textures;
	
	OctreeSegment* fullOctree = m_pDataManager->getFullOctree();
    
    info->maxOctreeDepth = m_pDataManager->getMaxOctreeDepth();
	
	for(int i = 0; i < devices; i++) {
		Device *thisDevice = getDevice(i);
		
		thisDevice->makeFrameBuffer(resolution);
		thisDevice->sendData(fullOctree);
		thisDevice->render(int2(),resolution,info);
		textures.push_back(thisDevice->getFrameBuffer());
	}
	
	return textures;
}
