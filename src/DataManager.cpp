#include "DataManager.h"

#include "RenderInfo.h"
#include "DeviceManager.h"

#include "Octree.h"
#include "Device.h"

DataManager::DataManager() {
	m_pOctree = Octree::getSimpleOctree();
}

DataManager::~DataManager() {
	
}

Octree* DataManager::getOctree() {
	return m_pOctree;
}

char* DataManager::renderFrame(DeviceManager* deviceManager, RenderInfo &info) {
	Device* dev = deviceManager->getDevice(0);

	char* buffer = getOctree()->flatten();
	dev->sendData(buffer);

	dev->render(info);

	return dev->getFrame();
}
