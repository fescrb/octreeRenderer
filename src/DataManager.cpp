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

int DataManager::getMaxOctreeDepth() {
	return getOctree()->getDepth();
}

char* DataManager::renderFrame(DeviceManager* deviceManager, RenderInfo &info) {
	Device* dev = deviceManager->getDevice(0);

	OctreeSegment* segment = getOctree()->flatten();
	dev->sendData(segment);

	dev->render(float2(0,0), float2(32,32), info);

	return dev->getFrame();
}
