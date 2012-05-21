#include "DataManager.h"

#include "Octree.h"
#include "OctreeSegment.h"

#include "Device.h"

DataManager::DataManager() {
	m_pOctree = Octree::getSimpleOctree();
}

DataManager::~DataManager() {
	
}

void DataManager::sendDataToDevice(Device* device) {
    device->sendData(getFullOctree());
}

Octree* DataManager::getOctree() {
	return m_pOctree;
}

int DataManager::getMaxOctreeDepth() {
	return getOctree()->getDepth();
}

Bin DataManager::getFullOctree() {
	return getOctree()->flatten();
}
