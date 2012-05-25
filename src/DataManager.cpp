#include "DataManager.h"

#include "Octree.h"
#include "OctreeSegment.h"

#include "Device.h"

DataManager::DataManager() {
	m_pOctree = Octree::getSimpleOctree();
}

DataManager::~DataManager() {
	
}

void DataManager::sendHeaderToDevice(Device* device) {
    Bin header = getOctree()->getHeader();
    device->sendHeader(header);
}

void DataManager::sendDataToDevice(Device* device) {
    Bin octree = getFullOctree();
    device->sendData(octree);
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

renderinfo DataManager::getInitialRenderInfo() {
    return getOctree()->getInitialRenderInfo();
}