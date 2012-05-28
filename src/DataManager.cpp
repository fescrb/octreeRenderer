#include "DataManager.h"

#include "ConcreteOctree.h"
#include "OctreeSegment.h"

#include "Device.h"

DataManager::DataManager(Octree* octree) {
	m_pOctree = octree;
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

Bin DataManager::getFullOctree() {
	return getOctree()->getRoot();
}

renderinfo DataManager::getInitialRenderInfo() {
    return getOctree()->getInitialRenderInfo();
}