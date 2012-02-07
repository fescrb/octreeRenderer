#include "DataManager.h"

#include "RenderInfo.h"
#include "DeviceManager.h"

#include "Octree.h"

DataManager::DataManager() {
	m_pOctree = Octree::getSimpleOctree();
}

DataManager::~DataManager() {
	
}

Octree* DataManager::getOctree() {
	return m_pOctree;
}

char* DataManager::renderFrame(DeviceManager* deviceManager, RenderInfo &info) {

}
