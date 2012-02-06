#include "DataManager.h"

#include "Octree.h"

DataManager::DataManager() {
	m_pOctree = Octree::getSimpleOctree();
}

DataManager::~DataManager() {
	
}

Octree* DataManager::getOctree() {
	return m_pOctree;
}
