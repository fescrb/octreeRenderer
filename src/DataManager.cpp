#include "DataManager.h"

#include "Octree.h"
#include "OctreeSegment.h"

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

OctreeSegment* DataManager::getFullOctree() {
	return getOctree()->flatten();
}
