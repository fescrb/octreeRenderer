#include "OctreeHeader.h"

#include "ConcreteOctree.h"

#include <cstdlib>

OctreeHeader::OctreeHeader() {
}

OctreeHeader::OctreeHeader(ConcreteOctree* octree) {
    setAttributeSize(octree->getAttributeSize());
    setOctreeDepth(octree->getDepth());
}
        
void OctreeHeader::setAttributeSize(int size) {
    m_attribute_size = size;
}

int OctreeHeader::getAttributeSize() {
    return m_attribute_size;
}

void OctreeHeader::setOctreeDepth(int depth) {
    m_octree_depth = depth;
}

int OctreeHeader::getOctreeDepth() {
    return m_octree_depth;
}
        
Bin OctreeHeader::flatten() {
    unsigned int size = sizeof(int)*2;
    char* data = (char*) malloc(size);
    
    int* data_int = (int*)data;
    data_int[0] = m_octree_depth;
    data_int[1] = m_attribute_size;
    
    return Bin(data, size); 
}