#include "OctreeHeader.h"

#include <cstdlib>

OctreeHeader::OctreeHeader() {
}
        
void OctreeHeader::setAttributeSize(int size) {
    m_attribute_size = size;
}

int OctreeHeader::getAttributeSize() {
    return m_attribute_size;
}
        
Bin OctreeHeader::flatten() {
    unsigned int size = sizeof(int)*2;
    char* data = (char*) malloc(size);
    
    int* data_int = (int*)data;
    data_int[1] = m_attribute_size;
    
    return Bin(data, size); 
}