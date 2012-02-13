#include "OctreeSegment.h"

OctreeSegment::OctreeSegment(char* data, size_t size)
:	m_size(size),
 	m_pData(data){

}

OctreeSegment::~OctreeSegment() {
	free(m_pData);
}

size_t OctreeSegment::getSize() {
	return m_size;
}

char* OctreeSegment::getData() {
	return m_pData;
}
