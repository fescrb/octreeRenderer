#ifndef _OCTREE_SEGMENT_H
#define _OCTREE_SEGMENT_H

#include <cstdlib>

class OctreeSegment {
	public:
		explicit				 OctreeSegment(char* data, size_t size);
								~OctreeSegment();

		size_t					 getSize(); //@ret size in bytes.
		char					*getData();

	private:
		// ID? Later?
		size_t					 m_size;
		char					*m_pData; // Becomes OWNER of memory.
};

#endif //_OCTREE_SEGMENT_H
