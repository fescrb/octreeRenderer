#ifndef _OCTREE_HEADER_H
#define _OCTREE_HEADER_H

#include "Bin.h"

class OctreeHeader {
    public:
        explicit    OctreeHeader();
        
        void        setAttributeSize(int size);
        
        int         getAttributeSize();
        
        Bin         flatten();
        
    private:
        int         m_attribute_size;
};

#endif //_OCTREE_HEADER_H