#ifndef _OCTREE_HEADER_H
#define _OCTREE_HEADER_H

#include "Bin.h"

class ConcreteOctree;

class OctreeHeader {
    public:
        explicit    OctreeHeader();
        explicit    OctreeHeader(ConcreteOctree* octree);
        
        void        setAttributeSize(int size);
        int         getAttributeSize();
        
        void        setOctreeDepth(int depth);
        int         getOctreeDepth();
        
        Bin         flatten();
        
    private:
        int         m_octree_depth;
        int         m_attribute_size;
};

#endif //_OCTREE_HEADER_H