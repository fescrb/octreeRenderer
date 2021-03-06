#ifndef _OCTREE_H
#define _OCTREE_H

#include "SizeMacros.h"

class OctreeHeader;
class OctreeNode;
class OctreeSegment;

#include "RenderInfo.h"
#include "Bin.h"

class Octree {
    public:
        explicit 				 Octree();
        
        virtual renderinfo       getInitialRenderInfo();
        
        
        virtual Bin              getHeader() = 0;
        virtual Bin              getRoot() = 0;

    protected:
        renderinfo               m_initial_renderinfo;
};

#endif //_OCTREE_H
