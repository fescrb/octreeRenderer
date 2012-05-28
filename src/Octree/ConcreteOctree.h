#ifndef _CONCRETE_OCTREE_H
#define _CONCRETE_OCTREE_H

#include "Octree.h"

class OctreeHeader;
class OctreeNode;
class OctreeSegment;

#include "RenderInfo.h"
#include "Bin.h"

class ConcreteOctree
:   public Octree {
    public:
        explicit                 ConcreteOctree();
        
        static Octree           *getSimpleOctree();
        
        void                     setInitialRenderInfo(renderinfo info);

        unsigned int             getDepth();
        unsigned int             getAttributeSize();
        unsigned int             getNumberOfNodes();
        
        
        Bin                      getHeader();
        Bin                      getRoot();
        Bin                      flatten();

    protected:
        OctreeHeader            *m_pHeader;
        OctreeNode              *m_pRootNode;
};

#endif //_CONCRETE_OCTREE_H