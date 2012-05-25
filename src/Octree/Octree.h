#ifndef _OCTREE_H
#define _OCTREE_H

class OctreeHeader;
class OctreeNode;
class OctreeSegment;

#include "RenderInfo.h"
#include "Bin.h"

class Octree {
	public:
		explicit 				 Octree();
		
		static Octree 			*getSimpleOctree();

		unsigned int			 getDepth();
        unsigned int             getAttributeSize();
		unsigned int			 getNumberOfNodes();
        
        renderinfo               getInitialRenderInfo();
        
        
        Bin                      getHeader();
		Bin			             flatten();

	private:
        OctreeHeader            *m_pHeader;
		OctreeNode				*m_pRootNode;
};

#endif //_OCTREE_H
