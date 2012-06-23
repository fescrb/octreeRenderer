#ifndef _OCTREE_NODE_H
#define _OCTREE_NODE_H

#include "Vector.h"

using namespace vector;

class OctreeNode {
	public:
        virtual OctreeNode      *getChildAt(int index) = 0;
        
		virtual char            *flatten(char* buffer, int depth) = 0;

		virtual unsigned int     getDepth() = 0;
		virtual unsigned int     getNumberOfNodes() = 0;
		virtual unsigned int     getAttributeSize() = 0;
        
        virtual float4           getColour() = 0;
        virtual float4           getNormal() = 0;
};

#endif //_OCTREE_NODE_H
