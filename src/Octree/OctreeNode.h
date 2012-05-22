#ifndef _OCTREE_NODE_H
#define _OCTREE_NODE_H

#include "Attributes.h"

class OctreeNode {

	public:
		explicit 				 OctreeNode();
		explicit 				 OctreeNode(Attributes att);

		enum PositionFlags {
			X = 1,
			Y = 2,
			Z = 4
		};

		void					 addChild(OctreeNode* node, unsigned int position_flag);
		void					 cleanChildrenPointers();

		void					 setAttributes(Attributes att);
        Attributes               getAttributes();

		char 					*flatten(char* buffer);

		unsigned int			 getDepth();
		unsigned int			 getNumberOfNodes();

	private:
		OctreeNode				*m_vChildren[8];
		unsigned int			 numberOfChildren;

		Attributes 				 m_attributes;
};

#endif //_OCTREE_NODE_H
