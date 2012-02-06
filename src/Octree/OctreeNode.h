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

		void					 setAttributes(Attributes att);

	private:
		OctreeNode				*m_vChildren[8];

		Attributes 				 m_attributes;
};

#endif //_OCTREE_NODE_H
