#ifndef _OCTREE_NODE_H
#define _OCTREE_NODE_H

class OctreeNode {

	public:
		explicit 				 OctreeNode();


		void					 addChild(OctreeNode* node, unsigned int position_flag);


	private:
		OctreeNode				*m_vChildren[8];


};

#endif //_OCTREE_NODE_H
