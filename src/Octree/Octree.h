#ifndef _OCTREE_H
#define _OCTREE_H

class OctreeNode;

class Octree {
	public:
		explicit 				 Octree();

	private:
		OctreeNode				*m_pRootNode;
};

#endif //_OCTREE_H
