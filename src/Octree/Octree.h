#ifndef _OCTREE_H
#define _OCTREE_H

class OctreeNode;
class OctreeSegment;

class Octree {
	public:
		explicit 				 Octree();
		
		static Octree 			*getSimpleOctree();

		unsigned int			 getDepth();
		unsigned int			 getNumberOfNodes();

		OctreeSegment			*flatten();

	private:
		OctreeNode				*m_pRootNode;
};

#endif //_OCTREE_H
