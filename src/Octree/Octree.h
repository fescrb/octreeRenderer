#ifndef _OCTREE_H
#define _OCTREE_H

class OctreeNode;

class Octree {
	public:
		explicit 				 Octree();
		
		static Octree 			*getSimpleOctree();

		unsigned int			 getDepth();
		unsigned int			 getNumberOfNodes();

		char* 					 flatten();

	private:
		OctreeNode				*m_pRootNode;
};

#endif //_OCTREE_H
