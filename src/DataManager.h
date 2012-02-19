#ifndef _DATA_MANAGER_H
#define _DATA_MANAGER_H

class Octree;
class OctreeSegment;

class DataManager {
	public:
		explicit 				 DataManager();
								~DataManager();

		Octree 				 	*getOctree();
		
		int 					 getMaxOctreeDepth();
		
		OctreeSegment			*getFullOctree();

	private:
	
		Octree					*m_pOctree;
};

#endif //_DATA_MANAGER_H
