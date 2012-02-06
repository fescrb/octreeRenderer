#ifndef _DATA_MANAGER_H
#define _DATA_MANAGER_H

class Octree;

class DataManager {
	public:
		explicit 				 DataManager();
								~DataManager();

		Octree 				 	*getOctree();
		
	private:
	
		Octree					*m_pOctree;
};

#endif //_DATA_MANAGER_H
