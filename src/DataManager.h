#ifndef _DATA_MANAGER_H
#define _DATA_MANAGER_H

class Octree;
class OctreeSegment;

class Device;

#include "RenderInfo.h"
#include "Bin.h"

class DataManager {
	public:
		explicit 				 DataManager();
								~DataManager();

		Octree 				 	*getOctree();
		
		int 					 getMaxOctreeDepth();
        
        void                     sendHeaderToDevice(Device* device);
        void                     sendDataToDevice(Device* device);
        
		
		Bin			             getFullOctree();
        
        renderinfo               getInitialRenderInfo();

	private:
	
		Octree					*m_pOctree;
};

#endif //_DATA_MANAGER_H
