#ifndef _DATA_MANAGER_H
#define _DATA_MANAGER_H

class Octree;
class DeviceManager;

#include "RenderInfo.h"

class DataManager {
	public:
		explicit 				 DataManager();
								~DataManager();

		Octree 				 	*getOctree();
		
		char					*renderFrame(DeviceManager* deviceManager, RenderInfo &info);

	private:
	
		Octree					*m_pOctree;
};

#endif //_DATA_MANAGER_H
