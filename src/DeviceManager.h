#ifndef _DEVICE_MANAGER_H
#define _DEVICE_MANAGER_H

#include "Vector.h"

#include <vector>

#include "Graphics.h"

class Device;
class Context;
class DataManager;
class renderinfo;

class DeviceManager {
	public:
		explicit 				 DeviceManager(DataManager *dataManager);
		virtual					~DeviceManager();
        
        void                     initialise();

		void					 printDeviceInfo();
		
		void 					 detectDevices();

		int						 getNumDevices();
		Device					*getDevice(int index);
		
		std::vector<Device*>     getDeviceList();
		
        void                     distributeHeaderAndOctreeRoot();
		std::vector<GLuint>		 renderFrame(renderinfo *info, int2 resolution);     
		
	private:
         struct device_tasks {
            int2                 total_window;
            std::vector<int2>    tasks;
        };
        
        device_tasks            *getPerDeviceTasks(int2 domain_resolution);
        
        
		std::vector<Context*>	 m_vContext;
		
		DataManager				*m_pDataManager;

};

#endif //_DEVICE_MANAGER_H
