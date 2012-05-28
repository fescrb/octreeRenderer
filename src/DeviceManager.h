#ifndef _DEVICE_MANAGER_H
#define _DEVICE_MANAGER_H

#include "Vector.h"

#include <vector>

#include "Graphics.h"
#include "FramebufferWindow.h"

#include "Rect.h"

class Device;
class Context;
class DataManager;
class renderinfo;

class DeviceManager {
	public:
		explicit 				         DeviceManager(DataManager *dataManager);
		virtual					        ~DeviceManager();

        void                             initialise();

		void					         printDeviceInfo();

		void 					         detectDevices();

		int						         getNumDevices();
		Device					        *getDevice(int index);

		std::vector<Device*>             getDeviceList();

        void                             distributeHeaderAndOctreeRoot();
		std::vector<framebuffer_window>  renderFrame(renderinfo *info, int2 resolution);

	private:
        struct device_tasks {
                                         device_tasks() : tasks() {}
            rect                         total_window;
            std::vector<rect>            tasks;
        };

        void                             setPerDeviceTasks(int2 domain_resolution);


		std::vector<Context*>	         m_vContext;

		DataManager				        *m_pDataManager;

};

#endif //_DEVICE_MANAGER_H
