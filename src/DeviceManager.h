#ifndef _DEVICE_MANAGER_H
#define _DEVICE_MANAGER_H

#include "Device.h"

#include "Rect.h"
#include "Vector.h"

#include <vector>

#include "Graphics.h"
#include "FramebufferWindow.h"

class Device;
class Context;
class DataManager;
class renderinfo;

class DeviceManager {
	public:
		explicit 				         DeviceManager(DataManager *dataManager, int2 resolution);
		virtual					        ~DeviceManager();

        void                             initialise();

		void					         printDeviceInfo();

		void 					         detectDevices();

		int						         getNumDevices();
		Device					        *getDevice(int index);

		std::vector<Device*>             getDeviceList();

        void                             setRenderMode(Device::RenderMode mode);
        
        void                             distributeHeaderAndOctreeRoot();
		std::vector<framebuffer_window>  renderFrame(renderinfo *info, int2 resolution);

	private:        
        struct device_characteristics {
            float                        it_per_second;
        };
        
        struct division_window {
            rect                         window;
            unsigned int                 cost;
        };

        void                             setPerDeviceTasks(int2 domain_resolution);
        void                             getFrameTimeResults(int2 domain_resolution);

		std::vector<Context*>	         m_vContext;
        std::vector<Device*>             m_vDeviceList;
        std::vector<device_characteristics>
                                         m_vDeviceCharacteristics;

        division_window                 *m_division_windows;
        int                              m_division_window_count;

        void                             createDivisionWindows(int2 domain_resolution);
        
		DataManager				        *m_pDataManager;

};

#endif //_DEVICE_MANAGER_H
