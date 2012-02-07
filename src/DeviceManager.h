#ifndef _DEVICE_MANAGER_H
#define _DEVICE_MANAGER_H

#include <vector>

class Device;
class Context;

class DeviceManager {
	public:
		explicit 				 DeviceManager();
		virtual					~DeviceManager();

		void					 printDeviceInfo();

		int						 getNumDevices();
		Device					*getDevice(int index);
	private:
		std::vector<Context*>	 m_vContext;

};

#endif //_DEVICE_MANAGER_H
