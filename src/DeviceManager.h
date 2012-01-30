#ifndef _DEVICE_MANAGER_H
#define _DEVICE_MANAGER_H

#include "Context.h"

#include <vector>

class DeviceManager {
	public:
		explicit 				 DeviceManager();
		virtual					~DeviceManager();

		void					 printDeviceInfo();

	private:
		std::vector<Context*>	 m_vContext;

};

#endif //_DEVICE_MANAGER_H
