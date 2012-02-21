#ifndef _DEVICE_MANAGER_H
#define _DEVICE_MANAGER_H

#include "Vector.h"

#include <vector>

#ifdef _LINUX
	#include <GL/gl.h>
#endif //_LINUX

#ifdef _OSX
	#include <OpenGL/gl.h>
#endif //_OSX

class Device;
class Context;
class DataManager;
class renderinfo;

class DeviceManager {
	public:
		explicit 				 DeviceManager(DataManager *dataManager);
		virtual					~DeviceManager();

		void					 printDeviceInfo();
		
		void 					 detectDevices();

		int						 getNumDevices();
		Device					*getDevice(int index);
		
		std::vector<GLuint>		 renderFrame(renderinfo *info, int2 resolution);
		
	private:
		std::vector<Context*>	 m_vContext;
		
		DataManager				*m_pDataManager;

};

#endif //_DEVICE_MANAGER_H
