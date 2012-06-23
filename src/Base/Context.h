#ifndef _CONTEXT_H
#define _CONTEXT_H

class Device;

#include <vector>

class Context {
	public:
        virtual                         ~Context() {}
        
		virtual void					 printDeviceInfo() = 0;

		virtual unsigned int	 		 getNumDevices() = 0;
		virtual Device					*getDevice(int index) = 0;
		
		virtual std::vector<Device*>	 getDeviceList() = 0;
};

#endif //_CONTEXT
