#ifndef _CONTEXT_H
#define _CONTEXT_H

class Device;

class Context {
	public:
		virtual void			 printDeviceInfo() = 0;

		virtual unsigned int	 getNumDevices() = 0;
		virtual Device			*getDevice(int index) = 0;
};

#endif //_CONTEXT
