#ifndef _SERIAL_CONTEXT_H
#define _SERIAL_CONTEXT_H

#include "Context.h"

class SerialDevice;

class SerialContext
:	public Context {

	public:
		explicit 				 SerialContext();
		void					 printDeviceInfo();	
		
		unsigned int			 getNumDevices();
		Device					*getDevice(int index);
		
		std::vector<Device*>	 getDeviceList();

	private:
		SerialDevice				*m_hostCPU;

};

#endif //_SERIAL_CONTEXT_H
