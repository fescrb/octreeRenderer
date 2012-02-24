#ifndef _TEST_CONTEXT_H
#define _TEST_CONTEXT_H

#include "Context.h"

class TestDevice;

class TestContext
:	public Context {

	public:
		explicit 				 TestContext();
		void					 printDeviceInfo();	
		
		unsigned int			 getNumDevices();
		Device					*getDevice(int index);
		
		std::vector<Device*>	 getDeviceList();

	private:
		TestDevice				*m_hostCPU;

};

#endif //_TEST_CONTEXT
