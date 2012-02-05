#ifndef _TEST_CONTEXT_H
#define _TEST_CONTEXT_H

#include "Context.h"

class TestDevice;

class TestContext
:	public Context {

	public:
		explicit 		 TestContext();
		void			 printDeviceInfo();	
		
	private:
		TestDevice		*m_hostCPU;

};

#endif //_TEST_CONTEXT
