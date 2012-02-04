#ifndef _TEST_CONTEXT_H
#define _TEST_CONTEXT_H

class TestDevice;

class TestContext {

	public:
		explicit 		 TestContext();
		void			 printDeviceInfo();	
		
	private:
		TestDevice		*m_hostCPU;

};

#endif //_TEST_CONTEXT
