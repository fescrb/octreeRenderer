#ifndef _TEST_DEVICE_H
#define _TEST_DEVICE_H

#include "TestDeviceInfo.h"

class TestDevice
{
    public:
		explicit		 TestDevice();
        virtual 		~TestDevice();

        virtual void	 printInfo();
    private:
		TestDeviceInfo   m_info;
};

#endif // _TEST_DEVICE_H
