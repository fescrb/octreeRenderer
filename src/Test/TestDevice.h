#ifndef _TEST_DEVICE_H
#define _TEST_DEVICE_H

#include "Device.h"

class TestDevice
:	public Device {
    public:
		explicit		 TestDevice();
        virtual 		~TestDevice();

        virtual void	 printInfo();
};

#endif // _TEST_DEVICE_H
