#ifndef _TEST_DEVICE_INFO_H
#define _TEST_DEVICE_INFO_H

#include "DeviceInfo.h"

class TestDeviceInfo
:	public DeviceInfo {
	public:
		explicit 			 TestDeviceInfo();
							~TestDeviceInfo();
		
		void				 printInfo();
        
        char                *getName();
};

#endif //_TEST_DEVICE_INFO_H
