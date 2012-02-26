#ifndef _SERIAL_DEVICE_INFO_H
#define _SERIAL_DEVICE_INFO_H

#include "DeviceInfo.h"

class SerialDeviceInfo
:	public DeviceInfo {
	public:
		explicit 			 SerialDeviceInfo();
							~SerialDeviceInfo();
		
		void				 printInfo();
        
        char                *getName();
};

#endif //_SERIAL_DEVICE_INFO_H
