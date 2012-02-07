#ifndef _DEVICE_H
#define _DEVICE_H

#include "RenderInfo.h"

class DeviceInfo;

class Device {
    public:
		explicit		 Device();
        virtual 		~Device();

        virtual void	 printInfo() = 0;

        virtual void 	 sendData(char* data) = 0;
        virtual void	 render(RenderInfo &info) = 0;
        virtual char    *getFrame() = 0;
    protected:

        DeviceInfo		*m_pDeviceInfo;
};

#endif // _DEVICE_H
