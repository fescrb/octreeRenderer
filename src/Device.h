#ifndef _DEVICE_H
#define _DEVICE_H

class DeviceInfo;

class Device {
    public:
		explicit		 Device();
        virtual 		~Device();

        virtual void	 printInfo() = 0;
    protected:

        DeviceInfo		*m_pDeviceInfo;
};

#endif // _DEVICE_H
