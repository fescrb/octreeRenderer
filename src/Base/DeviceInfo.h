#ifndef _DEVICE_INFO_H
#define _DEVICE_INFO_H

class DeviceInfo {
	public:
		explicit 			 DeviceInfo();
		virtual 			~DeviceInfo();

		virtual void		 printInfo() = 0;
        
        virtual char        *getName() = 0;
};

#endif //_DEVICE_INFO_H
