#include "Device.h"

#include "DeviceInfo.h"

Device::Device()
{
    //ctor
}

Device::~Device()
{
    //dtor
}

char* Device::getName() {
	return m_pDeviceInfo->getName();
}
