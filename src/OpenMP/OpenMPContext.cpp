#include "OpenMPContext.h"

#include "OpenMPDevice.h"

OpenMPContext::OpenMPContext(){
    m_hostCPU = new OpenMPDevice();
}
