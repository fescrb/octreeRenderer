#include "DeviceManager.h"

#include "DataManager.h"
#include "RenderInfo.h"

#include "Device.h"

#ifdef USE_OPENCL
	#include "OpenCLContext.h"
#endif

#ifdef USE_OPENMP
    #include "OpenMPContext.h"
#endif

#ifdef USE_SERIAL
	#include "SerialContext.h"
#endif

#include <cstdio>

DeviceManager::DeviceManager(DataManager *dataManager)
:	m_pDataManager(dataManager){
}

void DeviceManager::detectDevices() {
	#ifdef USE_OPENCL
		m_vContext.push_back(new OpenCLContext());
	#endif //USE_OPENCL
	
    #ifdef USE_OPENMP
        m_vContext.push_back(new OpenMPContext());
    #else
	#if USE_SERIAL
		m_vContext.push_back(new SerialContext());
    #endif //USE_SERIAL
	#endif //USE_OPENMP
	
	printDeviceInfo();
}

DeviceManager::~DeviceManager(){
	while(m_vContext.size()){
		delete m_vContext[m_vContext.size()-1];
		m_vContext.pop_back();
	}
}

void DeviceManager::printDeviceInfo() {
	for(int i = 0; i < m_vContext.size(); i++) {
		m_vContext[i]->printDeviceInfo();
	}
}

int	DeviceManager::getNumDevices(){
	unsigned int count = 0;
	for(int i = 0; i < m_vContext.size(); i++) {
		count += m_vContext[i]->getNumDevices();
	}
	return count;
}

Device* DeviceManager::getDevice(int index) {
	int start = -1;
	for(int i = 0; i < m_vContext.size(); i++) {
		int end = m_vContext[i]->getNumDevices() + start;
		if(start < index && index <= end)
			return m_vContext[i]->getDevice((index-1) - start);
		start = end;
	}

	// Index out of range.
	return 0;
}

//#include "Image.h"

void DeviceManager::distributeHeaderAndOctreeRoot() {
    for(int i = 0; i < getNumDevices(); i++) {
        m_pDataManager->sendHeaderToDevice(getDeviceList()[i]);
        m_pDataManager->sendDataToDevice(getDeviceList()[i]);
    }
}

std::vector<GLuint> DeviceManager::renderFrame(renderinfo *info, int2 resolution) {
	int devices = getNumDevices();
	
	std::vector<GLuint> textures;
	
	std::vector<Device*> device_list = getDeviceList();
	
	for(int i = 0; i < devices; i++) 
		device_list[i]->makeFrameBuffer(resolution);
	
	for(int i = 0; i < devices; i++) 
		device_list[i]->render(int2(),resolution,info);
	
	for(int i = 0; i < devices; i++) 
		textures.push_back(device_list[i]->getFrameBuffer());
    
    for(int i = 0; i < devices; i++) {
        printf("%d %f %f\n", i, (double)device_list[i]->getRenderTime(), (double)device_list[i]->getBufferToTextureTime());
    }
    //exit(0);
		
		//Image image(resolution[0], resolution[1], Image::RGB, thisDevice->getFrame());
		//image.toBMP("frame.bmp");
	
	
	return textures;
}

std::vector<Device*> DeviceManager::getDeviceList() {
	std::vector<Device*> ret;
    for (int i = 0; i < m_vContext.size(); i++) {
        std::vector<Device*> dev = m_vContext[i]->getDeviceList();
        ret.insert(ret.end(), dev.begin(), dev.end());
    }
    return ret;
}
