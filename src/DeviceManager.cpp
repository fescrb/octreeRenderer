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
    
    m_vDeviceList = getDeviceList();
}

DeviceManager::~DeviceManager(){
	while(m_vContext.size()){
		delete m_vContext[m_vContext.size()-1];
		m_vContext.pop_back();
	}
}

void DeviceManager::initialise() {
    detectDevices();
    distributeHeaderAndOctreeRoot();
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

void DeviceManager::setPerDeviceTasks(int2 domain_resolution) {
    int device_count = getNumDevices();
    if(m_vDeviceCharacteristics.size() != device_count) {
        for (int i = 0; i < device_count; i++) {
            getDevice(i)->clearTasks();
        }

        int2 origin = int2(0,0);
        int2 size = int2(domain_resolution.getX(),domain_resolution[1]);
        m_vDeviceList[m_vDeviceCharacteristics.size()]->addTask(rect(origin,size));
    } else {

        int start = 0;
        
        float total_pps = 0.0f;
        for (int i = 0; i < device_count; i++) {
            total_pps+=m_vDeviceCharacteristics[i].pixels_per_second;
        }

        for (int i = 0; i < device_count; i++) {
            getDevice(i)->clearTasks();

            int2 origin = int2(start, 0);
            int2 size = int2(domain_resolution.getX()*m_vDeviceCharacteristics[i].pixels_per_second/total_pps,domain_resolution[1]);

            printf("Device %d pps %f total_pps %f origin %d %d size %d %d\n", i, m_vDeviceCharacteristics[i].pixels_per_second, total_pps,
                                                                            origin.getX(), origin.getY(), size.getX(), size.getY()
            );
            
            //if(!i) {
            //    size.setX(size.getX()+(domain_resolution.getX()%device_count));
            //}

            start+=size.getX();

            rect window = rect(origin, size);

            getDevice(i)->addTask(window);
        }
    }
}

void DeviceManager::getFrameTimeResults(int2 domain_resolution) {
    if(m_vDeviceCharacteristics.size() != m_vDeviceList.size()) {
        float total_pixels = domain_resolution.getX() * domain_resolution.getY();
        device_characteristics dev_c;
        dev_c.pixels_per_second = total_pixels/((double)m_vDeviceList[m_vDeviceCharacteristics.size()]->getRenderTime());
        
        
        printf("New Device pps %f render time %f transfer time %f\n", dev_c.pixels_per_second, 
                                                                      ((double)m_vDeviceList[m_vDeviceCharacteristics.size()]->getRenderTime()), 
                                                                      ((double)m_vDeviceList[m_vDeviceCharacteristics.size()]->getBufferToTextureTime()));
        m_vDeviceCharacteristics.push_back(dev_c);
    } else {
        for(int i = 0; i < m_vDeviceList.size(); i++)
            printf("Device %d render time %f transfer time %f\n", i, ((double)m_vDeviceList[i]->getRenderTime()), 
                                                                     ((double)m_vDeviceList[i]->getBufferToTextureTime()));
    }
}

std::vector<framebuffer_window> DeviceManager::renderFrame(renderinfo *info, int2 resolution) {
	int devices = getNumDevices();

	std::vector<framebuffer_window> fb_windows;

	std::vector<Device*> device_list = getDeviceList();

    setPerDeviceTasks(resolution);

	for(int i = 0; i < devices; i++)
		device_list[i]->makeFrameBuffer(resolution);


	for(int i = 0; i < devices; i++)
        for(int j = 0; j < device_list[i]->getTaskCount(); j++)
            device_list[i]->renderTask(j,info);

	for(int i = 0; i < devices; i++)
		fb_windows.push_back(device_list[i]->getFrameBuffer());

    getFrameTimeResults(resolution);
    //exit(0);

		//Image image(resolution[0], resolution[1], Image::RGB, thisDevice->getFrame());
		//image.toBMP("frame.bmp");


	return fb_windows;
}

std::vector<Device*> DeviceManager::getDeviceList() {
	std::vector<Device*> ret;
    for (int i = 0; i < m_vContext.size(); i++) {
        std::vector<Device*> dev = m_vContext[i]->getDeviceList();
        ret.insert(ret.end(), dev.begin(), dev.end());
    }
    return ret;
}
