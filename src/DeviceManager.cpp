#include "DeviceManager.h"

#include "DataManager.h"
#include "RenderInfo.h"

#ifdef USE_OPENCL
	#include "OpenCLContext.h"
#endif

#ifdef USE_OPENMP
    #include "OpenMPContext.h"
#endif

#ifdef USE_SERIAL
	#include "SerialContext.h"
#endif

#include "SizeMacros.h"

#include <cstdio>

DeviceManager::DeviceManager(DataManager *dataManager, int2 resolution)
:	m_pDataManager(dataManager){
    createDivisionWindows(resolution);
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

void DeviceManager::setRenderMode(Device::RenderMode mode) {
    for(int i = 0; i < getNumDevices(); i++) {
        getDevice(i)->setRenderMode(mode);
    }
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
        float total_pps = 0.0f;
        for (int i = 0; i < device_count; i++) {
            total_pps+=m_vDeviceCharacteristics[i].pixels_per_second;
        }
        
        std::vector<division_window> unset_windows;
        
        //printf("division res %d %d\n", m_division_window_count[0], m_division_window_count[1]);
        
        for(int i = 0; i < m_division_window_count; i++) {
            unset_windows.push_back(m_division_windows[i]);   
        }

        for (int i = 0; i < device_count; i++) {
            getDevice(i)->clearTasks();

            int count = m_division_window_count * (m_vDeviceCharacteristics[i].pixels_per_second/total_pps);
            if(i == 0)
                count++;
            
            getDevice(i)->addTask(rect(unset_windows[0].window.getOrigin(),int2(count*RAY_BUNDLE_WINDOW_SIZE, domain_resolution.getY())));
            for(int j = 0; j < count; j++) {
                unset_windows.erase(unset_windows.begin());
            }
            
            /*printf("------------\n");
            printf("div count %d total %d\n", count, m_division_window_count);
            printf("Device %d totl window %d %d, %d %d\n", i, getDevice(i)->getTotalTaskWindow().getX(),
                                                              getDevice(i)->getTotalTaskWindow().getY(), 
                                                              getDevice(i)->getTotalTaskWindow().getWidth() ,
                                                              getDevice(i)->getTotalTaskWindow().getHeight());
            printf("------------\n");*/
        }
        
        while(unset_windows.size() != 0) {
            int count = 1;
            getDevice(0)->addTask(rect(unset_windows[0].window.getOrigin(),int2(count*RAY_BUNDLE_WINDOW_SIZE, domain_resolution.getY())));
            for(int j = 0; j < count; j++) {
                unset_windows.erase(unset_windows.begin());
            }
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
        device_list[i]->renderStart();
        
    
    #pragma omp parallel for 
	for(int i = 0; i < devices; i++) {
        device_list[i]->setRenderInfo(info);
        #pragma omp parallel for 
        for(int j = 0; j < device_list[i]->getTaskCount(); j++)
            device_list[i]->advanceTask(j);
        #pragma omp parallel for 
        for(int j = 0; j < device_list[i]->getTaskCount(); j++)
            device_list[i]->renderTask(j);
        #pragma omp parallel for 
        for(int j = 0; j < device_list[i]->getTaskCount(); j++)
            device_list[i]->calculateCostsForTask(j);
        
        device_list[i]->renderEnd();
    }

    std::vector<const unsigned int*> costs;
    
	for(int i = 0; i < devices; i++) {
		fb_windows.push_back(device_list[i]->getFrameBuffer());
        costs.push_back(device_list[i]->getCosts());
    }

    getFrameTimeResults(resolution);

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

void DeviceManager::createDivisionWindows(int2 domain_resolution) {
    m_division_window_count = domain_resolution[0]/RAY_BUNDLE_WINDOW_SIZE;
    
    m_division_windows = (division_window*) malloc (sizeof(division_window) * m_division_window_count);
    
    for(int x = 0; x < m_division_window_count; x++) {
        division_window this_window;
        this_window.window = rect(int2(x*RAY_BUNDLE_WINDOW_SIZE, 0), int2(RAY_BUNDLE_WINDOW_SIZE, domain_resolution[1]));
        m_division_windows[x] = this_window;
    }
}
