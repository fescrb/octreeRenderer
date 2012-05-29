#include "Device.h"

#include "DeviceInfo.h"

#include <cstdlib>

Device::Device()
:   m_pFrame(0) {
    m_tasks = std::vector<rect>();
    m_frameBufferResolution = int2();
}

Device::~Device() {
    clearTasks();
}

char* Device::getName() {
	return m_pDeviceInfo->getName();
}

void Device::makeFrameBuffer(int2 size) {
    // Generate frame buffer if non-existant or not the same size;
    if (size != m_frameBufferResolution) {
        if(m_pFrame) {
            free(m_pFrame);
            free(m_pDepthBuffer);
            free(m_pIterations);
            free(m_pOctreeDepth);
        }
        m_pFrame = (char*)malloc(4*size[0]*size[1]);
        m_pDepthBuffer = (float*)malloc(sizeof(float)*size[0]*size[1]);
        m_pIterations = (unsigned char*)malloc(sizeof(unsigned char)*size[0]*size[1]);
        m_pOctreeDepth = (unsigned char*)malloc(sizeof(unsigned char)*size[0]*size[1]);
        m_frameBufferResolution = size;
    }

    // Clear.
    int i = 0;
    int bufferSize = m_frameBufferResolution[0]*m_frameBufferResolution[1];
    while ( i < bufferSize) {
        m_pFrame[(i*4)  ]=0;
        m_pFrame[(i*4)+1]=0;
        m_pFrame[(i*4)+2]=0;
        m_pFrame[(i*4)+3]=0;
        m_pDepthBuffer[i] = 0.0f;
        m_pIterations[i] = 0;
        m_pOctreeDepth[i] = 0;
        i++;
    }
}

void Device::clearTasks() {
    m_tasks.clear();
}

void Device::addTask(rect task) {
    m_tasks.push_back(task); 
    // Add task to the maximum task window.
    // TODO
    m_tasksWindow = task;
}

rect* Device::getTask(int index) {
    return &m_tasks[index];
}

std::vector<rect> Device::getTasks() {
    return m_tasks;
}

int Device::getTaskCount() {
    return m_tasks.size();
}

rect Device::getTotalTaskWindow() {
    return m_tasksWindow;
}