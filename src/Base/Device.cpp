#include "Device.h"

#include "DeviceInfo.h"

#include <cstdlib>

Device::Device()
:   m_pFrame(0),
    m_renderMode(COLOUR){
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
        m_pFrame = (unsigned char*)malloc(4*size[0]*size[1]);
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
    m_tasksWindow = rect(int2(), int2());
}

void Device::addTask(rect task) {
    m_tasks.push_back(task); 
    
    if(m_tasksWindow.getWidth() == 0)
        m_tasksWindow = task;
    else {
        if(m_tasksWindow.getX() > task.getX()) {
            int diff = m_tasksWindow.getX() - task.getX();
            m_tasksWindow.setX(task.getX());
            m_tasksWindow.setWidth(m_tasksWindow.getWidth()+diff);
        }
        
        if(m_tasksWindow.getY() > task.getY()) {
            int diff = m_tasksWindow.getY() - task.getY();
            m_tasksWindow.setY(task.getY());
            m_tasksWindow.setHeight(m_tasksWindow.getHeight()+diff);
        }
        
        int x_end_task = task.getX() + task.getWidth();
        int x_end_m_window = m_tasksWindow.getX() + m_tasksWindow.getWidth();
        if(x_end_m_window < x_end_task)
            m_tasksWindow.setWidth(m_tasksWindow.getWidth() + (x_end_task - x_end_m_window));
        
        int y_end_task = task.getY() + task.getHeight();
        int y_end_m_window = m_tasksWindow.getY() + m_tasksWindow.getHeight();
        if(y_end_m_window < y_end_task)
            m_tasksWindow.setHeight(m_tasksWindow.getHeight() + (y_end_task - y_end_m_window));
    }
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

void Device::setRenderMode(RenderMode mode) {
    m_renderMode = mode;
}