#include "Device.h"

#include "DeviceInfo.h"

Device::Device() {
    m_tasks = std::vector<rect>();
}

Device::~Device() {
    clearTasks();
}

char* Device::getName() {
	return m_pDeviceInfo->getName();
}

void Device::clearTasks() {
    m_tasks.clear();
}

void Device::addTask(rect task) {
    m_tasks.push_back(task); 
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
    rect window = m_tasks[0];
    for(int i = 1; i < getTaskCount(); i++) {
        //TODO
    }
    return window;
}