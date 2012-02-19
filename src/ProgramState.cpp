#include "ProgramState.h"

#include "DataManager.h"
#include "DeviceManager.h"
#include "RenderInfo.h"

ProgramState::ProgramState(int argc, char** argv) {
    m_pDataManager = new DataManager;
    
    m_pDeviceManager = new DeviceManager(m_pDataManager);
	
	m_pDeviceManager->printDeviceInfo();
    
    // Setup the render info.
    m_pRenderInfo = new RenderInfo;
    
    m_pRenderInfo->eyePos.setX(0); //x
	m_pRenderInfo->eyePos.setY(0); //y
	m_pRenderInfo->eyePos.setZ(-256.0f); //z
    
	m_pRenderInfo->viewDir.setX(0); //x
	m_pRenderInfo->viewDir.setY(0); //y
	m_pRenderInfo->viewDir.setZ(1.0f); //z
    
    m_pRenderInfo->up.setX(0); //x
	m_pRenderInfo->up.setY(1.0f); //y
	m_pRenderInfo->up.setZ(0); //z
    
	m_pRenderInfo->eyePlaneDist = 1.0f; //Parallel projection, neither of these matter.
	m_pRenderInfo->fov = 1.0f;
}

ProgramState::~ProgramState() {
    delete m_pRenderInfo;
}

RenderInfo* ProgramState::getRenderInfo() {
    return m_pRenderInfo;
}

DataManager* ProgramState::getDataManager() {
    return m_pDataManager;
}

DeviceManager* ProgramState::getDeviceManager() {
    return m_pDeviceManager;
}