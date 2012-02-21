#include "ProgramState.h"

#include "DataManager.h"
#include "DeviceManager.h"
#include "RenderInfo.h"

ProgramState::ProgramState(int argc, char** argv) {
    m_pDataManager = new DataManager;
    
    m_pDeviceManager = new DeviceManager(m_pDataManager);
    
    // Setup the render info.
    m_prenderinfo = new renderinfo;
    
    m_prenderinfo->eyePos.setX(0); //x
	m_prenderinfo->eyePos.setY(0); //y
	m_prenderinfo->eyePos.setZ(-256.0f); //z
    
	m_prenderinfo->viewDir.setX(0); //x
	m_prenderinfo->viewDir.setY(0); //y
	m_prenderinfo->viewDir.setZ(1.0f); //z
    
    m_prenderinfo->up.setX(0); //x
	m_prenderinfo->up.setY(1.0f); //y
	m_prenderinfo->up.setZ(0); //z
    
	m_prenderinfo->eyePlaneDist = 1.0f; //Parallel projection, neither of these matter.
	m_prenderinfo->fov = 1.0f;
}

ProgramState::~ProgramState() {
    delete m_prenderinfo;
	delete m_pDataManager;
	delete m_pDeviceManager;
}

renderinfo* ProgramState::getrenderinfo() {
    return m_prenderinfo;
}

DataManager* ProgramState::getDataManager() {
    return m_pDataManager;
}

DeviceManager* ProgramState::getDeviceManager() {
    return m_pDeviceManager;
}