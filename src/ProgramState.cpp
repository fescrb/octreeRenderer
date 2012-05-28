#include "ProgramState.h"

#include "DataManager.h"
#include "DeviceManager.h"
#include "RenderInfo.h"

#include "ConcreteOctree.h"
#include "OctreeReader.h"

ProgramState::ProgramState(int argc, char** argv) {
    
    Octree *octree;
    
    if(argc > 1) {
        octree = new OctreeReader(argv[1]);
    } else {
        octree = ConcreteOctree::getSimpleOctree();
    }
    
    m_pDataManager = new DataManager(octree);
    
    m_pDeviceManager = new DeviceManager(m_pDataManager);
    
    // Setup the render info.
    m_prenderinfo = new renderinfo;
    
    m_prenderinfo[0] = m_pDataManager->getInitialRenderInfo();
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