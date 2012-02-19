/**
 * @file ProgramState.h
 * @brief Contains all global variables, and initializes them.
 */

#ifndef _PROGRAM_STATE_H
#define _PROGRAM_STATE_H

class DataManager;
class DeviceManager;
class RenderInfo;

class ProgramState {
    public:
        explicit                 ProgramState(int argc, char** argv);
                                ~ProgramState();
        
        RenderInfo              *getRenderInfo();
    
        DataManager             *getDataManager();
    
        DeviceManager           *getDeviceManager();
    
    private:
        RenderInfo              *m_pRenderInfo;
    
        DataManager             *m_pDataManager;
        DeviceManager           *m_pDeviceManager;
};

#endif //_PROGRAM_STATE_H
