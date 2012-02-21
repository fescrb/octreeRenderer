/**
 * @file ProgramState.h
 * @brief Contains all global variables, and initializes them.
 */

#ifndef _PROGRAM_STATE_H
#define _PROGRAM_STATE_H

class DataManager;
class DeviceManager;
class renderinfo;

class ProgramState {
    public:
        explicit                 ProgramState(int argc, char** argv);
                                ~ProgramState();
        
        renderinfo              *getrenderinfo();
    
        DataManager             *getDataManager();
    
        DeviceManager           *getDeviceManager();
    
    private:
        renderinfo              *m_prenderinfo;
    
        DataManager             *m_pDataManager;
        DeviceManager           *m_pDeviceManager;
};

#endif //_PROGRAM_STATE_H
