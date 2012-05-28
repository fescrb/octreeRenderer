#ifndef _BIN_READER_H
#define _BIN_READER_H

#include "Bin.h"

class BinReader {
    public:
        explicit             BinReader(char* path, const char* name);
        
        Bin                  readAll();
        
    private:
        char*                m_complete_path;
};

#endif //_BIN_READER_H