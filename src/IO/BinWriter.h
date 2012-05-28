#ifndef _BIN_WRITER_H
#define _BIN_WRITER_H

#include "Bin.h"

class BinWriter {
    public:
        explicit             BinWriter(Bin bin, char* path, const char* name);
        
        void                 writeAll();
        
    private:
        Bin                  m_bin;
        char*                m_complete_path;
};

#endif //_BIN_WRITER_H