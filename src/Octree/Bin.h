#ifndef _BIN_H
#define _BIN_H

class Bin {
    public:
                     Bin(char* data = 0, unsigned int size = 0);
        
        void         setData(char* data);
        void         setSize(unsigned int size);
        
        char*        getDataPointer();
        unsigned int getSize();
        
    private:
        
        char*        m_pData;
        unsigned int m_size;
};

#endif //_BIN_H