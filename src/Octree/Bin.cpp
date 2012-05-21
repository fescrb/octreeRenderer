#include "Bin.h"

Bin::Bin(char* data, unsigned int size) 
:   m_pData(data),
    m_size(size) {
}
        
void Bin::setData(char* data) {
    m_pData = data;
}

void Bin::setSize(unsigned int size) {
    m_size = size;
}
        
char* Bin::getDataPointer() {
    return m_pData;
}

unsigned int Bin::getSize() {
    return m_size;
}