#include "BinWriter.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

BinWriter::BinWriter(Bin bin, char* path, const char* name) 
:   m_bin(bin){
    m_complete_path = (char*)malloc(sizeof(char)*(strlen(path)+strlen(name)+1)+1);
    sprintf(m_complete_path, "%s/%s",path,name);
}
        
void  BinWriter::writeAll() {
    int fd = open(m_complete_path, O_WRONLY | O_CREAT | O_TRUNC);
    int size_to_go = m_bin.getSize();
    while(size_to_go) {
        size_to_go-=write(fd, &(m_bin.getDataPointer()[m_bin.getSize()-size_to_go]),size_to_go);
    }
    close(fd);
}