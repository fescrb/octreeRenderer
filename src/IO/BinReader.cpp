#include "BinReader.h"

#include "IOUtils.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

BinReader::BinReader(char* path, const char* name) {
    m_complete_path = (char*)malloc(sizeof(char)*(strlen(path)+strlen(name)+1)+1);
    sprintf(m_complete_path, "%s/%s",path,name);
}
        
Bin BinReader::readAll() {
    struct stat st;
    stat(m_complete_path, &st);
    int size = st.st_size;
    
    //printf("size %d\n", size);
    
    char* data = (char*)malloc(size+1);

    Bin bin = Bin(data, size);
    
    int fd = open(m_complete_path, O_RDONLY);
    if(isIOError(fd)) {
        printIOError();
        exit(1);
    }

    int size_to_go = size;
    while(size_to_go) {
        //printf("still reading %d\n",size_to_go);
        int res =read(fd, &(bin.getDataPointer()[size-size_to_go]),size_to_go);
        if(isIOError(res)) {
            printIOError();
            exit(1);
        }
        size_to_go-=res;
    }
    close(fd);
    
    return bin;
}