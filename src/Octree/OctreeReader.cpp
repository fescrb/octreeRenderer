#include "OctreeReader.h"

#include "Path.h"

#include <cstring>
#include <cstdio>

OctreeReader::OctreeReader(char* name) {
    bool name_contains_file_extension = !strcmp(&name[strlen(name) - 5],"vert");
    int size = strlen(name);
    if(!name_contains_file_extension)
        size+=5;
    m_sPath = (char*) malloc(sizeof(char)*size+1);
    if(name_contains_file_extension)
        strcpy(m_sPath, name);
    else
        sprintf(m_sPath,"%s.vert", name);
    
    if(path_exists(m_sPath)) {
        printf("Path exists!!\n"); exit(0);
    } else {
        printf("Path doesn't exists!!\n"); exit(1);
    }
}
        
Bin OctreeReader::getHeader() {
    
}

Bin OctreeReader::getRoot() {
    
}