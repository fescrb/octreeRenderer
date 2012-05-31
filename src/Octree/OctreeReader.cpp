#include "OctreeReader.h"

#include "Path.h"
#include "BinReader.h"
#include "RenderInfoReader.h"

#include <cstring>
#include <cstdio>

OctreeReader::OctreeReader(char* name)
:   m_header_loaded(false),
    m_root_loaded(false) {
    bool name_contains_file_extension = !strcmp(&name[strlen(name) - 5],".voct");
    int size = strlen(name);
    if(!name_contains_file_extension)
        size+=5;
    m_sPath = (char*) malloc(sizeof(char)*size+1);
    if(name_contains_file_extension)
        strcpy(m_sPath, name);
    else
        sprintf(m_sPath,"%s.voct", name);
    
    if(!path_exists(m_sPath)) {
        printf("OctreeReader Error: Path %s doesn't exist\n", m_sPath); exit(1);
    }
    
    RenderInfoReader renderinfo_reader = RenderInfoReader(m_sPath);
    m_initial_renderinfo = renderinfo_reader.read();
    
    //exit(0);
}
        
Bin OctreeReader::getHeader() {
    if(!m_header_loaded) {
        BinReader reader = BinReader(m_sPath, "header");
        m_header = reader.readAll();
        m_header_loaded = true;
    }
    return m_header;
}

Bin OctreeReader::getRoot() {
    if(!m_root_loaded) {
        BinReader reader = BinReader(m_sPath, "0");
        m_root = reader.readAll();
        m_root_loaded = true;
    }
    return m_root;
}