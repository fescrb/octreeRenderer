#include "OctreeReader.h"

#include "Path.h"
#include "BinReader.h"
#include "RenderInfoReader.h"

#include <cstring>
#include <cstdio>

OctreeReader::OctreeReader(char* name) {
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
    
    /*printf("renderinfo\neyePos %f %f %f\n", m_initial_renderinfo.eyePos[0], m_initial_renderinfo.eyePos[1], m_initial_renderinfo.eyePos[2]);
    printf("viewDir %f %f %f\n", m_initial_renderinfo.viewDir[0], m_initial_renderinfo.viewDir[1], m_initial_renderinfo.viewDir[2]);
    printf("up %f %f %f\n", m_initial_renderinfo.up[0], m_initial_renderinfo.up[1], m_initial_renderinfo.up[2]);
    printf("viewPortStart %f %f %f\n", m_initial_renderinfo.viewPortStart[0], m_initial_renderinfo.viewPortStart[1], m_initial_renderinfo.viewPortStart[2]);
    printf("viewStep %f %f %f\n", m_initial_renderinfo.viewStep[0], m_initial_renderinfo.viewStep[1], m_initial_renderinfo.viewStep[2]);
    printf("eyePlaneDist %f\n", m_initial_renderinfo.eyePlaneDist);
    printf("fov %f\n", m_initial_renderinfo.fov);
    printf("lightPos %f %f %f\n", m_initial_renderinfo.lightPos[0], m_initial_renderinfo.lightPos[1], m_initial_renderinfo.lightPos[2]);
    printf("lightBrightness %f\n", m_initial_renderinfo.lightBrightness);*/
    //exit(0);
}
        
Bin OctreeReader::getHeader() {
    BinReader reader = BinReader(m_sPath, "header");
    return reader.readAll();
}

Bin OctreeReader::getRoot() {
    BinReader reader = BinReader(m_sPath, "0");
    return reader.readAll();
}