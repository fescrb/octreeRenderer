#include "RenderInfoWriter.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <fstream>

RenderInfoWriter::RenderInfoWriter(renderinfo info, char* path, const char* name)
:   m_info(info){
    m_complete_path = (char*)malloc(sizeof(char)*(strlen(path)+strlen(name)+1)+1);
    sprintf(m_complete_path, "%s/%s",path,name);
}

void RenderInfoWriter::writeAll() {
    std::ofstream out;
    out.open(m_complete_path);
    
    out << "eyePos " << m_info.eyePos[0] << " " << m_info.eyePos[1] << " " << m_info.eyePos[2] << std::endl;
    out << "viewDir " << m_info.viewDir[0] << " " << m_info.viewDir[1] << " " << m_info.viewDir[2] << std::endl;
    out << "up " << m_info.up[0] << " " << m_info.up[1] << " " << m_info.up[2] << std::endl;
    out << "viewPortStart " << m_info.viewPortStart[0] << " " << m_info.viewPortStart[1] << " " << m_info.viewPortStart[2] << std::endl;
    out << "viewStep " << m_info.viewStep[0] << " " << m_info.viewStep[1] << " " << m_info.viewStep[2] << std::endl;
    out << "eyePlaneDist " << m_info.eyePlaneDist << std::endl;
    out << "fov " << m_info.fov << std::endl;
    
    out << "lightPos " << m_info.lightPos[0] << " " << m_info.lightPos[1] << " " << m_info.lightPos[2] << std::endl;
    out << "lightBrightness " << m_info.lightBrightness;
    
    out.close();
}

