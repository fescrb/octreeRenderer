#include "MTLFileReader.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <fstream>

MTLFileReader::MTLFileReader(const char* filename) {
    m_filename = (char*)malloc(strlen(filename)+1);
    strcpy(m_filename, filename);
}

std::map<std::string,float4> MTLFileReader::getMaterials() {
    std::map<std::string,float4> materials;
    
    std::ifstream in;
    in.open(m_filename);
    
    std::string current_mtl;
    
    while(!in.eof()) {
        char line[1024];

        in.getline(line, 1024);
        
        char* mtl_name;
        float r, g, b;
        
        switch(getLineType(line)){
            case TYPE_NEWMTL:
                strtok(line," ");
                mtl_name = strtok(NULL, " \n\0");
                current_mtl = std::string(mtl_name);
                break;
            case TYPE_KD:
                strtok(line," ");
                r = atof(strtok(NULL," "));
                g = atof(strtok(NULL," "));
                b = atof(strtok(NULL," \n\0"));
                materials[current_mtl] = float4(r, g, b, 1.0f);
                break;
            default:
                ; // We do nothing.
        }
    }
    
    return materials;
}

MTLFileReader::LineType MTLFileReader::getLineType(const char* line) {
    if(line[0] == 'K' && line[1] == 'd')
        return TYPE_KD;
    if(line[0] == 'n' && line[1] == 'e' && line[2] == 'w' && line[3] == 'm' && line[4] == 't' && line[5] == 'l')
        return TYPE_NEWMTL;
    return TYPE_UNKOWN;
}