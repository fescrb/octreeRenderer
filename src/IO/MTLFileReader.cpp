#include "MTLFileReader.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <fstream>

#include "PNGReader.h"

MTLFileReader::MTLFileReader(const char* filename) {
    m_filename = (char*)malloc(strlen(filename)+1);
    strcpy(m_filename, filename);
}

std::map<std::string,material> MTLFileReader::getMaterials() {
    std::map<std::string,material> materials;
    
    std::ifstream in;
    in.open(m_filename);
    
    std::string current_mtl;
    
    std::string mtl_folder = std::string(m_filename);
    size_t backslash_loc = mtl_folder.rfind("/");
    if(backslash_loc == mtl_folder.npos)
        backslash_loc = -1;
    mtl_folder.erase(backslash_loc, mtl_folder.size()-1);
    //printf("mtl folder %s\n",mtl_folder.c_str());
    
    PNGReader *png_reader;
    
    while(!in.eof()) {
        char line[1024];

        in.getline(line, 1024);
        
        char* mtl_name;
        char* texture_name;
        float r, g, b;
        
        switch(getLineType(line)){
            case TYPE_NEWMTL:
                strtok(line," ");
                mtl_name = strtok(NULL, " \n\0");
                current_mtl = std::string(mtl_name);
                materials[current_mtl] = material();
                break;
            case TYPE_KD:
                strtok(line," ");
                r = atof(strtok(NULL," "));
                g = atof(strtok(NULL," "));
                b = atof(strtok(NULL," \n\0"));
                materials[current_mtl].diffuse = float4(r, g, b, 1.0f);
                break;
            case TYPE_KD_TEXTURE:
                strtok(line," ");
                texture_name = strtok(NULL," \n\0");
                //printf("material %s uses text %s\n", current_mtl.c_str() ,texture_name);
                png_reader = new PNGReader(mtl_folder.c_str(), texture_name);
                materials[current_mtl].texture = png_reader->readImage();
                delete png_reader;
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
    if(line[0] == 'm' && line[1] == 'a' && line[2] == 'p' && line[3] == '_' && line[4] == 'K' && line[5] == 'd')
        return TYPE_KD_TEXTURE;
    return TYPE_UNKOWN;
}