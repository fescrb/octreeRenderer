#include "RenderInfoReader.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <fstream>

RenderInfoReader::RenderInfoReader(const char* path, const char* name) {
    m_sCompletePath = (char*)malloc(sizeof(char)*(strlen(path)+strlen(name)+1)+1);
    sprintf(m_sCompletePath, "%s/%s",path,name);
}
        
renderinfo RenderInfoReader::read() {
    renderinfo info;
    
    char line[255];
    
    std::ifstream in;
    in.open(m_sCompletePath);
    
    while(!in.eof()) {
        in.getline(line, 255);
        
        char* typeWord = strtok(line, " ");
        
        float scalar, x, y, z;
        
        switch(getLineType(typeWord)) {
            case EyePos:
                x = atof(strtok(NULL," "));
                y = atof(strtok(NULL," "));
                z = atof(strtok(NULL," \n\0"));
                info.eyePos = float3(x,y,z);
                break;
            case ViewDir:
                x = atof(strtok(NULL," "));
                y = atof(strtok(NULL," "));
                z = atof(strtok(NULL," \n\0"));
                info.viewDir = float3(x,y,z);
                break;
            case Up:
                x = atof(strtok(NULL," "));
                y = atof(strtok(NULL," "));
                z = atof(strtok(NULL," \n\0"));
                info.up = float3(x,y,z);
                break;
            case ViewPortStart:
                x = atof(strtok(NULL," "));
                y = atof(strtok(NULL," "));
                z = atof(strtok(NULL," \n\0"));
                info.viewPortStart = float3(x,y,z);
                break;
            case ViewStep:
                x = atof(strtok(NULL," "));
                y = atof(strtok(NULL," "));
                z = atof(strtok(NULL," \n\0"));
                info.viewStep = float3(x,y,z);
                break;
            case EyePlaneDist:
                scalar = atof(strtok(NULL," \n\0"));
                info.eyePlaneDist = scalar;
                break;
            case FOV:
                scalar = atof(strtok(NULL," \n\0"));
                info.fov = scalar;
                break;
            case LightPos:
                x = atof(strtok(NULL," "));
                y = atof(strtok(NULL," "));
                z = atof(strtok(NULL," \n\0"));
                info.lightPos = float3(x,y,z);
                break;
            case LightBrightness:
                scalar = atof(strtok(NULL," \n\0"));
                info.lightBrightness = scalar;
                break;
            default:
                break;
        }
    }
    
    return info;
}

RenderInfoReader::LineType RenderInfoReader::getLineType(char* word) {
    if(!strcmp(word,"eyePos"))
        return EyePos;
    if(!strcmp(word,"viewDir"))
        return ViewDir;
    if(!strcmp(word,"up"))
        return Up;
    if(!strcmp(word,"viewPortStart"))
        return ViewPortStart;
    if(!strcmp(word,"viewStep"))
        return ViewStep;
    if(!strcmp(word,"eyePlaneDist"))
        return EyePlaneDist;
    if(!strcmp(word,"fov"))
        return FOV;
    if(!strcmp(word,"lightPos"))
        return LightPos;
    if(!strcmp(word,"lightBrightness"))
        return LightBrightness;
    return ERROR;
}