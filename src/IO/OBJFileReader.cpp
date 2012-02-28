#include "OBJFileReader.h"

#include <cstring>
#include <fstream>
#include <cmath>

OBJFileReader::OBJFileReader(const char* filename) {
    m_filename = (char*)malloc(strlen(filename)+1);
}
        
mesh OBJFileReader::getMesh() {
    std::ifstream in;
    in.open(m_filename);
    
    mesh objMesh;
    std::vector<float4> vertexList;
    std::vector<float4> normalList;
    int normalCounter = 0;
    
    while(!in.eof()) {
        char line[1024];

        in.getline(line, 1024);
        
        float4 tmp;
        
        switch(getLineType(line)){
            case TYPE_VERTEX_DECLARATION:
                vertexList.push_back(getVertexFromLine(line));
            case TYPE_NORMAL_DECLARATION:
                tmp = getVertexFromLine(line); // tmp = normal
                tmp.setW(0.0f);
                normalList.push_back(tmp);
            case TYPE_FACE_DECLARATION:
                ;
            default:
                ; // We do nothing.
        }
    }
    
    return objMesh;
}

OBJFileReader::LineType OBJFileReader::getLineType(const char* line) {
    switch(line[0]) {
        case '#':
            return TYPE_COMMENT;
        case 'v':
            if(line[1] == ' ')
                return TYPE_VERTEX_DECLARATION;
            if(line[1] == 'n')
                return TYPE_NORMAL_DECLARATION;
        case 'f':
            return TYPE_FACE_DECLARATION;
        default:
            return TYPE_UNKOWN;
    }
}

float4 OBJFileReader::getVertexFromLine(char* line) {
    strtok(line, " ");
    char* x = strtok(NULL, " ");
    char* y = strtok(NULL, " ");
    char* z = strtok(NULL, " ");
    return float4(atof(x),atof(y),atof(z),1.0f);
}