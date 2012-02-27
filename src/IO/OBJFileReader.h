#ifndef _OBJ_FILE_READER_H
#define _OBJ_FILE_READER_H

#include "Mesh.h"

class OBJFileReader {
    public:
        explicit                 OBJFileReader(const char* filename);
        
        mesh                     getMesh();
        
        enum                     LineType {
            TYPE_COMMENT, 
            TYPE_VERTEX_DECLARATION,
            TYPE_NORMAL_DECLARATION,
            TYPE_FACE_DECLARATION,
            TYPE_UNKOWN
        };
        
        LineType                 getLineType(const char* line);
        
        float4                   getVertexFromLine(char* line);
        
    private:
        char*                    m_filename;
};

#endif _OBJ_FILE_READER_H