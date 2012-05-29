#ifndef _MTL_FILE_READER_H
#define _MTL_FILE_READER_H

#include <map>
#include <string>

#include "Vector.h"

class MTLFileReader {
    public:
        explicit                         MTLFileReader(const char* filename);
        
        enum                     LineType {
            TYPE_NEWMTL, 
            TYPE_KD,
            TYPE_UNKOWN
        };
        
        std::map<std::string,float4>     getMaterials();
     
    private:
        LineType                         getLineType(const char* line);
        
        char*                            m_filename;
};

#endif //_MTL_FILE_READER_H