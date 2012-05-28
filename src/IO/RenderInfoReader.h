#ifndef _RENDER_INFO_READER_H
#define _RENDER_INFO_READER_H

#include "RenderInfo.h"

class RenderInfoReader {
    public:
                         RenderInfoReader(const char* path, const char* name = "renderinfo");
        renderinfo       read();
        
        enum             LineType {
            EyePos, ViewDir, Up, ViewPortStart, ViewStep, EyePlaneDist, FOV, LightPos, LightBrightness, ERROR
        };
        
        LineType         getLineType(char* word);
        
    private:
        char            *m_sCompletePath;
};

#endif //_RENDER_INFO_READER_H