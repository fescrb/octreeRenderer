#ifndef _RENDER_INFO_WRITER_H
#define _RENDER_INFO_WRITER_H

#include "RenderInfo.h"

class RenderInfoWriter {
    public:
        explicit             RenderInfoWriter(renderinfo info, char* path, const char* name = "renderinfo");
        
        void                 writeAll();
        
    private:
        renderinfo           m_info;
        char*                m_complete_path;
};

#endif //_RENDER_INFO_WRITER_H