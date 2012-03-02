#ifndef _SHADER_H
#define _SHADER_H

#include "Graphics.h"

class Shader {
    public:
                         Shader() {};
        explicit         Shader(GLenum type, const char* fileName);
        
        inline operator  GLuint() {
            return m_shaderID;
        }
        
    private:
        GLuint           m_shaderID;
};

#endif //_SHADER_H