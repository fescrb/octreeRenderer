#ifndef _PROGRAM_H
#define _PROGRAM_H

#include "Shader.h"

#include "Graphics.h"

class Program {
    public:
                         Program() {};
        explicit         Program(Shader vertexShader, Shader fragmentShader);
        
        inline operator  GLuint() {
            return m_programID;
        }
        
    private:
        GLuint           m_programID;
};

#endif //_PROGRAM_H