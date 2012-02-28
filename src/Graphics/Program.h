#ifndef _PROGRAM_H
#define _PROGRAM_H

#include "Shader.h"

#ifdef _LINUX
    #include <GL/glut.h>
    #include <GL/gl.h>
    #include <GL/glext.h>
#endif //_LINUX

#ifdef _OSX
    #include <glut.h>
    #include <OpenGL/gl.h>
    #include <OpenGL/glext.h>
#endif //_OSX

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