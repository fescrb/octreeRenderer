#ifndef _SHADER_H
#define _SHADER_H

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