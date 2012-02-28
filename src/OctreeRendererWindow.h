#ifndef _OCTREE_RENDERER_WINDOW_H
#define _OCTREE_RENDERER_WINDOW_H

#include "Window.h"

class OctreeRendererWindow
:   public Window {
    public:
        explicit                 OctreeRendererWindow(int argc, char** argv, int2 dimensions, ProgramState* state);

        void                     render();
    
        void                     initGL();
        
        GLuint                   compileShader(GLenum type, const char* fileName);
        GLuint                   linkProgram(GLuint vertexShader, GLuint fragmentShader);
    
        void                     resize(GLint width, GLint height);
    
        void                     recalculateViewportVectors();
    
    private:
    
        ProgramState            *m_pProgramState;
        
        GLuint                   m_vertexShader;
        GLuint                   m_fragmentShader;
        
        GLuint                   m_programObject;
        
        GLint                    m_vertAttr;
        GLint                    m_textAttr;
        
        GLint                    m_textUniform;
};

#endif //_OCTREE_RENDERER_WINDOW_H