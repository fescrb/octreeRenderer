#ifndef _OCTREE_RENDERER_WINDOW_H
#define _OCTREE_RENDERER_WINDOW_H

#include "Window.h"

#include "Shader.h"
#include "Program.h"

class OctreeRendererWindow
:   public Window {
    public:
        explicit                 OctreeRendererWindow(int argc, char** argv, int2 dimensions, ProgramState* state);

        void                     render();
    
        void                     initGL();
        
        void                     idle();
    
        void                     resize(GLint width, GLint height);
    
        void                     recalculateViewportVectors();
        
        void                     mouseEvent(int button, int state, int x, int y);
    
    private:
    
        ProgramState            *m_pProgramState;
        
        Shader                   m_vertexShader;
        Shader                   m_fragmentShader;
        
        Program                  m_programObject;
        
        GLint                    m_textUniform;
};

#endif //_OCTREE_RENDERER_WINDOW_H