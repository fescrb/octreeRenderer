#ifndef _WINDOW_H
#define _WINDOW_H

#include "Vector.h"

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

class ProgramState;

class Window {
	public:
		explicit				 Window(int argc, char** argv, int2 dimensions, ProgramState* state);

		void				
		render();
    
        void                     initGL();
		
		GLuint 					 compileShader(GLenum type, const char* fileName);
		GLuint					 linkProgram(GLuint vertexShader, GLuint fragmentShader);
    
		void					 resize(GLint width, GLint height);
        int2                     getSize();
    
        void                     run();
    
        void                     setRenderWindow(Window *window);
    
        void                     recalculateViewportVectors();
    
    private:
    
        int2                     m_size;
    
        ProgramState            *m_pProgramState;
		
		GLuint					 m_vertexShader;
		GLuint					 m_fragmentShader;
		
		GLuint					 m_programObject;
		
		GLint 					 m_vertAttr;
		GLint					 m_textAttr;
		
		GLint					 m_textUniform;
};

#endif //_WINDOW_H
