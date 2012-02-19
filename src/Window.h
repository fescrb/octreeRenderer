#ifndef _WINDOW_H
#define _WINDOW_H

#include "Vector.h"

#ifdef _LINUX
	#include <GL/gl.h>
#endif //_LINUX

#ifdef _OSX
	#include <OpenGL/gl.h>
#endif //_OSX

class ProgramState;

class Window {
	public:
		explicit				 Window(int argc, char** argv, int2 dimensions, ProgramState* state);

		void					 render();
    
        void                     initGL();
    
		void					 resize(GLint width, GLint height);
        int2                     getSize();
    
        void                     run();
    
        void                     setRenderWindow(Window *window);
    
        void                     recalculateViewportVectors();
    
    private:
    
        int2                     m_size;
    
        ProgramState            *m_pProgramState;
};

#endif //_WINDOW_H
