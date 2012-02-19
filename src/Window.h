#ifndef _WINDOW_H
#define _WINDOW_H

#include "Vector.h"

#include <OpenGL/gl.h>

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
    
    private:
    
        int2                     m_size;
    
        ProgramState            *m_pProgramState;
};

#endif //_WINDOW_H
