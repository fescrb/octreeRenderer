#ifndef _WINDOW_H
#define _WINDOW_H

#include "Vector.h"

#include "Graphics.h"

class ProgramState;

class Window {
	public:
		explicit				 Window(int argc, char** argv, int2 dimensions);

		virtual void             render() = 0;
    
        virtual void             initGL() = 0;
    
		virtual void			 resize(GLint width, GLint height);
        int2                     getSize();
    
        void                     run();
    
        void                     setRenderWindow(Window *window);
    
    protected:
    
        int2                     m_size;
};

#endif //_WINDOW_H
