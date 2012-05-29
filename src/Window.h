#ifndef _WINDOW_H
#define _WINDOW_H

#include "Vector.h"

#include "Graphics.h"

class ProgramState;

class Window {
	public:
		explicit				 Window(int argc, char** argv, int2 dimensions, bool useDepthBuffer = true);

		virtual void             render() = 0;
        virtual void             idle() = 0;
    
        virtual void             initGL() = 0;
    
		virtual void			 resize(GLint width, GLint height);
        int2                     getSize();
        
        virtual void             mouse(int button, int state, int x, int y);
    
        void                     run();
    
        void                     setRenderWindow(Window *window);
    
    protected:
    
        int2                     m_size;
};

#endif //_WINDOW_H
