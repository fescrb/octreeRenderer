#ifndef _WINDOW_H
#define _WINDOW_H

#include "Vector.h"

#include <OpenGL/gl.h>

class Window {
	public:
		explicit				 Window(int argc, char** argv, float2 dimensions);

		void					 render();
    
		void					 resize(GLint width, GLint height);
        int2                     getSize();
    
    private:
    
        int2                     m_size;
};

#endif //_WINDOW_H
