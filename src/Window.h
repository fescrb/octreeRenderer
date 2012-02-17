#ifndef _WINDOW_H
#define _WINDOW_H

#include "Vector.h"

#include <glut.h>

class Window {
	public:
		explicit				 Window(int argc, char** argv, float2 dimensions);

		void					 render();
		void					 resize(GLInt width, GLInt height);
};

#endif //_WINDOW_H
