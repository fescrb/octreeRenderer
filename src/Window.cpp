#include "Window.h"

#include "SourceFileManager.h"
#include "SourceFile.h"

#include <cstdio>

/**
 * Statically define methods.
 *
 * Perhaps later we can do away with GLUT and use GLX, Cocoa, etc.
 */

Window *renderWindow = 0;

void Window::setRenderWindow(Window *window) {
    renderWindow = window;
}

void staticRender() {
    renderWindow->render();
}

void staticResize(GLint width, GLint height) {
    renderWindow->resize(width, height);
}

Window::Window(int argc, char** argv, int2 dimensions)
:	m_size(dimensions){
	glutInit(&argc, argv);
	glutInitWindowSize(dimensions[0], dimensions[1]);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

	glutCreateWindow("Octree Renderer");
    
	glutReshapeFunc(staticResize);
	
    glutDisplayFunc(staticRender);
    
    glutIdleFunc(staticRender);
}

void Window::resize(GLint width, GLint height) {
    m_size = int2(width, height);
}

int2 Window::getSize() {
    return m_size;
}

void Window::run() {
    glutMainLoop();
}