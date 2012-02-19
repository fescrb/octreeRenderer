#include "Window.h"

#include <glut.h>

Window::Window(int argc, char** argv, int2 dimensions, ProgramState* state) {
	glutInit(&argc, argv);
	glutInitWindowSize(dimensions[0], dimensions[1]);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

	glutCreateWindow("Octree Renderer");
    
    //glutDisplayFunc(&this->render);
    
    //glutReshapeFunc(&this->resize);
}

void Window::render() {
    glClear(GL_COLOR_BUFFER_BIT);
    
    glutSwapBuffers();
}

void Window::resize(GLint width, GLint height) {
    m_size = int2(width, height);
    
    // Set up viewport and parallel projection.
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glOrtho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
}

int2 Window::getSize() {
    return m_size;
}

void Window::run() {
    glutMainLoop();
}