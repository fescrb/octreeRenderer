#include "GeometryOctreeWindow.h"

GeometryOctreeWindow::GeometryOctreeWindow(int argc, char** argv, int2 dimensions)
:   Window(argc, argv, dimensions){
    
}

void GeometryOctreeWindow::initGL() {
    
}

void GeometryOctreeWindow::resize(GLint width, GLint height) {
    Window::resize(width, height);
    
    // Set up viewport and parallel projection.
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0f, 1.0f, 0.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void GeometryOctreeWindow::render() {
    
}