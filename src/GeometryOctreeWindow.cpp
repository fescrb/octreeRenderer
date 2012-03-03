#include "GeometryOctreeWindow.h"

#include "OctreeCreator.h"

GeometryOctreeWindow::GeometryOctreeWindow(int argc, char** argv, int2 dimensions, OctreeCreator* octreeCreator)
:   Window(argc, argv, dimensions),
    m_octreeCreator(octreeCreator){
    
}

void GeometryOctreeWindow::initGL() {
    
}

void GeometryOctreeWindow::resize(GLint width, GLint height) {
    Window::resize(width, height);
    
    // Set up viewport and parallel projection.
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //gluPerspective(45.0f, (double)width/(double)height, );
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void GeometryOctreeWindow::render() {
    glClear();
    
}