#include "OctreeWindow.h"

#include "SourceFileManager.h"
#include "SourceFile.h"

#include <cstdio>

/**
 * Statically define methods.
 *
 * Perhaps later we can do away with GLUT and use GLX, Cocoa, etc.
 */

OctreeWindow *renderWindow = 0;

void OctreeWindow::setRenderWindow(OctreeWindow *window) {
    renderWindow = window;
}
    
void staticRender() {
    renderWindow->render();
}

void staticIdle() {
    renderWindow->idle();
}

void staticResize(GLint width, GLint height) {
    renderWindow->resize(width, height);
}

void staticMouse(int button, int state, int x, int y) {
    renderWindow->mouse(button, state, x, y);
}

void staticMouseMotion(int x, int y) {
    renderWindow->mouseMotion(x, y);
}

void staticKey(unsigned char key, int x, int y) {
    renderWindow->key(key,x,y);
}

OctreeWindow::OctreeWindow(int argc, char** argv, int2 dimensions, bool useDepthBuffer)
:	m_size(dimensions){
	glutInit(&argc, argv);
	glutInitWindowSize(dimensions[0], dimensions[1]);

	unsigned int depth_buffer = useDepthBuffer ? GLUT_DEPTH : 0;
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | depth_buffer);

	glutCreateWindow("Octree Renderer");

	glutReshapeFunc(staticResize);

    glutDisplayFunc(staticRender);

    glutIdleFunc(staticIdle);
    
    glutMouseFunc(staticMouse);
    
    glutMotionFunc(staticMouseMotion);
    
    glutKeyboardFunc(staticKey);
}

void OctreeWindow::resize(GLint width, GLint height) {
    m_size = int2(width, height);
}

int2 OctreeWindow::getSize() {
    return m_size;
}

void OctreeWindow::run() {
    glutMainLoop();
}

void OctreeWindow::mouse(int button, int state, int x, int y) {
    if(button == 0) {
        if(state == GLUT_DOWN) {
            m_lastMouseLocation = int2(x, y);
        }
    }
    mouseEvent(button, state, x, y);
    printf("mouse button %d state %d x %d y %d\n", button, state, x, y);
}

//Empty mouse event for classes that don't need it
void OctreeWindow::mouseEvent(int button, int state, int x, int y) {
    
}

void OctreeWindow::mouseMotion(int x, int y) {
    printf("mouse motion x %d y %d\n", x, y);
    int x_displacement = x - m_lastMouseLocation[0];
    int y_displacement = y - m_lastMouseLocation[1];
    printf("displacement x %d y %d\n", x_displacement, y_displacement);
    mouseDragEvent(x_displacement, y_displacement);
    m_lastMouseLocation = int2(x, y);
}

//Empty mouse event for classes that don't need it
void OctreeWindow::mouseDragEvent(int x_displacement, int y_displacement) {
    
}

void OctreeWindow::key(unsigned char key, int x, int y) {
    printf("keypress %c\n", key);
    keyPressEvent(key);
}

//Empty key event for classes that don't need it
void OctreeWindow::keyPressEvent(unsigned char key) {
    
}