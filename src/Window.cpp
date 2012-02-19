#include "Window.h"

#include <glut.h>

#include "Vector.h"
#include "Matrix.h"

#include "ProgramState.h"
#include "RenderInfo.h"

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

Window::Window(int argc, char** argv, int2 dimensions, ProgramState* state) {
	glutInit(&argc, argv);
	glutInitWindowSize(dimensions[0], dimensions[1]);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

	glutCreateWindow("Octree Renderer");
    
    glutDisplayFunc(staticRender);
    
    glutReshapeFunc(staticResize);
    
    glutIdleFunc(staticRender);
    
    float4 res = float4x4::rotationAroundVector(float4(0.0f,1.0f,0.0f,0.0f), _PI/2.0f) * float4(0.0f,0.0f,1.0f,0.0f);
    
    printf("x %f y %f z %f w %f \n", res[0], res[1], res[2], res[3]);
    
    if(!renderWindow)
        setRenderWindow(this);
}

void Window::initGL() {
    glEnable(GL_TEXTURE_2D);
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

void Window::recalculateViewportVectors() {
    float4 viewDir = float4x4::rotationAroundVector( direction(m_pProgramState->getRenderInfo()->up), _PI/2.0f) * direction(m_pProgramState->getRenderInfo()->viewDir);
}