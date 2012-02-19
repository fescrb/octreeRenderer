#include "Window.h"

#ifdef _LINUX
	#include <GL/glut.h>
	#include <GL/gl.h>
	#include <GL/glext.h>
#endif //_LINUX

#ifdef _OSX
	#include <glut.h>
	#include <OpenGL/gl.h>
	#include <OpenGL/glext.h>
#endif //_OSX

#include "Vector.h"
#include "Matrix.h"

#include "ProgramState.h"
#include "RenderInfo.h"
#include "SourceFileManager.h"
#include "SourceFile.h"

#include <cstdio>

#include <vector>

/**
 * Statically define methods.
 *
 * Perhaps later we can do away with GLUT and use GLX, Cocoa, etc.
 */

Window *renderWindow = 0;

GLfloat square[] = {0.0f, 0.0f, 0.0f,
					1.0f, 0.0f, 0.0f,
					1.0f, 1.0f, 0.0f,
					0.0f, 1.0f, 0.0f
};

void Window::setRenderWindow(Window *window) {
    renderWindow = window;
}

void staticRender() {
    renderWindow->render();
}

void staticResize(GLint width, GLint height) {
    renderWindow->resize(width, height);
}

Window::Window(int argc, char** argv, int2 dimensions, ProgramState* state)
:	m_size(dimensions){
	if(!renderWindow)
        setRenderWindow(this);
	
	glutInit(&argc, argv);
	glutInitWindowSize(dimensions[0], dimensions[1]);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

	glutCreateWindow("Octree Renderer");
	
	initGL();
    
	glutReshapeFunc(staticResize);
	
    glutDisplayFunc(staticRender);
    
    glutIdleFunc(staticRender);
}

void Window::initGL() {
    glEnable(GL_TEXTURE_2D);
	
	m_vertexShader = compileShader(GL_VERTEX_SHADER, "NoTransform.vert");
	m_fragmentShader = compileShader(GL_FRAGMENT_SHADER, "Coalesce.frag");
	
	m_programObject = linkProgram(m_vertexShader, m_fragmentShader);
	
	resize(m_size[0],m_size[1]);
}

GLuint Window::compileShader(GLenum type, const char* fileName) {
	GLuint shaderID = glCreateShader(type);
	
	const GLchar* source = SourceFileManager::getSource(fileName)->getSource();
	
	glShaderSource(shaderID, 1, &source, NULL);
	
	glCompileShader(shaderID);
	
	GLint status;
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &status);
	
	if(!status) {
		char log[512];
		
		glGetShaderInfoLog(shaderID, 512, NULL, log);
		
		printf("Error compiling shader from file %s:\n%s", fileName, log);
	}
}

GLuint Window::linkProgram(GLuint vertexShader, GLuint fragmentShader) {
	GLuint programID = glCreateProgram();
	
	glAttachShader(programID, vertexShader);
	glAttachShader(programID, fragmentShader);
	
	
	
	glLinkProgram(programID);
	
	GLint status;
	glGetProgramiv(programID, GL_LINK_STATUS, &status);
	
	if(!status) {
		char log[512];
		
		glGetProgramInfoLog(programID, 512, NULL, log);
		
		printf("Error linking shader:\n%s", log);
	}
}

void Window::render() {
    glClear(GL_COLOR_BUFFER_BIT);
    
    //std::vector<GLuint>
    
    glutSwapBuffers();
}

void Window::resize(GLint width, GLint height) {
    m_size = int2(width, height);
    
    // Set up viewport and parallel projection.
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glOrtho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
	
	recalculateViewportVectors();
}

int2 Window::getSize() {
    return m_size;
}

void Window::run() {
    glutMainLoop();
}

void Window::recalculateViewportVectors() {
	RenderInfo *info = m_pProgramState->getRenderInfo(); 
	
	float4 eyepos = position(info->eyePos);
	float4 up = direction(info->up);
	float4 viewDir = direction(info->viewDir);
	
    float4 viewportStep = float4x4::rotationAroundVector( up, _PI/2.0f) * viewDir;
	
	float stepMagnitude = ((info->eyePlaneDist * tan(info->fov/2.0f))*2.0f)/(float)m_size[0];
	viewportStep = viewportStep * stepMagnitude;
	
	up = normalize(up) * stepMagnitude;
	float4 viewportStart(( (viewDir*info->eyePlaneDist) - (viewportStep*((float)m_size[0]/2.0f)) ) - (up*((float)m_size[1]/2.0f)) );
	
	info->up = up;
	info->viewPortStart = viewportStart;
	info->viewStep = viewportStep;
}