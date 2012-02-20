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

#include "DeviceManager.h"

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

GLuint	indices[] = { 1, 2, 3, 4 };

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
:	m_size(dimensions),
	m_pProgramState(state){
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
	
	//glGenBuffers(1, &m_vertexAndTextureBuffer);
	//glBindBuffer(GL_ARRAY_BUFFER, m_vertexAndTextureBuffer);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*3*4, square, GL_STATIC_DRAW);
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
	
	GLint numAtts, numUni;
	glGetProgramiv(programID, GL_ACTIVE_ATTRIBUTES, &numAtts);
	glGetProgramiv(programID, GL_ACTIVE_UNIFORMS, &numUni);
	
	printf("atts %d uni %d\n",numAtts, numUni);
	
	m_vertAttr = glGetAttribLocation(programID, "vertex");
	
	//glBindAttribLocation(programID, 1, "vertex");
	glVertexAttribPointer(m_vertAttr, 3, GL_FLOAT, GL_FALSE, 0, square);
	glEnableVertexAttribArray(m_vertAttr);
	
	printf("vert %d \n", m_vertAttr);
	
	m_textUniform = glGetUniformLocation(programID, "texture");
	
	if(m_textUniform < 0 ) {
		GLenum error = glGetError();
		if(error == GL_INVALID_VALUE) {
			printf("invalid value\n");
		} else if (error == GL_INVALID_OPERATION) {
			printf("invalid operation");
		} else {
			printf("unkown error");
		}
	}
	
	printf("ul %d \n", m_textUniform);
}

void Window::render() {
    glClear(GL_COLOR_BUFFER_BIT);
    
	glUseProgram(m_programObject);
	
	glEnableVertexAttribArray(m_vertAttr);
	
	//std::vector<GLuint> textures = m_pProgramState->getDeviceManager()->renderFrame(m_pProgramState->getRenderInfo(), m_size);
	
	//glUniform1i(m_textUniform, textures[0]);
	
	glColor3f(1.0f,1.0f,0.0f);
	
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    
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