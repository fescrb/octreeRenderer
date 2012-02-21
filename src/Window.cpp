#include "Window.h"

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

GLfloat square[] = {0.0f, 0.0f,
					1.0f, 0.0f,
					1.0f, 1.0f,
					0.0f, 1.0f
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
    /*GLenum err = glewInit();
    if (GLEW_OK != err || !(GLEW_ARB_vertex_shader && GLEW_ARB_fragment_shader && GLEW_ARB_vertex_program && GLEW_ARB_shader_objects)){
        printf("Error: %s\n", glewGetErrorString(err));
    }*/
    
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
	
	return shaderID;
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
	
	m_vertAttr = glGetAttribLocation(programID, "vertex");
	
	glVertexAttribPointer(m_vertAttr, 2, GL_FLOAT, GL_FALSE, 0, square);
	glEnableVertexAttribArray(m_vertAttr);
	
	m_textUniform = glGetUniformLocation(programID, "myTexture");
    GLenum error = glGetError();
    
	if(m_textUniform < 0 || error != GL_NO_ERROR) {
		if(error == GL_INVALID_VALUE) {
			printf("invalid value\n");
		} else if (error == GL_INVALID_OPERATION) {
			printf("invalid operation");
		} else {
			printf("unkown error");
		}
	}
	
    glValidateProgram(programID);
    glGetProgramiv(programID, GL_VALIDATE_STATUS, &status);
    
    if(!status) {
        printf("Error Invalid Program\n");
    }
    
    return programID;
}

void Window::render() {
    glClear(GL_COLOR_BUFFER_BIT);
	
	std::vector<GLuint> textures = m_pProgramState->getDeviceManager()->renderFrame(m_pProgramState->getrenderinfo(), m_size);
    
    glUseProgram(m_programObject);
    
    glEnableVertexAttribArray(m_vertAttr);
    
    glActiveTexture(GL_TEXTURE1);
    
    glUniform1i(m_textUniform, 1);
    GLint val;
    glGetUniformiv(m_programObject, m_textUniform, &val);
    //printf("Value is %d\n", val);
    
	
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    
    glUseProgram(m_programObject);
    
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    
    glutSwapBuffers();
}

void Window::resize(GLint width, GLint height) {
    m_size = int2(width, height);
    
    // Set up viewport and parallel projection.
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0f, 1.0f, 0.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	
	recalculateViewportVectors();
}

int2 Window::getSize() {
    return m_size;
}

void Window::run() {
    glutMainLoop();
}

void Window::recalculateViewportVectors() {
	renderinfo *info = m_pProgramState->getrenderinfo(); 
	
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