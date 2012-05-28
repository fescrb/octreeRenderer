#include "OctreeRendererWindow.h"

#include "Vector.h"
#include "Matrix.h"

#include "ProgramState.h"
#include "RenderInfo.h"

#include "GLUtils.h"
#include "DeviceManager.h"

#include <cstdio>

#include <vector>

OctreeRendererWindow::OctreeRendererWindow(int argc, char** argv, int2 dimensions, ProgramState* state)
:   Window(argc, argv, dimensions, false),
    m_pProgramState(state){
    setRenderWindow(this);

    initGL();
}

void OctreeRendererWindow::initGL() {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_TEXTURE_2D);

    m_vertexShader = Shader(GL_VERTEX_SHADER, "NoTransform.vert");
    m_fragmentShader = Shader(GL_FRAGMENT_SHADER, "Coalesce.frag");

    m_programObject = Program(m_vertexShader, m_fragmentShader);

    GLint numAtts, numUni;
    GLint status;
    glGetProgramiv(m_programObject, GL_ACTIVE_ATTRIBUTES, &numAtts);
    glGetProgramiv(m_programObject, GL_ACTIVE_UNIFORMS, &numUni);

    m_textUniform = glGetUniformLocation(m_programObject, "myTexture");
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

    glValidateProgram(m_programObject);
    glGetProgramiv(m_programObject, GL_VALIDATE_STATUS, &status);

    if(!status) {
        printf("Error Invalid Program\n");
    }

    glBlendEquation(GL_MAX);
    checkGLerror();
    glBlendFunc(GL_ONE,GL_ONE);

    resize(m_size[0],m_size[1]);
}

void OctreeRendererWindow::render() {
    glClear(GL_COLOR_BUFFER_BIT);

    std::vector<framebuffer_window> fb_windows = m_pProgramState->getDeviceManager()->renderFrame(m_pProgramState->getrenderinfo(), m_size);

    glUseProgram(m_programObject);

    for(int i = 0; i < fb_windows.size(); i++) {

        glActiveTexture(GL_TEXTURE0);
        glUniform1i(m_textUniform, 0);
        glBindTexture(GL_TEXTURE_2D, fb_windows[i].texture);

        GLint val;
        glGetUniformiv(m_programObject, m_textUniform, &val);
        printf("Value is %d texture is %d\n", val, fb_windows[i].texture);

        //glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        float x_start = fb_windows[i].window.getX();
        float x_end = fb_windows[i].window.getX() + fb_windows[i].window.getWidth();
        float y_start = fb_windows[i].window.getY();
        float y_end = fb_windows[i].window.getY() + fb_windows[i].window.getHeight();

        printf("x start %f end %f y start %f y end %f\n",x_start,x_end,y_start,y_end);

        glBegin(GL_TRIANGLE_FAN);
        glTexCoord2f(0.0f,0.0f);
        glVertex2f(x_start,y_start);
        glTexCoord2f(1.0f, 0.0f);
        glVertex2f(x_end, y_start);
        glTexCoord2f(1.0f, 1.0f);
        glVertex2f(x_end, y_end);
        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(x_start, y_end);
        glEnd();
    }

    glutSwapBuffers();
}

void OctreeRendererWindow::idle() {
    render();
};

void OctreeRendererWindow::resize(GLint width, GLint height) {
    Window::resize(width, height);

    // Set up viewport and parallel projection.
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0f, width, 0.0f, height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    recalculateViewportVectors();
}

void OctreeRendererWindow::recalculateViewportVectors() {
    renderinfo *info = m_pProgramState->getrenderinfo();

    float4 eyepos = position(info->eyePos);
    float4 up = direction(info->up);
    float4 viewDir = direction(info->viewDir);

    up = normalize(up);

    float4 viewportStep = float4x4::rotationAroundVector( up, _PI/2.0f) * viewDir;

    //printf("%f %f %f %f\n", viewportStep[0], viewportStep[1], viewportStep[2], viewportStep[3]);
    
    float aspec_ratio = (float)m_size[0] / (float)m_size[1];

    float stepMagnitude = ((info->eyePlaneDist * tan((info->fov*aspec_ratio)/2.0f))*2.0f)/(float)m_size[0];
    viewportStep = viewportStep * stepMagnitude;

    up = normalize(up) * stepMagnitude;
    float4 viewportStart(( (eyepos + (viewDir*info->eyePlaneDist)) - (viewportStep*((float)m_size[0]/2.0f)) ) - (up*((float)m_size[1]/2.0f)) );

    info->up = up;
    info->viewPortStart = viewportStart;
    info->viewStep = viewportStep;
}
