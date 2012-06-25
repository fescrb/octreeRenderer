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
:   OctreeWindow(argc, argv, dimensions, false),
    m_pProgramState(state){
    setRenderWindow(this);

    initGL();
}

void OctreeRendererWindow::initGL() {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_BLEND);
    
    glBlendEquation(GL_MAX);
    checkGLerror();
    glBlendFunc(GL_ONE,GL_ONE);
    
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

    resize(m_size[0],m_size[1]);
}

void OctreeRendererWindow::render() {
    high_res_timer start;
    high_res_timer end;
    start.reset();
    
    glClear(GL_COLOR_BUFFER_BIT);

    //high_res_timer get_things;
    //get_things.reset();
    std::vector<framebuffer_window> fb_windows = m_pProgramState->getDeviceManager()->renderFrame(m_pProgramState->getrenderinfo(), m_size);
    //end.reset();
    //printf("get %f\n", end - get_things);
    
    glUseProgram(m_programObject);

    for(int i = 0; i < fb_windows.size(); i++) {

        glActiveTexture(GL_TEXTURE0);
        glUniform1i(m_textUniform, 0);
        glBindTexture(GL_TEXTURE_2D, fb_windows[i].texture);

        GLint val;
        glGetUniformiv(m_programObject, m_textUniform, &val);
        //printf("Value is %d texture is %d\n", val, fb_windows[i].texture);

        //glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        float x_start = fb_windows[i].window.getX();
        float x_end = fb_windows[i].window.getX() + fb_windows[i].window.getWidth();
        float y_start = fb_windows[i].window.getY();
        float y_end = fb_windows[i].window.getY() + fb_windows[i].window.getHeight();

        //printf("x start %f end %f y start %f y end %f\n",x_start,x_end,y_start,y_end);

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

    glBindTexture(GL_TEXTURE_2D, 0);
    glutSwapBuffers();
    end.reset();
    
    printf("%f\n", end - start);
}

void OctreeRendererWindow::idle() {
    render();
};

void OctreeRendererWindow::resize(GLint width, GLint height) {
    OctreeWindow::resize(width, height);

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
    
    /*renderinfo render_info = info[0];
    
    printf("---------\n");
    printf("Before changing\n");
    printf("renderinfo\neyePos %f %f %f\n", render_info.eyePos[0], render_info.eyePos[1], render_info.eyePos[2]);
    printf("viewDir %f %f %f\n", render_info.viewDir[0], render_info.viewDir[1], render_info.viewDir[2]);
    printf("up %f %f %f\n", render_info.up[0], render_info.up[1], render_info.up[2]);
    printf("viewPortStart %f %f %f\n", render_info.viewPortStart[0], render_info.viewPortStart[1], render_info.viewPortStart[2]);
    printf("viewStep %f %f %f\n", render_info.viewStep[0], render_info.viewStep[1], render_info.viewStep[2]);
    printf("eyePlaneDist %f\n", render_info.eyePlaneDist);
    printf("fov %f\n", render_info.fov);
    printf("lightPos %f %f %f\n", render_info.lightPos[0], render_info.lightPos[1], render_info.lightPos[2]);
    printf("lightBrightness %f\n", render_info.lightBrightness);
    printf("---------\n");*/

    float4 eyepos = position(info->eyePos);
    float4 up = direction(info->up);
    float4 viewDir = direction(info->viewDir);

    up = normalize(up);
    float4 viewportStep = cross(viewDir, up);

    float stepMagnitude = fabs((info->eyePlaneDist * tan(info->fov/2.0f))/(float)m_size[1]);
    
    viewportStep.setY(0.0f);
    
    viewportStep = normalize(viewportStep) * stepMagnitude;
    
    up = normalize(up) * stepMagnitude;
    float4 viewportStart(( (eyepos + (viewDir*info->eyePlaneDist)) - (viewportStep*((float)m_size[0]/2.0f)) ) - (up*((float)m_size[1]/2.0f)) );

    info->up = up;
    info->viewPortStart = viewportStart;
    info->viewStep = viewportStep;
    info->pixel_half_size = mag(info->viewStep)/2.0f;
}

void OctreeRendererWindow::mouseEvent(int button, int state, int x, int y) {
    if((button == 3) || (button == 4)) { //Scroll wheel event
        renderinfo *info = m_pProgramState->getrenderinfo();
        if(button == 3) { //UP
            if(state == GLUT_DOWN) {
                info->eyePos = info->eyePos+(info->viewDir/5.0f);
                recalculateViewportVectors();
            }
        } else { //DOWN
            if(state == GLUT_DOWN) {
                info->eyePos = info->eyePos-(info->viewDir/5.0f);
                recalculateViewportVectors();
            }
        }
    }
}

void OctreeRendererWindow::mouseDragEvent(int x_displacement, int y_displacement) {
    renderinfo *info = m_pProgramState->getrenderinfo();
    float scale = 4.0f;
    float conversor = (info->fov*scale)/(float)m_size[1];
    float x_angle_change = (float)x_displacement*conversor;
    float y_angle_change = (float)y_displacement*conversor;
    
    printf("angle changes x %f y %f\n", x_angle_change, y_angle_change);
    
    if(x_angle_change!=0.0f) {
        info->viewDir = info->viewDir + (info->viewStep * x_angle_change);
        info->viewDir = normalize(info->viewDir);
        recalculateViewportVectors();
    }
    
    if(y_angle_change!=0.0f) {
        info->viewDir = info->viewDir + (info->up * y_angle_change);
        info->viewDir = normalize(info->viewDir);
        info->up = cross(info->viewStep, info->viewDir) * mag(info->up);
        recalculateViewportVectors();
    }
}

void OctreeRendererWindow::keyPressEvent(unsigned char key) {
    renderinfo *info = m_pProgramState->getrenderinfo();
    if(key == 'w' || key == 'W') {
        info->eyePos = info->eyePos + (normalize(info->up)/20.0f);
        recalculateViewportVectors();
    }
    if(key == 's' || key == 'S') {
        info->eyePos = info->eyePos - (normalize(info->up)/20.0f);
        recalculateViewportVectors();
    }
    if(key == 'd' || key == 'd') {
        info->eyePos = info->eyePos + (cross(info->viewDir,normalize(info->up))/20.0f);
        recalculateViewportVectors();
    }
    if(key == 'a' || key == 'A') {
        info->eyePos = info->eyePos - (cross(info->viewDir,normalize(info->up))/20.0f);
        recalculateViewportVectors();
    }
    if(key == 'c' || key == 'C') {
        m_pProgramState->getDeviceManager()->setRenderMode(Device::COLOUR);
    }
    if(key == 'z' || key == 'Z') {
        m_pProgramState->getDeviceManager()->setRenderMode(Device::DEPTH);
    }
    if(key == 'i' || key == 'I') {
        m_pProgramState->getDeviceManager()->setRenderMode(Device::ITERATIONS);
    }
    if(key == 'o' || key == 'O') {
        m_pProgramState->getDeviceManager()->setRenderMode(Device::OCTREE_DEPTH);
    }
    if(key == 'v' || key == 'V') {
        m_pProgramState->getrenderinfo()->print();
    }
}
