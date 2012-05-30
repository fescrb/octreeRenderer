#include "GeometryOctreeWindow.h"

#include "OctreeCreator.h"
#include "OctreeWriter.h"

#include "Matrix.h"

#include <cstdio>

GeometryOctreeWindow::GeometryOctreeWindow(int argc, char** argv, int2 dimensions, OctreeCreator* octreeCreator, OctreeWriter *octreeWriter)
:   Window(argc, argv, dimensions),
    m_octreeCreator(octreeCreator),
    m_octreeWriter(octreeWriter){

    setRenderWindow(this);

    initGL();
    
    m_fov = 30.0f;
    m_near_plane = 0.01f;
    m_far_plane =  mag(m_octreeCreator->getMeshAxisAlignedBoundingBox().getSizes());
    m_eye_position = float4(1.0f * m_far_plane, 1.0f * m_far_plane, 1.0f * m_far_plane, 1.0f);
    m_far_plane*=2.0f;
        
    m_viewDir = float4()-m_eye_position;
    m_viewDir.setW(0.0f);
    m_up = normalize(m_viewDir);
    m_up.setY(m_up.getY()*-1.0f);
}

void GeometryOctreeWindow::initGL() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);
}

void GeometryOctreeWindow::resize(GLint width, GLint height) {
    Window::resize(width, height);
    
    float light_pos[] ={ m_octreeCreator->getMeshAxisAlignedBoundingBox().getSizes()[0] * 2.0f, 0.0f, 0.0f} ;
    float color[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float ambient[] = {0.2f, 0.2f, 0.2f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, &(light_pos[0]));
    //glMaterialfv( );
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, color);
    glEnable(GL_LIGHT0);

    // Set up viewport and parallel projection.
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    //gluPerspective( fov, aspect, zNear, zFar);
    gluPerspective(m_fov, (double)width/(double)height, m_near_plane, m_far_plane );
    //float4 eye_pos = normalize(mesh_bounding_box.getCorner()) * center_distance_to_camera ;
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //gluLookAt( eye_pos, center, up)
    float4 centre = m_eye_position + m_viewDir;
    gluLookAt(m_eye_position[0], m_eye_position[1], m_eye_position[2],
              centre[0]        , centre[1]        , centre[2],
              0.0f             , 1.0f             , 0.0f);
}

void GeometryOctreeWindow::render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    m_octreeCreator->render();
    
    glutSwapBuffers();
}

void GeometryOctreeWindow::idle() {
    if(!m_octreeCreator->isConverted()) {
        m_octreeCreator->convert();
        m_octreeWriter->write();
    }
    render();
}

void GeometryOctreeWindow::mouseEvent(int button, int state, int x, int y) {
    if((button == 3) || (button == 4)) { //Scroll wheel event
        if(button == 3) { //UP
            if(state == GLUT_DOWN) {
                m_eye_position=m_eye_position+(m_viewDir/20.0f);
                resize(m_size[0], m_size[1]);
                //render();
            }
        } else { //DOWN
            if(state == GLUT_DOWN) {
                m_eye_position=m_eye_position-(m_viewDir/20.0f);
                resize(m_size[0], m_size[1]);
                //render();
            }
        }
    }
}

void GeometryOctreeWindow::mouseDragEvent(int x_displacement, int y_displacement) {
    float scale = 4.0f;
    float conversor = (m_fov*scale)/(float)m_size[1];
    float x_angle_change = (float)x_displacement*conversor;
    float y_angle_change = (float)y_displacement*conversor;
    
    printf("angle changes x %f y %f\n", x_angle_change, y_angle_change);
    
    float viewDirMag = mag(m_viewDir);
    float degree_step_mag = (viewDirMag * tan(m_fov/2.0f))/(m_fov/2.0f);
    float4 camera_x_axis = cross(m_up,m_viewDir/viewDirMag);
    
    if(x_angle_change!=0.0f) {
        
        m_viewDir = m_viewDir + (camera_x_axis * x_angle_change * degree_step_mag);
        m_viewDir = normalize(m_viewDir) * viewDirMag;
        printf("new viewDir %f %f %f\n", m_viewDir.getX(), m_viewDir.getY(), m_viewDir.getZ());
        resize(m_size[0], m_size[1]);
    }
    
    if(y_angle_change!=0.0f) {
        m_viewDir = m_viewDir + (m_up * x_angle_change * degree_step_mag);
        m_viewDir = normalize(m_viewDir) * viewDirMag;
        m_up = cross(m_viewDir/viewDirMag,camera_x_axis);
        printf("new viewDir %f %f %f\n", m_viewDir.getX(), m_viewDir.getY(), m_viewDir.getZ());
        printf("new up %f %f %f\n", m_up.getX(), m_up.getY(), m_up.getZ());
        resize(m_size[0], m_size[1]);
    }
}

void GeometryOctreeWindow::keyPressEvent(unsigned char key) {
    float viewDirMag = mag(m_viewDir);
    if(key == 'w' || key == 'W') {
        m_eye_position = m_eye_position + (m_up*viewDirMag/20.0f);
        resize(m_size[0], m_size[1]);
    }
    if(key == 's' || key == 'S') {
        m_eye_position = m_eye_position - (m_up*viewDirMag/20.0f);
        resize(m_size[0], m_size[1]);
    }
    if(key == 'd' || key == 'd') {
        m_eye_position = m_eye_position + (cross(m_viewDir/viewDirMag,m_up)*viewDirMag/20.0f);
        resize(m_size[0], m_size[1]);
    }
    if(key == 'a' || key == 'A') {
        m_eye_position = m_eye_position - (cross(m_viewDir/viewDirMag,m_up)*viewDirMag/20.0f);
        resize(m_size[0], m_size[1]);
    }
}
