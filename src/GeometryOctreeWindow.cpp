#include "GeometryOctreeWindow.h"

#include "OctreeCreator.h"
#include "OctreeWriter.h"

#include <cstdio>

GeometryOctreeWindow::GeometryOctreeWindow(int argc, char** argv, int2 dimensions, OctreeCreator* octreeCreator, OctreeWriter *octreeWriter)
:   Window(argc, argv, dimensions),
    m_octreeCreator(octreeCreator),
    m_octreeWriter(octreeWriter){

    setRenderWindow(this);

    initGL();
    
    m_near_plane = 0.01f;
    m_far_plane =  mag(m_octreeCreator->getMeshAxisAlignedBoundingBox().getSizes());
    m_eye_position = float4(1.0f * m_far_plane, 1.0f * m_far_plane, 1.0f * m_far_plane, 1.0f);
    m_far_plane*=2.0f;
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
    gluPerspective(30.0f, (double)width/(double)height, m_near_plane, m_far_plane );
    //float4 eye_pos = normalize(mesh_bounding_box.getCorner()) * center_distance_to_camera ;
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //gluLookAt( eye_pos, center, up)
    gluLookAt(m_eye_position[0], m_eye_position[1], m_eye_position[2],
              0.0f, 0.0f, 0.0f,
              0.0f, 1.0f, 0.0f);
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

void GeometryOctreeWindow::mouse(int button, int state, int x, int y) {
    printf("mouse b %d s %d u %d d %d\n",button, state, GLUT_UP, GLUT_DOWN);
    if((button == 3) || (button == 4)) { //Scroll wheel event
        if(button == 3) { //UP
            if(state == GLUT_DOWN) {
                m_eye_position=m_eye_position+direction(m_octreeCreator->getMeshAxisAlignedBoundingBox().getSizes().neg()/20.0f);
                resize(m_size[0], m_size[1]);
                render();
            }
        } else { //DOWN
            if(state == GLUT_DOWN) {
                m_eye_position=m_eye_position-direction(m_octreeCreator->getMeshAxisAlignedBoundingBox().getSizes().neg()/20.0f);
                resize(m_size[0], m_size[1]);
                render();
            }
        }
    }
}
