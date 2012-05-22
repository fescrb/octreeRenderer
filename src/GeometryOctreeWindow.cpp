#include "GeometryOctreeWindow.h"

#include "OctreeCreator.h"

GeometryOctreeWindow::GeometryOctreeWindow(int argc, char** argv, int2 dimensions, OctreeCreator* octreeCreator)
:   Window(argc, argv, dimensions),
    m_octreeCreator(octreeCreator){

    setRenderWindow(this);

    initGL();
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

    aabox mesh_bounding_box = m_octreeCreator->getMeshAxisAlignedBoundingBox();

    float end_to_end_distance = mag(mesh_bounding_box.getSizes());
    float half_end_to_end_distance = end_to_end_distance/2.0f;
    float center_distance_to_camera = end_to_end_distance*10.0f;
    float near_distance = half_end_to_end_distance;
    
    float light_pos[] ={ mesh_bounding_box.getSizes()[0] * 2.0f, 0.0f, 0.0f} ;
    float color[] = {1.0f, 1.0f, 1.0f};
    float ambient[] = {0.0f, 0.0f, 0.0f};
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
    gluPerspective(90.0f, (double)width/(double)height, near_distance, near_distance + end_to_end_distance*12.0f );
    float4 eye_pos = normalize(mesh_bounding_box.getCorner()) * center_distance_to_camera ;
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //gluLookAt( eye_pos, center, up)
    gluLookAt(eye_pos[0], eye_pos[1], eye_pos[2],
              0.0f, 0.0f, 0.0f,
              0.0f, 1.0f, 0.0f);
    glLoadIdentity();
}

void GeometryOctreeWindow::render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    m_octreeCreator->convert();
    m_octreeCreator->render();
    
    glutSwapBuffers();
}

void GeometryOctreeWindow::idle() {
    
}
