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
}

void GeometryOctreeWindow::resize(GLint width, GLint height) {
    Window::resize(width, height);

    aabox mesh_bounding_box = m_octreeCreator->getMeshAxisAlignedBoundingBox();

    float end_to_end_distance = mag(mesh_bounding_box.getSizes());
    float half_end_to_end_distance = end_to_end_distance/2.0f;
    float center_distance_to_camera = (half_end_to_end_distance) / tan(45.0f/2.0f);
    float near_distance = center_distance_to_camera-half_end_to_end_distance;

    // Set up viewport and parallel projection.
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (double)width/(double)height, near_distance, near_distance + end_to_end_distance );
    float4 eye_pos = normalize(mesh_bounding_box.getCorner()) * center_distance_to_camera * 100.0f;
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(eye_pos[0], eye_pos[1], eye_pos[2],
              0.0f, 0.0f, 0.0f,
              0.0f, 1.0f, 0.0f);
    glLoadIdentity();
}

void GeometryOctreeWindow::render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    m_octreeCreator->render();
    
    glutSwapBuffers();
}

void GeometryOctreeWindow::idle() {
    
}
