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
    
    float4 centre = mesh_bounding_box.getCentre();
    float3 size = mesh_bounding_box.getSizes();
    printf("centre (%f %f %f %f) size (%f %f %f)\n", centre[0], centre[1], centre[2], centre[3], size[0], size[1], size[2]);
    
    float end_to_end_distance = mag(mesh_bounding_box.getSizes());
    float half_end_to_end_distance = end_to_end_distance/2.0f;
    float center_distance_to_camera = end_to_end_distance;
    float near_distance = center_distance_to_camera-half_end_to_end_distance;
    
    float light_pos[] ={ mesh_bounding_box.getSizes()[0] * 2.0f, 0.0f, 0.0f} ;
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
    float far_plane = near_distance + center_distance_to_camera*2.0f;
    //gluPerspective( fov, aspect, zNear, zFar);
    gluPerspective(30.0f, (double)width/(double)height, near_distance, far_plane );
    //float4 eye_pos = normalize(mesh_bounding_box.getCorner()) * center_distance_to_camera ;
    float4 eye_pos = float4(1.0f * center_distance_to_camera, 1.0f * center_distance_to_camera, 1.0f * center_distance_to_camera, 1.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //gluLookAt( eye_pos, center, up)
    gluLookAt(eye_pos[0], eye_pos[1], eye_pos[2],
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
