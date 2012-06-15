#ifndef _GEOMETRY_OCTREE_WINDOW_H
#define _GEOMETRY_OCTREE_WINDOW_H

#include "OctreeWindow.h"

#include "Vector4.h"

class OctreeCreator;
class OctreeWriter;

class GeometryOctreeWindow
:   public OctreeWindow {
    public:
        explicit                 GeometryOctreeWindow(int argc, char** argv, int2 dimensions, OctreeCreator* octreeCreator, OctreeWriter *octreeWriter);

        void                     initGL();

        void                     resize(GLint width, GLint height);

        void                     render();
        void                     idle();
        
        void                     mouseEvent(int button, int state, int x, int y);
        void                     mouseDragEvent(int x_displacement, int y_displacement);
        void                     keyPressEvent(unsigned char key);

    private:

        OctreeCreator           *m_octreeCreator;
        OctreeWriter            *m_octreeWriter;
        
        float4                   m_eye_position;
        //! Note: not normalized
        float4                   m_viewDir;
        float4                   m_up;
        float                    m_near_plane;
        float                    m_far_plane;
        float                    m_fov;
};

#endif //_GEOMETRY_OCTREE_WINDOW_H
