#ifndef _GEOMETRY_OCTREE_WINDOW_H
#define _GEOMETRY_OCTREE_WINDOW_H

#include "Window.h"

#include "Vector4.h"

class OctreeCreator;
class OctreeWriter;

class GeometryOctreeWindow
:   public Window {
    public:
        explicit                 GeometryOctreeWindow(int argc, char** argv, int2 dimensions, OctreeCreator* octreeCreator, OctreeWriter *octreeWriter);

        void                     initGL();

        void                     resize(GLint width, GLint height);

        void                     render();
        void                     idle();
        
        void                     mouse(int button, int state, int x, int y);

    private:

        OctreeCreator           *m_octreeCreator;
        OctreeWriter            *m_octreeWriter;
        
        float4                   m_eye_position;
        float                    m_near_plane;
        float                    m_far_plane;
};

#endif //_GEOMETRY_OCTREE_WINDOW_H
