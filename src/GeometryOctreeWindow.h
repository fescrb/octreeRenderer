#ifndef _GEOMETRY_OCTREE_WINDOW_H
#define _GEOMETRY_OCTREE_WINDOW_H

#include "Window.h"

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

    private:

        OctreeCreator           *m_octreeCreator;
        OctreeWriter            *m_octreeWriter;
};

#endif //_GEOMETRY_OCTREE_WINDOW_H
