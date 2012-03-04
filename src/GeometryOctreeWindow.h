#ifndef _GEOMETRY_OCTREE_WINDOW_H
#define _GEOMETRY_OCTREE_WINDOW_H

#include "Window.h"

class OctreeCreator;

class GeometryOctreeWindow
:   public Window {
    public:
        explicit                 GeometryOctreeWindow(int argc, char** argv, int2 dimensions, OctreeCreator* octreeCreator);

        void                     initGL();

        void                     resize(GLint width, GLint height);

        void                     render();

    private:

        OctreeCreator           *m_octreeCreator;
};

#endif //_GEOMETRY_OCTREE_WINDOW_H
