#ifndef _GEOMETRY_OCTREE_WINDOW_H
#define _GEOMETRY_OCTREE_WINDOW_H

#include "Window.h"

class GeometryOctreeWindow 
:   public Window {
    public:
        explicit                 GeometryOctreeWindow(int argc, char** argv, int2 dimensions);
        
        void                     initGL();
        
        void                     resize(GLint width, GLint height);
        
        void                     render();
        
    private:
};

#endif //_GEOMETRY_OCTREE_WINDOW_H