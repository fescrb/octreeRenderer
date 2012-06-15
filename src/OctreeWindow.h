#ifndef _OCTREE_WINDOW_H
#define _OCTREE_WINDOW_H

#include "Vector.h"

#include "Graphics.h"

class ProgramState;

class OctreeWindow {
	public:
		explicit				 OctreeWindow(int argc, char** argv, int2 dimensions, bool useDepthBuffer = true);

		virtual void             render() = 0;
        virtual void             idle() = 0;
    
        virtual void             initGL() = 0;
    
		virtual void			 resize(GLint width, GLint height);
        int2                     getSize();
        
        virtual void             mouseEvent(int button, int state, int x, int y);
        virtual void             mouseDragEvent(int x_displacement, int y_displacement);
        virtual void             keyPressEvent(unsigned char key);
    
        void                     run();
    
        void                     setRenderWindow(OctreeWindow *window);
        
        void                     mouse(int button, int state, int x, int y);
        void                     mouseMotion(int x, int y);
        void                     key(unsigned char key, int x, int y);
    
    protected:
    
        int2                     m_size;
        
        int2                     m_lastMouseLocation;
};

#endif //_OCTREE_WINDOW_H
