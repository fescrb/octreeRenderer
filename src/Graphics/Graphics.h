#ifndef _GRAPHICS_H
#define _GRAPHICS_H

#ifdef _LINUX
    #include <GL/glut.h>
    #include <GL/gl.h>
    #include <GL/glext.h>
#endif //_LINUX

#ifdef _OSX
    #include <glut.h>
    #include <OpenGL/gl.h>
    #include <OpenGL/glext.h>
#endif //_OSX

#endif //_GRAPHICS_H