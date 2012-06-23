#include "GLUtils.h"

#include "Graphics.h"

#include <cstdio>

void checkGLerror() {
    GLenum error = glGetError();
    if(error != GL_NO_ERROR) {
        switch(error) {
            case GL_INVALID_ENUM:
                printf("GL_INVALID_ENUM\n"); break;
            case GL_INVALID_VALUE:
                printf("GL_INVALID_VALUE\n"); break;
            case GL_INVALID_OPERATION:
                printf("GL_INVALID_OPERATION\n"); break;
            default:
                printf("GlError unknown\n");
        }
    }
}
