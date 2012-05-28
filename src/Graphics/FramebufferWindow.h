#ifndef _FRAMEBUFFER_WINDOW_H
#define _FRAMEBUFFER_WINDOW_H

#include "Graphics.h"
#include "Rect.h"

struct framebuffer_window {
    rect         window;
    GLuint       texture;
};

#endif //_FRAMEBUFFER_WINDOW_H
