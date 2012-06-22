#ifndef _TEXTURE_H
#define _TEXTURE_H

#include "Graphics.h"

#include "Image.h"

class Texture
:   public Image {
    public:
                         Texture(unsigned int width, 
                                 unsigned int height, 
                                 ImageFormat buffer_format, 
                                 const char* buffer);
        virtual         ~Texture();
        
        GLuint           getGLTexture();
        
    private:
        void             transferToGL();
        
        GLuint           m_texture;
};

#endif //_TEXTURE_H