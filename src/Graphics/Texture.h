#ifndef _TEXTURE_H
#define _TEXTURE_H

#include "Graphics.h"

#include "Image.h"

#include "Vector.h"

using namespace vector;

class Texture
:   public Image {
    public:
                         Texture(unsigned int width, 
                                 unsigned int height, 
                                 ImageFormat buffer_format, 
                                 const char* buffer);
        virtual         ~Texture();
        
        GLuint           getGLTexture();
        
        float4           getColourAt(float2 coordinates) const;
        
    private:
        void             transferToGL();
        
        GLuint           m_texture;
};

#endif //_TEXTURE_H