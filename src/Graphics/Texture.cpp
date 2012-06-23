#include "Texture.h"

#include "GLUtils.h"

#include "MathUtil.h"

Texture::Texture(unsigned int width, unsigned int height, Image::ImageFormat buffer_format, const char* buffer)
:   Image(width, height, buffer_format, buffer),
    m_texture(0) {

}

Texture::~Texture() {

}

GLuint Texture::getGLTexture() {
    if(!m_texture)
        transferToGL();
    return m_texture;
}

void Texture::transferToGL() {
    glGenTextures(1, &m_texture);
    
    checkGLerror();

    glBindTexture(GL_TEXTURE_2D, m_texture);

    glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
    
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGB,
                 m_width,
                 m_height,
                 0,
                 GL_RGB,
                 GL_UNSIGNED_BYTE,
                 m_pData);
}

#include <cstdio>

float4 Texture::getColourAt(float2 coordinates) const {
    // Textue mode is Repeat, so bring all coordinates to [0,1]
    float tex_x = coordinates.getX();
    while(tex_x>1.0f)
        tex_x-=1.0f;
    while(tex_x<0.0f)
        tex_x+=1.0f;
    
    float tex_y = coordinates.getY();
    while(tex_y>1.0f)
        tex_y-=1.0f;
    while(tex_y<0.0f)
        tex_y+=1.0f;
    
    float x = tex_x * (float)m_width;
    float y = tex_y * (float)m_height;
    
    int int_x = x;
    int int_y = y;
    
    char* data_pointer = m_pData + ((int_x + (int_y*m_width))*3);
    float red = unsigned_8bit_fixed_point_to_float(data_pointer[0]);
    float green = unsigned_8bit_fixed_point_to_float(data_pointer[1]);
    float blue = unsigned_8bit_fixed_point_to_float(data_pointer[2]);
    
    //printf("tex_x %f tex_y %f x %f y %f red %f green %f blue %f\n", tex_x, tex_y, x, y, red, green, blue);
    
    return float4(red, green, blue, 1.0f);
}
