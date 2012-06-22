#include "Texture.h"

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
    //TODO
}

