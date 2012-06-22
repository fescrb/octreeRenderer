#include "PNGReader.h"

#include <cstdlib>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#include <png.h>

#include "Texture.h"

void print_colour_type(png_byte colour_type) {
    switch(colour_type) {
        case PNG_COLOR_TYPE_GRAY:
            printf("PNG_COLOR_TYPE_GRAY"); break;
        case PNG_COLOR_TYPE_PALETTE:
            printf("PNG_COLOR_TYPE_PALETTE"); break;
        case PNG_COLOR_TYPE_RGB:
            printf("PNG_COLOR_TYPE_RGB"); break;
        case PNG_COLOR_TYPE_RGB_ALPHA:
            printf("PNG_COLOR_TYPE_RGB_ALPHA"); break;
        case PNG_COLOR_TYPE_GRAY_ALPHA:
            printf("PNG_COLOR_TYPE_GRAY_ALPHA"); break;
        default:
            printf("Color type unknown");
    }
}

PNGReader::PNGReader(const char* filename) {
    m_filename = (char*)malloc(strlen(filename)+1);
    strcpy(m_filename, filename);
}

PNGReader::~PNGReader() {

}

Texture *PNGReader::readImage() {
    char header[8];
    
    FILE *fp = fopen(m_filename, "rb");
    if(!fp) {
        printf("PNGReader: Problem reading %s\n", m_filename); exit(1);
    }
    fread(header, 1, 8, fp);
    
    if(png_sig_cmp((unsigned char*)header, 0, 8)) {
        printf("PNGReader: %s not a png file\n", m_filename); exit(1);
    }
    
    // Initialization
    png_structp png_pointer = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    
    png_infop png_info_pointer = png_create_info_struct(png_pointer);
    
    png_init_io(png_pointer, fp);
    png_set_sig_bytes(png_pointer, 8);
    png_read_info(png_pointer, png_info_pointer);
    
    int width = png_get_image_width(png_pointer, png_info_pointer);
    int height = png_get_image_height(png_pointer, png_info_pointer);
    png_byte colour_type = png_get_color_type(png_pointer, png_info_pointer);
    png_byte bit_depth = png_get_bit_depth(png_pointer, png_info_pointer);
    
    printf("Image %s width %d height %d colour_type", m_filename, width, height);
    print_colour_type(colour_type);
    printf(" bit depth %d\n", bit_depth);
    
    
    int number_of_passes = png_set_interlace_handling(png_pointer);
    png_read_update_info(png_pointer, png_info_pointer);
    
    int image_channels;
    
    switch(colour_type) {
        case PNG_COLOR_TYPE_RGB:
            image_channels = 3;
            break;
        default:
            printf("PNGReader Error: Image not RGB\n");
            exit(1);
    }
    
    // Read
    unsigned char *data = (unsigned char*) malloc (sizeof(unsigned char)*height*width*image_channels+1);
    png_bytep row_pointers[height];
    
    unsigned int rowBytes = png_get_rowbytes(png_pointer, png_info_pointer);
    
    for(int i = 0; i < height; i ++) {
        row_pointers[i] = data + (i*rowBytes);
    }
    
    png_read_image(png_pointer, row_pointers);
    png_read_end(png_pointer, NULL);
    
    /*
     * Image seems to get image upside-down. With RGB backwards.
     */
    
    Texture *input = new Texture(width, height, Image::RGB, (char*)data);
    input->toBMP("test_out.bmp");
    
    return input;
}

