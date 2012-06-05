#include "Attributes.h"

#include <cmath>
#include <stdio.h>

#include "MathUtil.h"

Attributes::Attributes() {

}

void Attributes::setColour(unsigned char red,
                           unsigned char green,
                           unsigned char blue,
                           unsigned char alpha) {
	m_red = red;
	m_green = green;
	m_blue = blue;
	m_alpha = alpha;
}

void Attributes::setColour(float4 colour) {
    setColour(
        float_to_8_bit_unsigned_fixed_point(colour[0]),
        float_to_8_bit_unsigned_fixed_point(colour[1]),
        float_to_8_bit_unsigned_fixed_point(colour[2]),
        float_to_8_bit_unsigned_fixed_point(colour[3])
    );
    //printf("colors %d\n",m_red);
}

float4 Attributes::getColour() {
    return float4( unsigned_8bit_fixed_point_to_float(m_red),
                   unsigned_8bit_fixed_point_to_float(m_green),
                   unsigned_8bit_fixed_point_to_float(m_blue),
                   unsigned_8bit_fixed_point_to_float(m_alpha)
    );
}

void Attributes::setNormal(float x,
                           float y,
                           float z ) {
    m_x = float_to_fixed_point_8bit(x);
    m_y = float_to_fixed_point_8bit(y);
    m_z = float_to_fixed_point_8bit(z);
    m_w = 0;
}

float4 Attributes::getNormal() {
    return float4( fixed_point_8bit_to_float(m_x),
                   fixed_point_8bit_to_float(m_y),
                   fixed_point_8bit_to_float(m_z),
                   fixed_point_8bit_to_float(m_w)
    );
}


unsigned int Attributes::getSize() {
    //unsigned int size = sizeof(m_red)+sizeof(m_green)+sizeof(m_blue)+sizeof(m_alpha);
    //if(m_has_normal)
        //size += sizeof(m_x)+sizeof(m_y)+sizeof(m_z)+sizeof(m_w);
	return sizeof(char)*4;
}

unsigned short Attributes::getColourAsShort() {
    unsigned short rgb_565 = 0;

    unsigned short red = m_red  & ~7;
    rgb_565 |= (red << 8) ;

    unsigned short green = m_green  & ~3;
    rgb_565 |= (green << 3) ;

    unsigned short blue = m_blue & ~7;
    rgb_565 |= (blue >> 3) ;

    //printf("red %d green %d blue %d short %d\n", m_red, m_green, m_blue, rgb_565);

    return rgb_565;
}

unsigned short Attributes::getNormalAsShort() {
    unsigned short normals = 0;

    unsigned short x = m_x <<8;
    normals|= x ;//<< 8;

    normals|= ((unsigned char)m_y);

    if(m_z >= 0)
        normals &= ~1;
    else
        normals |= 1;

    //printf("x %d %d y %d z %d short %d\n", m_x, x, m_y, m_z, normals);

    return normals;
}

char* Attributes::flatten(char* buffer) {
    unsigned short* buffer_short = (unsigned short*) buffer;

    buffer_short[0] = getColourAsShort();
	buffer_short[1] = getNormalAsShort();

	return buffer + 4;
}
