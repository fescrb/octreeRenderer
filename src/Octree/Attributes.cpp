#include "Attributes.h"

#include <cmath>
#include <stdio.h>

#include "MathUtil.h"

Attributes::Attributes() 
:   m_has_normal(false) {

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
    printf("colors %d\n",m_red);
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
    m_has_normal=true;
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
    unsigned int size = sizeof(m_red)+sizeof(m_green)+sizeof(m_blue)+sizeof(m_alpha);
    if(m_has_normal)
        size += sizeof(m_x)+sizeof(m_y)+sizeof(m_z)+sizeof(m_w);
	return size;
}

char* Attributes::flatten(char* buffer) {
	buffer[0] = m_red;
	buffer++;
	buffer[0] = m_green;
	buffer++;
	buffer[0] = m_blue;
	buffer++;
	buffer[0] = m_alpha;
	buffer++;
    if(m_has_normal) {
        buffer[0] = m_x;
        buffer++;
        buffer[0] = m_y;
        buffer++;
        buffer[0] = m_z;
        buffer++;
        buffer[0] = m_w;
        buffer++;
    }

	return buffer;
}
