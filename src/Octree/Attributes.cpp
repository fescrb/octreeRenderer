#include "Attributes.h"

#include <cmath>

#include <stdio.h>

Attributes::Attributes() 
:   m_has_normal(false) {

}

void Attributes::setColour(char red,
                           char green,
                           char blue,
                           char alpha) {
	m_red = red;
	m_green = green;
	m_blue = blue;
	m_alpha = alpha;
}

void Attributes::setNormal(float x,
                           float y,
                           float z ) {
    m_has_normal=true;
    float range =127.0f; // Max value of a 7 bit unsigned integer.
    float step = 1.0f/range;
    m_x = x/step;
    m_y = y/step;
    m_z = z/step;
    //printf("range %f step %f %d\n", range, step);
    //printf("x %f %d y %f %d z %f %d\n", x, m_x, y, m_y, z, m_z);
    m_w = 0;
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
