#include "Attributes.h"

Attributes::Attributes(){

}

void Attributes::setAttributes(char red,
							   char green,
							   char blue,
							   char alpha) {
	m_red = red;
	m_green = green;
	m_blue = blue;
	m_alpha = alpha;
}


unsigned int Attributes::getSize() {
	return sizeof(m_red)+sizeof(m_green)+sizeof(m_blue)+sizeof(m_alpha);
}
