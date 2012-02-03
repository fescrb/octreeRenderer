#include "Attributes.h"

Attributes::Attributes(){

}

unsigned int Attributes::getSize() {
	return sizeof(m_red)+sizeof(m_green)+sizeof(m_blue)+sizeof(m_alpha);
}
