#ifndef _ATTRIBUTES_H
#define _ATTRIBUTES_H

#include "Vector.h"

class Attributes {

	public:
		explicit 				 Attributes();

		void					 setColour(unsigned char red,
                                           unsigned char green,
										   unsigned char blue,
										   unsigned char alpha);
        void                     setColour(float4 colour);

        float4                   getColour();

        /**
         * All floats must be [-1,1]
         */
        void                     setNormal(float x,
                                           float y,
                                           float z );

        float4                   getNormal();

		unsigned int			 getSize(); // Return size in chars

		char*					 flatten(char* buffer);

	private:
        unsigned char            m_red, m_green, m_blue, m_alpha;
        char                     m_x, m_y, m_z, m_w;

        /**
         * Converts the 8-bit colours to a single RGB565 short
         */
        unsigned short           getColourAsShort();
        unsigned short           getNormalAsShort();

};

#endif //_ATTRIBUTES_H
