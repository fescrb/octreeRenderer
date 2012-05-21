#ifndef _ATTRIBUTES_H
#define _ATTRIBUTES_H

class Attributes {

	public:
		explicit 				 Attributes();
		
		void					 setColour(char red,
                                           char green,
										   char blue,
										   char alpha);
        
        /**
         * All floats must be [-1,1]
         */
        void                     setNormal(float x,
                                           float y,
                                           float z );

		unsigned int			 getSize(); // Return size in chars

		char*					 flatten(char* buffer);

	private:
		char					 m_red, m_green, m_blue, m_alpha;
        char                     m_x, m_y, m_z, m_w;
        
        bool                     m_has_normal;


};

#endif //_ATTRIBUTES_H
