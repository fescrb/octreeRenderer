#ifndef _ATTRIBUTES_H
#define _ATTRIBUTES_H

class Attributes {

	public:
		explicit 				 Attributes();
		
		void					 setAttributes(char red,
											   char green,
											   char blue,
											   char alpha);

		unsigned int			 getSize(); // Return size in chars

		char*					 flatten(char* buffer);

	private:
		char					 m_red, m_green, m_blue, m_alpha;


};

#endif //_ATTRIBUTES_H
