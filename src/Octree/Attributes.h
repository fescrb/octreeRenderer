#ifndef _ATTRIBUTES_H
#define _ATTRIBUTES_H

class Attributes {

	public:
		explicit 				 Attributes();

		unsigned int			 getSize(); // Return size in chars

	private:
		char					 m_red, m_green, m_blue, m_alpha;


};

#endif //_ATTRIBUTES_H
