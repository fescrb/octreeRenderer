#ifndef _IMAGE_H
#define _IMAGE_H

class Image {
	public:
		enum ImageFormat {
			RGB,
			RGBA
		};
	
		explicit 		 Image(unsigned int width, unsigned int length);
		explicit 		 Image(unsigned int width, 
							   unsigned int length, 
							   ImageFormat buffer_format, 
							   const char* buffer);
		explicit 		 Image(Image* image);
		
		void			 toBMP(const char* filename);
		
	private:
		unsigned int 	 m_width;
		unsigned int	 m_height;
		char 			*m_pData;
};

#endif //_IMAGE_H
