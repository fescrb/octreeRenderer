#include "Image.h"

#include <cstdlib>

#include <cstdio>

Image::Image(unsigned int width, unsigned int length)
:	m_width(width),
	m_height(length) {
	m_pData = (char*) malloc (3 * m_width * m_height);
	
	for(int i = 0; i < 3 * m_height * m_width; i++) {
		m_pData[i] = 254;
	}
}

Image::Image(unsigned int width, 
			 unsigned int length, 
			 ImageFormat buffer_format,  
			 const char* buffer) 
:	m_width(width),
	m_height(length) {
	m_pData = (char*) malloc (3 * m_width * m_height);
	
	char* data = m_pData;
	
	while(data != (m_pData + ( 3 * m_width * m_height ))) {
		data[0] = buffer[0];
		data++; buffer++;
		data[0] = buffer[0];
		data++; buffer++;
		data[0] = buffer[0];
		data++; buffer++;
		if( buffer_format == RGBA) 
			buffer++;
	} 
}

Image::Image(Image* image) {
	
}

#pragma pack(push,1)
typedef struct {
     char magicword[2];
     long fileSize;
     long padding;
     long headerSize;
     long infoSize;
     long width;
     long height;
     short planes;
     short bpp;
     long compression;
     long compressedSize;
     long xPixelsPerMeter;
     long yPixelsPerMeter;
     long colourUsed;
     long colourImportant;
} BMPHeader;
#pragma pack(pop)
		
void Image::toBMP(const char* filename) {
	BMPHeader head;
	
	int linePadding = (3 * m_width) % 4;
	
	head.magicword[0] = 'B';
	head.magicword[1] = 'M';
	head.headerSize = 54;
	head.padding = 0;
	head.fileSize = head.headerSize + ( ( 3 * m_width * m_height ) + ( linePadding * m_height ));
	head.infoSize = 40;
	head.width = m_width;
	head.height = m_height;
	head.planes = 1;
	head.bpp = 24;
	head.compression = 0;
	head.compressedSize = 0;
	head.xPixelsPerMeter = 0;
	head.yPixelsPerMeter = 0;
	head.colourUsed = 0;
	head.colourImportant = 0;
	
	// Print data to check.
	printf("padding %d, header %d, sizeof %d, file %d\n", linePadding, head.headerSize, sizeof(head), head.fileSize);
	
	FILE* bmpFile = fopen(filename, "wb");
	fwrite(&head, sizeof(head), 1, bmpFile);
	
	//Remember, this won work unless width % 4 = 0
	fwrite(m_pData, 1,( 3 * m_width * m_height ), bmpFile );
	
	fclose(bmpFile);
}