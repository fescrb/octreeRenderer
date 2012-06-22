#ifndef _PNG_READER_H
#define _PNG_READER_H

class Texture;

class PNGReader {
    public:
                         PNGReader(const char* filename);
        virtual         ~PNGReader();
        
        Texture         *readImage();
        
    private:
        char            *m_filename;
};

#endif //_PNG_READER_H