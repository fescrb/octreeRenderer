#ifndef _RECT_H
#define _RECT_H

#include "Vector.h"

struct rect {
    public:
                         rect(){};
        explicit         rect(int2 origin, int2 size): m_origin(origin), m_size(size){};
        explicit         rect(int x, int y, int width, int height):m_origin(x,y), m_size(width,height){};
    
        int2&            getOrigin(){
            return m_origin;
        }
    
        int2&            getSize(){
            return m_size;
        }
    private:
        int2             m_origin, m_size;
};

#endif //_RECT_H
