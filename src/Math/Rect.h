#ifndef _RECT_H
#define _RECT_H

#include "Vector.h"

struct rect {
    public:
                         rect(){};
        explicit         rect(vector::int2 origin, vector::int2 size): m_origin(origin), m_size(size){};
        explicit         rect(int x, int y, int width, int height):m_origin(x,y), m_size(width,height){};
    
        vector::int2     getOrigin() const{
            return m_origin;
        }
    
        vector::int2     getSize() const{
            return m_size;
        }
        
        int              getX() const {
            return m_origin.getX();
        }
        
        int              getY() const {
            return m_origin.getY();
        }
        
        int              getWidth() const{
            return m_size.getX();
        }
        
        int              getHeight() const {
            return m_size.getY();
        }
        
        void             setX(int newX) {
            m_origin.setX(newX);
        }
        
        void             setY(int newY) {
            m_origin.setY(newY);
        }
        
        void             setWidth(int newWidth) {
            m_size.setX(newWidth);
        }
        
        void             setHeight(int newHeight) {
            m_size.setY(newHeight);
        }
    private:
        vector::int2     m_origin, m_size;
};

#endif //_RECT_H
