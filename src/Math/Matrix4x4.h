#ifndef _MATRIX_4_4_H
#define _MATRIX_4_4_H

#include "Vector.h"

#include "AABox.h"

template <class t>
struct matrix4x4 {
    public:
        matrix4x4(){}
                             matrix4x4(const matrix4x4<t>& other)
        :	m_column1(other.m_column1),
        m_column2(other.m_column2),
        m_column3(other.m_column3),
        m_column4(other.m_column4) {
        }
        
        explicit		 	 matrix4x4(const vector4<t>& col1,
                                       const vector4<t>& col2,
                                       const vector4<t>& col3,
                                       const vector4<t>& col4)
        :	m_column1(col1),
        m_column2(col2),
        m_column3(col3),
        m_column4(col4) {
        }
        
        inline matrix4x4 	 operator+(const matrix4x4& rightHandSide) {
            return matrix4x4(m_column1+rightHandSide.m_column1,
                            m_column2+rightHandSide.m_column2,
                            m_column3+rightHandSide.m_column3,
                            m_column4+rightHandSide.m_column4 );
        }
        
        inline matrix4x4 	 operator-(const matrix4x4& rightHandSide) {
            return matrix4x4(m_column1-rightHandSide.m_column1,
                            m_column2-rightHandSide.m_column2,
                            m_column3-rightHandSide.m_column3,
                            m_column4-rightHandSide.m_column4 );
        }
        
        inline matrix4x4 	 operator*(const matrix4x4& rightHandSide) {
            return matrix4x4(operator*(rightHandSide.m_column1),
                            operator*(rightHandSide.m_column2),
                            operator*(rightHandSide.m_column3),
                            operator*(rightHandSide.m_column4) );
        }
        
        inline vector4<t> 	 operator*(const vector4<t>& rightHandSide) {
            vector4<t> x = rightHandSide.replicateX();
            vector4<t> y = rightHandSide.replicateY();
            vector4<t> z = rightHandSide.replicateZ();
            vector4<t> w = rightHandSide.replicateW();
            return (m_column1*x) + (m_column2*y) + (m_column3*z) + (m_column4*w);
        }
        
        
        inline vertex        operator*(const vertex& rhs) {
            vertex vert(operator*(rhs.getPosition()), operator*(rhs.getNormal()), rhs.getColour());
            
            return vert;
        }
        
        inline triangle      operator*(const triangle& rhs) {
            triangle tri(
                operator*(rhs.getVertex(0)),
                operator*(rhs.getVertex(1)),
                operator*(rhs.getVertex(2))
            );
            
            return tri;
        }
        
        inline mesh          operator*(const mesh& rhs) {
            mesh new_mesh();
            
            std::vector<triangle> triangles = rhs.getTriangleList();
            
            for(int i = 0; i < triangles.size(); i++)
                new_mesh.appendTriangle(operator*(triangles[i]));
            
            return new_mesh;
        }
        
        inline aabox         operator*(const aabox& rhs) {
            mesh new_mesh();
            
            std::vector<triangle> triangles = rhs.getTriangleList();
            
            for(int i = 0; i < triangles.size(); i++)
                new_mesh.appendTriangle(operator*(triangles[i]));
            
            return new_mesh;
        }
        
        inline matrix4x4 	 transpose() {
            vector4<t> col1Copy(m_column1), 
            col2Copy(m_column2), 
            col3Copy(m_column3), 
            col4Copy(m_column4);
            m_column1 = vector4<t>(col1Copy[0],col2Copy[0],col3Copy[0],col4Copy[0]);
            m_column2 = vector4<t>(col1Copy[1],col2Copy[1],col3Copy[1],col4Copy[1]);
            m_column3 = vector4<t>(col1Copy[2],col2Copy[2],col3Copy[2],col4Copy[2]);
            m_column4 = vector4<t>(col1Copy[3],col2Copy[3],col3Copy[3],col4Copy[3]);
        }
        
        inline matrix4x4 	 getTransposed() const {
            matrix4x4 transposed(*this);
            transposed.transpose();
            return transposed;
        }
        
        /* *************************
         * Static member functions *
         ***************************/
        
        static inline matrix4x4		 
        identityMatrix() {
            return matrix4x4(vector4<t>( 1, 0, 0, 0),
                            vector4<t>( 0, 1, 0, 0),
                            vector4<t>( 0, 0, 1, 0),
                            vector4<t>( 0, 0, 0, 1) );
        }
        
        static inline matrix4x4		 
        translationMatrix(t x, t y, t z) {
            return matrix4x4(vector4<t>( 1, 0, 0, 0),
                            vector4<t>( 0, 1, 0, 0),
                            vector4<t>( 0, 0, 1, 0),
                            vector4<t>( x, y, z, 1) );
        }
        
        static inline matrix4x4		 
        rotationAroundX(F32 radians) {
            return matrix4x4(vector4<t>( 1,             0,            0, 0),
                            vector4<t>( 0,  cos(radians), sin(radians), 0),
                            vector4<t>( 0, -sin(radians), cos(radians), 0),
                            vector4<t>( 0,             0,            0, 1) );
        }
        
        static inline matrix4x4		 
        rotationAroundY(F32 radians) {
            return matrix4x4(vector4<t>( cos(radians), 0, -sin(radians), 0),
                            vector4<t>(            0, 1,             0, 0),
                            vector4<t>( sin(radians), 0,  cos(radians), 0),
                            vector4<t>(            0, 0,             0, 1) );
        }
        
        static inline matrix4x4		 
        rotationAroundZ(F32 radians) {
            return matrix4x4(vector4<t>( cos(radians), sin(radians), 0, 0),
                            vector4<t>(-sin(radians), cos(radians), 0, 0),
                            vector4<t>(            0,            0, 1, 0),
                            vector4<t>(            0,            0, 0, 1) );
        }
    
        static inline matrix4x4		 
        rotationAroundVector(vector4<t> vector, F32 radians) {
            vector2<t> twoComponenVector(vector[0], vector[1]);
            vector2<t> twoComponenYAxis(0.0f, 1.0f);
            
            float alpha = acos(dot(twoComponenVector, twoComponenYAxis));
            
            matrix4x4<t> rotateAroundZ = rotationAroundZ(alpha);
            matrix4x4<t> rotateAroundZneg = rotationAroundZ(-alpha); 
            
            // We now rotate up to be equal to k.
            vector = rotateAroundZ*vector;
            
            float beta = acos(dot(vector, vector4<t>(0.0f,0.0f,1.0f,0.0f)));
            
            matrix4x4<t> rotateAroundX = rotationAroundX(beta);
            matrix4x4<t> rotateAroundXneg = rotationAroundX(-beta); 
            
            return rotateAroundZneg * ( rotateAroundXneg * ( matrix4x4<t>::rotationAroundZ(radians) * ( rotateAroundX * rotateAroundZ ) ) );
        }
        
        static inline matrix4x4		 
        scalingMatrix(F32 scalingFactor) {
            return matrix4x4(vector4<t>( scalingFactor,             0,             0, 0),
                            vector4<t>(             0, scalingFactor,             0, 0),
                            vector4<t>(             0,             0, scalingFactor, 0),
                            vector4<t>(             0,             0,             0, 1) );
        }
        
    private:
        vector4<t> 	 		 m_column1;
        vector4<t>  		 m_column2;
        vector4<t>  	 	 m_column3;
        vector4<t> 	 		 m_column4;
};

#endif //_MATRIX_4_4_H