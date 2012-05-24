#include "OBJFileReader.h"

#include <cstring>
#include <fstream>
#include <cmath>

OBJFileReader::OBJFileReader(const char* filename) {
    m_filename = (char*)malloc(strlen(filename)+1);
    strcpy(m_filename, filename);
}
        
mesh OBJFileReader::getMesh() {
    std::ifstream in;
    in.open(m_filename);
    
    mesh objMesh;
    OBJFileData* data = new OBJFileData;
    // Indexing is different, so we pad.
    data->vertexList.push_back(float4());
    data->normalList.push_back(float4());
    
    int normalCounter = 0;
    
    while(!in.eof()) {
        char line[1024];

        in.getline(line, 1024);
        
        float4 tmp;
        
        switch(getLineType(line)){
            case TYPE_VERTEX_DECLARATION:
                data->vertexList.push_back(getVertexFromLine(line)); 
                break;
            case TYPE_NORMAL_DECLARATION:
                tmp = getVertexFromLine(line); // tmp = normal
                tmp.setW(0.0f);
                data->normalList.push_back(tmp);
                break;
            case TYPE_FACE_DECLARATION:objMesh.appendTriangles(getFacesFromLine(line,data));
            default:
                ; // We do nothing.
        }
    }
    
    return objMesh;
}

OBJFileReader::LineType OBJFileReader::getLineType(const char* line) {
    switch(line[0]) {
        case '#':
            return TYPE_COMMENT;
        case 'v':
            if(line[1] == ' ')
                return TYPE_VERTEX_DECLARATION;
            if(line[1] == 'n')
                return TYPE_NORMAL_DECLARATION;
        case 'f':
            return TYPE_FACE_DECLARATION;
        default:
            return TYPE_UNKOWN;
    }
}

float4 OBJFileReader::getVertexFromLine(char* line) {
    strtok(line, " ");
    char* x = strtok(NULL, " ");
    char* y = strtok(NULL, " ");
    char* z = strtok(NULL, " \n\0");
    return float4(atof(x),atof(y),atof(z),1.0f);
}

std::vector<triangle> OBJFileReader::getFacesFromLine(char* line, const OBJFileData* data) {
    std::vector<triangle> triangles;
    
    int vertex_count = countCharacter(' ', line);
    
    char **all_vertices = (char**) malloc (sizeof(char*)*vertex_count + 1);
    
    strtok(line, " ");
    
    for(int i = 0; i < vertex_count-1; i++) {
        all_vertices[i] = strtok(NULL, " ");
    }
    
    all_vertices[vertex_count-1] = strtok(NULL, " \n\0");
    
    for(int vert = 1; vert < vertex_count - 1; vert++) {
        char* vertices[3] = {all_vertices[0], all_vertices[vert], all_vertices[vert+1]};
        
        int   vertex_indices[3] = {-1, -1, -1};
        int   normal_indices[3] = {-1, -1, -1};
        
        // Boolean containing whether we haven't declared a normal.
        bool  undeclared_normal = false;
        
        for(int i = 0; i < 3; i++) {
            int size = strlen(vertices[i]);
            std::string data[] = {std::string(), std::string() ,std::string()};
            int counter = 0; // The amount of '/' we have encountered.
            
            for(int j = 0; j < size; j++) {
                if(vertices[i][j] == '/') {
                    counter++;
                } else {
                    data[counter].push_back(vertices[i][j]);
                }
            }
            
            vertex_indices[i] = atoi(data[0].data());
            if(counter == 2)
                normal_indices[i] = atoi(data[2].data());
            else
                undeclared_normal = true;
        }
        
        vertex first(data->vertexList[vertex_indices[0]]);
        vertex secnd(data->vertexList[vertex_indices[1]]);
        vertex third(data->vertexList[vertex_indices[2]]);
        
        triangle new_triangle(first, secnd, third);
        
        if(undeclared_normal) {
            new_triangle.generateNormals();
        } else {
            new_triangle[0].setNormal(data->normalList[normal_indices[0]]);
            new_triangle[1].setNormal(data->normalList[normal_indices[1]]);
            new_triangle[2].setNormal(data->normalList[normal_indices[2]]);
        }
        
        triangles.push_back(new_triangle);
    }
    
    return triangles;
}

int OBJFileReader::countCharacter(char character, const char* line) {
    int counter = 0;
    int lineSize = strlen(line);
    
    for(int i = 0; i < lineSize; i++)
        if(line[i] == character)
            counter++;
    
    return counter;
}