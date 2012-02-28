#include "Shader.h"

#include "SourceFileManager.h"
#include "SourceFile.h"

#include <cstdio>

Shader::Shader(GLenum type, const char* fileName) {
    m_shaderID = glCreateShader(type);
    SourceFile *sourceFile = SourceFileManager::getSource(fileName);
    const GLchar** source = sourceFile->getSource();
    
    glShaderSource(m_shaderID, sourceFile->getNumLines(), source, NULL);
    
    glCompileShader(m_shaderID);
    
    GLint status;
    glGetShaderiv(m_shaderID, GL_COMPILE_STATUS, &status);
    
    if(!status) {
        printf("Error compiling shader from file %s.\n", fileName);
    }
    
    char log[512];
        
    glGetShaderInfoLog(m_shaderID, 512, NULL, log);
    
    printf("Shader compile log:\n%s\n", log);
}