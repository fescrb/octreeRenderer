#include "Program.h"

#include <cstdio>

Program::Program(Shader vertexShader, Shader fragmentShader) {
    m_programID = glCreateProgram();
    
    glAttachShader(m_programID, vertexShader);
    glAttachShader(m_programID, fragmentShader);
    
    glLinkProgram(m_programID);
    
    GLint status;
    glGetProgramiv(m_programID, GL_LINK_STATUS, &status);
    
    if(!status) {
        
        printf("Error linking shader.\n");
    }
    
    char log[512];

    glGetProgramInfoLog(m_programID, 512, NULL, log);
        
    printf("Program link log:\n%s\n", log);
}