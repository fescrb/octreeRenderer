#include "SourceFileManager.h"

#include "SourceFile.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <unistd.h>

#ifdef _OSX
#include <mach-o/dyld.h>
#endif //_OSX

using namespace std;

SourceFileManager *SourceFileManager::m_pDefaultInstance = 0;

SourceFileManager::SourceFileManager() {

#ifdef _LINUX
	// We get the pid.
	pid_t pid;
	pid = getpid();

	// We use it to find the symlink location.
	char path[512];
	sprintf(path,"/proc/%d/exe", pid);

	int size = readlink(path, path, 512);
#endif //_LINUX
#ifdef _OSX
    char* path = (char*) malloc(512);;
    uint32_t size = 512;
    
    _NSGetExecutablePath(path,&size);
#endif _OSX
    
    size--;
    while(path[size] != '/')
		size--;
    
    strcpy(path+size, "/shaders/");
	size += strlen("/shaders/");
    
	// path is not null terminated, we make it so.
	path[size] = 0;
    
    m_sShaderLocation = (char*) malloc(size);
    strcpy(m_sShaderLocation,path);
}

SourceFileManager::SourceFileManager(char* shaderLocation)
:	m_sShaderLocation(shaderLocation){

}

SourceFileManager::~SourceFileManager() {
	map<const char*,SourceFile*>::iterator it = m_map.begin();

	while(it != m_map.end()) {
		delete it->second;
		it++;
	}
}

SourceFile* SourceFileManager::openSource(const char* name) {
	if(!m_map.count(name)) {
		char path[256];
		sprintf(path,"%s%s", m_sShaderLocation, name);
        //printf("res %s\n", path);
		m_map[name] = new SourceFile(path);
	}
	return m_map[name];
}

SourceFile* SourceFileManager::getSource(const char* name) {
	if(!m_pDefaultInstance)
		m_pDefaultInstance = new SourceFileManager();
	return m_pDefaultInstance->openSource(name);
}

void SourceFileManager::setDefaultInstance(SourceFileManager *defaultInstance) {
	m_pDefaultInstance = defaultInstance;
}
