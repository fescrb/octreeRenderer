#include "SourceFileManager.h"

#include "SourceFile.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <unistd.h>

using namespace std;

SourceFileManager *SourceFileManager::m_pDefaultInstance = 0;

SourceFileManager::SourceFileManager() {
	// We get the pid.
	pid_t pid;
	pid = getpid();

	// We use it to find the symlink location.
	char symlink[128];
	sprintf(symlink,"/proc/%d/exe", pid);

	int sizeOfBuffer = 256;
	m_sShaderLocation = (char*) malloc (sizeOfBuffer);
	int size = readlink(symlink, m_sShaderLocation, sizeOfBuffer);

	while(m_sShaderLocation[size] != '/')
		size--;

	strcpy(&m_sShaderLocation[size], "/shaders/");
	size += strlen("/shaders/");

	// m_sShaderLocation is not null terminated, we make it so.
	m_sShaderLocation[size] = 0;
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
