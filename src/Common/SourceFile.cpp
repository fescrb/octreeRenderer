#include "SourceFile.h"

#include <fstream>
#include <string>
#include <cstdlib>
#include <cstring>

SourceFile::SourceFile(const char* path) {
	std::ifstream in;
	in.open(path);

	while(!in.eof()) {
		char line[1024];

		in.getline(line, 1024);
		
		char *sLine = (char*) malloc(strlen(line)+1);
		strcpy(sLine, line);
		
		m_vsSourceLines.push_back(sLine);
	}
}

SourceFile::~SourceFile() {
	unsigned int num = getNumLines();
	for(int i = 0; i<num; i++)
		free (m_vsSourceLines[i]);
	m_vsSourceLines.clear();
}

const char** SourceFile::getSource() {
	return (const char**)&m_vsSourceLines[0];
}

unsigned int SourceFile::getNumLines() {
	return m_vsSourceLines.size();
}
