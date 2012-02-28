#include "SourceFile.h"

#include <fstream>
#include <string>
#include <cstring>

SourceFile::SourceFile(const char* path) {
	std::ifstream in;
	in.open(path);

	while(!in.eof()) {
		char line[1024];

		in.getline(line, 1024);
		
		int size = strlen(line);
		
		line[size] = '\n';
		line[size+1] = '\0';
		
		char *sLine = (char*) malloc(size+2);
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

std::vector<size_t> SourceFile::getLineLength() {
	std::vector<size_t> sizes;
	for(int i = 0; i < m_vsSourceLines.size(); i++) 
		sizes.push_back(strlen(m_vsSourceLines[i]));
	return sizes;
}