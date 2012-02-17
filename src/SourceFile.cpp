#include "SourceFile.h"

#include <fstream>
#include <string>
#include <cstdlib>
#include <cstring>

SourceFile::SourceFile(const char* path):
	m_sSource(0) {
	std::ifstream in;
	in.open(path);

	std::string all_text;

	while(!in.eof()) {
		char line[1024];

		in.getline(line, 1024);

		all_text.append(line);
	}

	m_sSource = (char*) malloc(all_text.length()+1);

	strcpy(m_sSource, all_text.data());
}

SourceFile::~SourceFile() {
	if(m_sSource)
		free(m_sSource);
}
