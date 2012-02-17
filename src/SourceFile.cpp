#include "SourceFile.h"

#include <fstream>
#include <string>
#include <cstdlib>

SourceFile::SourceFile(const char* path):
	m_sSource(0) {

}

SourceFile::~SourceFile() {
	if(m_sSource)
		free(m_sSource);
}
