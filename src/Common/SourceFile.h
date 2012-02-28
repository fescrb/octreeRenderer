#ifndef _SOURCE_FILE_H
#define _SOURCE_FILE_H

#include <vector>
#include <cstdlib>

class SourceFile {
	public:
		explicit 			 SourceFile(const char* path);
							~SourceFile();
							
		const char 		   **getSource();
		unsigned int	     getNumLines();
		std::vector<size_t>  getLineLength();

	private:
		std::vector<char*>	m_vsSourceLines;
		unsigned int		m_lines;

};

#endif //_SOURCE_FILE_H
