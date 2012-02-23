#ifndef _SOURCE_FILE_H
#define _SOURCE_FILE_H

#include <vector>

class SourceFile {
	public:
		explicit 			 SourceFile(const char* path);
							~SourceFile();
							
		const char 		   **getSource();
		unsigned int	     getNumLines();

	private:
		std::vector<char*>	m_vsSourceLines;
		unsigned int		m_lines;

};

#endif //_SOURCE_FILE_H
