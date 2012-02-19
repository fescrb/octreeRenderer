#ifndef _SOURCE_FILE_H
#define _SOURCE_FILE_H

class SourceFile {
	public:
		explicit 			 SourceFile(const char* path);
							~SourceFile();
							
		char 				*getSource();

	private:
		char*				 m_sSource;
		unsigned int		 m_length;

};

#endif //_SOURCE_FILE_H
