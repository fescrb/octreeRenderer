#ifndef _SOURCE_FILE_MANAGER_H
#define _SOURCE_FILE_MANAGER_H

#include <map>

class SourceFile;

class SourceFileManager {
	public:
		/**
		 * Contructor that creates a SourceFile manager where the
		 * shader location is <path_to_binary>/shaders.
		 */
		explicit					 SourceFileManager();
		explicit					 SourceFileManager(char* shaderLocation);
									~SourceFileManager();

		SourceFile					*openSource(const char* name);

		static SourceFile			*getSource(const char* name);
		static void					 setDefaultInstance(SourceFileManager *defaultInstance);

	private:
		static SourceFileManager  	*m_pDefaultInstance;

		char						*m_sShaderLocation;
		std::map<const char*,SourceFile*>
									 m_map;
};

#endif //_SOURCE_FILE_MANAGER_H
