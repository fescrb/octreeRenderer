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
        
        char                        *getShaderLocation();

		static SourceFile			*getSource(const char* name);
		static void					 setDefaultInstance(SourceFileManager *defaultInstance);
        static SourceFileManager    *getDefaultInstance();

	private:
		static SourceFileManager  	*m_pDefaultInstance;

		char						*m_sShaderLocation;
		std::map<const char*,SourceFile*>
									 m_map;
};

#endif //_SOURCE_FILE_MANAGER_H
