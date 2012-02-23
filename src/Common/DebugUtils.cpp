#include "DebugUtils.h"

#include <cstdlib>
#include <cstdio>

#include "execinfo.h"

void printStackTrace(int levels_to_print, int levels_ommited) {
	// We omit the call to printStackTrace
	levels_ommited+=1;
	
	
	void **array = (void**) malloc (sizeof(void*)*(levels_to_print+levels_ommited));
	
	int obtained = backtrace(array, (levels_to_print+levels_ommited));
	obtained -= levels_ommited;
	
	if(obtained<0) {
		printf("Error: We have ommited too many levels"); 
	}
	
	array = &array[levels_ommited];
	
	char** strings = backtrace_symbols(array, obtained);
	
	for(int i = 0; i < obtained; i++)
		printf("%s\n", strings[i]);
}