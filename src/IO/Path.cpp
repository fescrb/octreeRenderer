#include "Path.h"

#include <sys/stat.h>

int make_directory(char* name) {
    return mkdir(name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}