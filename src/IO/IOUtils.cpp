#include "IOUtils.h"

#include <cerrno>
#include <cstdio>

bool isIOError(int return_value) {
    return return_value==-1;
}

void printIOError() {
    switch(errno) {
        case EAGAIN:
            printf("IO Error: EAGAIN\n");
            break;
        case EBADF:
            printf("IO Error: EBADF\n");
            break;
        case EINTR:
            printf("IO Error: EINTR\n");
            break;
        case EIO:
            printf("IO Error: EIO\n");
            break;
        case EINVAL:
            printf("IO Error: EINVAL\n");
            break;
        case EACCES:
            printf("IO Error: EACCES\n");
            break;
        case EEXIST:
            printf("IO Error: EEXIST\n");
            break;
        case EISDIR:
            printf("IO Error: EISDIR\n");
            break;
        case EMFILE:
            printf("IO Error: EMFILE\n");
            break;
        case ENFILE:
            printf("IO Error: ENFILE\n");
            break;
        case ENOENT:
            printf("IO Error: ENOENT\n");
            break;
        case ENOSPC:
            printf("IO Error: ENOSPC\n");
            break;
        case ENXIO:
            printf("IO Error: ENXIO\n");
            break;
        case EROFS:
            printf("IO Error: EROFS\n");
            break;
        default:
            printf("IO Error Unkown\n");
            break;
    }
}