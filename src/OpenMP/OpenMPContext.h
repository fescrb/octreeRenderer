#ifndef _OPENMP_CONTEXT_H
#define _OPENMP_CONTEXT_H

#include "SerialContext.h"

class OpenMPContext 
:   public SerialContext {
    public:
        explicit         OpenMPContext();
};

#endif //_OPENMP_CONTEXT_H
