#ifndef _DEBUG_UTILS_H
#define _DEBUG_UTILS_H

/**
 * Prints the immediate stack trace, ommiting the call to this
 * funtion.
 * @param levels_to_print The amount of calls we should print.
 * @param levels_ommited The amount of levels of the stack you
 * want to omit from printing. Default is 0.
 */
void printStackTrace(int levels_to_print = 5, int levels_ommited = 0);

#endif //_DEBUG_UTILS_H
