/**
 * @file BasicTypes.h
 * @brief Declares basic integer and floating point types.
 * 
 * This file declares basic integer and floating point types. Using them
 * we can make sure that they have the determined number of bits.
 */

#ifndef _BASIC_TYPES_H
#define _BASIC_TYPES_H

#include <cstddef>

/** @typedef I64 
 * 
 * 8 byte signed integer.
 */
typedef long long int I64;
/** @typedef I32 
 * 
 * 4 byte signed integer.
 */
typedef int I32;
/** @typedef I16 
 * 
 * 2 byte signed integer.
 */
typedef short I16;
/** @typedef I8 
 * 
 * 1 byte signed integer.
 */
typedef char I8;

/** @typedef U64 
 * 
 * 8 byte unsigned integer.
 */
typedef unsigned long  long int U64;
/** @typedef U32 
 * 
 * 4 byte unsigned integer.
 */
typedef unsigned int U32;
/** @typedef U16
 * 
 * 2 byte unsigned integer.
 */
typedef unsigned short U16;
/** @typedef U8
 * 
 * 4 byte unsigned integer.
 */
typedef unsigned char U8;

/** @typedef F64
 * 
 * 8 byte signed floating point.
 */
typedef double F64;
/** @typedef F32 
 * 
 * 4 byte signed floating point.
 */
typedef float F32;

/* *********************************
 *   Minimum & maximim definitions
 ***********************************/

/** @def I64_MAX 
 * 
 * Maximum 8 byte signed integer value.
 */
#define I64_MAX 0x7fffffffffffffff
/** @def I64_MIN 
 * 
 * Minimum 8 byte signed integer value.
 */
#define I64_MIN (-I64_MAX-1)
/** @def I32_MAX 
 * 
 * Maximum 4 byte signed integer value.
 */
#define I32_MAX 0x7fffffff
/** @def I32_MIN 
 * 
 * Minimum 4 byte signed integer value.
 */
#define I32_MIN (-I32_MAX-1)
/** @def I16_MAX 
 * 
 * Maximum 2 byte signed integer value.
 */
#define I16_MAX 0x7fff
/** @def I16_MIN 
 * 
 * Minimum 2 byte signed integer value.
 */
#define I16_MIN (-I16_MAX-1)
/** @def I8_MAX 
 * 
 * Maximum 1 byte signed integer value.
 */
#define I8_MAX 0x7f
/** @def I8_MIN 
 * 
 * Minimum 1 byte signed integer value.
 */
#define I8_MIN (-I8_MAX-1)

/** @def U64_MAX 
 * 
 * Maximum 8 byte unsigned integer value.
 */
#define U64_MAX 0xffffffffffffffff
/** @def U64_MIN 
 * 
 * Minimum 8 byte unsigned integer value.
 */
#define U64_MIN 0x0
/** @def U32_MAX 
 * 
 * Maximum 4 byte unsigned integer value.
 */
#define U32_MAX 0xffffffff
/** @def U32_MIN 
 * 
 * Minimum 4 byte unsigned integer value.
 */
#define U32_MIN 0x0
/** @def U16_MAX 
 * 
 * Maximum 2 byte unsigned integer value.
 */
#define U16_MAX 0xffff
/** @def U16_MIN 
 * 
 * Minimum 2 byte unsigned integer value.
 */
#define U16_MIN 0x0
/** @def U8_MAX 
 * 
 * Maximum 1 byte unsigned integer value.
 */
#define U8_MAX 0xff
/** @def U8_MIN 
 * 
 * Minimum 1 byte unsigned integer value.
 */
#define U8_MIN 0x0

/** @def F64_MAX 
 * 
 * Maximum 8 byte signed floating point value.
 */
#define F64_MAX 1E+37

/** @def F64_MIN 
 * 
 * Minimum 8 byte signed floating point value.
 */
#define F64_MIN -1E+37
/** @def F32_MAX 
 * 
 * Maximum 4 byte signed floating point value.
 */
#define F32_MAX 1E+37
/** @def F32_MIN 
 * 
 * Minimum 4 byte signed floating point value.
 */
#define F32_MIN -1E+37

/* *********************************
 *         Size definitions
 ***********************************/

/**
 * @def I64_SIZE
 * 
 * Size in bytes of a 64bit signed integer.
 */
#define I64_SIZE 8
/**
 * @def I32_SIZE
 * 
 * Size in bytes of a 32bit signed integer.
 */
#define I32_SIZE 4
/**
 * @def I16_SIZE
 * 
 * Size in bytes of a 16bit signed integer.
 */
#define I16_SIZE 2
/**
 * @def I8_SIZE
 * 
 * Size in bytes of a 8bit signed integer.
 */
#define I8_SIZE 1

/**
 * @def U64_SIZE
 * 
 * Size in bytes of a 64bit unsigned integer.
 */
#define U64_SIZE 8
/**
 * @def U32_SIZE
 * 
 * Size in bytes of a 32bit unsigned integer.
 */
#define U32_SIZE 4
/**
 * @def U16_SIZE
 * 
 * Size in bytes of a 16bit unsigned integer.
 */
#define U16_SIZE 2
/**
 * @def U8_SIZE
 * 
 * Size in bytes of a 8bit unsigned integer.
 */
#define U8_SIZE 1

/* *********************************
 * Epsilon & small value definitions
 ***********************************/

/** @def F64_SMALLEST 
 * 
 * Smallest 8 byte signed floating point non-zero positive value.
 */
#define F64_SMALLEST 1E-37
/** @def F32_SMALLEST 
 * 
 * Smallest 4 byte signed floating point non-zero positive value.
 */
#define F32_SMALLEST 1E-37

/** @def F64_EPSILON 
 * 
 * Smallest 8 byte signed floating point non-zero positive value which
 * added to one will produce a machine-representable number which is not
 * 1. 
 */
#define F64_EPSILON 1E-9
/** @def F32_EPSILON 
 * 
 * Smallest 4 byte signed floating point non-zero positive value which
 * added to one will produce a machine-representable number which is not
 * 1. 
 */
#define F32_EPSILON 1E-5

#endif //BASIC_TYPES_H
