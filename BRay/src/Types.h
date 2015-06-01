//=====================================================================================================================
//
//   Types.h
//
//   Internal data structures and constants used by B-Ray
//   
//   Copyright 2015 Joshua Barczak
//
//   LICENSE:  This source code is distributed under the terms of the GNU GPL
//          
//
//=====================================================================================================================

#ifndef _TYPES_H_
#define _TYPES_H_

#include "BRay.h"

#define BRAY_MEM_ALIGNMENT 32

namespace BRay{
    typedef unsigned char uint8;
    typedef unsigned short uint16;
    typedef unsigned __int64 uint64;
    typedef unsigned int uint32;
    typedef int int32;

    static_assert( sizeof(uint8)  == 1 && 
                   sizeof(uint16) == 2 && 
                   sizeof(uint32) == 4 && 
                   sizeof(int32)  == 4 &&
                   sizeof(uint64) == 8,   "Your compiler disagrees with our assumptions");

namespace _INTERNAL{

    typedef size_t uint;

    struct __declspec(align(BRAY_MEM_ALIGNMENT)) Object
    {
        virtual ~Object(){}
        Object* pNext;
        Object* pPrev;
    };
   
   

}}


#endif