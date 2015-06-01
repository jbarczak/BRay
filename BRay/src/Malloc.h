//=====================================================================================================================
//
//   Malloc.h
//
//   Internal malloc used by B-Ray
//   
//   Copyright 2015 Joshua Barczak
//
//   LICENSE:  This source code is distributed under the terms of the GNU GPL
//          
//
//=====================================================================================================================

#ifndef _MALLOC_H_
#define _MALLOC_H_

namespace BRay{
    class Allocator;
namespace _INTERNAL{

    void InitMalloc( Allocator* pAlloc );
    void ShutdownMalloc();
    void* Malloc( size_t bytes, size_t align );
    void Free( void* bytes );

}}

#endif