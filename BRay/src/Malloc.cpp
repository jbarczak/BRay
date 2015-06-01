//=====================================================================================================================
//
//   Malloc.cpp
//
//   Internal malloc used by B-Ray
//   
//   Copyright 2015 Joshua Barczak
//
//   LICENSE:  This source code is distributed under the terms of the GNU GPL
//          
//
//=====================================================================================================================

#include "BRay.h"
#include <intrin.h>

namespace BRay{
namespace _INTERNAL{

    static Allocator* g_pAlloc=0;

    class DefaultAllocator : public Allocator
    {
    public:
        virtual void* Malloc( size_t nBytes, size_t nAlign ) { return _aligned_malloc(nBytes,nAlign); }
        virtual void Free( void* pBytes ) { return _aligned_free(pBytes); }
    };

    static DefaultAllocator g_DefaultAlloc;

    void InitMalloc( Allocator* p )
    {
        if( !p )
            p = &g_DefaultAlloc;
        g_pAlloc = p;
    }

    void ShutdownMalloc()
    {
        g_pAlloc=0;
    }

    void* Malloc( size_t bytes, size_t align )
    {
        return g_pAlloc->Malloc(bytes,align);
    }

    void Free( void* bytes )
    {
        g_pAlloc->Free(bytes);
    }

}}