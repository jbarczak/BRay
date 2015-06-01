//=====================================================================================================================
//
//   PoolAllocator.h
//
//   Pooled memory allocator for batching small mallocs
//
//   The lazy man's utility library
//   Joshua Barczak
//   Copyright 2014 Joshua Barczak
//
//   LICENSE:  See Doc\License.txt for terms and conditions
//
//=====================================================================================================================


#ifndef _POOL_ALLOCATOR_H_
#define _POOL_ALLOCATOR_H_

namespace Simpleton
{
    /// Simple block allocator which allocates storage in chunks and never releases any of it
    /// This is intended for aggregating large numbers of temporary allocs which are all released at once
    /// It is by no means a general purpose allocator.
    class PoolAllocator
    {
    public:

        class ScopedFree
        {
        public:
            ScopedFree( PoolAllocator& alloc ) : m_rAlloc(alloc){}
            ~ScopedFree() { m_rAlloc.FreeAll(); };
        private:        
            PoolAllocator& m_rAlloc;        
        };

        class ScopedRecycler
        {
        public:
            ScopedRecycler( PoolAllocator& alloc ) : m_rAlloc(alloc){}
            ~ScopedRecycler() { m_rAlloc.Recycle(); };
        private:        
            PoolAllocator& m_rAlloc;        
        };

        PoolAllocator() : m_pHead(0) {}
        ~PoolAllocator() { FreeAll(); }

        template< class T > T* GetA() { return reinterpret_cast<T*>( GetMore( sizeof(T) ) ); }
        template< class T > T* GetSome( size_t n ) { return reinterpret_cast<T*>( GetMore( sizeof(T)*n ) ); };
        void* GetMore( size_t nSize );

        void FreeAll( );

        void Recycle( );

    private:

        struct AllocHeader
        {
            AllocHeader* pNext;     ///< Next block in linked list
            size_t nCapacity;       ///< Total Size of block (including header)
            size_t nOffset;         ///< Offset of next free byte     
        };


        enum
        {
            ALIGN       = 64,       ///< Align all 'GetMore' calls to this size
            CHUNKSIZE   = 16*1024,  ///< Align all mallocs to this size
            HEADER_SIZE = (sizeof(AllocHeader) + ALIGN-1) & ~(ALIGN-1)
        };

        AllocHeader* m_pHead;
    };
}



#endif