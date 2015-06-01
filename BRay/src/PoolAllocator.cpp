//=====================================================================================================================
//
//   PoolAllocator.cpp
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

#include "Types.h"
#include "PoolAllocator.h"
#include "Malloc.h"

namespace Simpleton
{
    void* PoolAllocator::GetMore( size_t nSize )
    {
        // pad allocs 
        nSize = (nSize + ALIGN-1) & ~(ALIGN-1);
        
        // frontmost block has enough room?
        if( !m_pHead || (m_pHead->nOffset + nSize > m_pHead->nCapacity) )
        {
            // no, search for a block that fits and has maximum available space
            AllocHeader* pBlock = m_pHead;
            AllocHeader* pPrev = 0;
            AllocHeader* pBest = 0;
            AllocHeader* pBestPred=0;
            while( pBlock )
            {
                if( pBlock->nOffset + nSize <= pBlock->nCapacity )
                {
                    if( !pBest || ((pBest->nCapacity-pBest->nOffset) < (pBlock->nCapacity-pBlock->nOffset)) )
                    {
                        pBest = pBlock;
                        pBestPred = pPrev;
                    }
                }
            
                pPrev = pBlock;
                pBlock = pBlock->pNext;
            }

            // found suitable block?
            if( pBest )
            {
                // YES: if chosen block isn't the head, make it the head to try and avoid search next time
                if( pBestPred )
                {
                    pBestPred->pNext = pBest->pNext;
                    pBest->pNext = m_pHead;
                    m_pHead = pBest;
                }
            }
            else
            {
                // NO: suitable block not found, must allocate...

                // add space for a header, and pad alloc to multiple of pool chunk size              
                size_t nBytesNeeded = HEADER_SIZE + nSize;
                nBytesNeeded = (nBytesNeeded + CHUNKSIZE-1) & ~(CHUNKSIZE-1);

                // allocate and set up new block
                pBlock = (AllocHeader*) BRay::_INTERNAL::Malloc(nBytesNeeded,ALIGN);
                pBlock->pNext = m_pHead;
                pBlock->nCapacity = nBytesNeeded;
                pBlock->nOffset = HEADER_SIZE;
                m_pHead = pBlock;                
            }
        }
   
        // alloc to frontmost block
        size_t nOffs = m_pHead->nOffset;
        m_pHead->nOffset += nSize;
        return ((char*)m_pHead) + nOffs;                 
    }


    void PoolAllocator::FreeAll()
    {
        AllocHeader* pBlock = m_pHead;
        while( pBlock )
        {
            AllocHeader* pPrev = pBlock;
            pBlock = pBlock->pNext;
            BRay::_INTERNAL::Free(pPrev);
        }
        m_pHead=0;
    }

    void PoolAllocator::Recycle()
    {
        AllocHeader* pBlock = m_pHead;
        while( pBlock )
        {
            pBlock->nOffset = HEADER_SIZE;
            pBlock = pBlock->pNext;
        }
    }
}