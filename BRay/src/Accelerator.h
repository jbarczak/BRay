//=====================================================================================================================
//
//   BVHBuilder.h
//
//   B-Ray BVH builder
//   
//   Copyright 2015 Joshua Barczak
//
//   LICENSE:  This source code is distributed under the terms of the GNU GPL v2
//         
//
//=====================================================================================================================

#ifndef _BVH_BUILDER_H_
#define _BVH_BUILDER_H_

#include "Types.h"
#include "PoolAllocator.h"

namespace BRay{
    struct Mesh;

namespace _INTERNAL{

    using BRay::PoolAllocator;

    struct PreprocessedTri;

    struct TriList
    {
        uint32 GetTriCount() const { return m_nTris; };

        const PreprocessedTri* GetTriList() const { return (const PreprocessedTri*) (this+1); }
        PreprocessedTri* GetTriList() { return (PreprocessedTri*) (this+1); }
        uint32 m_nTris; 
    };


    struct BVHNode
    {
        bool IsLeaf() const { return (m_pNext & 4)!=0; }

        size_t GetSplitAxis() const { return m_pNext & 3; }
        const char* GetPrefetch() const { return (const char*)m_pNext; }
        const float* GetAABB() const { return m_AABB; }
        TriList* GetTriList() const { return (TriList*)(m_pNext&~7); }
        BVHNode* GetLeftChild() const { return (BVHNode*)(m_pNext&~7); }
        BVHNode* GetRightChild() const { return GetLeftChild()+1; }
        
        void SetAABB( const float* p )
        {
            for( int i=0; i<6; i++ )
                m_AABB[i] = p[i];
        }

        void CreateLeaf( TriList* pList )
        {
            m_pNext = (uint64)pList;
            m_pNext |= 4;
        }

        void CreateInnerNode( BVHNode* pKids, uint nAxis )
        {
            m_pNext = (uint64) pKids;
            m_pNext |= (nAxis);
        }


        float  m_AABB[6];
        uint64 m_pNext; ///< Pointer to pair of child nodes (inner), or triangle list (leaf)
                        ///< low-order bits contain 2-bit split axis and 1 bit (is-leaf) flag 
    };

    

    struct Accelerator : public Object
    {
        BVHNode* pBVHRoot;
        size_t nStackDepth;
        size_t nRefCount;
        PoolAllocator Memory;
    };

    void BuildBVH( Accelerator* pAccel, const Mesh* pMesh, float fIntersectCost );

}}

#endif