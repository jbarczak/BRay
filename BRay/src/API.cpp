//=====================================================================================================================
//
//   API.cpp
//
//   B-Ray API implementation
//   
//   Copyright 2015 Joshua Barczak
//
//   LICENSE:  This source code is distributed under the terms of the GNU GPL
//          
//
//=====================================================================================================================

#include "BRay.h"
#include "Types.h"
#include "Malloc.h"

#include "Tracer.h"
#include "Accelerator.h"

#include <new.h>
#include <intrin.h>

namespace BRay
{
    namespace _INTERNAL
    {
        // All API objects ever constructed
        //  used for final cleanup on shutdown
        Object* m_pObjects=0;

        void FreeObject( Object* pObj )
        {
            if( pObj->pPrev )
                pObj->pPrev->pNext = pObj->pNext;           
            if( pObj->pNext )
                pObj->pNext->pPrev = pObj->pPrev;
            if( pObj == m_pObjects )
                m_pObjects = m_pObjects->pNext;

            pObj->~Object();
            Free(pObj);            
        }

        void RetainObject( Object* pObj )
        {
            pObj->pNext = m_pObjects;
            pObj->pPrev = 0;
            if( m_pObjects )
                m_pObjects->pPrev = pObj;
            m_pObjects = pObj;
        }

    }

    void Init( Allocator* pAlloc )
    {
        _INTERNAL::InitMalloc(pAlloc);
    }

    void Shutdown( )
    {
        _INTERNAL::Object* pObj = _INTERNAL::m_pObjects;
        while( pObj )
        {
            _INTERNAL::Object* pNext = pObj->pNext;
            pObj->~Object();
            _INTERNAL::Free(pObj);
            pObj = pNext;
        }
        _INTERNAL::m_pObjects=0;
        _INTERNAL::ShutdownMalloc();
    }

    AcceleratorHandle CreateAccelerator( const Mesh* pMesh )
    {
        _INTERNAL::Accelerator* pAccel = (_INTERNAL::Accelerator*) _INTERNAL::Malloc( sizeof(_INTERNAL::Accelerator), BRAY_MEM_ALIGNMENT );
        
        pAccel = new(pAccel) _INTERNAL::Accelerator();
        pAccel->nRefCount = 1;
        _INTERNAL::BuildBVH(pAccel,pMesh,1.4f);

        _INTERNAL::RetainObject(pAccel);
        return pAccel;
    }

    void ReleaseAccelerator( AcceleratorHandle hAccel )
    {
        if( hAccel )
        {
            hAccel->nRefCount--;
            if( !hAccel->nRefCount)
            {
                _INTERNAL::FreeObject(hAccel);
            }
        }
    } 

    TracerHandle CreateTracer( size_t nMaxRays, AcceleratorHandle hAccel )
    {
        hAccel->nRefCount++;
        
        size_t nBytes = sizeof(_INTERNAL::Tracer) +
                        nMaxRays*sizeof(_INTERNAL::Ray) + 
                        hAccel->nStackDepth*sizeof(_INTERNAL::StackEntry) +
                        nMaxRays*sizeof(uint8);

        uint8* pStorage = (uint8*) _INTERNAL::Malloc( nBytes, BRAY_MEM_ALIGNMENT );
        _INTERNAL::Tracer* pTracer = new(pStorage) _INTERNAL::Tracer();
        pTracer->nMaxRays = nMaxRays;
        pTracer->nRays = 0;
        pTracer->pAccelerator = hAccel;
        pTracer->pBVHRoot = hAccel->pBVHRoot;
        pTracer->pRays = (_INTERNAL::Ray*)(pTracer+1);
        pTracer->pTraversalStack = (_INTERNAL::StackEntry*) (pTracer->pRays + nMaxRays);
        pTracer->pRayOctants = (uint8*) (pTracer->pTraversalStack + hAccel->nStackDepth);
        for( size_t i=0; i<8; i++ )
            pTracer->pOctantCounts[i] = 0;

        _INTERNAL::RetainObject(pTracer);
        return pTracer;
    }

    void ReleaseTracer( TracerHandle hStream )
    {
        if( hStream )
        {
            ReleaseAccelerator(hStream->pAccelerator);
            _INTERNAL::FreeObject(hStream);
        }
    }

    void ResetTracer( TracerHandle hTracer )
    {
        hTracer->nRays=0;
        for( size_t i=0; i<8; i++ )
            hTracer->pOctantCounts[i] = 0;
    }

    

    void AddRay( TracerHandle hTracer, const RayData* pRay )
    {
        size_t nRay = hTracer->nRays;
        size_t nOffs = nRay*sizeof(_INTERNAL::Ray);

        char* pRayLocation = ((char*)hTracer->pRays)+nOffs;
     
        // read direction and compute octant based on direction signs
        const uint32* pRayIn = (const uint32*)pRay;
        size_t ox = pRayIn[0];
        size_t oy = pRayIn[1];
        size_t oz = pRayIn[2];
        size_t t  = pRayIn[3];
        size_t dx = pRayIn[4];
        size_t dy = pRayIn[5];
        size_t dz = pRayIn[6];

        // move the ray.  
        uint32* pRayOut = (uint32*) (pRayLocation);
        pRayOut[0] = ox;
        pRayOut[1] = oy;
        pRayOut[2] = oz;
        pRayOut[3] = t;
        pRayOut[4] = dx;
        pRayOut[5] = dy;
        pRayOut[6] = dz;
        pRayOut[7] = nOffs;
       
        size_t octant = ((dx>>31) | ((dy>>30)&2) | ((dz>>29)&4) );

        hTracer->pRayOctants[nRay] = octant;
        hTracer->pOctantCounts[octant]++;
        hTracer->nRays = nRay+1;
    }

    void ReadRayData( TracerHandle hTracer, size_t nRayIndex, RayData* pRayOut )
    {
        _INTERNAL::Ray* pR = &hTracer->pRays[nRayIndex];
        pRayOut->O[0] = pR->ox;
        pRayOut->O[1] = pR->oy;
        pRayOut->O[2] = pR->oz;
        pRayOut->D[0] = pR->dx;
        pRayOut->D[1] = pR->dy;
        pRayOut->D[2] = pR->dz;
        pRayOut->TMax = pR->tmax;
           
    }

    void ReadRayOrigin( TracerHandle hTracer, float* pOrigin, size_t nRayIndex )
    {
        float ox = hTracer->pRays[nRayIndex].ox;
        float oy = hTracer->pRays[nRayIndex].oy;
        float oz = hTracer->pRays[nRayIndex].oz;
        pOrigin[0] = ox;
        pOrigin[1] = oy;
        pOrigin[2] = oz;
    }
    void ReadRayDirection( TracerHandle hTracer, float* pDir, size_t nRayIndex )
    {
        float dx = hTracer->pRays[nRayIndex].dx;
        float dy = hTracer->pRays[nRayIndex].dy;
        float dz = hTracer->pRays[nRayIndex].dz;
        pDir[0] = dx;
        pDir[1] = dy;
        pDir[2] = dz;
    }

    size_t GetRayCount( TracerHandle hTracer )
    {
        return hTracer->nRays;
    }

    void Trace( TracerHandle hTracer, BRay::RayHitInfo* pHitsOut )
    {
        _INTERNAL::DoTrace(hTracer, pHitsOut);
    }

    void OcclusionTrace( TracerHandle hTracer, unsigned char* pOcclusion )
    {
        _INTERNAL::DoOcclusionTrace(hTracer,pOcclusion);
    }

}