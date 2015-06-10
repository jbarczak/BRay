
#ifndef _NO_EMBREE

#include "BRay.h"
#include "Simpleton.h"
#include "embree.h"
#include "embree/common/stream_ray.h"
typedef unsigned int uint;

#include <Windows.h>

static embree::IntersectorStream* m_pIntersector=0;
static LARGE_INTEGER m_Timer;

void CreateEmbreeScene( uint nTris, uint nVerts, float* pVerts, uint* pIndices )
{
    embree::rtcInit();
    embree::RTCGeometry* pMesh = embree::rtcNewTriangleMesh( nTris, nVerts, "bvh4.triangle8" );
    embree::RTCVertex* pVB = embree::rtcMapPositionBuffer( pMesh );
    embree::RTCTriangle* pIB = embree::rtcMapTriangleBuffer( pMesh );

    for( uint i=0; i<nVerts; i++ )
    {
        pVB[i].x = pVerts[3*i];
        pVB[i].y = pVerts[3*i+1];
        pVB[i].z = pVerts[3*i+2];
    }

    for( uint i=0; i<nTris; i++ )
    {
        pIB[i].v0 = pIndices[3*i];
        pIB[i].v1 = pIndices[3*i+1];
        pIB[i].v2 = pIndices[3*i+2];
        pIB[i].id0 = i;
        pIB[i].id1 = 0;
    }

    embree::rtcUnmapPositionBuffer( pMesh );
    embree::rtcUnmapTriangleBuffer( pMesh );

    embree::rtcStartThreads(0);
    embree::rtcBuildAccel(pMesh,"default");
    m_pIntersector = embree::rtcQueryIntersectorStream( pMesh, "stream.moeller" );
    embree::rtcStopThreads();
    
    m_Timer.QuadPart=0;
}

void ShootEmbreeRays( BRay::RayData* pRays, BRay::RayHitInfo* pHits, uint nRays )
{
    embree::StreamRay rays[BRay::MAX_TRACER_SIZE];
    embree::StreamRayExtra extras[BRay::MAX_TRACER_SIZE];
    embree::StreamHit hits[BRay::MAX_TRACER_SIZE];
    
    for( uint i=0; i<nRays; i++ )
    {
        rays[i].set( embree::Vector3f(pRays[i].O[0], pRays[i].O[1],pRays[i].O[2]),
                     embree::Vector3f(pRays[i].D[0], pRays[i].D[1],pRays[i].D[2]),
                     0,pRays[i].TMax);
        extras[i].set(embree::Vector3f(pRays[i].O[0], pRays[i].O[1],pRays[i].O[2]),
                     embree::Vector3f(pRays[i].D[0], pRays[i].D[1],pRays[i].D[2]),
                     pRays[i].TMax);
        hits[i].id0=BRay::RayHitInfo::NULL_PRIMID;
    }
    
    LARGE_INTEGER tm;
    LARGE_INTEGER tm1;
    QueryPerformanceCounter(&tm);

    m_pIntersector->intersect( rays, extras, hits, nRays, 0 );

    QueryPerformanceCounter(&tm1);
    m_Timer.QuadPart += (tm1.QuadPart-tm.QuadPart);
    
    for( uint i=0; i<nRays; i++ )
    {
        pRays[i].TMax = rays[i].tfar;
        pHits[i].u = hits[i].u;
        pHits[i].v = hits[i].v;
        pHits[i].nPrimID = hits[i].id0;
        pHits[i].t = rays[i].tfar;
      
    }
}

double GetEmbreeTime()
{
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return m_Timer.QuadPart / (double)(freq.QuadPart);;
}


#endif