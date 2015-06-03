//=====================================================================================================================
//
//   BRay.h
//
//   B-Ray raytracer API
//   
//   Copyright 2015 Joshua Barczak
//
//   LICENSE:  This source code is distributed under the terms of the GNU GPL v2
//         
//
//=====================================================================================================================

#ifndef _BRAY_H_
#define _BRAY_H_

namespace BRay
{
    typedef unsigned int uint32;
    static_assert( sizeof(uint32) == 4, "Your compiler disagrees with our assumptions");

    namespace _INTERNAL{
        struct Accelerator;
        struct Tracer;
    };


    typedef _INTERNAL::Accelerator* AcceleratorHandle;
    typedef _INTERNAL::Tracer* TracerHandle;

    enum
    {
        MAX_TRACER_SIZE = 4096,
    };

    struct RayHitInfo
    {
        enum 
        {
            NULL_PRIMID = 0xffffffff
        };
        float u;
        float v;
        uint32 nPrimID;
        float t;
    };


    class Allocator
    {
    public:
        virtual void* Malloc( size_t nBytes, size_t nAlign ) = 0;
        virtual void Free( void* pBytes ) = 0;
    };

    
    struct Mesh
    {
        const float* pVertexPositions;
        const uint32* pIndices;
        uint32 nVertexStride;
        uint32 nTriangleStride;
        uint32 nTriangles;
    };

    struct RayData
    {
        float O[3]; float TMax;
        float D[3]; float _Pad;
    };

    void Init( Allocator* pAlloc );
    void Shutdown( );

    AcceleratorHandle CreateAccelerator( const Mesh* pMesh );
    void ReleaseAccelerator( AcceleratorHandle hAccel );

    TracerHandle CreateTracer( size_t nMaxRays, AcceleratorHandle hAccel );
    void ReleaseTracer( TracerHandle pStream );

    void ResetTracer( TracerHandle hTracer );
    void AddRay( TracerHandle hTracer, const RayData* pRay );
    void ReadRayOrigin( TracerHandle hTracer,float* pOrigin,  size_t nRay );
    void ReadRayDirection( TracerHandle hTracer,  float* pDir, size_t nRay );
    void ReadRayData( TracerHandle hTracer, size_t nRay, RayData* pRayOut );
    size_t GetRayCount( TracerHandle hTracer );

    void Trace( TracerHandle hTracer, BRay::RayHitInfo* pHitsOut );
    void OcclusionTrace( TracerHandle hTracer, unsigned char* pOcclusionOut );
}


#endif