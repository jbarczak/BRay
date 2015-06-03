



#include "PlyLoader.h"
#include "Timer.h"
#include "BRay.h"

#include <vector>

#include "Rand.h"
#include "VectorMath.h"
#include "Matrix.h"
#include "PPMImage.h"

using Simpleton::Vec3f;
using Simpleton::Timer;


#define NUM_PHOTONS 20000000
#define BOUNCE_LIMIT 12

#define MESH_FILE "kitchen.ply"

#define _NO_EMBREE
//#define CHECK_ANSWER


struct Photon
{
    Vec3f Pos;
    Vec3f Normal;
    Vec3f Dir;
};




int main(int argc, char* argv[])
{
    Simpleton::PlyMesh ply;
    if( !Simpleton::LoadPly( MESH_FILE, ply, 0 ) )
        return false;
    
    Timer tm;
    BRay::AcceleratorHandle hAccel=0;
    BRay::TracerHandle hTrace0=0;
    BRay::TracerHandle hTrace1=0;
    tm.Reset();
    {
        BRay::Mesh m;
        m.nTriangles = ply.nTriangles;
        m.nTriangleStride = 3*sizeof(uint32);
        m.nVertexStride = sizeof(Vec3f);
        m.pVertexPositions = (float*) ply.pPositions;
        m.pIndices = &ply.pVertexIndices[0];
        BRay::Init(0);
        hAccel = BRay::CreateAccelerator(&m);
        hTrace0 = BRay::CreateTracer(BRay::MAX_TRACER_SIZE,hAccel);
        hTrace1 = BRay::CreateTracer(BRay::MAX_TRACER_SIZE,hAccel);
    }
    printf("BRay init time: %u\n", tm.Tick() );

    Photon* pPhotons = new Photon[NUM_PHOTONS];
    uint nPhotons=0;


#ifndef _NO_EMBREE
    CreateEmbreeScene( mesh.GetObjectCount(), ply.nVertices, (float*) ply.pPositions, &ply.pVertexIndices[0] );
#endif

    
    
    float LIGHT_SIZE = 0.5f;
    Vec3f vLightCenter = Vec3f(2.15,2.1,2.15);
    Vec3f vLightCorners[4] = { vLightCenter + Vec3f(-LIGHT_SIZE,0,LIGHT_SIZE),
                               vLightCenter + Vec3f(LIGHT_SIZE,0,LIGHT_SIZE),
                               vLightCenter + Vec3f(LIGHT_SIZE,0,-LIGHT_SIZE),
                               vLightCenter + Vec3f(-LIGHT_SIZE,0,-LIGHT_SIZE) };


    printf("Done setup\n");

    tm.Reset();

    Timer ttrace;
    long nTotalRays =0;
    long nPackets=0;
    long nTraceTime = 0;
    while( nPhotons < NUM_PHOTONS )
    {
        // generate rays
        BRay::ResetTracer(hTrace0);
        for( uint i=0; i<BRay::MAX_TRACER_SIZE; i++ )
        {
            float u  = Simpleton::Rand();
            float v  = Simpleton::Rand();
            Vec3f v0 = Simpleton::Lerp3(
                           Simpleton::Lerp3( vLightCorners[0], vLightCorners[1], u ),
                           Simpleton::Lerp3( vLightCorners[2], vLightCorners[3], u ), v );

            float du  = Simpleton::Rand();
            float dv  = Simpleton::Rand();
            Vec3f dir = Simpleton::UniformSampleHemisphere(du,dv);
            dir.y *= -1;

            BRay::RayData r;
            r.O[0] = v0.x;
            r.O[1] = v0.y;
            r.O[2] = v0.z;
            r.D[0] = dir.x;
            r.D[1] = dir.y;
            r.D[2] = dir.z;
            r.TMax = 999999999999999;
            BRay::AddRay( hTrace0, &r );
        }

        uint nBounce = 0;
        do
        {
            // shoot rays
            BRay::RayHitInfo pHits[BRay::MAX_TRACER_SIZE];
            ttrace.Reset();
            BRay::Trace(hTrace0,pHits);
            nTraceTime += ttrace.TickMicroSeconds();

            // for each ray, store photon hit 
            uint nFirstPhoton = nPhotons;

            uint nParentRays = BRay::GetRayCount(hTrace0);
            nTotalRays += nParentRays;
            nPackets++;
            for( uint r=0; r<nParentRays; r++ )
            {
                if( pHits[r].nPrimID != BRay::RayHitInfo::NULL_PRIMID )
                {
                    BRay::RayData ray;
                    BRay::ReadRayData( hTrace0, r, &ray );

                    pPhotons[nPhotons].Pos.x = ray.O[0] + ray.TMax*ray.D[0];
                    pPhotons[nPhotons].Pos.y = ray.O[1] + ray.TMax*ray.D[1];
                    pPhotons[nPhotons].Pos.z = ray.O[2] + ray.TMax*ray.D[2];
                    pPhotons[nPhotons].Dir.x = ray.D[0];
                    pPhotons[nPhotons].Dir.y = ray.D[1];
                    pPhotons[nPhotons].Dir.z = ray.D[2];

                    uint id = pHits[r].nPrimID;
                    Vec3f v0 = Vec3f(ply.pPositions[ ply.pVertexIndices[3*id+0] ] );
                    Vec3f v1 = Vec3f(ply.pPositions[ ply.pVertexIndices[3*id+1] ] );
                    Vec3f v2 = Vec3f(ply.pPositions[ ply.pVertexIndices[3*id+2] ] );
                    Vec3f N = Simpleton::Normalize3( Simpleton::Cross3( v1-v0, v2-v0 ) );
                    pPhotons[nPhotons].Normal = N;
                    nPhotons++;

                    if( nPhotons == NUM_PHOTONS )
                        break;
                }
            }

            if( nPhotons == NUM_PHOTONS )
                break;
            
            nBounce++;
            if( nBounce < BOUNCE_LIMIT )
            {
                // create secondary rays
                BRay::ResetTracer(hTrace1);
                for( uint p=nFirstPhoton; p<nPhotons; p++ )
                {
                    float rng = Simpleton::Rand();
                    if( rng < 0.7f )
                    {
                        float s = Simpleton::Rand();
                        float t = Simpleton::Rand();
                        Vec3f v = Simpleton::UniformSampleHemisphere(s,t);
                        Vec3f T,B;
                        Simpleton::BuildTangentFrame( pPhotons[p].Normal, T, B );
                        Vec3f Dir = (pPhotons[p].Normal * v.z) +
                                    (T * v.x) +
                                    (B * v.y) ;

                        BRay::RayData rd;
                        rd.O[0] = Dir.x*0.0001f + pPhotons[p].Pos.x;
                        rd.O[1] = Dir.y*0.0001f + pPhotons[p].Pos.y;
                        rd.O[2] = Dir.z*0.0001f + pPhotons[p].Pos.z;
                        rd.D[0] = Dir.x;
                        rd.D[1] = Dir.y;
                        rd.D[2] = Dir.z;
                        rd.TMax = 999999999999;

                        BRay::AddRay( hTrace1, &rd );
                    }
                }

                std::swap( hTrace0, hTrace1 );
            }

        }while( nBounce < BOUNCE_LIMIT );
    }

    long nTime = tm.TickMicroSeconds();
    
    double fKRays = (nTotalRays/1000.0) / (nTime/1000000.0);
    double fKRaysAdjusted = (nTotalRays/1000.0) / (nTraceTime/1000000.0);
    long nMS = nTime/1000;

    double fTraceTime = nTraceTime;
    double fTotalTime = nTime;

    printf("Rays shot: %.2fM.  Packs shot: %u\n", nTotalRays/(1000000.0), nPackets );
    printf("Mean rays/pack: %.2f\n", nTotalRays / (double)nPackets );
    printf("total: %.2f s (%.2f KRays/s) \n", nMS/1000.0, fKRays );
    printf("trace: %.2f s (%.2f KRays/s). (%.2f ratio)\n", nTraceTime/(1000000.0), fKRaysAdjusted, fTraceTime/fTotalTime );

    printf("Making debug image...\n");
    Vec3f vCameraPosition(3.750000, 2.100000, 3.600000);
    Vec3f vLookAt( 3.436536, 1.880186, 2.748239 );

    Simpleton::Matrix4f mView = Simpleton::MatrixLookAtLH(vCameraPosition, vLookAt, Vec3f(0,1,0) );
    Simpleton::Matrix4f mProj = Simpleton::MatrixPerspectiveFovLH( 1.0f, 60.0f, 1, 100 );
   
    Simpleton::Matrix4f mViewProj = mProj*mView;
    Simpleton::PPMImage img(512,512);
    
    for( uint i=0; i<nPhotons; i++ )
    {
        Vec3f view = Simpleton::AffineTransformPoint(mView,pPhotons[i].Pos );
        if( view.z < 0 )
            continue;

        float diff = -Simpleton::Dot3( pPhotons[i].Dir, pPhotons[i].Normal);
        diff = Simpleton::Max(diff,0.0f);

        Vec3f vec = Simpleton::TransformPoint(mViewProj,pPhotons[i].Pos);
        float u = vec.x*0.5f + 0.5f;
        float v = (-vec.y)*0.5f + 0.5f;
        if(  u > 0 && u < 1 && v > 0 && v < 1 )
        {
            float rgb[3];
            int x = u*img.GetWidth();
            int y = v*img.GetHeight();
            img.GetPixel(x,y,rgb);
            for( int j=0; j<3; j++ )
                rgb[j] = Simpleton::Min(1.0f,rgb[j]+diff*0.01f);
            img.SetPixel(x,y,rgb[0],rgb[1],rgb[2]);
        }
    }
    img.SaveFile("debug.ppm");

#ifndef _NO_EMBREE
    uint nTimeEmbree = GetEmbreeTime()/1000;
    uint nKRaysEmbree = (N*PACKET_SIZE)/nTimeEmbree;
    printf("embree: %u ms( %u KRays/s)\n", nTimeEmbree, nKRaysEmbree );
#endif

    return 0;
}

