#ifndef _EMBREE_H_
#define _EMBREE_H_


typedef unsigned int uint;
struct RayStream;
void CreateEmbreeScene( uint nTris, uint nVerts, float* pVerts, uint* pIndices );
void ShootEmbreeRays( BRay::RayData* pRays, BRay::RayHitInfo* pHits, uint nRays );
double GetEmbreeTime();

#endif