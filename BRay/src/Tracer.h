

#include "Types.h"

namespace BRay{
namespace _INTERNAL{

    struct BVHNode;

    struct Ray
    {
        float ox;
        float oy;
        float oz;
        float tmax;
        float dx;
        float dy;
        float dz;
        uint32 offset; ///< Byte offset of this ray from start of stream
    };

    
    struct PreprocessedTri
    {
        uint32 nID;
        float P0[3];
        float v10[3];
        float v02[3];
        float v10x02[3];
    };

    struct StackEntry
    {
        BVHNode* pNode;
        size_t nRayPop;
        size_t nGroups;
    };


    struct Tracer : public Object
    {
        Accelerator* pAccelerator;
        uint32 nMaxRays;
        
        Ray* pRays;
        BVHNode* pBVHRoot;
        StackEntry* pTraversalStack;

        uint32 nRays;
        uint16 pOctantCounts[8];
        uint8* pRayOctants;

    };

    struct PreprocessedTri;
    void AssembleTri( PreprocessedTri* __restrict pTri, uint32 nID, const float* P0, const float* P1, const float* P2 );

    void DoTrace( Tracer* p, BRay::RayHitInfo* pHitsOut );
   
    void DoOcclusionTrace( Tracer* p, unsigned char* pOcclusionOut );
}};