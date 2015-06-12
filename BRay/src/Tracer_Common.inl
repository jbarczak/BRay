
//=====================================================================================================================
//
//   Tracer.cpp
//
//   Miscellaneous code common to both the primary and occlusion tracers
//   
//   Copyright 2015 Joshua Barczak
//
//   LICENSE:  This source code is distributed under the terms of the GNU GPL v2
//         
//
//=====================================================================================================================


#include "Types.h"
#include "Tracer.h"
#include "Accelerator.h"

#include <stdio.h>
#include <intrin.h>


//#define NO_REORDERING
//#define GATHERS

namespace BRay{
namespace _INTERNAL{

    enum
    {
        MAX_PACKETS_IN_FLIGHT = MAX_TRACER_SIZE/8
    };

    typedef size_t uint;
    typedef uint8 byte;

    
    struct __declspec(align(64)) RayPacket
    {
        __m256 Ox; __m256 DInvx;  // NOTE: Order matters.  Each line here is 1 cacheline
        __m256 Oy; __m256 DInvy;  // The O/DInv pairs are always read together during traversal
        __m256 Oz; __m256 DInvz;
        __m256 TMax; uint32 RayOffsets[8];
    };



    struct StackFrame
    {
        BVHNode* pBVH;        
        Ray* pRays;
        size_t nRays;
        size_t nPackets;
    
        RayPacket* pAllPackets;
        RayHitInfo* pHitInfo;
        uint8* pOcclusion;
        float pAABB[6];
        RayPacket* pActivePackets[MAX_PACKETS_IN_FLIGHT];    ///< Indices of active packets in pPackets
        uint8 pMasks[MAX_PACKETS_IN_FLIGHT];        ///< Ray mask for each active packet
                                                  ///<  This is indexed over the packets, not the active groups in pGroupIDs
    };


    static uint RoundUp8( uint n )
    {
        return (n+7)&~7;
    }

    static __m256 __forceinline RCPNR( __m256 f )
    {
        __m256 rcp    = _mm256_rcp_ps(f);
        __m256 rcp_sq = _mm256_mul_ps(rcp,rcp);
        __m256 rcp_x2 = _mm256_add_ps(rcp,rcp);
        return _mm256_fnmadd_ps( rcp_sq, f, rcp_x2 );
    }

    static __m256i __forceinline BROADCASTINT( size_t x )
    {
        return _mm256_broadcastd_epi32(_mm_cvtsi32_si128((int)x));
    }
    

    #define SHUFFLE(a,b,c,d) (a|(b<<2)|(c<<4)|(d<<6))

    static void ReadDirs( __m256 D[3], RayPacket* pPacket, Ray* pRays )
    {
#ifdef GATHERS
        __m256i idx = _mm256_load_si256((__m256i*)pPacket->RayOffsets);
        __m256 Dx = _mm256_i32gather_ps( &pRays->dx, idx, 1 );
        __m256 Dy = _mm256_i32gather_ps( &pRays->dy, idx, 1 );
        __m256 Dz = _mm256_i32gather_ps( &pRays->dz, idx, 1 );
        D[0] = Dx;
        D[1] = Dy;
        D[2] = Dz;
#else
        char* pBytes = ((char*)pRays)+16;
        #define LOADPS(x) _mm_load_ps( (float*)(x) )
        __m256 v0 =  _mm256_set_m128( LOADPS( pBytes + pPacket->RayOffsets[4] ),
                                      LOADPS( pBytes + pPacket->RayOffsets[0] ) );  // 0000 4444
        __m256 v1 =  _mm256_set_m128( LOADPS( pBytes + pPacket->RayOffsets[6] ),
                                      LOADPS( pBytes + pPacket->RayOffsets[2] ) );  // 2222 6666
        __m256 v2 =  _mm256_set_m128( LOADPS( pBytes + pPacket->RayOffsets[5] ),
                                      LOADPS( pBytes + pPacket->RayOffsets[1] ) );  // 1111 5555
        __m256 v3 =  _mm256_set_m128( LOADPS( pBytes + pPacket->RayOffsets[7] ),
                                      LOADPS( pBytes + pPacket->RayOffsets[3] ) );  // 3333 7777
        #undef LOADPS

        __m256 t0 = _mm256_unpacklo_ps(v0,v1); // 02 02 46 46
        __m256 t1 = _mm256_unpackhi_ps(v0,v1); // 02 02 46 46
        __m256 t2 = _mm256_unpacklo_ps(v2,v3); // 13 13 57 57
        __m256 t3 = _mm256_unpackhi_ps(v2,v3); // 13 13 57 57
        __m256 X  = _mm256_unpacklo_ps(t0,t2);  // 01 23 45 67
        __m256 Y  = _mm256_unpackhi_ps(t0,t2);
        __m256 Z  = _mm256_unpacklo_ps(t1,t3);
        D[0] = X;
        D[1] = Y;
        D[2] = Z;
#endif
    }

    static void ReadOrigins( __m256 O[3], RayPacket* pPacket, Ray* pRays )
    {
#ifdef GATHERS
        __m256i idx = _mm256_load_si256((__m256i*)pPacket->RayOffsets);
        __m256 Ox = _mm256_i32gather_ps( &pRays->ox, idx, 1 );
        __m256 Oy = _mm256_i32gather_ps( &pRays->oy, idx, 1 );
        __m256 Oz = _mm256_i32gather_ps( &pRays->oz, idx, 1 );
        O[0] = Ox;
        O[1] = Oy;
        O[2] = Oz;
#else
        char* pBytes = ((char*)pRays);
        #define LOADPS(x) _mm_load_ps( (float*)(x) )
        __m256 v0 =  _mm256_set_m128( LOADPS( pBytes + pPacket->RayOffsets[4] ),
                                      LOADPS( pBytes + pPacket->RayOffsets[0] ) );  // 0000 4444
        __m256 v1 =  _mm256_set_m128( LOADPS( pBytes + pPacket->RayOffsets[6] ),
                                      LOADPS( pBytes + pPacket->RayOffsets[2] ) );  // 2222 6666
        __m256 v2 =  _mm256_set_m128( LOADPS( pBytes + pPacket->RayOffsets[5] ),
                                      LOADPS( pBytes + pPacket->RayOffsets[1] ) );  // 1111 5555
        __m256 v3 =  _mm256_set_m128( LOADPS( pBytes + pPacket->RayOffsets[7] ),
                                      LOADPS( pBytes + pPacket->RayOffsets[3] ) );  // 3333 7777
        #undef LOADPS

        __m256 t0 = _mm256_unpacklo_ps(v0,v1); // 02 02 46 46
        __m256 t1 = _mm256_unpackhi_ps(v0,v1); // 02 02 46 46
        __m256 t2 = _mm256_unpacklo_ps(v2,v3); // 13 13 57 57
        __m256 t3 = _mm256_unpackhi_ps(v2,v3); // 13 13 57 57
        __m256 X  = _mm256_unpacklo_ps(t0,t2);  // 01 23 45 67
        __m256 Y  = _mm256_unpackhi_ps(t0,t2);
        __m256 Z  = _mm256_unpacklo_ps(t1,t3);
        O[0] = X;
        O[1] = Y;
        O[2] = Z;
#endif
    }


    static void ReadRays( RayPacket* __restrict pPacket, const byte* __restrict pRays, const uint32* __restrict pOffsets )
    {
    
#ifdef GATHERS
        Ray* pR = (Ray*)pRays;
         __m256i idx = _mm256_loadu_si256((__m256i*)pOffsets);
        __m256 Ox   = _mm256_i32gather_ps( &pR->ox, idx, 1 );
        __m256 Oy   = _mm256_i32gather_ps( &pR->oy, idx, 1 );
        __m256 Oz   = _mm256_i32gather_ps( &pR->oz, idx, 1 );
        __m256 Tmax = _mm256_i32gather_ps( &pR->tmax, idx, 1 );
        __m256 Dx   = _mm256_i32gather_ps( &pR->dx, idx, 1 );
        __m256 Dy   = _mm256_i32gather_ps( &pR->dy, idx, 1 );
        __m256 Dz   = _mm256_i32gather_ps( &pR->dz, idx, 1 );
        __m256 RID  = _mm256_i32gather_ps( (float*)&(pR->offset), idx,1 );
        Dx = RCPNR(Dx);
        Dy = RCPNR(Dy);
        Dz = RCPNR(Dz);
        pPacket->Ox = Ox;
        pPacket->Oy = Oy;
        pPacket->Oz = Oz;
        pPacket->DInvx = Dx;
        pPacket->DInvy = Dy;
        pPacket->DInvz = Dz;
        pPacket->TMax = Tmax;
        _mm256_store_ps((float*)(pPacket->RayOffsets), RID );
#else
        // unpacklo(x,y) -->   x0 y0 x1 y1 x4 y4 x5 y5
        // 0 1 2 3   0 1 2 3
        //  
        //   0 1 2 3   0 1 2 3
        //   0 1 2 3   0 1 2 3
        //   0 1 2 3   0 1 2 3
        //   0 1 2 3   0 1 2 3
   
        #define LOADPS(x) _mm_load_ps((float*)(x))
        __m256 l0 = _mm256_set_m128( LOADPS(pRays + pOffsets[1]), LOADPS(pRays+pOffsets[0]));
        __m256 l1 = _mm256_set_m128( LOADPS(pRays + pOffsets[3]), LOADPS(pRays+pOffsets[2]));
        __m256 l2 = _mm256_set_m128( LOADPS(pRays + pOffsets[5]), LOADPS(pRays+pOffsets[4]));
        __m256 l3 = _mm256_set_m128( LOADPS(pRays + pOffsets[7]), LOADPS(pRays+pOffsets[6]));
        __m256 l4 = _mm256_set_m128( LOADPS(pRays + pOffsets[1]+16), LOADPS(pRays+pOffsets[0]+16));
        __m256 l5 = _mm256_set_m128( LOADPS(pRays + pOffsets[3]+16), LOADPS(pRays+pOffsets[2]+16));
        __m256 l6 = _mm256_set_m128( LOADPS(pRays + pOffsets[5]+16), LOADPS(pRays+pOffsets[4]+16));
        __m256 l7 = _mm256_set_m128( LOADPS(pRays + pOffsets[7]+16), LOADPS(pRays+pOffsets[6]+16));
        #undef LOADPS

       __m256 t0 = _mm256_unpacklo_ps(l0,l1); // 00 11 00 11
       __m256 t1 = _mm256_unpacklo_ps(l2,l3); // 00 11 00 11
       __m256 t2 = _mm256_unpackhi_ps(l0,l1); // 22 33 22 33
       __m256 t3 = _mm256_unpackhi_ps(l2,l3); // 22 33 22 33
       __m256 t4 = _mm256_unpacklo_ps(l4,l5); // 44 55 44 55
       __m256 t5 = _mm256_unpacklo_ps(l6,l7); // 44 55 44 55
       __m256 t6 = _mm256_unpackhi_ps(l4,l5); // 66 77 66 77 
       __m256 t7 = _mm256_unpackhi_ps(l6,l7); // 66 77 66 77
   
        __m256 Ox    = _mm256_unpacklo_ps(t0,t1); // 00 00 00 00
        __m256 Oy    = _mm256_unpackhi_ps(t0,t1); // 11 11 11 11
        __m256 Oz    = _mm256_unpacklo_ps(t2,t3); // 22 22 22 22
        __m256 TMax  = _mm256_unpackhi_ps(t2,t3); // 33 33 33 33
        __m256 Dx    = _mm256_unpacklo_ps(t4,t5); 
        __m256 Dy    = _mm256_unpackhi_ps(t4,t5); 
        __m256 Dz    = _mm256_unpacklo_ps(t6,t7);
        __m256 RID   = _mm256_unpackhi_ps(t6,t7);
   
       _mm256_store_ps( (float*)&pPacket->Ox, Ox );
       _mm256_store_ps( (float*)&pPacket->DInvx, RCPNR(Dx) );
       _mm256_store_ps( (float*)&pPacket->Oy, Oy );    
       _mm256_store_ps( (float*)&pPacket->DInvy, RCPNR(Dy) );
       _mm256_store_ps( (float*)&pPacket->Oz, Oz );    
       _mm256_store_ps( (float*)&pPacket->DInvz, RCPNR(Dz) );
       _mm256_store_ps( (float*)&pPacket->TMax, TMax );
       _mm256_store_ps( (float*)pPacket->RayOffsets, RID );
 
        // unpacklo(x,y) -->   x0 y0 x1 y1 x4 y4 x5 y5
        // 0 1 2 3   0 1 2 3
        //  
        //   0 1 2 3   0 1 2 3
        //   0 1 2 3   0 1 2 3
        //   0 1 2 3   0 1 2 3
        //   0 1 2 3   0 1 2 3
#endif
    }

 

    struct ReadRaysLoopArgs
    {
        uint32* pRayIDs;
        RayPacket** pPackets;
        const byte* pRays;
    };

    static void ReadRaysLoop( ReadRaysLoopArgs& l, size_t nReorder )
    {
        const byte* __restrict pRays      = l.pRays;
        for( size_t i=0; i<nReorder; i++ )
        {
            const uint32* __restrict pOffsets = l.pRayIDs + 8*i;
        
#ifdef GATHERS
        Ray* pR = (Ray*)pRays;
         __m256i idx = _mm256_loadu_si256((__m256i*)pOffsets);
        __m256 Ox   = _mm256_i32gather_ps( &pR->ox, idx, 1 );
        __m256 Oy   = _mm256_i32gather_ps( &pR->oy, idx, 1 );
        __m256 Oz   = _mm256_i32gather_ps( &pR->oz, idx, 1 );
        __m256 Tmax = _mm256_i32gather_ps( &pR->tmax, idx, 1 );
        __m256 Dx   = _mm256_i32gather_ps( &pR->dx, idx, 1 );
        __m256 Dy   = _mm256_i32gather_ps( &pR->dy, idx, 1 );
        __m256 Dz   = _mm256_i32gather_ps( &pR->dz, idx, 1 );
        __m256 RID  = _mm256_i32gather_ps( (float*)&(pR->offset), idx,1 );
        Dx = RCPNR(Dx);
        Dy = RCPNR(Dy);
        Dz = RCPNR(Dz);
        
        RayPacket* __restrict pPacket = l.pPackets[i];
        pPacket->Ox = Ox;
        pPacket->Oy = Oy;
        pPacket->Oz = Oz;
        pPacket->DInvx = Dx;
        pPacket->DInvy = Dy;
        pPacket->DInvz = Dz;
        pPacket->TMax = Tmax;
        _mm256_store_ps((float*)(pPacket->RayOffsets), RID );

#else

            // Load lower halves into L0-L3, and upper halves into L4-L7
            //   Lower half contains origin (x,y,z) and TMax
            //   Upper half contains directions(x,y,z), and byte offset from start of ray stream
            //
            // Using 128-bit loads and inserts is preferable to 256-bit loads and cross-permutes
            //  The inserts can be fused with the loads, and Haswell can issue them on more ports that way
            #define LOADPS(x) _mm_load_ps((float*)(x))
           __m256 l0 = _mm256_set_m128( LOADPS(pRays + pOffsets[1]), LOADPS(pRays+pOffsets[0]));
           __m256 l1 = _mm256_set_m128( LOADPS(pRays + pOffsets[3]), LOADPS(pRays+pOffsets[2]));
           __m256 l2 = _mm256_set_m128( LOADPS(pRays + pOffsets[5]), LOADPS(pRays+pOffsets[4]));
           __m256 l3 = _mm256_set_m128( LOADPS(pRays + pOffsets[7]), LOADPS(pRays+pOffsets[6]));
           __m256 l4 = _mm256_set_m128( LOADPS(pRays + pOffsets[1]+16), LOADPS(pRays+pOffsets[0]+16));
           __m256 l5 = _mm256_set_m128( LOADPS(pRays + pOffsets[3]+16), LOADPS(pRays+pOffsets[2]+16));
           __m256 l6 = _mm256_set_m128( LOADPS(pRays + pOffsets[5]+16), LOADPS(pRays+pOffsets[4]+16));
           __m256 l7 = _mm256_set_m128( LOADPS(pRays + pOffsets[7]+16), LOADPS(pRays+pOffsets[6]+16));
            #undef LOADPS

           __m256 t4 = _mm256_unpacklo_ps(l4,l5); // 44 55 44 55
           __m256 t5 = _mm256_unpacklo_ps(l6,l7); // 44 55 44 55       
           t4 = RCPNR(t4); // both 4 and 5 get rcp'd eventually, so we can start them earlier
           t5 = RCPNR(t5); // to give the other ports something to do during this monstrous blob of unpacks
           
           __m256 t0     = _mm256_unpacklo_ps(l0,l1); // 00 11 00 11
           __m256 t1     = _mm256_unpacklo_ps(l2,l3); // 00 11 00 11
           __m256 t2     = _mm256_unpackhi_ps(l0,l1); // 22 33 22 33
           __m256 t3     = _mm256_unpackhi_ps(l2,l3); // 22 33 22 33
           __m256 t6     = _mm256_unpackhi_ps(l4,l5); // 66 77 66 77 
           __m256 t7     = _mm256_unpackhi_ps(l6,l7); // 66 77 66 77
            __m256 Ox    = _mm256_unpacklo_ps(t0,t1); // 00 00 00 00
            __m256 Oy    = _mm256_unpackhi_ps(t0,t1); // 11 11 11 11
            __m256 Oz    = _mm256_unpacklo_ps(t2,t3); // 22 22 22 22
            __m256 TMax  = _mm256_unpackhi_ps(t2,t3); // 33 33 33 33
            __m256 Dx    = _mm256_unpacklo_ps(t4,t5); 
            __m256 Dy    = _mm256_unpackhi_ps(t4,t5); 
            __m256 Dz    = _mm256_unpacklo_ps(t6,t7);
            __m256 RID   = _mm256_unpackhi_ps(t6,t7);
   
            RayPacket* __restrict pPacket = l.pPackets[i];
           _mm256_store_ps( (float*)&pPacket->Ox, Ox );
           _mm256_store_ps( (float*)&pPacket->DInvx, (Dx) );
           _mm256_store_ps( (float*)&pPacket->Oy, Oy );    
           _mm256_store_ps( (float*)&pPacket->DInvy, (Dy) );
           _mm256_store_ps( (float*)&pPacket->Oz, Oz );    
           _mm256_store_ps( (float*)&pPacket->DInvz, RCPNR(Dz) );
           _mm256_store_ps( (float*)&pPacket->TMax, TMax );
           _mm256_store_ps( (float*)pPacket->RayOffsets, RID );
#endif
        }
    }


    static size_t GroupTest2X( StackFrame* pFrame, size_t nGroups )
    {
        RayPacket** pPackets = pFrame->pActivePackets;
        uint8* pMasks        = pFrame->pMasks;
        const float* pAABB   = pFrame->pAABB;
    
        size_t g;
        size_t nTwos = (nGroups&~1); 
        size_t nHitPopulation=0;
        for( g=0;  g<nTwos; g += 2 )
        {
            RayPacket& pack0 = *pPackets[g];
            RayPacket& pack1 = *pPackets[g+1];

            ///////////////////////////////////////////////////////////////////////////////////
            // VERSION 1:  sub, mul.    
            //  MSVC likes to spill unless you spell things out for out very explicitly...
            ///////////////////////////////////////////////////////////////////////////////////
            // __m256 Bmin  = _mm256_broadcast_ss( pAABB+0 );
            // __m256 Bmax  = _mm256_broadcast_ss( pAABB+3 );
            //  __m256 O0 = _mm256_load_ps( (float*)&pack0.Ox );
            //  __m256 O1 = _mm256_load_ps( (float*)&pack1.Ox );
            //  __m256 D0 = _mm256_load_ps( (float*)&pack0.DInvx );
            //  __m256 D1 = _mm256_load_ps( (float*)&pack1.DInvx );
            // __m256 t0 =  _mm256_sub_ps( Bmin, O0 );
            // __m256 t1 =  _mm256_sub_ps( Bmax, O0 );
            // __m256 t2 =  _mm256_sub_ps( Bmin, O1 );
            // __m256 t3 =  _mm256_sub_ps( Bmax, O1 );
            // __m256 tmin0 = _mm256_mul_ps(t0,D0 );
            // __m256 tmax0 = _mm256_mul_ps(t1,D0 );
            // __m256 tmin1 = _mm256_mul_ps(t2,D1 );
            // __m256 tmax1 = _mm256_mul_ps(t3,D1 );
            // Bmin   = _mm256_broadcast_ss( pAABB+1 );
            // Bmax   = _mm256_broadcast_ss( pAABB+4 );
            // 
            // O0 = _mm256_load_ps( (float*)&pack0.Oy );
            // O1 = _mm256_load_ps( (float*)&pack1.Oy );
            // D0 = _mm256_load_ps( (float*)&pack0.DInvy );
            // D1 = _mm256_load_ps( (float*)&pack1.DInvy );
            // t0     = _mm256_sub_ps( Bmin, O0 );
            // t1     = _mm256_sub_ps( Bmax, O0 );
            // t2     = _mm256_sub_ps( Bmin, O1 );
            // t3     = _mm256_sub_ps( Bmax, O1 );
            // t0     = _mm256_mul_ps( t0, D0 );
            // t1     = _mm256_mul_ps( t1, D0 );
            // t2     = _mm256_mul_ps( t2, D1 );
            // t3     = _mm256_mul_ps( t3, D1 );
            // tmin0  = _mm256_max_ps( tmin0, t0 );
            // tmax0  = _mm256_min_ps( tmax0, t1 );
            // tmin1  = _mm256_max_ps( tmin1, t2 );
            // tmax1  = _mm256_min_ps( tmax1, t3 );
            //
            // O0 = _mm256_load_ps( (float*)&pack0.Oz );
            // O1 = _mm256_load_ps( (float*)&pack1.Oz );
            // D0 = _mm256_load_ps( (float*)&pack0.DInvz );
            // D1 = _mm256_load_ps( (float*)&pack1.DInvz );
            // Bmin   = _mm256_broadcast_ss( pAABB+2 );
            // Bmax   = _mm256_broadcast_ss( pAABB+5 ); 
            // t0     = _mm256_sub_ps( Bmin, O0 );
            // t1     = _mm256_sub_ps( Bmax, O0 );
            // t2     = _mm256_sub_ps( Bmin, O1 );
            // t3     = _mm256_sub_ps( Bmax, O1 );
            // t0     = _mm256_mul_ps( t0, D0 );
            // t1     = _mm256_mul_ps( t1, D0 );
            // t2     = _mm256_mul_ps( t2, D1 );
            // t3     = _mm256_mul_ps( t3, D1 );
            // tmin0  = _mm256_max_ps( tmin0, t0 );
            // tmax0  = _mm256_min_ps( tmax0, t1 );
            // tmin1  = _mm256_max_ps( tmin1, t2 );
            // tmax1  = _mm256_min_ps( tmax1, t3 );

          ///////////////////////////////////////////////////////////////////////////////////
          // VERSION 2:  same thing, but using fmsub   
          //  MSVC likes to spill unless you spell things out for out very explicitly...
          ///////////////////////////////////////////////////////////////////////////////////
          // __m256 ONE =  _mm256_broadcastss_ps( _mm_set_ss(1.0f) );
          // __m256 Bmin  = _mm256_broadcast_ss( pAABB+0 );
          // __m256 Bmax  = _mm256_broadcast_ss( pAABB+3 );
          //  __m256 O0 = _mm256_load_ps( (float*)&pack0.Ox );
          //  __m256 O1 = _mm256_load_ps( (float*)&pack1.Ox );
          //  __m256 D0 = _mm256_load_ps( (float*)&pack0.DInvx );
          //  __m256 D1 = _mm256_load_ps( (float*)&pack1.DInvx );
          // __m256 t0 =  _mm256_fmsub_ps( Bmin,ONE, O0 );
          // __m256 t1 =  _mm256_fmsub_ps( Bmax,ONE, O0 );
          // __m256 t2 =  _mm256_fmsub_ps( Bmin,ONE, O1 );
          // __m256 t3 =  _mm256_fmsub_ps( Bmax,ONE, O1 );
          // __m256 tmin0 = _mm256_mul_ps(t0,D0 );
          // __m256 tmax0 = _mm256_mul_ps(t1,D0 );
          // __m256 tmin1 = _mm256_mul_ps(t2,D1 );
          // __m256 tmax1 = _mm256_mul_ps(t3,D1 );
          // Bmin   = _mm256_broadcast_ss( pAABB+1 );
          // Bmax   = _mm256_broadcast_ss( pAABB+4 );
          // 
          // O0 = _mm256_load_ps( (float*)&pack0.Oy );
          // O1 = _mm256_load_ps( (float*)&pack1.Oy );
          // D0 = _mm256_load_ps( (float*)&pack0.DInvy );
          // D1 = _mm256_load_ps( (float*)&pack1.DInvy );
          // t0     = _mm256_fmsub_ps( Bmin,ONE, O0 );
          // t1     = _mm256_fmsub_ps( Bmax,ONE, O0 );
          // t2     = _mm256_fmsub_ps( Bmin,ONE, O1 );
          // t3     = _mm256_fmsub_ps( Bmax,ONE, O1 );
          // t0     = _mm256_mul_ps( t0, D0 );
          // t1     = _mm256_mul_ps( t1, D0 );
          // t2     = _mm256_mul_ps( t2, D1 );
          // t3     = _mm256_mul_ps( t3, D1 );
          // tmin0  = _mm256_max_ps( tmin0, t0 );
          // tmax0  = _mm256_min_ps( tmax0, t1 );
          // tmin1  = _mm256_max_ps( tmin1, t2 );
          // tmax1  = _mm256_min_ps( tmax1, t3 );
          // 
          // O0 = _mm256_load_ps( (float*)&pack0.Oz );
          // O1 = _mm256_load_ps( (float*)&pack1.Oz );
          // D0 = _mm256_load_ps( (float*)&pack0.DInvz );
          // D1 = _mm256_load_ps( (float*)&pack1.DInvz );
          // Bmin   = _mm256_broadcast_ss( pAABB+2 );
          // Bmax   = _mm256_broadcast_ss( pAABB+5 ); 
          // t0     = _mm256_fmsub_ps( Bmin,ONE, O0 );
          // t1     = _mm256_fmsub_ps( Bmax,ONE, O0 );
          // t2     = _mm256_fmsub_ps( Bmin,ONE, O1 );
          // t3     = _mm256_fmsub_ps( Bmax,ONE, O1 );
          // t0     = _mm256_mul_ps( t0, D0 );
          // t1     = _mm256_mul_ps( t1, D0 );
          // t2     = _mm256_mul_ps( t2, D1 );
          // t3     = _mm256_mul_ps( t3, D1 );
          // tmin0  = _mm256_max_ps( tmin0, t0 );
          // tmax0  = _mm256_min_ps( tmax0, t1 );
          // tmin1  = _mm256_max_ps( tmin1, t2 );
          // tmax1  = _mm256_min_ps( tmax1, t3 );

          ///////////////////////////////////////////////////////////////////////////////////
          // VERSION 3:  two muls, then fmsubs  
          //  MSVC likes to spill unless you spell things out for out very explicitly...
          ///////////////////////////////////////////////////////////////////////////////////
      
           __m256 Bmin  = _mm256_broadcast_ss( pAABB+0 );
           __m256 Bmax  = _mm256_broadcast_ss( pAABB+3 );
           __m256 O0    = _mm256_load_ps( (float*)&pack0.Ox );
           __m256 O1    = _mm256_load_ps( (float*)&pack1.Ox );
           __m256 D0    = _mm256_load_ps( (float*)&pack0.DInvx );
           __m256 D1    = _mm256_load_ps( (float*)&pack1.DInvx );
           __m256 r0    = _mm256_mul_ps( O0, D0 );
           __m256 r1    = _mm256_mul_ps( O1, D1 );
           __m256 tmin0 = _mm256_fmsub_ps( Bmin, D0, r0 );
           __m256 tmax0 = _mm256_fmsub_ps( Bmax, D0, r0 );
           __m256 tmin1 = _mm256_fmsub_ps( Bmin, D1, r1 );
           __m256 tmax1 = _mm256_fmsub_ps( Bmax, D1, r1 );

           Bmin  = _mm256_broadcast_ss( pAABB+1 );
           Bmax  = _mm256_broadcast_ss( pAABB+4 );
           O0    = _mm256_load_ps( (float*)&pack0.Oy );
           O1    = _mm256_load_ps( (float*)&pack1.Oy );
           D0    = _mm256_load_ps( (float*)&pack0.DInvy );
           D1    = _mm256_load_ps( (float*)&pack1.DInvy );
           r0 = _mm256_mul_ps( O0, D0 );
           r1 = _mm256_mul_ps( O1, D1 );
           __m256 t0 = _mm256_fmsub_ps( Bmin, D0, r0 ) ;
           __m256 t1 = _mm256_fmsub_ps( Bmax, D0, r0 ) ;
           __m256 t2 = _mm256_fmsub_ps( Bmin, D1, r1 ) ;
           __m256 t3 = _mm256_fmsub_ps( Bmax, D1, r1 ) ;
           tmin0 = _mm256_max_ps( tmin0, t0 );
           tmax0 = _mm256_min_ps( tmax0, t1 );
           tmin1 = _mm256_max_ps( tmin1, t2 );
           tmax1 = _mm256_min_ps( tmax1, t3 );

           Bmin  = _mm256_broadcast_ss( pAABB+2 );        
           Bmax  = _mm256_broadcast_ss( pAABB+5 ); 
           O0    = _mm256_load_ps( (float*)&pack0.Oz );
           O1    = _mm256_load_ps( (float*)&pack1.Oz );
           D0    = _mm256_load_ps( (float*)&pack0.DInvz );
           D1    = _mm256_load_ps( (float*)&pack1.DInvz );
           r0     = _mm256_mul_ps( O0, D0 );
           r1     = _mm256_mul_ps( O1, D1 );
           t0     = _mm256_fmsub_ps( Bmin, D0, r0 ) ;
           t1     = _mm256_fmsub_ps( Bmax, D0, r0 ) ;
           t2     = _mm256_fmsub_ps( Bmin, D1, r1 ) ;
           t3     = _mm256_fmsub_ps( Bmax, D1, r1 ) ;
           tmin0  = _mm256_max_ps( tmin0,t0);
           tmax0  = _mm256_min_ps( tmax0,t1);
           tmin1  = _mm256_max_ps( tmin1,t2);
           tmax1  = _mm256_min_ps( tmax1,t3);
            
        
            // andnot -> uses sign-bit trick for tmax>=0
            __m256 l0   = _mm256_min_ps( tmax0, pack0.TMax );
            __m256 l1   = _mm256_min_ps( tmax1, pack1.TMax );
            __m256 hit0 = _mm256_andnot_ps( tmax0, _mm256_cmp_ps(tmin0,l0,_CMP_LE_OQ) );
            __m256 hit1 = _mm256_andnot_ps( tmax1, _mm256_cmp_ps(tmin1,l1,_CMP_LE_OQ) );
        
            size_t mask0 = (size_t)_mm256_movemask_ps(hit0);
            size_t mask1 = (size_t)_mm256_movemask_ps(hit1);
            size_t nMergedMask = (mask1<<8)|mask0;
       
            *((uint16*)(pMasks+g)) = (uint16) nMergedMask;
            nHitPopulation += _mm_popcnt_u64(nMergedMask);        
        }
        
        for( ; g<nGroups; g++ )
        {
            RayPacket& pack = *pPackets[g];
            __m256 Bmin0  = _mm256_broadcast_ss( pAABB+0 );
            __m256 Bmax0  = _mm256_broadcast_ss( pAABB+3 );
            __m256 tmin0 = _mm256_mul_ps( _mm256_sub_ps( Bmin0, pack.Ox ),pack.DInvx );
            __m256 tmax0 = _mm256_mul_ps( _mm256_sub_ps( Bmax0, pack.Ox ),pack.DInvx );
            
            Bmin0  = _mm256_broadcast_ss( pAABB+1 );
            Bmax0  = _mm256_broadcast_ss( pAABB+4 );
            tmin0  = _mm256_max_ps( tmin0, _mm256_mul_ps( _mm256_sub_ps( Bmin0, pack.Oy ),pack.DInvy  ) );
            tmax0  = _mm256_min_ps( tmax0, _mm256_mul_ps( _mm256_sub_ps( Bmax0, pack.Oy ),pack.DInvy  ) );
                
            Bmin0  = _mm256_broadcast_ss( pAABB+2 );
            Bmax0  = _mm256_broadcast_ss( pAABB+5 );
            tmin0  = _mm256_max_ps( tmin0, _mm256_mul_ps( _mm256_sub_ps( Bmin0, pack.Oz ),pack.DInvz ) );
            tmax0  = _mm256_min_ps( tmax0, _mm256_mul_ps( _mm256_sub_ps( Bmax0, pack.Oz ),pack.DInvz ) );
        
            __m256 l0  = _mm256_min_ps( tmax0, pack.TMax );
                
            __m256 hit = _mm256_andnot_ps( tmax0, _mm256_cmp_ps( tmin0, l0, _CMP_LE_OQ ) );

            size_t mask = (size_t)_mm256_movemask_ps(hit);
            nHitPopulation += _mm_popcnt_u64(mask);
            pMasks[g] = (uint8) mask;
        }

        return nHitPopulation;
    }


    static const __declspec(align(32))uint32 SHIFTS[]  = {0,1,2,3,4,5,6,7};

    static void __fastcall ReorderRays( StackFrame& frame, size_t nGroups )
    {        
        RayPacket** pPackets = frame.pActivePackets;

        for( size_t i=0; i<nGroups; i++ )
            _mm_prefetch( (char*)(pPackets[i]->RayOffsets), _MM_HINT_T0 );
   
        uint32 pIDs[MAX_TRACER_SIZE];
   
        size_t nHitLoc  = 0;
        size_t nMissLoc = 8*nGroups;

        const char* pRays = (const char*) frame.pRays;
        for( size_t i=0; i<nGroups; i++ )
        {
            uint32* __restrict pPacketRayIDs = pPackets[i]->RayOffsets;
   
            // Turn the 8-bit mask into 8 packed bytes
            const unsigned __int64 ONE_BYTES = 0x0101010101010101;
            unsigned __int64 hit    = _pdep_u64( frame.pMasks[i], ONE_BYTES);
            unsigned __int64 miss   = hit ^ (ONE_BYTES);
            __m128i vhit      = _mm_cvtsi64_si128(hit);
            __m128i vmiss     = _mm_cvtsi64_si128(miss);

            __m128i vhit_mask = _mm_sub_epi8(vmiss,_mm_cvtsi64_si128(ONE_BYTES)); // 0 if miss, 0xff if hit
                    vhit_mask = _mm_cvtepi8_epi16(vhit_mask); // 0 or 0xffff

            // prefix sum via shifts+adds.  Thanks to @rygorous for the idea
            //  We could also do this by packing into an __m256 and doing hit/miss in parallel
            //  
            //  But using __m128 is nicer bc Haswell can dual-issue nearly all of these ops
            //    and it avoids the expensive cross-lane pack/unpack
            //
            __m128i prefix_hit  = _mm_add_epi8(vhit,   _mm_slli_si128(vhit,1));  
            __m128i prefix_miss = _mm_add_epi8(vmiss,  _mm_slli_si128(vmiss,1));
                    prefix_hit  = _mm_add_epi8(prefix_hit,   _mm_slli_si128(prefix_hit,2));
                    prefix_miss = _mm_add_epi8(prefix_miss,  _mm_slli_si128(prefix_miss,2));
                    prefix_hit  = _mm_add_epi8(prefix_hit,   _mm_slli_si128(prefix_hit,4));
                    prefix_miss = _mm_add_epi8(prefix_miss,  _mm_slli_si128(prefix_miss,4));
                    prefix_hit  = _mm_sub_epi8(prefix_hit, vhit);  // exclude ray itself from the prefix sum
                    prefix_miss = _mm_sub_epi8(prefix_miss,vmiss);
                    prefix_hit  = _mm_cvtepi8_epi16(prefix_hit);
                    prefix_miss = _mm_cvtepi8_epi16(prefix_miss);

            __m128i hitBase  = _mm_broadcastw_epi16(_mm_cvtsi64_si128(nHitLoc));
            __m128i missBase = _mm_broadcastw_epi16(_mm_cvtsi64_si128(nMissLoc-1));
            prefix_hit  = _mm_add_epi16(prefix_hit,hitBase);
            prefix_miss = _mm_sub_epi16(missBase, prefix_miss);

            size_t nHitPop = _mm_popcnt_u64(frame.pMasks[i]);
            nHitLoc  += nHitPop ;
            nMissLoc -= (8-nHitPop);

            __m128i addr = _mm_blendv_epi8(prefix_miss,prefix_hit,vhit_mask);
           
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[0],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[1],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[2],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[3],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[4],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[5],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[6],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[7],_MM_HINT_T0 );

            pIDs[ _mm_extract_epi16(addr,0) ] = pPacketRayIDs[0]; 
            pIDs[ _mm_extract_epi16(addr,1) ] = pPacketRayIDs[1]; 
            pIDs[ _mm_extract_epi16(addr,2) ] = pPacketRayIDs[2]; 
            pIDs[ _mm_extract_epi16(addr,3) ] = pPacketRayIDs[3]; 
            pIDs[ _mm_extract_epi16(addr,4) ] = pPacketRayIDs[4]; 
            pIDs[ _mm_extract_epi16(addr,5) ] = pPacketRayIDs[5]; 
            pIDs[ _mm_extract_epi16(addr,6) ] = pPacketRayIDs[6]; 
            pIDs[ _mm_extract_epi16(addr,7) ] = pPacketRayIDs[7]; 

              
            /*
            __m256i ONES    = BROADCASTINT(1);
            __m256i INDEX   = _mm256_load_si256((__m256i*)SHIFTS);

            __m256i masks   = BROADCASTINT(frame.pMasks[i]);
                    masks  = _mm256_srlv_epi32(masks, INDEX ); // mask >> lane_idx
            __m256i hits   = _mm256_and_si256(masks,ONES);     // value is 1 for each lane if it hit
        
       
            // Hit ray IDs go in from the start
            // Miss ray IDs go in from the end.  In both cases we can figure out
            //   the address of each ray using a prefix sum across the hit/miss masks
            //  
           
            // We want to sum things as follows:  where: hxy is sum(hx...hy)       
            //
            // <--- Columns are SIMD lanes ---->
            // --------------------------------
            //  0   0  0  0     0   0   0    0
            //  0  h0  0  h2    0   h4  0    h6
            //  0   0 h01 h01   0   0  h45  h45  
            //  0   0  0   0  h03 h03  h03  h03
            //
        



            __m256i Doubles = _mm256_hadd_epi32(hits,hits);        // h0+h1, h2+h3, h0+h1,h2+h3,  h4+h5, h6+h7, h4+h5, h6+h7
            __m256i Quads   = _mm256_hadd_epi32(Doubles,Doubles);  // h0-h3, h0-h3, h0-h3,h0-h3,  h4-h7, h4-h7, h4-h7, h4-h7
            Quads = _mm256_inserti128_si256(_mm256_setzero_si256(),_mm256_castsi256_si128(Quads),0x1); // 0,0,0,0, h0-h3 .....
     
            __m256i m0  = _mm256_shuffle_epi32( hits,     SHUFFLE(0,0,2,2) ); // h0, h0, h2, h2, h4, h4, h6, h6
            __m256i m1  = _mm256_shuffle_epi32( Doubles,  SHUFFLE(0,0,0,0) ); // h0+h1 ...., h4+h5,....
            m0  = _mm256_blend_epi32( _mm256_setzero_si256(), m0, 0xAA ); // 10101010 ->  0 h0  0   h2 0 h4  0   h6
            m1  = _mm256_blend_epi32( _mm256_setzero_si256(), m1, 0xCC ); // 11001100 ->  0  0 h01 h01 0  0 h45 h45

            __m256i hitPrefix  = _mm256_add_epi32(Quads,_mm256_add_epi32(m0,m1));  // prefix sum on hit rays
            __m256i missPrefix = _mm256_sub_epi32(INDEX,hitPrefix); // prefix sum on missed rays
            __m256i missMask   = _mm256_sub_epi32(hits,ONES);       // 0xffffffff if lane is a miss, 0 otherwise
            __m256i hitAddr    = BROADCASTINT(nHitLoc);
            __m256i missAddr   = BROADCASTINT(nMissLoc-1);
            hitAddr            = _mm256_add_epi32(hitAddr,hitPrefix);
            missAddr           = _mm256_sub_epi32(missAddr,missPrefix);
            __m256i addr       = _mm256_blendv_epi8(hitAddr,missAddr,missMask);
        
            
            size_t nHitPop = _mm_popcnt_u64(frame.pMasks[i]);
            nHitLoc  += nHitPop ;
            nMissLoc -= (8-nHitPop);
        
            __m128i hi = _mm256_extracti128_si256(addr,1);
            __m128i lo = _mm256_extracti128_si256(addr,0);
            
        
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[0],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[1],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[2],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[3],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[4],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[5],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[6],_MM_HINT_T0 );
            _mm_prefetch( (char*)(pRays)+pPacketRayIDs[7],_MM_HINT_T0 );

            pIDs[ _mm_extract_epi32(lo,0) ] = pPacketRayIDs[0]; 
            pIDs[ _mm_extract_epi32(lo,1) ] = pPacketRayIDs[1]; 
            pIDs[ _mm_extract_epi32(lo,2) ] = pPacketRayIDs[2]; 
            pIDs[ _mm_extract_epi32(lo,3) ] = pPacketRayIDs[3]; 
            pIDs[ _mm_extract_epi32(hi,0) ] = pPacketRayIDs[4]; 
            pIDs[ _mm_extract_epi32(hi,1) ] = pPacketRayIDs[5]; 
            pIDs[ _mm_extract_epi32(hi,2) ] = pPacketRayIDs[6]; 
            pIDs[ _mm_extract_epi32(hi,3) ] = pPacketRayIDs[7]; 
        */
        
        
           // The bit-twiddling loops that it replaced...
           /*
            // misses in back
            size_t mmiss = (~frame.pMasks[i])&0xff;
            while( mmiss )
            {
                size_t k      = _tzcnt_u64(mmiss);
                mmiss         = _blsr_u64(mmiss);
                size_t nID = pPacketRayIDs[k];
                uint32* p = pIDs + (--nMissLoc);
                *p = nID;
                _mm_prefetch( pRays + nID, _MM_HINT_T0 );
            } 

            // hits in fromt
            size_t mhit    = frame.pMasks[i];
            while( mhit )
            {
                size_t k     = _tzcnt_u64(mhit);
                mhit         = _blsr_u64(mhit);
                size_t nID = pPacketRayIDs[k];
                uint32* p = pIDs + (nHitLoc++);
                *p = nID;
                _mm_prefetch( pRays + nID, _MM_HINT_T0 );
            }
            */
        }

    
            
        ReadRaysLoopArgs args;
        args.pPackets = pPackets;
        args.pRayIDs = pIDs;
        args.pRays = (const byte*)pRays;
        ReadRaysLoop(args,nGroups);
    }



    
    static void __forceinline TransposePacket( __m256* pOut, __m256* p )
    {
        // Transpose a set of 8 m256's  
        //  a 0 1 2 3 4 5 6 7
        //  b ....
        //  c ....

        //  ===>
        //  a0 b0 c0 d0 e0 f0 g0 h0
        //  a1 b1 c1 d1 e1 f1 g1 h1 
        //  ....
        //
        __m256* pLower = (__m256*)p;
        __m256* pUpper = (__m256*)(((char*)p)+16);
        
        #define LOADPS(x) _mm_load_ps((float*)(x))
        __m256 l0 = _mm256_set_m128( LOADPS(pLower + 4), LOADPS(pLower+0) ); //0123(a) 0123(e)
        __m256 l1 = _mm256_set_m128( LOADPS(pLower + 5), LOADPS(pLower+1) ); //0123(b) 0123(f)
        __m256 l2 = _mm256_set_m128( LOADPS(pLower + 6), LOADPS(pLower+2) ); //0123(c) 0123(g)
        __m256 l3 = _mm256_set_m128( LOADPS(pLower + 7), LOADPS(pLower+3) ); //0123(d) 0123(h)
        __m256 l4 = _mm256_set_m128( LOADPS(pUpper + 4), LOADPS(pUpper+0) ); //4567(a) 4567(e)
        __m256 l5 = _mm256_set_m128( LOADPS(pUpper + 5), LOADPS(pUpper+1) ); //4567(b) 4567(f)
        __m256 l6 = _mm256_set_m128( LOADPS(pUpper + 6), LOADPS(pUpper+2) ); //4567(c) 4567(g)
        __m256 l7 = _mm256_set_m128( LOADPS(pUpper + 7), LOADPS(pUpper+3) ); //4567(d) 4567(h)
        #undef LOADPS

        __m256 t0 = _mm256_shuffle_ps( l0, l1, SHUFFLE(0,1,0,1) ); // a0a1 b0b1  e0e1  f0f1
        __m256 t1 = _mm256_shuffle_ps( l2, l3, SHUFFLE(0,1,0,1) ); // c0c1 d0d1  g0g1  h0h1
        __m256 t2 = _mm256_unpacklo_ps(t0,t1);                     // a0c0  a1c1  e0 g0  e1g1
        __m256 t3 = _mm256_unpackhi_ps(t0,t1);                     // a0c0  a1c1  e0 g0  e1g1
        t0 = _mm256_unpacklo_ps(t2,t3);                            // a0c0  a1c1  e0 g0  e1g1
        t1 = _mm256_unpackhi_ps(t2,t3);                            // b0d0  b1d1  f0 h0  f1h1
        _mm256_store_ps( (float*)(pOut+0),t0);                     // a0 b0 c0 d0 e0 f0 g0 h0
        _mm256_store_ps( (float*)(pOut+1),t1);                     // a1 b1 c1 d1 e1 f1 g1 h1

        t0 = _mm256_shuffle_ps( l0, l1, SHUFFLE(2,3,2,3) ); // 2 and 3
        t1 = _mm256_shuffle_ps( l2, l3, SHUFFLE(2,3,2,3) ); // 
        t2 = _mm256_unpacklo_ps(t0,t1);                     // 
        t3 = _mm256_unpackhi_ps(t0,t1);                     // 
        t0 = _mm256_unpacklo_ps(t2,t3);                     // 
        t1 = _mm256_unpackhi_ps(t2,t3);                     // 
        _mm256_store_ps( (float*)(pOut+2),t0);              // 
        _mm256_store_ps( (float*)(pOut+3),t1);              // 

        t0 = _mm256_shuffle_ps( l4, l5, SHUFFLE(0,1,0,1) ); // 4 and 5
        t1 = _mm256_shuffle_ps( l6, l7, SHUFFLE(0,1,0,1) ); // 
        t2 = _mm256_unpacklo_ps(t0,t1);                     // 
        t3 = _mm256_unpackhi_ps(t0,t1);                     // 
        t0 = _mm256_unpacklo_ps(t2,t3);                     // 
        t1 = _mm256_unpackhi_ps(t2,t3);                     // 
        _mm256_store_ps( (float*)(pOut+4),t0);              // 
        _mm256_store_ps( (float*)(pOut+5),t1);              // 

        t0 = _mm256_shuffle_ps( l4, l5, SHUFFLE(2,3,2,3) ); // 6 and 7
        t1 = _mm256_shuffle_ps( l6, l7, SHUFFLE(2,3,2,3) ); // 
        t2 = _mm256_unpacklo_ps(t0,t1);                     // 
        t3 = _mm256_unpackhi_ps(t0,t1);                     // 
        t0 = _mm256_unpacklo_ps(t2,t3);                     // 
        t1 = _mm256_unpackhi_ps(t2,t3);                     // 
        _mm256_store_ps( (float*)(pOut+6),t0);              // 
        _mm256_store_ps( (float*)(pOut+7),t1);              // 
    }


    static size_t RemoveMissedGroups( RayPacket** __restrict pGroups, uint8* __restrict pMasks, size_t nGroups )
    {
        size_t nHit=0;
        while(1)
        {
            // skip in-place hits at beginning
            while(pMasks[nHit])
            {
                nHit++;
                if( nHit == nGroups )
                    return nGroups;
            }
        
            // skip in-place misses at end
            size_t mask;
            do
            {
                --nGroups;
                if( nHit == nGroups )
                    return nGroups;
                mask = pMasks[nGroups];
            
            } while( !mask );

            RayPacket* h = pGroups[nHit];
            RayPacket* m = pGroups[nGroups];
            pGroups[nHit] = m;
            pGroups[nGroups] = h;
            pMasks[nHit] = (uint8) mask;
        } 
    }


    static void BuildPacketsByOctant( RayPacket* __restrict pPackets, Tracer* pTracer, uint* __restrict pOctantPacketCounts )
    {
        
        uint8* pRayOctants = pTracer->pRayOctants;
        uint16* pOctantRayCounts = pTracer->pOctantCounts;

        uint nRays                     = pTracer->nRays;
        Ray* __restrict pRays     = pTracer->pRays;

        // counts into offsets via prefix sum
        uint pOctantOffsets[8];
        pOctantOffsets[0] = 0;
        for( size_t i=1; i<8; i++ )
            pOctantOffsets[i] = pOctantOffsets[i-1] + pOctantRayCounts[i-1];

        // bin rays by octant
        uint16 pIDsByOctant[MAX_TRACER_SIZE];
        for( size_t i=0; i<nRays; i++ )
            pIDsByOctant[pOctantOffsets[pRayOctants[i]]++] = (uint16) i;

        for( size_t i=0; i<8; i++ )
        {
            uint nOctantRays = pOctantRayCounts[i];
            if( !nOctantRays )
                continue;

            uint offs = pOctantOffsets[i] - pOctantRayCounts[i];
            uint nPacks = RoundUp8(nOctantRays)/8;

             // build a padded ID list
            __declspec(align(16)) uint32 IDs[MAX_TRACER_SIZE];
            for( uint k=0; k<nOctantRays; k++ )
                IDs[k] = pIDsByOctant[offs+k]*sizeof(Ray);
            uint32 last = IDs[nOctantRays-1];
            while( nOctantRays & 7 )
                IDs[nOctantRays++] = last;

            for( uint p=0; p<nPacks; p++ )
            {
                RayPacket* pPack = pPackets++;
                ReadRays(pPack,(const byte*)pRays,IDs + 8*p );
            }

            pOctantPacketCounts[i] = nPacks;
        }
    }

    



}}