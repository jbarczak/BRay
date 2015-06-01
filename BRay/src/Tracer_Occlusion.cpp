//=====================================================================================================================
//
//   Tracer.cpp
//
//   B-Ray occlusion tracer
//   
//   Copyright 2015 Joshua Barczak
//
//   LICENSE:  This source code is distributed under the terms of the GNU GPL v2
//         
//
//=====================================================================================================================

#include "Tracer_Common.inl"


namespace BRay{
namespace _INTERNAL{

    

    static size_t TriISectPacket_Preproc_List_Occ( size_t mask, const TriList* __restrict pList, RayPacket* __restrict pPacket, StackFrame* __restrict pFrame )
    {
        __m256 h    = _mm256_setzero_ps();
        __m256 TMax = pPacket->TMax;
    
        __m256 D[3];
        ReadDirs(D,pPacket, pFrame->pRays);

        size_t m = mask;
        size_t nTris = pList->GetTriCount();
        for( size_t i=0; i<nTris; i++ )
        {
            const PreprocessedTri* __restrict pTri = pList->GetTriList()+i;
            const float* __restrict P0 = pTri->P0;
            const float* __restrict v10 = pTri->v10;
            const float* __restrict v02 = pTri->v02;
            const float* __restrict v10x02 = pTri->v10x02;

            _mm_prefetch( (char*) (pTri+1), _MM_HINT_T0 );

            __m256 vDx = D[0];
            __m256 vDy = D[1];
            __m256 vDz = D[2];
            __m256 v0A[3] = {
                _mm256_sub_ps( _mm256_broadcast_ss( &P0[0] ), pPacket->Ox ),
                _mm256_sub_ps( _mm256_broadcast_ss( &P0[1] ), pPacket->Oy ),
                _mm256_sub_ps( _mm256_broadcast_ss( &P0[2] ), pPacket->Oz ),
            };
            __m256 v0AxD[] = {
                _mm256_fmsub_ps( v0A[1], vDz, _mm256_mul_ps( v0A[2], vDy ) ),
                _mm256_fmsub_ps( v0A[2], vDx, _mm256_mul_ps( v0A[0], vDz ) ),
                _mm256_fmsub_ps( v0A[0], vDy, _mm256_mul_ps( v0A[1], vDx ) )
            };

           //V = ((p1 - p0)x(p0 -p2)).d
           //Va = ((p1 - p0)x(p0 -p2)).(p0 -a)
           //V1 = ((p0 - a)×d).(p0 -p2)
           //V2 = ((p0 - a)×d).(p1 -p0)

            __m256 v10x02_x = _mm256_broadcast_ss(&v10x02[0]);
            __m256 v10x02_y = _mm256_broadcast_ss(&v10x02[1]);
            __m256 v10x02_z = _mm256_broadcast_ss(&v10x02[2]);

            __m256 T =  _mm256_fmadd_ps( v10x02_z, v0A[2], 
                                          _mm256_fmadd_ps( v10x02_y, v0A[1],
                                          _mm256_mul_ps( v10x02_x, v0A[0] ) ));

            __m256 V = _mm256_fmadd_ps( v10x02_z, vDz,
                            _mm256_fmadd_ps( v10x02_y, vDy, 
                                                _mm256_mul_ps( v10x02_x, vDx ) ) );
            V = RCPNR(V);

            __m256 v02_x = _mm256_broadcast_ss(&v02[0]);
            __m256 v02_y = _mm256_broadcast_ss(&v02[1]);
            __m256 v02_z = _mm256_broadcast_ss(&v02[2]);
            
            __m256 A = _mm256_fmadd_ps( v0AxD[2], v02_z,
                                         _mm256_fmadd_ps( v0AxD[1],  v02_y,
                                         _mm256_mul_ps( v0AxD[0],  v02_x )));

            __m256 v10_x = _mm256_broadcast_ss(&v10[0]);
            __m256 v10_y = _mm256_broadcast_ss(&v10[1]);
            __m256 v10_z = _mm256_broadcast_ss(&v10[2]);
            __m256 B = _mm256_fmadd_ps( v0AxD[2], v10_z,
                                         _mm256_fmadd_ps( v0AxD[1],  v10_y,
                                         _mm256_mul_ps( v0AxD[0],  v10_x )));


            A = _mm256_mul_ps( A, V );
            B = _mm256_mul_ps( B, V );
            T = _mm256_mul_ps( T, V );
 
            __m256 front = _mm256_and_ps( _mm256_cmp_ps( T,_mm256_setzero_ps(), _CMP_GT_OQ),
                                          _mm256_cmp_ps(TMax,T,_CMP_GT_OQ) );
            __m256 in    = _mm256_and_ps( _mm256_cmp_ps(A,_mm256_setzero_ps(),_CMP_GE_OQ ),
                                          _mm256_cmp_ps(B,_mm256_setzero_ps(),_CMP_GE_OQ ) );

            in = _mm256_and_ps( in, _mm256_cmp_ps( _mm256_add_ps(A,B), _mm256_set1_ps(1.0f), _CMP_LE_OQ ) );

            __m256 hit = _mm256_and_ps( in, front ) ;
            h = _mm256_or_ps(h,hit);

            m = _mm256_movemask_ps(h);
            if( m == 0xff )
                break;
        }

        // force TMax=-1 for hit rays so that all subsequent intersection tests on them will miss
        __m256 MINUS_ONE = _mm256_broadcastss_ps( _mm_set_ss(-1.0f) );
        TMax = _mm256_blendv_ps( TMax, MINUS_ONE, h );
        _mm256_store_ps( (float*)&(pPacket->TMax), TMax );

        // mark hits in output array and set t=0 in ray structs
        size_t k = m;
        size_t idx = _tzcnt_u64(k);
        while( idx < 64 )
        {
            size_t rid = pPacket->RayOffsets[idx]/sizeof(Ray);
            k = _blsr_u64(k);
            idx = _tzcnt_u64(k);
            pFrame->pOcclusion[rid] = 0xff;
            pFrame->pRays[rid].tmax = -1.0f;
        }


        return m;
    }


    static size_t PacketTrace_Octant_Occ( RayPacket& pack, size_t mask, uint octant, void* pStackBottom, BVHNode* pRoot, StackFrame& frame )
    {

        BVHNode** pStack = (BVHNode**) pStackBottom;
        *(pStack++) = pRoot;

        // positive sign:  first=0, last=3
        // negative sign:  first=3, last=0
        uint xfirst = (octant&1) ? 3 : 0;
        uint yfirst = (octant&2) ? 3 : 0;
        uint zfirst = (octant&4) ? 3 : 0;

    
        __m256 OD[] = { 
            _mm256_mul_ps( pack.DInvx, pack.Ox ),
            _mm256_mul_ps( pack.DInvy, pack.Oy ),
            _mm256_mul_ps( pack.DInvz, pack.Oz ),
        };
          
            // pick min/max if octant sign is zero, reverse otherwise
        __m128i vOctant = _mm_set_epi32( 0, octant&4 ? 0xffffffff : 0, 
                                            octant&2 ? 0xffffffff : 0,
                                            octant&1 ? 0xffffffff : 0 );
        __m256 octant_select = _mm256_castsi256_ps(_mm256_broadcastsi128_si256(vOctant));
         

        while( pStack != pStackBottom && mask != 0xff )
        {
            BVHNode* pN = *(--pStack);
            if( pN->IsLeaf() )
            {
                // visit leaf
                const TriList* pList = pN->GetTriList();
                mask |= TriISectPacket_Preproc_List_Occ( mask, pList, &pack, &frame );
            }
            else
            {
                BVHNode* pNLeft  = pN->GetLeftChild();
                BVHNode* pNRight = pN->GetRightChild();
                const float* __restrict pLeft  = (const float*)pNLeft->GetAABB();
                const float* __restrict pRight = (const float*)pNRight->GetAABB();
                _mm_prefetch( (char*) pNLeft->GetPrefetch(),  _MM_HINT_T0 );
                _mm_prefetch( (char*) pNRight->GetPrefetch(), _MM_HINT_T0 );

                /*
                // test children, push far ones, descend to near ones
                __m256 Bmin0  = _mm256_broadcast_ss( pLeft  + xfirst      );
                __m256 Bmax0  = _mm256_broadcast_ss( pLeft  + (xfirst^3)  );
                __m256 Bmin1  = _mm256_broadcast_ss( pRight + xfirst      );
                __m256 Bmax1  = _mm256_broadcast_ss( pRight + (xfirst^3)  );

                __m256 tmin0 = _mm256_mul_ps( _mm256_sub_ps( Bmin0, pack.Ox ),pack.DInvx );
                __m256 tmax0 = _mm256_mul_ps( _mm256_sub_ps( Bmax0, pack.Ox ),pack.DInvx );
                __m256 tmin1 = _mm256_mul_ps( _mm256_sub_ps( Bmin1, pack.Ox ),pack.DInvx );
                __m256 tmax1 = _mm256_mul_ps( _mm256_sub_ps( Bmax1, pack.Ox ),pack.DInvx );
                                                 
                Bmin0  = _mm256_broadcast_ss( pLeft  + 1 + yfirst      );
                Bmax0  = _mm256_broadcast_ss( pLeft  + 1 + (yfirst^3)  );
                Bmin1  = _mm256_broadcast_ss( pRight + 1 + yfirst      );
                Bmax1  = _mm256_broadcast_ss( pRight + 1 + (yfirst^3)  );
                tmin0  = _mm256_max_ps( tmin0, _mm256_mul_ps( _mm256_sub_ps( Bmin0, pack.Oy ),pack.DInvy  ) );
                tmax0  = _mm256_min_ps( tmax0, _mm256_mul_ps( _mm256_sub_ps( Bmax0, pack.Oy ),pack.DInvy  ) );
                tmin1  = _mm256_max_ps( tmin1, _mm256_mul_ps( _mm256_sub_ps( Bmin1, pack.Oy ),pack.DInvy  ) );
                tmax1  = _mm256_min_ps( tmax1, _mm256_mul_ps( _mm256_sub_ps( Bmax1, pack.Oy ),pack.DInvy  ) );
    
                Bmin0  = _mm256_broadcast_ss( pLeft  + 2 + zfirst     );
                Bmax0  = _mm256_broadcast_ss( pLeft  + 2 + (zfirst^3) );
                Bmin1  = _mm256_broadcast_ss( pRight + 2 + zfirst     );
                Bmax1  = _mm256_broadcast_ss( pRight + 2 + (zfirst^3) );
                tmin0  = _mm256_max_ps( tmin0, _mm256_mul_ps( _mm256_sub_ps( Bmin0, pack.Oz ),pack.DInvz ) );
                tmax0  = _mm256_min_ps( tmax0, _mm256_mul_ps( _mm256_sub_ps( Bmax0, pack.Oz ),pack.DInvz ) );
                tmin1  = _mm256_max_ps( tmin1, _mm256_mul_ps( _mm256_sub_ps( Bmin1, pack.Oz ),pack.DInvz ) );
                tmax1  = _mm256_min_ps( tmax1, _mm256_mul_ps( _mm256_sub_ps( Bmax1, pack.Oz ),pack.DInvz ) );
            */

                /*
                // test children, push far ones, descend to near ones
                __m256 Bmin0  = _mm256_broadcast_ss( pLeft  + xfirst      );
                __m256 Bmax0  = _mm256_broadcast_ss( pLeft  + (xfirst^3)  );
                __m256 Bmin1  = _mm256_broadcast_ss( pRight + xfirst      );
                __m256 Bmax1  = _mm256_broadcast_ss( pRight + (xfirst^3)  );
               
                __m256 tmin0 = _mm256_fmsub_ps( Bmin0, pack.DInvx, rx );
                __m256 tmax0 = _mm256_fmsub_ps( Bmax0, pack.DInvx, rx );
                __m256 tmin1 = _mm256_fmsub_ps( Bmin1, pack.DInvx, rx );
                __m256 tmax1 = _mm256_fmsub_ps( Bmax1, pack.DInvx, rx );

                Bmin0  = _mm256_broadcast_ss( pLeft  + 1 + yfirst      );
                Bmax0  = _mm256_broadcast_ss( pLeft  + 1 + (yfirst^3)  );
                Bmin1  = _mm256_broadcast_ss( pRight + 1 + yfirst      );
                Bmax1  = _mm256_broadcast_ss( pRight + 1 + (yfirst^3)  );
                tmin0 = _mm256_max_ps( tmin0, _mm256_fmsub_ps( Bmin0, pack.DInvy, ry ) );
                tmax0 = _mm256_min_ps( tmax0, _mm256_fmsub_ps( Bmax0, pack.DInvy, ry ) );
                tmin1 = _mm256_max_ps( tmin1, _mm256_fmsub_ps( Bmin1, pack.DInvy, ry ) );
                tmax1 = _mm256_min_ps( tmax1, _mm256_fmsub_ps( Bmax1, pack.DInvy, ry ) );

                Bmin0  = _mm256_broadcast_ss( pLeft  + 2 + zfirst     );
                Bmax0  = _mm256_broadcast_ss( pLeft  + 2 + (zfirst^3) );
                Bmin1  = _mm256_broadcast_ss( pRight + 2 + zfirst     );
                Bmax1  = _mm256_broadcast_ss( pRight + 2 + (zfirst^3) );
                tmin0 = _mm256_max_ps( tmin0, _mm256_fmsub_ps( Bmin0, pack.DInvz, rz ) );
                tmax0 = _mm256_min_ps( tmax0, _mm256_fmsub_ps( Bmax0, pack.DInvz, rz ) );
                tmin1 = _mm256_max_ps( tmin1, _mm256_fmsub_ps( Bmin1, pack.DInvz, rz ) );
                tmax1 = _mm256_min_ps( tmax1, _mm256_fmsub_ps( Bmax1, pack.DInvz, rz ) );
                */


                
                // fetch AABB
                // bbmin(xyz), bbmax(xyz)
                BVHNode* pL = pN->GetLeftChild();
                BVHNode* pR = pN->GetRightChild();
                __m256 min0 = _mm256_broadcast_ps( (const __m128*)(pLeft)   );
                __m256 min1 = _mm256_broadcast_ps( (const __m128*)(pRight)   );
                __m256 max0 = _mm256_broadcast_ps( (const __m128*)(pLeft+3) );
                __m256 max1 = _mm256_broadcast_ps( (const __m128*)(pRight+3) );

                // swap planes based on octant
                __m256 bbmin0 = _mm256_blendv_ps(min0, max0, octant_select );
                __m256 bbmax0 = _mm256_blendv_ps(max0, min0, octant_select );
                __m256 bbmin1 = _mm256_blendv_ps(min1, max1, octant_select );
                __m256 bbmax1 = _mm256_blendv_ps(max1, min1, octant_select );
               
                _mm_prefetch( (char*) pNLeft->GetPrefetch(),  _MM_HINT_T0 );
                _mm_prefetch( (char*) pNRight->GetPrefetch(), _MM_HINT_T0 );


                // axis tests
                __m256 D = _mm256_load_ps( (float*)&pack.DInvx);
                __m256 O = _mm256_load_ps( (float*) (OD+0) );
                __m256 Bmin0 = _mm256_permute_ps(bbmin0, 0x00);
                __m256 Bmax0 = _mm256_permute_ps(bbmax0, 0x00);
                __m256 Bmin1 = _mm256_permute_ps(bbmin1, 0x00);
                __m256 Bmax1 = _mm256_permute_ps(bbmax1, 0x00);
                __m256 tmin0 = _mm256_fmsub_ps( Bmin0, D, O );
                __m256 tmax0 = _mm256_fmsub_ps( Bmax0, D, O );
                __m256 tmin1 = _mm256_fmsub_ps( Bmin1, D, O );
                __m256 tmax1 = _mm256_fmsub_ps( Bmax1, D, O );

                D = _mm256_load_ps( (float*)&pack.DInvy);
                O = _mm256_load_ps( (float*) (OD+1) );
                Bmin0 = _mm256_permute_ps(bbmin0, 0x55); // 0101
                Bmax0 = _mm256_permute_ps(bbmax0, 0x55);
                Bmin1 = _mm256_permute_ps(bbmin1, 0x55);
                Bmax1 = _mm256_permute_ps(bbmax1, 0x55);
                __m256 t0 = _mm256_fmsub_ps( Bmin0, D, O );
                __m256 t1 = _mm256_fmsub_ps( Bmax0, D, O );
                __m256 t2 = _mm256_fmsub_ps( Bmin1, D, O );
                __m256 t3 = _mm256_fmsub_ps( Bmax1, D, O );
                tmin0 = _mm256_max_ps( tmin0, t0 );
                tmax0 = _mm256_min_ps( tmax0, t1 );
                tmin1 = _mm256_max_ps( tmin1, t2 );
                tmax1 = _mm256_min_ps( tmax1, t3 );

                D = _mm256_load_ps( (float*)&pack.DInvz);
                O = _mm256_load_ps( (float*) (OD+2) );               
                Bmin0 = _mm256_permute_ps(bbmin0, 0xAA); // 1010
                Bmax0 = _mm256_permute_ps(bbmax0, 0xAA);
                Bmin1 = _mm256_permute_ps(bbmin1, 0xAA);
                Bmax1 = _mm256_permute_ps(bbmax1, 0xAA);
                t0 = _mm256_fmsub_ps( Bmin0, D, O );
                t1 = _mm256_fmsub_ps( Bmax0, D, O );
                t2 = _mm256_fmsub_ps( Bmin1, D, O );
                t3 = _mm256_fmsub_ps( Bmax1, D, O );
                tmin0 = _mm256_max_ps( tmin0, t0  );
                tmax0 = _mm256_min_ps( tmax0, t1  );
                tmin1 = _mm256_max_ps( tmin1, t2  );
                tmax1 = _mm256_min_ps( tmax1, t3  );

                __m256 limL  = _mm256_min_ps( tmax0, pack.TMax );
                __m256 limR  = _mm256_min_ps( tmax1, pack.TMax );

                // using sign-bit trick for tmax >= 0
                __m256 hitL = _mm256_andnot_ps( tmax0, _mm256_cmp_ps( tmin0, limL, _CMP_LE_OQ ) );
                __m256 hitR = _mm256_andnot_ps( tmax1, _mm256_cmp_ps( tmin1, limR, _CMP_LE_OQ ) );
                
            
                int maskhitL   = _mm256_movemask_ps(hitL);
                int maskhitR   = _mm256_movemask_ps(hitR);
                int maskhitB = maskhitL & maskhitR;
                
                if( maskhitR )
                    *(pStack++) = pNRight;
                if( maskhitL )
                    *(pStack++) = pNLeft;                
            }
        }

        return mask;
    }
    

    static size_t ReadOcclusionMask( RayPacket* pPack, uint8* pOcclusion )
    {
        size_t m0 = pOcclusion[pPack->RayOffsets[0]/sizeof(Ray)]&1;
        size_t m1 = pOcclusion[pPack->RayOffsets[1]/sizeof(Ray)]&2;
        size_t m2 = pOcclusion[pPack->RayOffsets[2]/sizeof(Ray)]&4;
        size_t m3 = pOcclusion[pPack->RayOffsets[3]/sizeof(Ray)]&8;
        size_t m4 = pOcclusion[pPack->RayOffsets[4]/sizeof(Ray)]&16;
        size_t m5 = pOcclusion[pPack->RayOffsets[5]/sizeof(Ray)]&32;
        size_t m6 = pOcclusion[pPack->RayOffsets[6]/sizeof(Ray)]&64;
        size_t m7 = pOcclusion[pPack->RayOffsets[7]/sizeof(Ray)]&128;
        return (m0|m1)|(m2|m3)|(m4|m5)|(m6|m7);
    }




   

    void AdaptiveTrace_Occlusion( void* pStackMem, StackFrame& frame, uint nRayOctant )
    {
        struct Stack
        {
            BVHNode* pN;
            size_t nGroups;
            size_t nRayPop;
        };

        BVHNode* pBVH       = frame.pBVH;
        RayPacket* pPackets = frame.pAllPackets;

        uint nPackets = frame.nPackets;
        for( size_t i=0; i<nPackets; i++ )
            frame.pActivePackets[i] = frame.pAllPackets + i;
    
        Stack* pStackBottom = (Stack*)pStackMem;
        Stack* pStack = pStackBottom;
        pStack->nGroups = nPackets;
        pStack->pN = pBVH;
        pStack->nRayPop = 8*nPackets;
        ++pStack;
    
    
        size_t xfirst = (nRayOctant&1) ? 3 : 0;
        size_t yfirst = (nRayOctant&2) ? 3 : 0;
        size_t zfirst = (nRayOctant&4) ? 3 : 0;  

        while( pStack != pStackBottom )
        {
            Stack* pS = (--pStack);
            BVHNode* pN = pS->pN;
            size_t nGroups  = pS->nGroups;
            if( !nGroups )
                continue;

            // pre-swizzle node AABB based on ray direction signs
            const float* pNodeAABB = pN->GetAABB();
            frame.pAABB[0] = pNodeAABB[0+xfirst];
            frame.pAABB[1] = pNodeAABB[1+yfirst];
            frame.pAABB[2] = pNodeAABB[2+zfirst];
            frame.pAABB[3] = pNodeAABB[0+(xfirst^3)];
            frame.pAABB[4] = pNodeAABB[1+(yfirst^3)];
            frame.pAABB[5] = pNodeAABB[2+(zfirst^3)];

            // intersection test with node, store hit masks
            size_t nHitPopulation = GroupTest2X( &frame, nGroups );
        
            // clean miss 
            if( !nHitPopulation )
                continue;
      
            _mm_prefetch( pN->GetPrefetch(), _MM_HINT_T1 );

            // skip ray reshuffling on a clean hit
            if( nHitPopulation < pS->nRayPop )
            {
                nGroups = RemoveMissedGroups(frame.pActivePackets,frame.pMasks,nGroups);
        
#ifndef NO_REORDERING
                // re-sort incoherent packets if coherency is low enough
                if( nGroups > 1 && (8*nGroups - nHitPopulation) >= 4*nGroups ) // 50% utilization
                {           
                    ReorderRays(frame,nGroups);

                    // packets are now fully coherent.  There is at most one underutilized packet.
                    //  Reduce the number of active groups and append reordered packets onto end of group list
                    nGroups = RoundUp8(nHitPopulation)/8;
                }
#endif
            }
        
            if( pN->IsLeaf() )
            {
                // visit leaf
                size_t nOut=0;
              
                for( size_t g=0; g<nGroups; g++ )
                {
                    RayPacket* p = frame.pActivePackets[g];
                    size_t m = ReadOcclusionMask(p,frame.pOcclusion);
                    TriISectPacket_Preproc_List_Occ(m,pN->GetTriList(), p, &frame );

                    if( m != 0xff )
                        frame.pActivePackets[nOut++] = p;
                }

                if( nOut < nGroups )
                {
                    // some packets were completely occluded and dropped out.  Shift the rest of the active packets array backwards
                    size_t nDropped = nGroups-nOut;
                    for( size_t g=nGroups; g<frame.nPackets; g++ )
                        frame.pActivePackets[g-nDropped] = frame.pActivePackets[g];
                    frame.nPackets -= nDropped;
                    
                    // adjust stack entries to make it look like the occluded packets never existed
                    for( Stack* s = pStackBottom; s != pStack; s++ )
                    {
                        s->nGroups -= nDropped;
                        s->nRayPop -= 8*nDropped;
                    }
                }

            }
            else if( nGroups == 1 )
            {
                // if we're down to a single group, dispatch single packet traversal 
                RayPacket* p = frame.pActivePackets[0];
                size_t m = ReadOcclusionMask(p,frame.pOcclusion);
                PacketTrace_Octant_Occ( *p, m, nRayOctant, pStack, pN,frame );    

                if( m == 0xff )
                {
                    // this packet is dead.  Shift it out
                    for( size_t g=1; g<frame.nPackets; g++ )
                        frame.pActivePackets[g-1] = frame.pActivePackets[g];
                    frame.nPackets--;

                     // adjust stack entries to make it look like the occluded packets never existed
                    for( Stack* s = pStackBottom; s != pStack; s++ )
                    {
                        s->nGroups -= 1;
                        s->nRayPop -= 8;
                    }
                }
                
            }
            else
            {
                // left, then right
                pStack[0].nGroups = nGroups;
                pStack[0].pN = pN->GetRightChild();
                pStack[0].nRayPop = nHitPopulation;
                pStack[1].nGroups = nGroups;
                pStack[1].pN = pN->GetLeftChild();
                pStack[1].nRayPop = nHitPopulation;
                pStack += 2;
            }
        }
    }


    void DoOcclusionTrace( Tracer* pTracer, unsigned char* pOcclusionOut )
    {
        __declspec(align(64)) RayPacket packs[MAX_TRACER_SIZE/8 + 8]; // up to one additional packet per octant
   
        StackFrame frame;
        frame.pRays       = pTracer->pRays;
        frame.nRays       = pTracer->nRays;
        frame.pAllPackets = packs;
        frame.pHitInfo    = 0;
        frame.pOcclusion  = pOcclusionOut;
        frame.pBVH        = pTracer->pBVHRoot;
        
        for( size_t i=0; i<frame.nRays; i++ )
            pOcclusionOut[i] = 0;

        uint pPacksByOctant[8] = {0};
        BuildPacketsByOctant(packs, pTracer, pPacksByOctant );
    
        RayPacket* pPacket = packs;
        for( uint i=0; i<8; i++ )
        {
            size_t n = pPacksByOctant[i];
            if( n )
            {
                frame.nPackets = n;
                AdaptiveTrace_Occlusion( pTracer->pTraversalStack, frame, i );
            }

            frame.pAllPackets += n;
        }
    }

}}