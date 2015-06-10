/* Copyright (c) 2013, Intel Corporation
*
* Redistribution and use in source and binary forms, with or without 
* modification, are permitted provided that the following conditions are met:
*
* - Redistributions of source code must retain the above copyright notice, 
*   this list of conditions and the following disclaimer.
* - Redistributions in binary form must reproduce the above copyright notice, 
*   this list of conditions and the following disclaimer in the documentation 
*   and/or other materials provided with the distribution.
* - Neither the name of Intel Corporation nor the names of its contributors 
*   may be used to endorse or promote products derived from this software 
*   without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
* POSSIBILITY OF SUCH DAMAGE.
*
*/

#include "bvh4_intersector_stream.h"
#include "geometry/triangles.h"
#include "../common/stream_ray.h"
#include "../common/stack_item.h"

using namespace embree;

__forceinline unsigned align(unsigned x) {
  static const unsigned width = 2;
  return (x + (width-1)) & (~(width-1));
}

__forceinline __m256 simd_float_combine(__m128 a, __m128 b) {
  __m256 c = _mm256_castps128_ps256(a);
  c = _mm256_insertf128_ps(c, b, 1);
  return c;
}

__forceinline void setupRays(unsigned short** buckets, unsigned rayCount) {
  // Fill with 0, 1, .., count.
  __m256i a0 = _mm256_setr_epi16( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15);
  __m256i a1 = _mm256_setr_epi16(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
  __m256i a2 = _mm256_setr_epi16(32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47);
  __m256i a3 = _mm256_setr_epi16(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
  __m256i r = _mm256_set1_epi16(64);
      
  __m256i* start = reinterpret_cast<__m256i*>(buckets[0]);
  __m256i* end = start + ((rayCount + 15) >> 4);
      
  for (__m256i* indices = start; indices < end; indices += 4) {
    _mm256_store_si256(indices + 0, a0);
    _mm256_store_si256(indices + 1, a1);
    _mm256_store_si256(indices + 2, a2);
    _mm256_store_si256(indices + 3, a3);
        
    a0 = _mm256_add_epi16(a0, r);
    a1 = _mm256_add_epi16(a1, r);
    a2 = _mm256_add_epi16(a2, r);
    a3 = _mm256_add_epi16(a3, r);
  }
}

__forceinline __m256 _mm256_cmple_ps(__m256 a, __m256 b) {
  return _mm256_cmp_ps(a, b, _CMP_LE_OS);
}

__forceinline __m256 _mm256_cmpge_ps(__m256 a, __m256 b) {
  return _mm256_cmp_ps(a, b, _CMP_GE_OS);
}

static void intersectChildrenAndBucketRays(const void* nodePtr, StackEntry* __restrict* __restrict stack, unsigned short* __restrict* __restrict buckets, unsigned* __restrict bucketSize, unsigned short* __restrict firstRay, unsigned short* __restrict lastRay, BVH4::NodeRef n, const StreamRay* __restrict rayData) {
  const BVH4::Node* __restrict node = n.node(nodePtr);

  unsigned nextIndices = *reinterpret_cast<unsigned*>(firstRay);

  unsigned short* __restrict bucket[9];

  for (unsigned i = 0; i < 9; ++i)
    bucket[i] = buckets[i] + bucketSize[i];

  __m256 dxDyDzTnOxOyOzTfA = _mm256_load_ps(&rayData[nextIndices & 0xffff].invDirX);
  __m256 dxDyDzTnOxOyOzTfB = _mm256_load_ps(&rayData[nextIndices >> 16].invDirX);

  __m256 minMinMinMinMaxMaxMaxMaxX = _mm256_load_ps(reinterpret_cast<const float*>(&node->lower_x));
  __m256 minMinMinMinMaxMaxMaxMaxY = _mm256_load_ps(reinterpret_cast<const float*>(&node->lower_y));
  __m256 minMinMinMinMaxMaxMaxMaxZ = _mm256_load_ps(reinterpret_cast<const float*>(&node->lower_z));

  __m256 bbX = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(minMinMinMinMaxMaxMaxMaxX), (0) | ((2) << 2) | ((1) << 4) | ((3) << 6)));
  __m256 bbY = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(minMinMinMinMaxMaxMaxMaxY), (0) | ((2) << 2) | ((1) << 4) | ((3) << 6)));
  __m256 bbZ = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(minMinMinMinMaxMaxMaxMaxZ), (0) | ((2) << 2) | ((1) << 4) | ((3) << 6)));

  __m128 pnl = _mm_castsi128_ps(_mm_setr_epi32(0x00000000, 0x00000000, 0x80000000, 0x80000000));
  __m256 pn = simd_float_combine(pnl, pnl);
  
  unsigned mask = _mm256_movemask_ps(dxDyDzTnOxOyOzTfA);

  __m256 dxDyDzTn = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (0) | ((2) << 4));
  __m256 oxOyOzTf = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (1) | ((3) << 4));

  if (mask & 1) bbX = _mm256_shuffle_ps(bbX, bbX, _MM_SHUFFLE(1,0,3,2));
  if (mask & 2) bbY = _mm256_shuffle_ps(bbY, bbY, _MM_SHUFFLE(1,0,3,2));
  if (mask & 4) bbZ = _mm256_shuffle_ps(bbZ, bbZ, _MM_SHUFFLE(1,0,3,2));

  bbX = _mm256_xor_ps(bbX, pn);
  bbY = _mm256_xor_ps(bbY, pn);
  bbZ = _mm256_xor_ps(bbZ, pn);

  __m256 neg = _mm256_set1_ps(-0.0f);

  __m256 rbbX0 = _mm256_permute2f128_ps(bbX, bbX, (0) | ((0) << 4));
  __m256 rbbY0 = _mm256_permute2f128_ps(bbY, bbY, (0) | ((0) << 4));
  __m256 rbbZ0 = _mm256_permute2f128_ps(bbZ, bbZ, (0) | ((0) << 4));

  __m256 rbbX1 = _mm256_permute2f128_ps(bbX, bbX, (1) | ((1) << 4));
  __m256 rbbY1 = _mm256_permute2f128_ps(bbY, bbY, (1) | ((1) << 4));
  __m256 rbbZ1 = _mm256_permute2f128_ps(bbZ, bbZ, (1) | ((1) << 4));

  unsigned orderMask = 0;
  unsigned indices = 0;

  unsigned* __restrict r = reinterpret_cast<unsigned*>(firstRay);
  ++r;
  lastRay += 2;

  const char* __restrict rayBytes = reinterpret_cast<const char*>(rayData);

  __align(32) unsigned short bucketOffsets[16] = { 0 };
  __m256i bucketOffset = _mm256_setzero_si256();
  
  do {
    unsigned ni = *(r++);
    unsigned indexB = indices >> 16;
    const float* ra = reinterpret_cast<const float*>(rayBytes + ((ni & 0xffff) << 5));
    
    __m256 oxOyOzTfneg = _mm256_xor_ps(oxOyOzTf, neg);

    __m256 rdirX = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(0,0,0,0));
    __m256 orgInvDirX = _mm256_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(0,0,0,0));
    __m256 tNearFarX0 = _mm256_fmsub_ps(rdirX, rbbX0, orgInvDirX);
    __m256 tNearFarX1 = _mm256_fmsub_ps(rdirX, rbbX1, orgInvDirX);

    __m256 dxDyDzTnOxOyOzTfA = _mm256_load_ps(ra);
    const float* rb = reinterpret_cast<const float*>(rayBytes + ((ni & 0xffff0000) >> 11));

    __m256 rdirY = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(1,1,1,1));
    __m256 orgInvDirY = _mm256_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(1,1,1,1));
    __m256 tNearFarY0 = _mm256_fmsub_ps(rdirY, rbbY0, orgInvDirY);
    __m256 tNearFarY1 = _mm256_fmsub_ps(rdirY, rbbY1, orgInvDirY);

    __m256 nearFar = _mm256_shuffle_ps(dxDyDzTn, oxOyOzTfneg, _MM_SHUFFLE(3,3,3,3));

    __m256 rdirZ = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(2,2,2,2));
    __m256 orgInvDirZ = _mm256_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(2,2,2,2));
    __m256 tNearFarZ0 = _mm256_fmsub_ps(rdirZ, rbbZ0, orgInvDirZ);
    __m256 tNearFarZ1 = _mm256_fmsub_ps(rdirZ, rbbZ1, orgInvDirZ);

    __m256 dxDyDzTnOxOyOzTfB = _mm256_load_ps(rb);

    tNearFarX0 = _mm256_max_ps(tNearFarX0, nearFar);
    tNearFarX1 = _mm256_max_ps(tNearFarX1, nearFar);

    __m256 tNearFar0 = _mm256_max_ps(_mm256_max_ps(tNearFarX0, tNearFarY0), tNearFarZ0);
    __m256 tNearFar1 = _mm256_max_ps(_mm256_max_ps(tNearFarX1, tNearFarY1), tNearFarZ1);

    if (orderMask & 0x10) {
      bucket[0][bucketOffsets[0]] = indexB;
      bucket[1][bucketOffsets[1]] = indexB;
      bucket[2][bucketOffsets[2]] = indexB;
    }
    else {
      bucket[6][bucketOffsets[8]] = indexB;
      bucket[7][bucketOffsets[9]] = indexB;
      bucket[8][bucketOffsets[10]] = indexB;
    }

    bucket[3][bucketOffsets[4]] = indexB;
    bucket[4][bucketOffsets[5]] = indexB;
    bucket[5][bucketOffsets[6]] = indexB;

    _mm256_store_si256(reinterpret_cast<__m256i*>(bucketOffsets), bucketOffset);

    __m256 s0 = _mm256_shuffle_ps(tNearFar0, _mm256_xor_ps(tNearFar0, neg), _MM_SHUFFLE(1,0,1,1));
    __m256 s1 = _mm256_shuffle_ps(tNearFar1, _mm256_xor_ps(tNearFar1, neg), _MM_SHUFFLE(1,0,1,1));

    orderMask = _mm256_movemask_ps(_mm256_cmple_ps(_mm256_min_ps(s0, tNearFar0), _mm256_min_ps(s1, tNearFar1)));
    // 0: 2lfirst

    __m256 r0 = _mm256_cmpge_ps(s0, tNearFar0);
    __m256 r1 = _mm256_cmpge_ps(s1, tNearFar1);
    // 0: lfirst, 1: 1, 2: lhit, 3: rhit

    __m256i r = _mm256_packs_epi32(_mm256_castps_si256(r0), _mm256_castps_si256(r1));

    __m256i lf = _mm256_shuffle_epi32(r, _MM_SHUFFLE(2,2,0,0));
    lf = _mm256_xor_si256(lf, _mm256_set1_epi64x(0x0000000100000000ull));
    // 0: lfirst, 1: 1, 2: ~lfirst, 3: 1

    __m256i lr = _mm256_shuffle_epi32(r, _MM_SHUFFLE(3,3,1,1));
    // 0: lhit, 1: rhit, 2: lhit, 3: rhit

    __m256i bucketIncrement = _mm256_and_si256(lf, lr);

    oxOyOzTf = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (1) | ((3) << 4));
    dxDyDzTn = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (0) | ((2) << 4));

    indices = nextIndices;
    nextIndices = ni;

    unsigned indexA = indices & 0xffff;

    __m256i incr = _mm256_permute4x64_epi64(bucketIncrement, (0) | ((1) << 2) | ((0) << 4) | ((1) << 6));

    __m256i incrMask;

    if (orderMask & 0x1) {
      incrMask = _mm256_setr_epi64x(0x100010001ull, 0x100010001ull, 0, 0);
      bucket[0][bucketOffsets[0]] = indexA;
      bucket[1][bucketOffsets[1]] = indexA;
      bucket[2][bucketOffsets[2]] = indexA;
    }
    else {
      incrMask = _mm256_setr_epi64x(0, 0x100010001ull, 0x100010001ull, 0);
      bucket[6][bucketOffsets[8]] = indexA;
      bucket[7][bucketOffsets[9]] = indexA;
      bucket[8][bucketOffsets[10]] = indexA;
    }

    bucketOffset = _mm256_add_epi16(bucketOffset, _mm256_and_si256(incr, incrMask));

    if (orderMask & 0x10)
      incrMask = _mm256_setr_epi64x(0x100010001ull, 0x100010001ull, 0, 0);
    else
      incrMask = _mm256_setr_epi64x(0, 0x100010001ull, 0x100010001ull, 0);

    bucket[3][bucketOffsets[4]] = indexA;
    bucket[4][bucketOffsets[5]] = indexA;
    bucket[5][bucketOffsets[6]] = indexA;

    _mm256_store_si256(reinterpret_cast<__m256i*>(bucketOffsets), bucketOffset);

    incr = _mm256_permute4x64_epi64(bucketIncrement, (2) | ((3) << 2) | ((2) << 4) | ((3) << 6));

    bucketOffset = _mm256_add_epi16(bucketOffset, _mm256_and_si256(incr, incrMask));
  } while (reinterpret_cast<unsigned short*>(r) < lastRay);

  unsigned indexA = indices & 0xffff;
  unsigned indexB = indices >> 16;

  if (indexA != indexB) {
    if (orderMask & 0x10) {
      bucket[0][bucketOffsets[0]] = indexB;
      bucket[1][bucketOffsets[1]] = indexB;
      bucket[2][bucketOffsets[2]] = indexB;
    }
    else {
      bucket[6][bucketOffsets[8]] = indexB;
      bucket[7][bucketOffsets[9]] = indexB;
      bucket[8][bucketOffsets[10]] = indexB;
    }

    bucket[3][bucketOffsets[4]] = indexB;
    bucket[4][bucketOffsets[5]] = indexB;
    bucket[5][bucketOffsets[6]] = indexB;

    _mm256_store_si256(reinterpret_cast<__m256i*>(bucketOffsets), bucketOffset);
  }

  unsigned size[] = {
    bucketOffsets[0],
    bucketOffsets[1],
    bucketOffsets[2],
    bucketOffsets[4],
    bucketOffsets[5],
    bucketOffsets[6],
    bucketOffsets[8],
    bucketOffsets[9],
    bucketOffsets[10],
  };

  // Push bucket intervals to stack.
  BVH4::NodeRef left0 = node->child(0);
  BVH4::NodeRef right0 = node->child(1);
  BVH4::NodeRef left1 = node->child(2);
  BVH4::NodeRef right1 = node->child(3);

  StackEntry* __restrict s = *stack;

#define PUSH_BUCKET(n, Node)\
  if (unsigned added = size[n]) {\
    unsigned newSize = bucketSize[n] + added;\
    StackEntry& e = *(s++);\
    e.node = Node;\
    e.bucket = (unsigned)n;\
    e.firstRay = bucketSize[n];\
    e.lastRay = newSize;\
    bucketSize[n] = align(newSize+2);\
    unsigned short* __restrict b = bucket[n];\
    b[added] = b[added+1] = b[added+2] = b[added-1];\
  }

  PUSH_BUCKET(8, left0);
  PUSH_BUCKET(7, right0);
  PUSH_BUCKET(6, left0);

  PUSH_BUCKET(5, left1);
  PUSH_BUCKET(4, right1);
  PUSH_BUCKET(3, left1);

  PUSH_BUCKET(2, left0);
  PUSH_BUCKET(1, right0);
  PUSH_BUCKET(0, left0);

  *stack = s;
}

static void intersectChildrenAndBucketRaysNoOrder(const void* nodePtr, StackEntry* __restrict* __restrict stack, unsigned short* __restrict* __restrict buckets, unsigned* __restrict bucketSize, unsigned short* __restrict firstRay, unsigned short* __restrict lastRay, BVH4::NodeRef n, const StreamRay* __restrict rayData) {
  const BVH4::Node* __restrict node = n.node(nodePtr);

  unsigned nextIndices = *reinterpret_cast<unsigned*>(firstRay);

  unsigned short* __restrict bucket[4];

  for (unsigned i = 0; i < 4; ++i)
    bucket[i] = buckets[i] + bucketSize[i];

  __m256 dxDyDzTnOxOyOzTfA = _mm256_load_ps(&rayData[nextIndices & 0xffff].invDirX);
  __m256 dxDyDzTnOxOyOzTfB = _mm256_load_ps(&rayData[nextIndices >> 16].invDirX);

  __m256 minMinMinMinMaxMaxMaxMaxX = _mm256_load_ps(reinterpret_cast<const float*>(&node->lower_x));
  __m256 minMinMinMinMaxMaxMaxMaxY = _mm256_load_ps(reinterpret_cast<const float*>(&node->lower_y));
  __m256 minMinMinMinMaxMaxMaxMaxZ = _mm256_load_ps(reinterpret_cast<const float*>(&node->lower_z));

  __m256 bbX = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(minMinMinMinMaxMaxMaxMaxX), (0) | ((2) << 2) | ((1) << 4) | ((3) << 6)));
  __m256 bbY = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(minMinMinMinMaxMaxMaxMaxY), (0) | ((2) << 2) | ((1) << 4) | ((3) << 6)));
  __m256 bbZ = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(minMinMinMinMaxMaxMaxMaxZ), (0) | ((2) << 2) | ((1) << 4) | ((3) << 6)));

  __m128 pnl = _mm_castsi128_ps(_mm_setr_epi32(0x00000000, 0x00000000, 0x80000000, 0x80000000));
  __m256 pn = simd_float_combine(pnl, pnl);

  unsigned mask = _mm256_movemask_ps(dxDyDzTnOxOyOzTfA);

  __m256 dxDyDzTn = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (0) | ((2) << 4));
  __m256 oxOyOzTf = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (1) | ((3) << 4));

  if (mask & 1) bbX = _mm256_shuffle_ps(bbX, bbX, _MM_SHUFFLE(1, 0, 3, 2));
  if (mask & 2) bbY = _mm256_shuffle_ps(bbY, bbY, _MM_SHUFFLE(1, 0, 3, 2));
  if (mask & 4) bbZ = _mm256_shuffle_ps(bbZ, bbZ, _MM_SHUFFLE(1, 0, 3, 2));

  bbX = _mm256_xor_ps(bbX, pn);
  bbY = _mm256_xor_ps(bbY, pn);
  bbZ = _mm256_xor_ps(bbZ, pn);

  __m256 neg = _mm256_set1_ps(-0.0f);

  __m256 rbbX0 = _mm256_permute2f128_ps(bbX, bbX, (0) | ((0) << 4));
  __m256 rbbY0 = _mm256_permute2f128_ps(bbY, bbY, (0) | ((0) << 4));
  __m256 rbbZ0 = _mm256_permute2f128_ps(bbZ, bbZ, (0) | ((0) << 4));

  __m256 rbbX1 = _mm256_permute2f128_ps(bbX, bbX, (1) | ((1) << 4));
  __m256 rbbY1 = _mm256_permute2f128_ps(bbY, bbY, (1) | ((1) << 4));
  __m256 rbbZ1 = _mm256_permute2f128_ps(bbZ, bbZ, (1) | ((1) << 4));

  unsigned indices = 0;

  unsigned* __restrict r = reinterpret_cast<unsigned*>(firstRay);
  ++r;
  lastRay += 2;

  const char* __restrict rayBytes = reinterpret_cast<const char*>(rayData);

  __align(16) unsigned bucketOffsets[4] = { 0 };
  __m128i bucketOffset = _mm_setzero_si128();

  do {
    unsigned ni = *(r++);
    unsigned indexB = indices >> 16;
    const float* ra = reinterpret_cast<const float*>(rayBytes + ((ni & 0xffff) << 5));
    
    __m256 oxOyOzTfneg = _mm256_xor_ps(oxOyOzTf, neg);

    __m256 rdirX = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(0, 0, 0, 0));
    __m256 orgInvDirX = _mm256_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(0, 0, 0, 0));
    __m256 tNearFarX0 = _mm256_fmsub_ps(rdirX, rbbX0, orgInvDirX);
    __m256 tNearFarX1 = _mm256_fmsub_ps(rdirX, rbbX1, orgInvDirX);

    __m256 dxDyDzTnOxOyOzTfA = _mm256_load_ps(ra);
    const float* rb = reinterpret_cast<const float*>(rayBytes + ((ni & 0xffff0000) >> 11));

    __m256 rdirY = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(1, 1, 1, 1));
    __m256 orgInvDirY = _mm256_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(1, 1, 1, 1));
    __m256 tNearFarY0 = _mm256_fmsub_ps(rdirY, rbbY0, orgInvDirY);
    __m256 tNearFarY1 = _mm256_fmsub_ps(rdirY, rbbY1, orgInvDirY);

    __m256 nearFar = _mm256_shuffle_ps(dxDyDzTn, oxOyOzTfneg, _MM_SHUFFLE(3, 3, 3, 3));

    __m256 rdirZ = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(2, 2, 2, 2));
    __m256 orgInvDirZ = _mm256_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(2, 2, 2, 2));
    __m256 tNearFarZ0 = _mm256_fmsub_ps(rdirZ, rbbZ0, orgInvDirZ);
    __m256 tNearFarZ1 = _mm256_fmsub_ps(rdirZ, rbbZ1, orgInvDirZ);

    __m256 dxDyDzTnOxOyOzTfB = _mm256_load_ps(rb);

    tNearFarX0 = _mm256_max_ps(tNearFarX0, nearFar);
    tNearFarX1 = _mm256_max_ps(tNearFarX1, nearFar);

    __m256 tNearFar0 = _mm256_max_ps(_mm256_max_ps(tNearFarX0, tNearFarY0), tNearFarZ0);
    __m256 tNearFar1 = _mm256_max_ps(_mm256_max_ps(tNearFarX1, tNearFarY1), tNearFarZ1);

    bucket[0][bucketOffsets[0]] = indexB;
    bucket[1][bucketOffsets[1]] = indexB;
    bucket[2][bucketOffsets[2]] = indexB;
    bucket[3][bucketOffsets[3]] = indexB;

    _mm_store_si128(reinterpret_cast<__m128i*>(bucketOffsets), bucketOffset);

    __m256i bucketIncrement = _mm256_castps_si256(_mm256_cmpge_ps(_mm256_xor_ps(_mm256_shuffle_ps(tNearFar0, tNearFar1, _MM_SHUFFLE(1, 0, 1, 0)), neg), _mm256_shuffle_ps(tNearFar0, tNearFar1, _MM_SHUFFLE(3, 2, 3, 2))));
    // 0: lhit, 1: rhit, 2: lhit, 3: rhit

    bucketIncrement = _mm256_and_si256(bucketIncrement, _mm256_set1_epi32(1));

    oxOyOzTf = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (1) | ((3) << 4));
    dxDyDzTn = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (0) | ((2) << 4));

    indices = nextIndices;
    nextIndices = ni;

    unsigned indexA = indices & 0xffff;

    __m128i incr = _mm256_castsi256_si128(bucketIncrement);

    bucket[0][bucketOffsets[0]] = indexA;
    bucket[1][bucketOffsets[1]] = indexA;

    bucketOffset = _mm_add_epi32(bucketOffset, incr);

    bucket[2][bucketOffsets[2]] = indexA;
    bucket[3][bucketOffsets[3]] = indexA;

    _mm_store_si128(reinterpret_cast<__m128i*>(bucketOffsets), bucketOffset);

    incr = _mm256_extracti128_si256(bucketIncrement, 1);

    bucketOffset = _mm_add_epi32(bucketOffset, incr);
  } while (reinterpret_cast<unsigned short*>(r) < lastRay);

  unsigned indexA = indices & 0xffff;
  unsigned indexB = indices >> 16;

  if (indexA != indexB) {
    bucket[0][bucketOffsets[0]] = indexB;
    bucket[1][bucketOffsets[1]] = indexB;
    bucket[2][bucketOffsets[2]] = indexB;
    bucket[3][bucketOffsets[3]] = indexB;

    _mm_store_si128(reinterpret_cast<__m128i*>(bucketOffsets), bucketOffset);
  }

  // Push bucket intervals to stack.
  for (unsigned i = 0; i < 4; ++i) {
    unsigned j = i;
    unsigned x = i | (bucketOffsets[i] << 2);

#if 1 // Sort.
    while (j > 0) {
      unsigned nextJ = j - 1;
      unsigned nextItem = bucketOffsets[nextJ];

      if (nextItem < x)
        break;

      bucketOffsets[j] = nextItem;
      j = nextJ;
    }
#endif

    bucketOffsets[j] = x;
  }

  StackEntry* __restrict s = *stack;

  for (unsigned i = 0; i < 4; ++i) {
    unsigned added = bucketOffsets[i] >> 2;
    unsigned n = bucketOffsets[i] & 3;

    if (!added)
      continue;

    unsigned newSize = bucketSize[n] + added;
    StackEntry& e = *(s++);
    e.node = node->child(n);
    e.bucket = n;
    e.firstRay = bucketSize[n];
    e.lastRay = newSize;
    bucketSize[n] = align(newSize + 2);
    unsigned short* __restrict b = bucket[n];
    b[added] = b[added + 1] = b[added + 2] = b[added - 1];
  }

  *stack = s;
}

template<class Triangle, class TriangleIntersector>
static void intersectTriangles(unsigned short* first, unsigned short* last, StreamRay* rayData, StreamRayExtra* rayExtras, StreamHit* hits, Triangle* triangles, unsigned triangleCount, const Vec3fa* vertices) {
  for (; first < last; ++first) {
    StreamRay& ray = rayData[*first];
    StreamRayExtra& rayExtra = rayExtras[*first];
    StreamHit& hit = hits[*first];
          
    for (size_t i = 0; i < triangleCount; ++i)
      TriangleIntersector::intersect(ray, rayExtra, hit, triangles[i], vertices);
  }
}

template<class Triangle, class TriangleIntersector>
static void occludeByTriangles(unsigned short* first, unsigned short* last, StreamRay* rayData, StreamRayExtra* rayExtras, unsigned char* occluded, Triangle* triangles, unsigned triangleCount, const Vec3fa* vertices) {
  for (; first < last; ++first) {
    StreamRay& ray = rayData[*first];
    StreamRayExtra& rayExtra = rayExtras[*first];
    unsigned char& rayOccluded = occluded[*first];

    if (rayOccluded)
      continue;

    for (size_t i = 0; i < triangleCount; ++i) {
      if (TriangleIntersector::occluded(ray, rayExtra, triangles[i], vertices)) {
        rayOccluded = 1;
        ray.tfar = 0.0f;
        ray.tnear = 1.0f;
        break;
      }
    }
  }
}

template<typename TriangleIntersector>
BVH4IntersectorStream<TriangleIntersector>::BVH4IntersectorStream(const BVH4* bvh) : IntersectorStream((intersectFunc)intersect, (occludedFunc)occluded), bvh(bvh) {
  threadStates.resize(TaskScheduler::getNumThreads());
}

template<typename TriangleIntersector>
void BVH4IntersectorStream<TriangleIntersector>::intersect(BVH4IntersectorStream* This, StreamRay* rayData, StreamRayExtra* rayExtras, StreamHit* hits, unsigned count, unsigned thread) {
  AVX_ZERO_UPPER();

  ThreadState& threadState = This->threadStates[thread];
  threadState.reserve(count);
  
  unsigned* bucketSize = threadState.bucketSize;
  unsigned short** buckets = threadState.buckets;
  
  StackEntry* stack = threadState.stack;
  StackEntry* const stackStart = stack;
  
  const BVH4* bvh = This->bvh;

  const void* nodePtr = bvh->nodePtr();
  const void* triPtr  = bvh->triPtr();
    
  NodeRef node = bvh->root;
  unsigned bucket = 0;
  unsigned firstRay = 0;
  unsigned lastRay = count;

  for (unsigned i = 0; i < BVH4IntersectorStream<TriangleIntersector>::bucketCount; ++i)
    bucketSize[i] = 0;

#if 0
  setupRays(buckets, count);

  buckets[0][count] = count-1;
  buckets[0][count+1] = count-1;
  buckets[0][count+2] = count-1;
#else
  // Bucket rays based on direction sign.
  unsigned short* orderBuckets[8];

  for (unsigned i = 0; i < 8; ++i)
    orderBuckets[i] = buckets[i];

  unsigned bucketRay = 0;

  for (; (int)bucketRay < (int)count-3; bucketRay += 4) {
    __m128 m0 = _mm_load_ps(&rayData[bucketRay+0].invDirX);
    __m128 m1 = _mm_load_ps(&rayData[bucketRay+1].invDirX);
    __m128 m2 = _mm_load_ps(&rayData[bucketRay+2].invDirX);
    __m128 m3 = _mm_load_ps(&rayData[bucketRay+3].invDirX);

    unsigned swap0 = _mm_movemask_ps(m0) & 7;
    unsigned swap1 = _mm_movemask_ps(m1) & 7;
    unsigned swap2 = _mm_movemask_ps(m2) & 7;
    unsigned swap3 = _mm_movemask_ps(m3) & 7;

    *(orderBuckets[swap0]++) = bucketRay+0;
    *(orderBuckets[swap1]++) = bucketRay+1;
    *(orderBuckets[swap2]++) = bucketRay+2;
    *(orderBuckets[swap3]++) = bucketRay+3;
  }

  for (; bucketRay < count; ++bucketRay) {
    __m128 m = _mm_load_ps(&rayData[bucketRay].invDirX);

    unsigned swap = _mm_movemask_ps(m) & 7;

    *(orderBuckets[swap]++) = bucketRay;
  }

  for (unsigned i = 0; i < 8; ++i) {
    if (orderBuckets[i] == buckets[i])
      continue;

    unsigned newSize = (unsigned)(orderBuckets[i] - buckets[i]);
    StackEntry& e = *(stack++);
    e.node = node;
    e.bucket = (unsigned)i;
    e.firstRay = 0;
    e.lastRay = newSize;
    bucketSize[i] = align(newSize+2);
    orderBuckets[i][0] = orderBuckets[i][1] = orderBuckets[i][2] = orderBuckets[i][-1];
  }

  if (stack == stackStart)
    return;
      
  const StackEntry& entry = *(--stack);

  bucket = entry.bucket;
  firstRay = entry.firstRay;
  lastRay = entry.lastRay;
      
  bucketSize[bucket] = firstRay;
#endif
  
  for (;;) {
    if (likely(node.isNode())) {
      if (lastRay - firstRay >= 16) {
        intersectChildrenAndBucketRays(nodePtr, &stack, buckets, bucketSize, buckets[bucket] + firstRay, buckets[bucket] + lastRay, node, rayData);
      }
      else {
        unsigned short* first =  buckets[bucket] + firstRay;
        unsigned short* last = buckets[bucket] + lastRay;

        __m256 dxDyDzTnOxOyOzTf = _mm256_load_ps(&rayData[*first].invDirX);
        unsigned mask = _mm256_movemask_ps(dxDyDzTnOxOyOzTf);

        const avxf pos_neg = avxf(ssef(+0.0f), ssef(-0.0f));
        const avxf neg_pos = avxf(ssef(-0.0f), ssef(+0.0f));

        unsigned swapX = mask & 1;
        unsigned swapY = mask & 2;
        unsigned swapZ = mask & 4;

        const avxf flipSignX = swapX ? neg_pos : pos_neg;
        const avxf flipSignY = swapY ? neg_pos : pos_neg;
        const avxf flipSignZ = swapZ ? neg_pos : pos_neg;
        
        do {
          StreamRay& ray = rayData[*first];
          StreamRayExtra& rayExtra = rayExtras[*first];
          StreamHit& hit = hits[*first];
          ++first;
          
          typedef typename TriangleIntersector::Triangle Triangle;
          typedef StackItemT<size_t> StackItem;
          typedef typename BVH4::NodeRef NodeRef;
          typedef typename BVH4::Node Node;

          __m256 dxDyDzTn = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTf, dxDyDzTnOxOyOzTf, (0) | ((0) << 4));
          __m256 oxOyOzTf = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTf, dxDyDzTnOxOyOzTf, (1) | ((1) << 4));

          /*! stack state */
          StackItem stack[1+3*BVH4::maxDepth];  //!< stack of nodes 
          StackItem* stackPtr = stack+1;        //!< current stack pointer
          stack[0].ptr  = node;
          stack[0].dist = neg_inf;

          __m256 pmRay = dxDyDzTnOxOyOzTf ^ pos_neg;

          /*! load the ray into SIMD registers */
          const avx3f rdir(
            _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(0,0,0,0)) ^ flipSignX,
            _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(1,1,1,1)) ^ flipSignY,
            _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(2,2,2,2)) ^ flipSignZ);

          const avx3f org_rdir(
            _mm256_shuffle_ps(oxOyOzTf, oxOyOzTf, _MM_SHUFFLE(0,0,0,0)) ^ flipSignX,
            _mm256_shuffle_ps(oxOyOzTf, oxOyOzTf, _MM_SHUFFLE(1,1,1,1)) ^ flipSignY,
            _mm256_shuffle_ps(oxOyOzTf, oxOyOzTf, _MM_SHUFFLE(2,2,2,2)) ^ flipSignZ);

          avxf rayNearFar = _mm256_shuffle_ps(pmRay, pmRay, _MM_SHUFFLE(3,3,3,3));//(ssef(ray.tnear),-ssef(ray.tfar));

          dxDyDzTnOxOyOzTf = _mm256_load_ps(&rayData[*first].invDirX);

          const void* nodePtr = bvh->nodePtr();
          const void* triPtr  = bvh->triPtr();
     
          /* pop loop */
          while (true) pop:
          {
            /*! pop next node */
            if (unlikely(stackPtr == stack)) break;
            stackPtr--;
            NodeRef cur = NodeRef(stackPtr->ptr);
      
            /*! if popped node is too far, pop next one */
            if (unlikely(stackPtr->dist > ray.tfar))
              continue;

            /* downtraversal loop */
            while (true)
            {
              /*! stop if we found a leaf */
              if (unlikely(cur.isLeaf())) break;
              
              /*! single ray intersection with 4 boxes */
              const Node* node = cur.node(nodePtr);

              avxf bbX = avxf::load(&node->lower_x);
              avxf bbY = avxf::load(&node->lower_y);
              avxf bbZ = avxf::load(&node->lower_z);

              const avxf tLowerUpperX = msub(bbX, rdir.x, org_rdir.x);
              const avxf tLowerUpperY = msub(bbY, rdir.y, org_rdir.y);
              const avxf tLowerUpperZ = msub(bbZ, rdir.z, org_rdir.z);

              const BVH4::Node* leftmost = node->child(1).node(nodePtr);
              const BVH4::Node* rightmost = node->child(2).node(nodePtr);

              _mm_prefetch(reinterpret_cast<const char*>(&leftmost->lower_x), _MM_HINT_T0);
              _mm_prefetch(reinterpret_cast<const char*>(&rightmost->lower_x), _MM_HINT_T0);
              
              const avxf tNearFarX = swapX ? shuffle<1,0>(tLowerUpperX) : tLowerUpperX;
              const avxf tNearFarY = swapY ? shuffle<1,0>(tLowerUpperY) : tLowerUpperY;
              const avxf tNearFarZ = swapZ ? shuffle<1,0>(tLowerUpperZ) : tLowerUpperZ;

              const avxf tNearFar = max(rayNearFar,tNearFarX,tNearFarY,tNearFarZ);
              const ssef tNear = extract<0>(tNearFar);
              const ssef tFar  = extract<1>(tNearFar);

              size_t mask = movemask(-tNear >= tFar);

              _mm_prefetch(reinterpret_cast<const char*>(&leftmost->lower_z), _MM_HINT_T0);
              _mm_prefetch(reinterpret_cast<const char*>(&rightmost->lower_z), _MM_HINT_T0);
                
              /*! if no child is hit, pop next node */
              if (unlikely(mask == 0))
                goto pop;

              /*! one child is hit, continue with that child */
              size_t r = __bsf(mask); mask = __btc(mask,r);
              if (likely(mask == 0)) {
                cur = node->child(r);
                continue;
              }

              /*! two children are hit, push far child, and continue with closer child */
              NodeRef c0 = node->child(r); const float d0 = tNear[r];
              r = __bsf(mask); mask = __btc(mask,r);
              NodeRef c1 = node->child(r); const float d1 = tNear[r];
              if (likely(mask == 0)) {
                if (d0 < d1) { stackPtr->ptr = c1; stackPtr->dist = d1; stackPtr++; cur = c0; continue; }
                else         { stackPtr->ptr = c0; stackPtr->dist = d0; stackPtr++; cur = c1; continue; }
              }

              /*! Here starts the slow path for 3 or 4 hit children. We push
                *  all nodes onto the stack to sort them there. */
              stackPtr->ptr = c0; stackPtr->dist = d0; stackPtr++;
              stackPtr->ptr = c1; stackPtr->dist = d1; stackPtr++;

              /*! three children are hit, push all onto stack and sort 3 stack items, continue with closest child */
              r = __bsf(mask); mask = __btc(mask,r);
              NodeRef c = node->child(r); float d = tNear[r]; stackPtr->ptr = c; stackPtr->dist = d; stackPtr++;
              if (likely(mask == 0)) {
                sort(stackPtr[-1],stackPtr[-2],stackPtr[-3]);
                cur = (NodeRef) stackPtr[-1].ptr; stackPtr--;
                continue;
              }

              /*! four children are hit, push all onto stack and sort 4 stack items, continue with closest child */
              r = __bsf(mask); mask = __btc(mask,r);
              c = node->child(r); d = tNear[r]; stackPtr->ptr = c; stackPtr->dist = d; stackPtr++;
              sort(stackPtr[-1],stackPtr[-2],stackPtr[-3],stackPtr[-4]);
              cur = (NodeRef) stackPtr[-1].ptr; stackPtr--;
            }

            /*! this is a leaf node */
            size_t num; Triangle* tri = (Triangle*) cur.leaf(triPtr,num);
            for (size_t i=0; i<num; i++)
              TriangleIntersector::intersect(ray,rayExtra,hit,tri[i],bvh->vertices);
      
            rayNearFar = insert<1>(rayNearFar,-ssef(ray.tfar));
          }
        } while (first < last);
      }
    }
    else {
      size_t triangleCount;
      Triangle* triangles = (Triangle*)node.leaf(0, triangleCount);
          
      intersectTriangles<Triangle, TriangleIntersector>(buckets[bucket] + firstRay, buckets[bucket] + lastRay, rayData, rayExtras, hits, triangles, (unsigned)triangleCount, bvh->vertices);
    }
      
    if (stack == stackStart)
      break;
      
    const StackEntry& entry = *(--stack);
      
    node = entry.node;
    bucket = entry.bucket;
    firstRay = entry.firstRay;
    lastRay = entry.lastRay;
      
    bucketSize[bucket] = firstRay;
  }

  AVX_ZERO_UPPER();
}

template<typename TriangleIntersector>
void BVH4IntersectorStream<TriangleIntersector>::occluded(BVH4IntersectorStream* This, StreamRay* rayData, StreamRayExtra* rayExtras, unsigned count, unsigned char* occluded, unsigned thread) {
  AVX_ZERO_UPPER();
  
  ThreadState& threadState = This->threadStates[thread];
  threadState.reserve(count);
  
  unsigned* bucketSize = threadState.bucketSize;
  unsigned short** buckets = threadState.buckets;

  StackEntry* stack = threadState.stack;
  StackEntry* const stackStart = stack;

  const BVH4* bvh = This->bvh;

  const void* nodePtr = bvh->nodePtr();
  const void* triPtr = bvh->triPtr();

  NodeRef node = bvh->root;
  unsigned bucket = 0;
  unsigned firstRay = 0;
  unsigned lastRay = count;

  for (unsigned i = 0; i < BVH4IntersectorStream<TriangleIntersector>::bucketCount; ++i)
    bucketSize[i] = 0;

#if 0
  setupRays(buckets, count);

  buckets[0][count] = count - 1;
  buckets[0][count + 1] = count - 1;
  buckets[0][count + 2] = count - 1;
#else
  // Bucket rays based on direction sign.
  unsigned short* orderBuckets[8];

  for (unsigned i = 0; i < 8; ++i)
    orderBuckets[i] = buckets[i];

  unsigned bucketRay = 0;

  for (; (int)bucketRay < (int)count - 3; bucketRay += 4) {
    __m128 m0 = _mm_load_ps(&rayData[bucketRay + 0].invDirX);
    __m128 m1 = _mm_load_ps(&rayData[bucketRay + 1].invDirX);
    __m128 m2 = _mm_load_ps(&rayData[bucketRay + 2].invDirX);
    __m128 m3 = _mm_load_ps(&rayData[bucketRay + 3].invDirX);

    unsigned swap0 = _mm_movemask_ps(m0) & 7;
    unsigned swap1 = _mm_movemask_ps(m1) & 7;
    unsigned swap2 = _mm_movemask_ps(m2) & 7;
    unsigned swap3 = _mm_movemask_ps(m3) & 7;

    *(orderBuckets[swap0]++) = bucketRay + 0;
    *(orderBuckets[swap1]++) = bucketRay + 1;
    *(orderBuckets[swap2]++) = bucketRay + 2;
    *(orderBuckets[swap3]++) = bucketRay + 3;

    *reinterpret_cast<unsigned*>(occluded + bucketRay) = 0;
  }

  for (; bucketRay < count; ++bucketRay) {
    __m128 m = _mm_load_ps(&rayData[bucketRay].invDirX);

    unsigned swap = _mm_movemask_ps(m) & 7;

    *(orderBuckets[swap]++) = bucketRay;

    occluded[bucketRay] = 0;
  }

  for (unsigned i = 0; i < 8; ++i) {
    if (orderBuckets[i] == buckets[i])
      continue;

    unsigned newSize = (unsigned)(orderBuckets[i] - buckets[i]);
    StackEntry& e = *(stack++);
    e.node = node;
    e.bucket = (unsigned)i;
    e.firstRay = 0;
    e.lastRay = newSize;
    bucketSize[i] = align(newSize + 2);
    orderBuckets[i][0] = orderBuckets[i][1] = orderBuckets[i][2] = orderBuckets[i][-1];
  }

  if (stack == stackStart) {
    return;
  }

  const StackEntry& entry = *(--stack);

  bucket = entry.bucket;
  firstRay = entry.firstRay;
  lastRay = entry.lastRay;

  bucketSize[bucket] = firstRay;
#endif

  for (;;) {
    if (likely(node.isNode())) {
      if (lastRay - firstRay >= 16) {
        intersectChildrenAndBucketRaysNoOrder(nodePtr, &stack, buckets, bucketSize, buckets[bucket] + firstRay, buckets[bucket] + lastRay, node, rayData);
      }
      else {
        unsigned short* first = buckets[bucket] + firstRay;
        unsigned short* last = buckets[bucket] + lastRay;

        __m256 dxDyDzTnOxOyOzTf = _mm256_load_ps(&rayData[*first].invDirX);
        unsigned mask = _mm256_movemask_ps(dxDyDzTnOxOyOzTf);

        const avxf pos_neg = avxf(ssef(+0.0f), ssef(-0.0f));
        const avxf neg_pos = avxf(ssef(-0.0f), ssef(+0.0f));

        unsigned swapX = mask & 1;
        unsigned swapY = mask & 2;
        unsigned swapZ = mask & 4;

        const avxf flipSignX = swapX ? neg_pos : pos_neg;
        const avxf flipSignY = swapY ? neg_pos : pos_neg;
        const avxf flipSignZ = swapZ ? neg_pos : pos_neg;

        do {
          StreamRay& ray = rayData[*first];
          StreamRayExtra& rayExtra = rayExtras[*first];
          unsigned char& rayOccluded = occluded[*first];
          ++first;

          typedef typename TriangleIntersector::Triangle Triangle;
          typedef typename BVH4::NodeRef NodeRef;
          typedef typename BVH4::Node Node;

          __m256 dxDyDzTn = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTf, dxDyDzTnOxOyOzTf, (0) | ((0) << 4));
          __m256 oxOyOzTf = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTf, dxDyDzTnOxOyOzTf, (1) | ((1) << 4));

          /*! stack state */
          NodeRef stack[1 + 3 * BVH4::maxDepth];  //!< stack of nodes 
          NodeRef* stackPtr = stack + 1;        //!< current stack pointer
          stack[0] = node;
          
          __m256 pmRay = dxDyDzTnOxOyOzTf ^ pos_neg;

          /*! load the ray into SIMD registers */
          const avx3f rdir(
            _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(0, 0, 0, 0)) ^ flipSignX,
            _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(1, 1, 1, 1)) ^ flipSignY,
            _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(2, 2, 2, 2)) ^ flipSignZ);

          const avx3f org_rdir(
            _mm256_shuffle_ps(oxOyOzTf, oxOyOzTf, _MM_SHUFFLE(0, 0, 0, 0)) ^ flipSignX,
            _mm256_shuffle_ps(oxOyOzTf, oxOyOzTf, _MM_SHUFFLE(1, 1, 1, 1)) ^ flipSignY,
            _mm256_shuffle_ps(oxOyOzTf, oxOyOzTf, _MM_SHUFFLE(2, 2, 2, 2)) ^ flipSignZ);

          avxf rayNearFar = _mm256_shuffle_ps(pmRay, pmRay, _MM_SHUFFLE(3, 3, 3, 3));//(ssef(ray.tnear),-ssef(ray.tfar));

          dxDyDzTnOxOyOzTf = _mm256_load_ps(&rayData[*first].invDirX);

          if (rayOccluded)
            continue;

          const void* nodePtr = bvh->nodePtr();
          const void* triPtr = bvh->triPtr();

          /* pop loop */
          while (true) pop:
          {
            /*! pop next node */
            if (unlikely(stackPtr == stack)) break;
            stackPtr--;
            NodeRef cur = *stackPtr;

            /* downtraversal loop */
            while (true)
            {
              /*! stop if we found a leaf */
              if (unlikely(cur.isLeaf())) break;

              /*! single ray intersection with 4 boxes */
              const Node* node = cur.node(nodePtr);

              avxf bbX = avxf::load(&node->lower_x);
              avxf bbY = avxf::load(&node->lower_y);
              avxf bbZ = avxf::load(&node->lower_z);

              const avxf tLowerUpperX = msub(bbX, rdir.x, org_rdir.x);
              const avxf tLowerUpperY = msub(bbY, rdir.y, org_rdir.y);
              const avxf tLowerUpperZ = msub(bbZ, rdir.z, org_rdir.z);

              /*const BVH4::Node* leftmost = node->child(1).node(nodePtr);
              const BVH4::Node* rightmost = node->child(2).node(nodePtr);

              _mm_prefetch(reinterpret_cast<const char*>(&leftmost->lower_x), _MM_HINT_T0);
              _mm_prefetch(reinterpret_cast<const char*>(&rightmost->lower_x), _MM_HINT_T0);*/

              const avxf tNearFarX = swapX ? shuffle<1, 0>(tLowerUpperX) : tLowerUpperX;
              const avxf tNearFarY = swapY ? shuffle<1, 0>(tLowerUpperY) : tLowerUpperY;
              const avxf tNearFarZ = swapZ ? shuffle<1, 0>(tLowerUpperZ) : tLowerUpperZ;

              const avxf tNearFar = max(rayNearFar, tNearFarX, tNearFarY, tNearFarZ);
              const ssef tNear = extract<0>(tNearFar);
              const ssef tFar = extract<1>(tNearFar);

              size_t mask = movemask(-tNear >= tFar);

              //_mm_prefetch(reinterpret_cast<const char*>(&leftmost->lower_z), _MM_HINT_T0);
              //_mm_prefetch(reinterpret_cast<const char*>(&rightmost->lower_z), _MM_HINT_T0);

              /*! if no child is hit, pop next node */
              if (unlikely(mask == 0))
                goto pop;

              /*! one child is hit, continue with that child */
              size_t r = __bsf(mask); mask = __btc(mask, r);
              if (likely(mask == 0)) {
                cur = node->child(r);
                continue;
              }

              /*! two children are hit, push far child, and continue with closer child */
              NodeRef c0 = node->child(r); const float d0 = tNear[r];
              r = __bsf(mask); mask = __btc(mask, r);
              NodeRef c1 = node->child(r); const float d1 = tNear[r];
              if (likely(mask == 0)) {
                if (d0 < d1) { *stackPtr = c1; stackPtr++; cur = c0; continue; }
                else         { *stackPtr = c0; stackPtr++; cur = c1; continue; }
              }
              *stackPtr = c0; stackPtr++;
              *stackPtr = c1; stackPtr++;

              /*! three children are hit */
              r = __bsf(mask); mask = __btc(mask, r);
              cur = node->child(r); *stackPtr = cur; stackPtr++;
              if (likely(mask == 0)) {
                stackPtr--;
                continue;
              }

              /*! four children are hit */
              cur = node->child(3);
            }

            /*! this is a leaf node */
            size_t num; Triangle* tri = (Triangle*)cur.leaf(triPtr, num);
            for (size_t i = 0; i < num; i++) {
              if (TriangleIntersector::occluded(ray, rayExtra, tri[i], bvh->vertices)) {
                rayOccluded = 1;
                ray.tfar = 0.0f;
                ray.tnear = 1.0f;
                goto doneWithRay;
              }
            }
          }
        doneWithRay:;
        } while (first < last);
      }
    }
    else {
      size_t triangleCount;
      Triangle* triangles = (Triangle*)node.leaf(0, triangleCount);

      occludeByTriangles<Triangle, TriangleIntersector>(buckets[bucket] + firstRay, buckets[bucket] + lastRay, rayData, rayExtras, occluded, triangles, (unsigned)triangleCount, bvh->vertices);
    }

    if (stack == stackStart)
      break;

    const StackEntry& entry = *(--stack);

    node = entry.node;
    bucket = entry.bucket;
    firstRay = entry.firstRay;
    lastRay = entry.lastRay;

    bucketSize[bucket] = firstRay;
  }
  
  AVX_ZERO_UPPER();
}

namespace embree {
  
  void BVH4IntersectorStreamRegister () {
    TriangleMesh::intersectorsStream.add("bvh4","triangle1" ,"stream","moeller" ,true ,BVH4IntersectorStream<Triangle1Intersector1MoellerTrumboreStream>::create);
    TriangleMesh::intersectorsStream.add("bvh4","triangle4" ,"stream","moeller" ,true ,BVH4IntersectorStream<Triangle4Intersector1MoellerTrumboreStream>::create);
    TriangleMesh::intersectorsStream.add("bvh4","triangle8" ,"stream","moeller" ,true ,BVH4IntersectorStream<Triangle8Intersector1MoellerTrumboreStream>::create);
    TriangleMesh::intersectorsStream.setAccelDefaultTraverser("bvh4","stream");
  }
  
}
