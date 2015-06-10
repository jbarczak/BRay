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

#include "bvh2_intersector_stream.h"
#include "geometry/triangles.h"
#include "../common/stream_ray.h"

using namespace embree;

__forceinline unsigned align(unsigned x) {
  static const unsigned width = 2;
  return (x + (width-1)) & (~(width-1));
}

__forceinline void setupRays(short** buckets, unsigned rayCount) {
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

__forceinline __m256 simd_float_combine(__m128 a, __m128 b) {
  __m256 c = _mm256_castps128_ps256(a);
  c = _mm256_insertf128_ps(c, b, 1);
  return c;
}

static void intersectChildrenAndBucketRays(StackEntry* __restrict* __restrict stack, short* __restrict* __restrict buckets, unsigned* __restrict bucketSize, short* __restrict firstRay, short* __restrict lastRay, BVH2::NodeRef n, StreamRay* __restrict rayData) {
  const BVH2::Node* __restrict node = n.node();
  
  __m256 minMinMaxMaxX = _mm256_broadcast_ps(&node->lower_upper_x.m128);
  __m256 minMinMaxMaxY = _mm256_broadcast_ps(&node->lower_upper_y.m128);
  __m256 minMinMaxMaxZ = _mm256_broadcast_ps(&node->lower_upper_z.m128);

  short* __restrict bucket0 = buckets[0] + bucketSize[0]; // left hit &  left first
  short* __restrict bucket1 = buckets[1] + bucketSize[1]; // right hit
  short* __restrict bucket2 = buckets[2] + bucketSize[2]; // left hit & right first
  
  short* r = firstRay;

  __m128 pnl = _mm_castsi128_ps(_mm_setr_epi32(0x00000000, 0x00000000, 0x80000000, 0x80000000));
  __m256 pn = simd_float_combine(pnl, pnl);
    
  __m256 bbX = minMinMaxMaxX;
  __m256 bbY = minMinMaxMaxY;
  __m256 bbZ = minMinMaxMaxZ;

  __m256 bbXs = _mm256_shuffle_ps(bbX, bbX, _MM_SHUFFLE(1,0,3,2));
  __m256 bbYs = _mm256_shuffle_ps(bbY, bbY, _MM_SHUFFLE(1,0,3,2));
  __m256 bbZs = _mm256_shuffle_ps(bbZ, bbZ, _MM_SHUFFLE(1,0,3,2));

  bbX = _mm256_xor_ps(bbX, pn);
  bbY = _mm256_xor_ps(bbY, pn);
  bbZ = _mm256_xor_ps(bbZ, pn);

  bbXs = _mm256_xor_ps(bbXs, pn);
  bbYs = _mm256_xor_ps(bbYs, pn);
  bbZs = _mm256_xor_ps(bbZs, pn);

  __m256 neg = _mm256_set1_ps(-0.0f);

  unsigned nextIndexA = r[0];
  unsigned nextIndexB = r[1];

  const StreamRay* rayA = &rayData[nextIndexA];
  const StreamRay* rayB = &rayData[nextIndexB];

  __m256 dxDyDzTnOxOyOzTfA = _mm256_load_ps(&rayA->invDirX);
  __m256 dxDyDzTnOxOyOzTfB = _mm256_load_ps(&rayB->invDirX);

  __m256 dxDyDzTn = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (0) | ((2) << 4));
  __m256 oxOyOzTf = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (1) | ((3) << 4));

  __m256 oxOyOzTfneg = _mm256_xor_ps(oxOyOzTf, neg);

  __m256 rdirX = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(0,0,0,0));
  __m256 rdirY = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(1,1,1,1));
  __m256 rdirZ = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(2,2,2,2));

  unsigned mask = 0;
  unsigned indexA = 0, indexB = 0;

  for (; r < lastRay; ) {
    unsigned lhit, rhit, lf;

    lhit = mask >> 4;
    rhit = (mask >> 5) & 1;
    lf = (mask >> 6) & rhit;

    __m256 rbbX = _mm256_blendv_ps(bbX, bbXs, rdirX);
    __m256 orgInvDirX = _mm256_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(0,0,0,0));

    __m256 rbbY = _mm256_blendv_ps(bbY, bbYs, rdirY);
    __m256 orgInvDirY = _mm256_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(1,1,1,1));

    __m256 rbbZ = _mm256_blendv_ps(bbZ, bbZs, rdirZ);
    __m256 orgInvDirZ = _mm256_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(2,2,2,2));

    *bucket0 = indexB;
    *bucket1 = indexB;
    *bucket2 = indexB;

    bucket0 += lhit & (1^lf);
    bucket1 += rhit;
    bucket2 += lhit & lf;

    indexA = nextIndexA;
    indexB = nextIndexB;

    r += 2;

    nextIndexA = r[0];
    nextIndexB = r[1];

    __m256 tNearFarX = _mm256_fmsub_ps(rbbX, rdirX, orgInvDirX);
    __m256 tNearFarY = _mm256_fmsub_ps(rbbY, rdirY, orgInvDirY);
    __m256 tNearFarZ = _mm256_fmsub_ps(rbbZ, rdirZ, orgInvDirZ);

    __m256 nearFar = _mm256_shuffle_ps(dxDyDzTn, oxOyOzTfneg, _MM_SHUFFLE(3,3,3,3));

    rayA = &rayData[nextIndexA];
    rayB = &rayData[nextIndexB];

    dxDyDzTnOxOyOzTfA = _mm256_load_ps(&rayA->invDirX);
    dxDyDzTnOxOyOzTfB = _mm256_load_ps(&rayB->invDirX);

    __m256 tNearFar = _mm256_max_ps(_mm256_max_ps(nearFar, tNearFarX), _mm256_max_ps(tNearFarY, tNearFarZ));

    mask = _mm256_movemask_ps(_mm256_cmple_ps(_mm256_shuffle_ps(tNearFar, tNearFar, _MM_SHUFFLE(0,1,1,0)), _mm256_shuffle_ps(_mm256_xor_ps(tNearFar, neg), tNearFar, _MM_SHUFFLE(0,0,3,2))));
    // 0: lhit, 1: rhit, 2: rfirst

    dxDyDzTn = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (0) | ((2) << 4));
    oxOyOzTf = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (1) | ((3) << 4));

    rdirX = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(0,0,0,0));
    rdirY = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(1,1,1,1));
    rdirZ = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(2,2,2,2));

    oxOyOzTfneg = _mm256_xor_ps(oxOyOzTf, neg);

    lhit = mask;
    rhit = (mask >> 1) & 1;
    lf = (mask >> 2) & rhit;

    *bucket0 = indexA;
    *bucket1 = indexA;
    *bucket2 = indexA;
    
    bucket0 += lhit & (1^lf);
    bucket1 += rhit;
    bucket2 += lhit & lf;
  }

  if (indexA != indexB) {
    unsigned lhit, rhit, lf;

    *bucket0 = indexB;
    *bucket1 = indexB;
    *bucket2 = indexB;

    lhit = mask >> 4;
    rhit = (mask >> 5) & 1;
    lf = (mask >> 6) & rhit;

    bucket0 += lhit & (1^lf);
    bucket1 += rhit;
    bucket2 += lhit & lf;
  }

  // Push bucket intervals to stack.
  BVH2::NodeRef left = node->child(0);
  BVH2::NodeRef right = node->child(1);

  StackEntry* __restrict s = *stack;

  if (bucket2 != buckets[2] + bucketSize[2]) {
    unsigned newSize = (unsigned)(bucket2 - buckets[2]);
    StackEntry& e = *(s++);

    e.node = left;
    e.bucket = (unsigned)2;
    e.firstRay = bucketSize[2];
    e.lastRay = newSize;

    bucketSize[2] = align(newSize+2);
    bucket2[0] = bucket2[1] = bucket2[2] = bucket2[-1];
  }

  if (bucket1 != buckets[1] + bucketSize[1]) {
    unsigned newSize = (unsigned)(bucket1 - buckets[1]);
    StackEntry& e = *(s++);

    e.node = right;
    e.bucket = (unsigned)1;
    e.firstRay = bucketSize[1];
    e.lastRay = newSize;

    bucketSize[1] = align(newSize+2);
    bucket1[0] = bucket1[1] = bucket1[2] = bucket1[-1];
  }

  if (bucket0 != buckets[0] + bucketSize[0]) {
    unsigned newSize = (unsigned)(bucket0 - buckets[0]);
    StackEntry& e = *(s++);

    e.node = left;
    e.bucket = (unsigned)0;
    e.firstRay = bucketSize[0];
    e.lastRay = newSize;

    bucketSize[0] = align(newSize+2);
    bucket0[0] = bucket0[1] = bucket0[2] = bucket0[-1];
  }

  *stack = s;
}

static void intersectChildrenAndBucketRaysNoOrder(StackEntry* __restrict* __restrict stack, short* __restrict* __restrict buckets, unsigned* __restrict bucketSize, short* __restrict firstRay, short* __restrict lastRay, BVH2::NodeRef n, StreamRay* __restrict rayData) {
  const BVH2::Node* __restrict node = n.node();

  __m256 minMinMaxMaxX = _mm256_broadcast_ps(&node->lower_upper_x.m128);
  __m256 minMinMaxMaxY = _mm256_broadcast_ps(&node->lower_upper_y.m128);
  __m256 minMinMaxMaxZ = _mm256_broadcast_ps(&node->lower_upper_z.m128);

  short* __restrict bucket0 = buckets[0] + bucketSize[0]; // left hit
  short* __restrict bucket1 = buckets[1] + bucketSize[1]; // right hit

  short* r = firstRay;

  __m128 pnl = _mm_castsi128_ps(_mm_setr_epi32(0x00000000, 0x00000000, 0x80000000, 0x80000000));
  __m256 pn = simd_float_combine(pnl, pnl);

  __m256 bbX = minMinMaxMaxX;
  __m256 bbY = minMinMaxMaxY;
  __m256 bbZ = minMinMaxMaxZ;

  __m256 bbXs = _mm256_shuffle_ps(bbX, bbX, _MM_SHUFFLE(1, 0, 3, 2));
  __m256 bbYs = _mm256_shuffle_ps(bbY, bbY, _MM_SHUFFLE(1, 0, 3, 2));
  __m256 bbZs = _mm256_shuffle_ps(bbZ, bbZ, _MM_SHUFFLE(1, 0, 3, 2));

  bbX = _mm256_xor_ps(bbX, pn);
  bbY = _mm256_xor_ps(bbY, pn);
  bbZ = _mm256_xor_ps(bbZ, pn);

  bbXs = _mm256_xor_ps(bbXs, pn);
  bbYs = _mm256_xor_ps(bbYs, pn);
  bbZs = _mm256_xor_ps(bbZs, pn);

  __m256 neg = _mm256_set1_ps(-0.0f);

  unsigned nextIndexA = r[0];
  unsigned nextIndexB = r[1];

  const StreamRay* rayA = &rayData[nextIndexA];
  const StreamRay* rayB = &rayData[nextIndexB];

  __m256 dxDyDzTnOxOyOzTfA = _mm256_load_ps(&rayA->invDirX);
  __m256 dxDyDzTnOxOyOzTfB = _mm256_load_ps(&rayB->invDirX);

  __m256 dxDyDzTn = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (0) | ((2) << 4));
  __m256 oxOyOzTf = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (1) | ((3) << 4));

  __m256 oxOyOzTfneg = _mm256_xor_ps(oxOyOzTf, neg);

  __m256 rdirX = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(0, 0, 0, 0));
  __m256 rdirY = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(1, 1, 1, 1));
  __m256 rdirZ = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(2, 2, 2, 2));

  unsigned mask = 0;
  unsigned indexA = 0, indexB = 0;

  for (; r < lastRay;) {
    unsigned lhit, rhit;

    lhit = (mask >> 6) & 1;
    rhit = (mask >> 7);

    __m256 rbbX = _mm256_blendv_ps(bbX, bbXs, rdirX);
    __m256 orgInvDirX = _mm256_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(0, 0, 0, 0));

    __m256 rbbY = _mm256_blendv_ps(bbY, bbYs, rdirY);
    __m256 orgInvDirY = _mm256_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(1, 1, 1, 1));

    __m256 rbbZ = _mm256_blendv_ps(bbZ, bbZs, rdirZ);
    __m256 orgInvDirZ = _mm256_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(2, 2, 2, 2));

    *bucket0 = indexB;
    *bucket1 = indexB;
    
    bucket0 += lhit;
    bucket1 += rhit;
    
    indexA = nextIndexA;
    indexB = nextIndexB;

    r += 2;

    nextIndexA = r[0];
    nextIndexB = r[1];

    __m256 tNearFarX = _mm256_fmsub_ps(rbbX, rdirX, orgInvDirX);
    __m256 tNearFarY = _mm256_fmsub_ps(rbbY, rdirY, orgInvDirY);
    __m256 tNearFarZ = _mm256_fmsub_ps(rbbZ, rdirZ, orgInvDirZ);

    __m256 nearFar = _mm256_shuffle_ps(dxDyDzTn, oxOyOzTfneg, _MM_SHUFFLE(3, 3, 3, 3));

    rayA = &rayData[nextIndexA];
    rayB = &rayData[nextIndexB];

    dxDyDzTnOxOyOzTfA = _mm256_load_ps(&rayA->invDirX);
    dxDyDzTnOxOyOzTfB = _mm256_load_ps(&rayB->invDirX);

    __m256 tNearFar = _mm256_max_ps(_mm256_max_ps(nearFar, tNearFarX), _mm256_max_ps(tNearFarY, tNearFarZ));

    mask = _mm256_movemask_ps(_mm256_cmple_ps(_mm256_shuffle_ps(tNearFar, tNearFar, _MM_SHUFFLE(1, 0, 1, 0)), _mm256_xor_ps(tNearFar, neg)));
    // ?, ?, 2: lhit, 3: rhit

    dxDyDzTn = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (0) | ((2) << 4));
    oxOyOzTf = _mm256_permute2f128_ps(dxDyDzTnOxOyOzTfA, dxDyDzTnOxOyOzTfB, (1) | ((3) << 4));

    rdirX = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(0, 0, 0, 0));
    rdirY = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(1, 1, 1, 1));
    rdirZ = _mm256_shuffle_ps(dxDyDzTn, dxDyDzTn, _MM_SHUFFLE(2, 2, 2, 2));

    oxOyOzTfneg = _mm256_xor_ps(oxOyOzTf, neg);

    lhit = (mask >> 2) & 1;
    rhit = (mask >> 3) & 1;
    
    *bucket0 = indexA;
    *bucket1 = indexA;
    
    bucket0 += lhit;
    bucket1 += rhit;
  }

  if (indexA != indexB) {
    unsigned lhit, rhit, lf;

    *bucket0 = indexB;
    *bucket1 = indexB;
    
    lhit = (mask >> 6) & 1;
    rhit = (mask >> 7);
    
    bucket0 += lhit;
    bucket1 += rhit;
  }

  unsigned size0 = (unsigned)(bucket0 - buckets[0]) - bucketSize[0];
  unsigned size1 = (unsigned)(bucket1 - buckets[1]) - bucketSize[1];

  unsigned size[] = {
    size0 + size0,
    size1 + size1 + 1,
  };

  if (size[0] > size[1])
    std::swap(size[0], size[1]);

  StackEntry* __restrict s = *stack;

  for (unsigned i = 0; i < 2; ++i) {
    unsigned added = size[i] >> 1;
    unsigned n = size[i] & 1;

    if (!added)
      continue;

    unsigned newSize = bucketSize[n] + added;
    StackEntry& e = *(s++);
    e.node = node->child(n);
    e.bucket = n;
    e.firstRay = bucketSize[n];
    e.lastRay = newSize;
    bucketSize[n] = align(newSize + 2);
    short* __restrict b = buckets[n];
    b[newSize] = b[newSize + 1] = b[newSize + 2] = b[newSize - 1];
  }

  *stack = s;
}

template<class Triangle, class TriangleIntersector>
static void intersectTriangles(short* first, short* last, StreamRay* rayData, StreamRayExtra* rayExtras, StreamHit* hits, Triangle* triangles, unsigned triangleCount, const Vec3fa* vertices) {
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
void BVH2IntersectorStream<TriangleIntersector>::intersect(BVH2IntersectorStream* This, StreamRay* rayData, StreamRayExtra* rayExtras, StreamHit* hits, unsigned count, unsigned thread) {
  AVX_ZERO_UPPER();
  
  ThreadState& threadState = This->threadStates[thread];
  threadState.reserve(count);
  
  unsigned* bucketSize = threadState.bucketSize;
  short** buckets = threadState.buckets;
  
  StackEntry* stack = threadState.stack;
  StackEntry* const stackStart = stack;
  
  const BVH2* bvh = This->bvh;
    
  NodeRef node = bvh->root;
  unsigned bucket = 0;
  unsigned firstRay = 0;
  unsigned lastRay = count;

  setupRays(buckets, count);

  buckets[0][count] = count-1;
  buckets[0][count+1] = count-1;
  buckets[0][count+2] = count-1;
    
  for (unsigned i = 0; i < BVH2IntersectorStream<TriangleIntersector>::bucketCount; ++i)
    bucketSize[i] = 0;
    
  for (;;) {
    if (likely(node.isNode())) {
      if (lastRay - firstRay >= 8) {
        intersectChildrenAndBucketRays(&stack, buckets, bucketSize, buckets[bucket] + firstRay, buckets[bucket] + lastRay, node, rayData);
      }
      else {
        __m128 pn = _mm_castsi128_ps(_mm_setr_epi32(0x00000000, 0x00000000, 0x80000000, 0x80000000));
        __m128 neg = _mm_set1_ps(-0.0f);

        const ssei identity = _mm_set_epi8(15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 0);
        const ssei swap     = _mm_set_epi8( 7,  6,  5,  4,  3,  2,  1,  0, 15, 14, 13, 12, 11, 10,  9, 8);

        short* first =  buckets[bucket] + firstRay;
        short* last = buckets[bucket] + lastRay;

        for (; first < last; ++first) {
          StreamRay& ray = rayData[*first];
          StreamRayExtra& rayExtra = rayExtras[*first];
          StreamHit& hit = hits[*first];

          STAT3(normal.travs,1,1,1);

          struct StackItem {
            NodeRef ptr;   //!< node pointer
            float dist;  //!< distance of node
          };

          /*! stack state */
          StackItem stack[1+BVH2::maxDepth];  //!< stack of nodes that still need to get traversed
          StackItem* stackPtr = stack;        //!< current stack pointer
          NodeRef cur = node;              //!< in cur we track the ID of the current node

          /*! load the ray into SIMD registers */
          __m256 dxDyDzTnOxOyOzTf = _mm256_load_ps(&ray.invDirX);
          __m128 dxDyDzTn = _mm256_castps256_ps128(dxDyDzTnOxOyOzTf);
          __m128 oxOyOzTf = _mm256_extractf128_ps(dxDyDzTnOxOyOzTf, 1);

          __m128 dxDyDzTnneg = _mm_xor_ps(dxDyDzTn, neg);
          __m128 oxOyOzTfneg = _mm_xor_ps(oxOyOzTf, neg);

          unsigned shuffleMask = _mm256_movemask_ps(dxDyDzTnOxOyOzTf);

          __m128 rdirX = _mm_shuffle_ps(dxDyDzTn, dxDyDzTnneg, _MM_SHUFFLE(0,0,0,0));
          __m128 rdirY = _mm_shuffle_ps(dxDyDzTn, dxDyDzTnneg, _MM_SHUFFLE(1,1,1,1));
          __m128 rdirZ = _mm_shuffle_ps(dxDyDzTn, dxDyDzTnneg, _MM_SHUFFLE(2,2,2,2));

          __m128 orgInvDirX = _mm_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(0,0,0,0));
          __m128 orgInvDirY = _mm_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(1,1,1,1));
          __m128 orgInvDirZ = _mm_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(2,2,2,2));

          const ssei shuffleX = shuffleMask & (   1) ? swap : identity;
          const ssei shuffleY = shuffleMask & (1<<1) ? swap : identity;
          const ssei shuffleZ = shuffleMask & (1<<2) ? swap : identity;

          const sse3f rorg(orgInvDirX, orgInvDirY, orgInvDirZ);
          const sse3f rdir = sse3f(rdirX, rdirY, rdirZ);
          ssef nearFar(_mm_shuffle_ps(dxDyDzTn, oxOyOzTfneg, _MM_SHUFFLE(3,3,3,3)));

          while (true)
          {
            /*! downtraversal loop */
            while (likely(cur.isNode()))
            {
              /*! single ray intersection with box of both children. */
              const Node* node = cur.node();
              const ssef tNearFarX = msub(shuffle8(node->lower_upper_x,shuffleX), rdir.x, rorg.x);
              const ssef tNearFarY = msub(shuffle8(node->lower_upper_y,shuffleY), rdir.y, rorg.y);
              const ssef tNearFarZ = msub(shuffle8(node->lower_upper_z,shuffleZ), rdir.z, rorg.z);
              const ssef tNearFar = max(tNearFarX,tNearFarY,tNearFarZ,nearFar);
              const sseb lrhit = (tNearFar ^ neg) >= shuffle8(tNearFar,swap);

              /*! if two children hit, push far node onto stack and continue with closer node */
              if (likely(lrhit[0] != 0 && lrhit[1] != 0)) {
                if (likely(tNearFar[0] < tNearFar[1])) {
                  stackPtr->ptr = node->child(1);
                  stackPtr->dist = tNearFar[1];
                  cur = node->child(0);
                  stackPtr++;
                }
                else {
                  stackPtr->ptr = node->child(0);
                  stackPtr->dist = tNearFar[0];
                  cur = node->child(1);
                  stackPtr++;
                }
              }

              /*! if one child hit, continue with that child */
              else {
                if      (likely(lrhit[0] != 0)) cur = node->child(0);
                else if (likely(lrhit[1] != 0)) cur = node->child(1);
                else goto pop_node;
              }
            }

            /*! leaf node, intersect all triangles */
            {
              STAT3(shadow.trav_leaves,1,1,1);
              size_t num; Triangle* tri = (Triangle*) cur.leaf(NULL,num);
              for (size_t i=0; i<num; i++)
                TriangleIntersector::intersect(ray,rayExtra,hit,tri[i],bvh->vertices);
              nearFar = shuffle<0,1,2,3>(nearFar,-ray.tfar);
            }

            /*! pop next node from stack */
pop_node:
            if (unlikely(stackPtr == stack)) break;
            --stackPtr;
            cur = stackPtr->ptr;
            if (unlikely(stackPtr->dist > ray.tfar)) goto pop_node;
          }
        }
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
void BVH2IntersectorStream<TriangleIntersector>::occluded(BVH2IntersectorStream* This, StreamRay* rayData, StreamRayExtra* rayExtras, unsigned count, unsigned char* occluded, unsigned thread) {
  AVX_ZERO_UPPER();
  
  ThreadState& threadState = This->threadStates[thread];
  threadState.reserve(count);

  unsigned* bucketSize = threadState.bucketSize;
  short** buckets = threadState.buckets;

  StackEntry* stack = threadState.stack;
  StackEntry* const stackStart = stack;

  const BVH2* bvh = This->bvh;

  NodeRef node = bvh->root;
  unsigned bucket = 0;
  unsigned firstRay = 0;
  unsigned lastRay = count;

  setupRays(buckets, count);

  buckets[0][count] = count-1;
  buckets[0][count+1] = count-1;
  buckets[0][count+2] = count-1;

  unsigned i = 0;
  __m256 a = _mm256_setzero_ps();
  for (; i < count; i += 64) {
    _mm256_store_ps((float*)(occluded + i), a);
    _mm256_store_ps((float*)(occluded + 32 + i), a);
  }

  for (unsigned i = 0; i < BVH2IntersectorStream<TriangleIntersector>::bucketCount; ++i)
    bucketSize[i] = 0;

  for (;;) {
    if (likely(node.isNode())) {
      if (lastRay - firstRay >= 8) {
        intersectChildrenAndBucketRaysNoOrder(&stack, buckets, bucketSize, buckets[bucket] + firstRay, buckets[bucket] + lastRay, node, rayData);
      }
      else {
        __m128 pn = _mm_castsi128_ps(_mm_setr_epi32(0x00000000, 0x00000000, 0x80000000, 0x80000000));
        __m128 neg = _mm_set1_ps(-0.0f);

        const ssei identity = _mm_set_epi8(15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 0);
        const ssei swap     = _mm_set_epi8( 7,  6,  5,  4,  3,  2,  1,  0, 15, 14, 13, 12, 11, 10,  9, 8);

        short* first =  buckets[bucket] + firstRay;
        short* last = buckets[bucket] + lastRay;

        for (; first < last; ++first) {
          StreamRay& ray = rayData[*first];
          StreamRayExtra& rayExtra = rayExtras[*first];
          unsigned char& rayOccluded = occluded[*first];

          if (rayOccluded)
            continue;

          STAT3(normal.travs,1,1,1);

          struct StackItem {
            NodeRef ptr;   //!< node pointer
            float dist;  //!< distance of node
          };

          /*! stack state */
          StackItem stack[1+BVH2::maxDepth];  //!< stack of nodes that still need to get traversed
          StackItem* stackPtr = stack;        //!< current stack pointer
          NodeRef cur = node;              //!< in cur we track the ID of the current node

          /*! load the ray into SIMD registers */
          __m256 dxDyDzTnOxOyOzTf = _mm256_load_ps(&ray.invDirX);
          __m128 dxDyDzTn = _mm256_castps256_ps128(dxDyDzTnOxOyOzTf);
          __m128 oxOyOzTf = _mm256_extractf128_ps(dxDyDzTnOxOyOzTf, 1);

          __m128 dxDyDzTnneg = _mm_xor_ps(dxDyDzTn, neg);
          __m128 oxOyOzTfneg = _mm_xor_ps(oxOyOzTf, neg);

          unsigned shuffleMask = _mm256_movemask_ps(dxDyDzTnOxOyOzTf);

          __m128 rdirX = _mm_shuffle_ps(dxDyDzTn, dxDyDzTnneg, _MM_SHUFFLE(0,0,0,0));
          __m128 rdirY = _mm_shuffle_ps(dxDyDzTn, dxDyDzTnneg, _MM_SHUFFLE(1,1,1,1));
          __m128 rdirZ = _mm_shuffle_ps(dxDyDzTn, dxDyDzTnneg, _MM_SHUFFLE(2,2,2,2));

          __m128 orgInvDirX = _mm_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(0,0,0,0));
          __m128 orgInvDirY = _mm_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(1,1,1,1));
          __m128 orgInvDirZ = _mm_shuffle_ps(oxOyOzTf, oxOyOzTfneg, _MM_SHUFFLE(2,2,2,2));

          const ssei shuffleX = shuffleMask & (   1) ? swap : identity;
          const ssei shuffleY = shuffleMask & (1<<1) ? swap : identity;
          const ssei shuffleZ = shuffleMask & (1<<2) ? swap : identity;

          const sse3f rorg(orgInvDirX, orgInvDirY, orgInvDirZ);
          const sse3f rdir = sse3f(rdirX, rdirY, rdirZ);
          ssef nearFar(_mm_shuffle_ps(dxDyDzTn, oxOyOzTfneg, _MM_SHUFFLE(3,3,3,3)));

          while (true)
          {
            /*! downtraversal loop */
            while (likely(cur.isNode()))
            {
              /*! single ray intersection with box of both children. */
              const Node* node = cur.node();
              const ssef tNearFarX = msub(shuffle8(node->lower_upper_x, shuffleX), rdir.x, rorg.x);
              const ssef tNearFarY = msub(shuffle8(node->lower_upper_y, shuffleY), rdir.y, rorg.y);
              const ssef tNearFarZ = msub(shuffle8(node->lower_upper_z, shuffleZ), rdir.z, rorg.z);
              const ssef tNearFar = max(tNearFarX, tNearFarY, tNearFarZ, nearFar);
              const sseb lrhit = (tNearFar ^ neg) >= shuffle8(tNearFar, swap);

              /*! if two children hit, push far node onto stack and continue with closer node */
              if (likely(lrhit[0] != 0 && lrhit[1] != 0)) {
                if (likely(tNearFar[0] < tNearFar[1])) {
                  stackPtr->ptr = node->child(1);
                  stackPtr->dist = tNearFar[1];
                  cur = node->child(0);
                  stackPtr++;
                }
                else {
                  stackPtr->ptr = node->child(0);
                  stackPtr->dist = tNearFar[0];
                  cur = node->child(1);
                  stackPtr++;
                }
              }

              /*! if one child hit, continue with that child */
              else {
                if (likely(lrhit[0] != 0)) cur = node->child(0);
                else if (likely(lrhit[1] != 0)) cur = node->child(1);
                else goto pop_node;
              }
            }

            /*! leaf node, intersect all triangles */
            {
              STAT3(shadow.trav_leaves, 1, 1, 1);
              size_t num; Triangle* tri = (Triangle*)cur.leaf(NULL, num);
              for (size_t i = 0; i < num; i++) {
                if (TriangleIntersector::occluded(ray, rayExtra, tri[i], bvh->vertices)) {
                  rayOccluded = 1;
                  ray.tfar = 0.0f;
                  ray.tnear = 1.0f;
                  goto doneWithRay;
                }
              }
            }

            /*! pop next node from stack */
          pop_node:
            if (unlikely(stackPtr == stack)) break;
            --stackPtr;
            cur = stackPtr->ptr;
            if (unlikely(stackPtr->dist > ray.tfar)) goto pop_node;
          }
        doneWithRay:;
        }
      }
    }
    else {
      size_t triangleCount;
      Triangle* triangles = (Triangle*)node.leaf(0, triangleCount);

      occludeByTriangles<Triangle, TriangleIntersector>((unsigned short*)buckets[bucket] + firstRay, (unsigned short*)buckets[bucket] + lastRay, rayData, rayExtras, occluded, triangles, (unsigned)triangleCount, bvh->vertices);
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
  
  void BVH2IntersectorStreamRegister () {
    TriangleMesh::intersectorsStream.add("bvh2","triangle4" ,"stream","moeller" ,true ,BVH2IntersectorStream<Triangle4Intersector1MoellerTrumboreStream>::create);
    TriangleMesh::intersectorsStream.add("bvh2","triangle8" ,"stream","moeller" ,true ,BVH2IntersectorStream<Triangle8Intersector1MoellerTrumboreStream>::create);
    TriangleMesh::intersectorsStream.setAccelDefaultTraverser("bvh2","stream");
  }
  
}
