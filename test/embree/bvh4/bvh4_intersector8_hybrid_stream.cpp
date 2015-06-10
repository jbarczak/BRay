// ======================================================================== //
// Copyright 2009-2013 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#if __AVX__

#include "bvh4_intersector8_hybrid_stream.h"
#include "bvh4_intersector8_hybrid.h"
#include "bvh4_intersector1.h"
#include "../geometry/triangles.h"

#define _MM256_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7) \
do {\
  __m256 t0, t1, t2, t3, t4, t5, t6, t7;\
  __m256 tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;\
  \
  t0 = _mm256_unpacklo_ps(row0, row1);\
  t1 = _mm256_unpackhi_ps(row0, row1);\
  t2 = _mm256_unpacklo_ps(row2, row3);\
  t3 = _mm256_unpackhi_ps(row2, row3);\
  t4 = _mm256_unpacklo_ps(row4, row5);\
  t5 = _mm256_unpackhi_ps(row4, row5);\
  t6 = _mm256_unpacklo_ps(row6, row7);\
  t7 = _mm256_unpackhi_ps(row6, row7);\
  \
  tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));\
  tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));\
  tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));\
  tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));\
  tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));\
  tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));\
  tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));\
  tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));\
  \
  row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);\
  row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);\
  row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);\
  row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);\
  row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);\
  row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);\
  row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);\
  row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);\
} while (0)

namespace embree
{
  template<typename TriangleIntersector8, typename TriangleIntersector1>
  void BVH4Intersector8HybridStream<TriangleIntersector8, TriangleIntersector1>::intersect(BVH4Intersector8HybridStream* This, StreamRay* rays, StreamRayExtra* rayExtras, StreamHit* hits, unsigned count, unsigned thread)
  {
    for (unsigned i = 0; i < count; i += 8) {
      __m256 r0 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+0]));
      __m256 r1 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+1]));
      __m256 r2 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+2]));
      __m256 r3 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+3]));
      __m256 r4 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+4]));
      __m256 r5 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+5]));
      __m256 r6 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+6]));
      __m256 r7 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+7]));

      _MM256_TRANSPOSE8_PS(r0, r1, r2, r3, r4, r5, r6, r7);

      __m256 re0 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+0]));
      __m256 re1 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+1]));
      __m256 re2 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+2]));
      __m256 re3 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+3]));
      __m256 re4 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+4]));
      __m256 re5 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+5]));
      __m256 re6 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+6]));
      __m256 re7 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+7]));

      _MM256_TRANSPOSE8_PS(re0, re1, re2, re3, re4, re5, re6, re7);

      Ray8 ray8;

      ray8.org.x = re0;
      ray8.org.y = re1;
      ray8.org.z = re2;

      ray8.time = re3;

      ray8.dir.x = re4;
      ray8.dir.y = re5;
      ray8.dir.z = re6;

      ray8.tnear = r3;
      ray8.tfar = r7;

      ray8.id0 = _mm256_set1_epi32(-1);
      ray8.id1 = _mm256_set1_epi32(-1);
      ray8.mask = _mm256_set1_epi32(0xffffffff);

      ray8.Ng.x = _mm256_setzero_ps();
      ray8.Ng.y = _mm256_setzero_ps();
      ray8.Ng.z = _mm256_setzero_ps();

      ray8.u = _mm256_setzero_ps();
      ray8.v = _mm256_setzero_ps();

      __m256i valid = _mm256_cmpgt_epi32(_mm256_set1_epi32(count), _mm256_add_epi32(_mm256_set1_epi32(i), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)));

      This->packetIntersector->intersect(_mm256_castsi256_ps(valid), ray8);

      r3 = ray8.tnear;
      r7 = ray8.tfar;

      _MM256_TRANSPOSE8_PS(r0, r1, r2, r3, r4, r5, r6, r7);

      __m256 h0 = ray8.Ng.x;
      __m256 h1 = ray8.Ng.y;
      __m256 h2 = ray8.Ng.z;
      __m256 h3 = _mm256_setzero_ps();
      __m256 h4 = ray8.u;
      __m256 h5 = ray8.v;
      __m256 h6 = _mm256_castsi256_ps(ray8.id0);
      __m256 h7 = _mm256_castsi256_ps(ray8.id1);

      _MM256_TRANSPOSE8_PS(h0, h1, h2, h3, h4, h5, h6, h7);

      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+0]), r0);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+1]), r1);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+2]), r2);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+3]), r3);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+4]), r4);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+5]), r5);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+6]), r6);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+7]), r7);

      _mm256_store_ps(reinterpret_cast<float*>(&hits[i+0]), h0);
      _mm256_store_ps(reinterpret_cast<float*>(&hits[i+1]), h1);
      _mm256_store_ps(reinterpret_cast<float*>(&hits[i+2]), h2);
      _mm256_store_ps(reinterpret_cast<float*>(&hits[i+3]), h3);
      _mm256_store_ps(reinterpret_cast<float*>(&hits[i+4]), h4);
      _mm256_store_ps(reinterpret_cast<float*>(&hits[i+5]), h5);
      _mm256_store_ps(reinterpret_cast<float*>(&hits[i+6]), h6);
      _mm256_store_ps(reinterpret_cast<float*>(&hits[i+7]), h7);
    }
  }

  template<typename TriangleIntersector8, typename TriangleIntersector1>
  void BVH4Intersector8HybridStream<TriangleIntersector8, TriangleIntersector1>::occluded(BVH4Intersector8HybridStream* This, StreamRay* rays, StreamRayExtra* rayExtras, unsigned count, unsigned char* occluded, unsigned thread)
  {
    for (unsigned i = 0; i < count; i += 8) {
      __m256 r0 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+0]));
      __m256 r1 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+1]));
      __m256 r2 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+2]));
      __m256 r3 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+3]));
      __m256 r4 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+4]));
      __m256 r5 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+5]));
      __m256 r6 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+6]));
      __m256 r7 = _mm256_load_ps(reinterpret_cast<const float*>(&rays[i+7]));

      _MM256_TRANSPOSE8_PS(r0, r1, r2, r3, r4, r5, r6, r7);

      __m256 re0 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+0]));
      __m256 re1 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+1]));
      __m256 re2 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+2]));
      __m256 re3 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+3]));
      __m256 re4 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+4]));
      __m256 re5 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+5]));
      __m256 re6 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+6]));
      __m256 re7 = _mm256_load_ps(reinterpret_cast<const float*>(&rayExtras[i+7]));

      _MM256_TRANSPOSE8_PS(re0, re1, re2, re3, re4, re5, re6, re7);

      Ray8 ray8;

      ray8.org.x = re0;
      ray8.org.y = re1;
      ray8.org.z = re2;

      ray8.time = re3;

      ray8.dir.x = re4;
      ray8.dir.y = re5;
      ray8.dir.z = re6;

      ray8.tnear = r3;
      ray8.tfar = r7;

      ray8.id0 = _mm256_set1_epi32(-1);
      ray8.id1 = _mm256_set1_epi32(-1);
      ray8.mask = _mm256_set1_epi32(0xffffffff);

      ray8.Ng.x = _mm256_setzero_ps();
      ray8.Ng.y = _mm256_setzero_ps();
      ray8.Ng.z = _mm256_setzero_ps();

      ray8.u = _mm256_setzero_ps();
      ray8.v = _mm256_setzero_ps();

      __m256i valid = _mm256_cmpgt_epi32(_mm256_set1_epi32(count), _mm256_add_epi32(_mm256_set1_epi32(i), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)));

      __m256i hit = _mm256_castps_si256(This->packetIntersector->occluded(_mm256_castsi256_ps(valid), ray8));

      hit = _mm256_packs_epi32(hit, hit);
      hit = _mm256_permute4x64_epi64(hit, (0) | ((2) << 2) | ((0) << 4) | ((2) << 6));

      __m128i hit128 = _mm256_castsi256_si128(hit);
      hit128 = _mm_packs_epi16(hit128, hit128);

      unsigned long long hit64 = _mm_extract_epi64(hit128, 0);
      *reinterpret_cast<unsigned long long*>(occluded+i) = hit64;

      r3 = ray8.tnear;
      r7 = ray8.tfar;

      _MM256_TRANSPOSE8_PS(r0, r1, r2, r3, r4, r5, r6, r7);

      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+0]), r0);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+1]), r1);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+2]), r2);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+3]), r3);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+4]), r4);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+5]), r5);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+6]), r6);
      _mm256_store_ps(reinterpret_cast<float*>(&rays[i+7]), r7);
    }
  }

  void BVH4Intersector8HybridStreamRegister () 
  {
    TriangleMesh::intersectorsStream.add("bvh4","triangle1i","hybrid","moeller" ,false, BVH4Intersector8HybridStream<Triangle1iIntersector8<Intersector8MoellerTrumbore>, Triangle1iIntersector1<Intersector1MoellerTrumbore> >::create);
    TriangleMesh::intersectorsStream.add("bvh4","triangle4" ,"hybrid","moeller" ,true ,BVH4Intersector8HybridStream<Triangle4Intersector8MoellerTrumbore, Triangle4Intersector1MoellerTrumbore>::create);
    TriangleMesh::intersectorsStream.setAccelDefaultTraverser("bvh4.triangle4","hybrid");
    TriangleMesh::intersectorsStream.setAccelDefaultTraverser("bvh4.triangle8","hybrid");
  }
}
#endif

