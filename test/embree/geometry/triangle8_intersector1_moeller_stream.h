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

#ifndef __EMBREE_ACCEL_TRIANGLE8_INTERSECTOR1_MOELLER_STREAM_H__
#define __EMBREE_ACCEL_TRIANGLE8_INTERSECTOR1_MOELLER_STREAM_H__

#include "triangle8.h"
#include "../common/stream_ray.h"

namespace embree {

  struct Triangle8Intersector1MoellerTrumboreStream {
    typedef Triangle8 Triangle;

    static const char* name() { return "moeller"; }

    static __forceinline void intersect(StreamRay& ray, StreamRayExtra& rayExtra, StreamHit& hit, const Triangle8& tri, const Vec3fa* vertices) {
      STAT3(normal.trav_tris,1,1,1);

      /* calculate determinant */
      const avx3f O = avx3f(rayExtra.orgX, rayExtra.orgY, rayExtra.orgZ);
      const avx3f D = avx3f(rayExtra.dirX, rayExtra.dirY, rayExtra.dirZ);
      const avx3f C = avx3f(tri.v0) - O;
      const avx3f R = cross(D,C);
      const avxf det = dot(avx3f(tri.Ng),D);
      const avxf absDet = abs(det);
      const avxf sgnDet = signmsk(det);

      /* perform edge tests */
      const avxf U = dot(R,avx3f(tri.e2)) ^ sgnDet;
      const avxf V = dot(R,avx3f(tri.e1)) ^ sgnDet;
      avxb valid = (det != avxf(zero)) & (U >= 0.0f) & (V >= 0.0f) & (U+V<=absDet);
      if (unlikely(none(valid))) return;
      
      /* perform depth test */
      const avxf T = dot(avx3f(tri.Ng),C) ^ sgnDet;
      valid &= (T > absDet*avxf(ray.tnear)) & (T < absDet*avxf(ray.tfar));
      if (unlikely(none(valid))) return;

      /* update hit information */
      const avxf rcpAbsDet = rcp(absDet);
      const avxf u = U * rcpAbsDet;
      const avxf v = V * rcpAbsDet;
      const avxf t = T * rcpAbsDet;
      const size_t i = select_min(valid,t);
      hit.u   = u[i];
      hit.v   = v[i];
      ray.tfar = t[i];
      hit.Ng[0] = tri.Ng.x[i];
      hit.Ng[1] = tri.Ng.y[i];
      hit.Ng[2] = tri.Ng.z[i];
      hit.id0 = tri.id0[i];
      hit.id1 = tri.id1[i];
    }

    static __forceinline bool occluded(const StreamRay& ray, const StreamRayExtra& rayExtra, const Triangle8& tri, const Vec3fa* vertices = NULL) {
      STAT3(shadow.trav_tris,1,1,1);

      /* calculate determinant */
      const avx3f O = avx3f(rayExtra.orgX, rayExtra.orgY, rayExtra.orgZ);
      const avx3f D = avx3f(rayExtra.dirX, rayExtra.dirY, rayExtra.dirZ);
      const avx3f C = avx3f(tri.v0) - O;
      const avx3f R = cross(D,C);
      const avxf det = dot(avx3f(tri.Ng),D);
      const avxf absDet = abs(det);
      const avxf sgnDet = signmsk(det);

      /* perform edge tests */
      const avxf U = dot(R,avx3f(tri.e2)) ^ sgnDet;
      const avxf V = dot(R,avx3f(tri.e1)) ^ sgnDet;
      const avxf W = absDet-U-V;
      avxb valid = (U >= 0.0f) & (V >= 0.0f) & (W >= 0.0f);
      if (unlikely(none(valid))) return false;
      
      /* perform depth test */
      const avxf T = dot(avx3f(tri.Ng),C) ^ sgnDet;
      valid &= (det != avxf(zero)) & (T >= absDet*avxf(ray.tnear)) & (absDet*avxf(ray.tfar) >= T);
      if (unlikely(none(valid))) return false;

      return true;
    }
  };
  
}

#endif


