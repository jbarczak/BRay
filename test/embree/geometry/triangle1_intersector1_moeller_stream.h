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

#ifndef __EMBREE_ACCEL_TRIANGLE1_INTERSECTOR1_MOELLER_STREAM_H__
#define __EMBREE_ACCEL_TRIANGLE1_INTERSECTOR1_MOELLER_STREAM_H__

#include "triangle1.h"
#include "../common/stream_ray.h"

namespace embree
{
  /*! Intersector for a single ray with individual precomputed
   *  triangles. This intersector implements a modified version of the
   *  Moeller Trumbore intersector from the paper "Fast, Minimum
   *  Storage Ray-Triangle Intersection". In contrast to the paper we
   *  precalculate some factors and factor the calculations
   *  differently to allow precalculating the cross product e1 x
   *  e2. The resulting algorithm is similar to the fastest one of the
   *  paper "Optimizing Ray-Triangle Intersection via Automated
   *  Search". */
  struct Triangle1Intersector1MoellerTrumboreStream
  {
    typedef Triangle1 Triangle;

    /*! Name of intersector */
    static const char* name() { return "moeller"; }

    /*! Intersect a ray with the triangle and updates the hit. */
    static __forceinline void intersect(StreamRay& ray, StreamRayExtra& rayExtra, StreamHit& hit, const Triangle1& tri, const Vec3fa* vertices)
    {
      STAT3(normal.trav_tris,1,1,1);

      /* calculate determinant */
      const Vector3f O = rayExtra.org();
      const Vector3f D = rayExtra.dir();
      const Vector3f C = tri.v0 - O;
      const Vector3f R = cross(D,C);
      const float det = dot(tri.Ng,D);
      const float absDet = abs(det);
      const float sgnDet = signmsk(det);

      /* perform edge tests */
      const float U = xorf(dot(R,tri.e2),sgnDet);
      if (unlikely(U < 0.0f)) return;
      const float V = xorf(dot(R,tri.e1),sgnDet);
      if (unlikely(V < 0.0f)) return;
      const float W = absDet-U-V;
      if (unlikely(W < 0.0f)) return;
      
      /* perform depth test */
      const float T = xorf(dot(tri.Ng,C),sgnDet);
      if (unlikely(absDet*float(ray.tfar) < T)) return;
      if (unlikely(T < absDet*float(ray.tnear))) return;
      if (unlikely(det == float(zero))) return;

      /* update hit information */
      const float rcpAbsDet = rcp(absDet);
      hit.u   = U * rcpAbsDet;
      hit.v   = V * rcpAbsDet;
      ray.tfar = T * rcpAbsDet;
      hit.Ng[0]  = tri.Ng.x;
      hit.Ng[1]  = tri.Ng.y;
      hit.Ng[2]  = tri.Ng.z;
      hit.id0 = tri.e1.a;
      hit.id1 = tri.e2.a;
    }

    /*! Test if the ray is occluded by one of the triangles. */
    static __forceinline bool occluded(const StreamRay& ray, const StreamRayExtra& rayExtra, const Triangle1& tri, const Vec3fa* vertices = NULL)
    {
      STAT3(shadow.trav_tris,1,1,1);

      /* calculate determinant */
      const Vector3f O = Vector3f(rayExtra.org());
      const Vector3f D = Vector3f(rayExtra.dir());
      const Vector3f C = tri.v0 - O;
      const Vector3f R = cross(D,C);
      const float det = dot(tri.Ng,D);
      const float absDet = abs(det);
      const float sgnDet = signmsk(det);

      /* perform edge tests */
      const float U = xorf(dot(R,tri.e2),sgnDet);
      if (unlikely(U < 0.0f)) return false;
      const float V = xorf(dot(R,tri.e1),sgnDet);
      if (unlikely(V < 0.0f)) return false;
      const float W = absDet-U-V;
      if (unlikely(W < 0.0f)) return false;
      
      /* perform depth test */
      const float T = xorf(dot(tri.Ng,C),sgnDet);
      if (unlikely(absDet*float(ray.tfar) < T)) return false;
      if (unlikely(T < absDet*float(ray.tnear))) return false;
      if (unlikely(det == float(zero))) return false;
      return true;
    }
  };
}

#endif


