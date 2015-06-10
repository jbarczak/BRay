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

#ifndef __EMBREE_STREAM_RAY_H__
#define __EMBREE_STREAM_RAY_H__

#include "../common/default.h"

namespace embree {
  
  struct __align(32) StreamRay {
    float invDirX;
    float invDirY;
    float invDirZ;
    float tnear;
    float orgInvDirX;
    float orgInvDirY;
    float orgInvDirZ;
    float tfar;
    
    __forceinline void set(const Vector3f& org, const Vector3f& dir, float tnear, float tfar) {
      __m128 o = org;
      __m128 d = dir;
      __m128 rd = rcp_safe(d);

      _mm_store_ps(&invDirX, _mm_insert_ps(rd, _mm_set_ss(tnear), 3<<4));
      _mm_store_ps(&orgInvDirX, _mm_insert_ps(_mm_mul_ps(o, rd), _mm_set_ss(tfar), 3<<4));
    }
  };

  struct __align(32) StreamRayExtra {
    float orgX;
    float orgY;
    float orgZ;
    float time;
    float dirX;
    float dirY;
    float dirZ;
    float dummy;

    __forceinline void set(const Vector3f& org, const Vector3f& dir, float time) {
      __m128 o = org;
      __m128 d = dir;

      _mm_store_ps(&orgX, _mm_insert_ps(o, _mm_set_ss(time), 3<<4));
      _mm_store_ps(&dirX, d);
    }

    __forceinline Vector3f org() const {
      return Vector3f(_mm_load_ps(&orgX));
    }

    __forceinline Vector3f dir() const {
      return Vector3f(_mm_load_ps(&dirX));
    }
  };
  
  struct __align(32) StreamHit {
    float Ng[3];
    float dummy;
    float u;
    float v;
    int32 id0;
    int32 id1;

    __forceinline operator bool() const { return id0 != -1; }
  };
  
}

#endif
