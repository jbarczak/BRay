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

#ifndef __EMBREE_INTERSECTOR_STREAM_H__
#define __EMBREE_INTERSECTOR_STREAM_H__

#include <stddef.h>
#include <immintrin.h>

namespace embree {
  
  class Accel;
  struct Ray;
  struct StreamRay;
  struct StreamRayExtra;
  struct StreamHit;

  class IntersectorStream {
  public:
    static const char* const name;

    typedef void (*intersectFunc)(const IntersectorStream* This, StreamRay* rays, StreamRayExtra* rayExtras, StreamHit* hits, unsigned count, unsigned thread);
    typedef void (*occludedFunc)(const IntersectorStream* This, StreamRay* rays, StreamRayExtra* rayExtras, unsigned count, unsigned char* occluded, unsigned thread);

  public:
    __forceinline IntersectorStream() : intersectPtr(0), occludedPtr(0) {}
    
    __forceinline IntersectorStream(intersectFunc intersect, occludedFunc occluded) : intersectPtr(intersect), occludedPtr(occluded) {}

    __forceinline void intersect(StreamRay* rays, StreamRayExtra* rayExtras, StreamHit* hits, unsigned count, unsigned thread) const {
      return intersectPtr(this, rays, rayExtras, hits, count, thread);
    }

    __forceinline void occluded(StreamRay* rays, StreamRayExtra* rayExtras, unsigned count, unsigned char* occluded, unsigned thread) const {
      occludedPtr(this, rays, rayExtras, count, occluded, thread);
    }

  public:
    intersectFunc intersectPtr;
    occludedFunc occludedPtr;
  };

  typedef IntersectorStream RTCIntersectorStream;
  
}

#endif
