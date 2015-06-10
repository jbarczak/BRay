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

#ifndef __EMBREE_BVH4_INTERSECTOR_STREAM__
#define __EMBREE_BVH4_INTERSECTOR_STREAM__

#include "bvh4.h"
#include "../include/intersector_stream.h"
#include "../common/alignedallocator.h"
#include <vector>

namespace embree {
  struct __align(16) StackEntry {
    BVH4::NodeRef node;
    unsigned short bucket;
    unsigned short firstRay;
    unsigned short lastRay;
  };
  
  template<typename TriangleIntersector>
  class BVH4IntersectorStream : public IntersectorStream {
    typedef typename TriangleIntersector::Triangle Triangle;
    typedef typename BVH4::NodeRef NodeRef;
    typedef typename BVH4::Node Node;

  public:
    static const unsigned bucketCount = 9;

  private:
    struct __align(32) ThreadState {
      __align(32) unsigned short* buckets[bucketCount];
      __align(32) unsigned bucketSize[bucketCount];
      unsigned maxRayCount;
      StackEntry* stack;
      
      __forceinline ThreadState() {
        maxRayCount = 0;
        
        for (unsigned i = 0; i < bucketCount; ++i) {
          buckets[i] = 0;
          bucketSize[i] = 0;
        }

        stack = 0;
      }
      
      __forceinline void reserve(unsigned maxRayCount) {
        if (this->maxRayCount >= maxRayCount)
          return;
        
        this->maxRayCount = maxRayCount;
        
        for (unsigned i = 0; i < bucketCount; ++i) {
          if (buckets[i])
            alignedFree(buckets[i]);
        }

        const unsigned maxDepth = 32;
        
        for (unsigned i = 0; i < bucketCount; ++i)
          buckets[i] = static_cast<unsigned short*>(alignedMalloc(maxRayCount*maxDepth*sizeof(short), 128));
        
        if (stack)
          alignedFree(stack);

        stack = static_cast<StackEntry*>(alignedMalloc(maxDepth*3*sizeof(StackEntry)));
      }
      
      __forceinline ~ThreadState() {
        for (unsigned i = 0; i < bucketCount; ++i) {
          if (buckets[i])
            alignedFree(buckets[i]);
        }
        if (stack)
          alignedFree(stack);
      }
    };
    
    const BVH4* bvh;
    std::vector<ThreadState, AlignedAllocator<ThreadState, 64> > threadStates;

  public:
    BVH4IntersectorStream(const BVH4* bvh);

    static IntersectorStream* create(const Accel* bvh) { 
      return new BVH4IntersectorStream((const BVH4*)bvh); 
    }

    static void intersect(BVH4IntersectorStream* This, StreamRay* rays, StreamRayExtra* rayExtras, StreamHit* hits, unsigned count, unsigned thread);
    static void occluded(BVH4IntersectorStream* This, StreamRay* rays, StreamRayExtra* rayExtras, unsigned count, unsigned char* occluded, unsigned thread);
  };
  
}

#endif
