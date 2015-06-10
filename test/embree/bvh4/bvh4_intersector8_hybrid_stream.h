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

#ifndef __EMBREE_BVH4_INTERSECTOR8_HYBRID_STREAM_H__
#define __EMBREE_BVH4_INTERSECTOR8_HYBRID_STREAM_H__

#include "bvh4.h"
#include "../include/intersector_stream.h"
#include "bvh4_intersector8_hybrid.h"

namespace embree
{
  /*! BVH4 Traverser. Packet traversal implementation for a Quad BVH. */
  template<typename TriangleIntersector8, typename TriangleIntersector1>
    class BVH4Intersector8HybridStream : public IntersectorStream
  {
    /* shortcuts for frequently used types */
    typedef typename TriangleIntersector8::Triangle Triangle;
    typedef typename BVH4::NodeRef NodeRef;
    typedef typename BVH4::Node Node;
    Intersector8* packetIntersector;

  public:
    BVH4Intersector8HybridStream (const BVH4* bvh) 
      : IntersectorStream((intersectFunc)intersect,(occludedFunc)occluded), bvh(bvh), packetIntersector(BVH4Intersector8Hybrid<TriangleIntersector8, TriangleIntersector1>::create(bvh)) {}

    static IntersectorStream* create(const Accel* bvh) {
      return new BVH4Intersector8HybridStream((const BVH4*)bvh); 
    }

    static void intersect(BVH4Intersector8HybridStream* This, StreamRay* rays, StreamRayExtra* rayExtras, StreamHit* hits, unsigned count, unsigned thread);
    static void occluded(BVH4Intersector8HybridStream* This, StreamRay* rays, StreamRayExtra* rayExtras, unsigned count, unsigned char* occluded, unsigned thread);

    ~BVH4Intersector8HybridStream() {
      delete packetIntersector;
    }

  private:
    const BVH4* bvh;
  };
}

#endif
