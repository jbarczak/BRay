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

#ifndef __EMBREE_ALIGNED_ALLOCATOR_H__
#define __EMBREE_ALIGNED_ALLOCATOR_H__

#include <new>
#include <cstdio>
#include <xmmintrin.h>

namespace embree {
  
  template <typename T, std::size_t Alignment>
  class AlignedAllocator {
  public:
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef std::size_t size_type;
    typedef ptrdiff_t difference_type;
    
    template <typename U>
    struct rebind {
      typedef AlignedAllocator<U, Alignment> other;
    };
    
    __forceinline AlignedAllocator() {}
    
    __forceinline AlignedAllocator(const AlignedAllocator&) {}
    
    template <typename U>
    __forceinline AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}
    
    __forceinline std::size_t max_size() const {
      return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
    }
    
    __forceinline T* address(T& r) const {
      return &r;
    }
    
    __forceinline const T* address(const T& s) const {
      return &s;
    }
    
    __forceinline void construct(T* const p, const T& t) const {
      void* const pv = static_cast<void*>(p);
      new (pv) T(t);
    }
    
    __forceinline void destroy(T* const p) const {
      p->~T();
    }
    
    __forceinline bool operator == (const AlignedAllocator& other) const {
      return true;
    }
    
    __forceinline bool operator != (const AlignedAllocator& other) const {
      return !(*this == other);
    }
    
    __forceinline T* allocate(const std::size_t n) const {
      if (n == 0) {
        return NULL;
      }
      
      if (n > max_size())
        throw std::bad_alloc();
      
      void* const pv = _mm_malloc(n * sizeof(T), Alignment);
      
      if (pv == NULL)
        throw std::bad_alloc();
      
      return static_cast<T*>(pv);
    }
    
    __forceinline void deallocate(T* const p, const std::size_t n) const {
      _mm_free(p);
    }
    
    template <typename U>
    __forceinline T* allocate(const std::size_t n, const U*) const {
      return allocate(n);
    }
  };

}

#endif
