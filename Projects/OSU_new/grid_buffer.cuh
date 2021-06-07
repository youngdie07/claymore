#ifndef __GRID_BUFFER_CUH_
#define __GRID_BUFFER_CUH_
#include "mgmpm_kernels.cuh"
#include "settings.h"
#include <MnSystem/Cuda/HostUtils.hpp>
#include "utility_funcs.hpp"
#include <MnBase/Meta/Polymorphism.h>


namespace mn {

// Grid-block data structure (device). 
// Each channel is a value (mass, momentum/velocity) for a grid-node
using grid_block_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               BlockDomain, attrib_layout::soa, f32_, f32_, f32_, f32_>;

using grid_block_layer_0_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               BlockDomainLayer0, attrib_layout::soa, f32_, f32_, f32_, f32_>;
using grid_block_layer_1_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               BlockDomainLayer1, attrib_layout::soa, f32_, f32_, f32_, f32_>;
using grid_block_layer_2_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               BlockDomainLayer2, attrib_layout::soa, f32_, f32_, f32_, f32_>;


// Grid buffer data structure (device). Each channel (one) contains a grid-block
// Dense (non-dynamic) structure
using grid_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               GridDomain, attrib_layout::aos, grid_block_>;

// Grid buffer data structure (device). Each channel (one) contains a grid-block
// Dynamic structure
using grid_buffer_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               GridBufferDomain, attrib_layout::aos, grid_block_>;

// Two colocated grid buffers (device). Each channel (two) contains a grid-block
// Dynamic structure
using grid_buffer_double_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               GridBufferDomain, attrib_layout::aos, grid_block_, grid_block_>;

// Three colocated grid buffers (device). Each channel (three) contains a grid-block
// Dynamic structure
using grid_buffer_triple_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               GridBufferDomain, attrib_layout::aos, grid_block_, grid_block_, grid_block_>;

// Structure to hold downsampled grid-block values for ouput (device)
// Dynamic structure
using grid_array_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               GridArrayDomain, attrib_layout::aos, f32_, f32_, f32_, f32_, f32_, f32_, f32_>;
               // x, y, z, mass, Mx, My, Mz

// Structure to hold grid-cell target values for ouput (device)
// Dynamic structure
using grid_target_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               GridTargetDomain, attrib_layout::aos, f32_, f32_, f32_, 
               f32_, f32_, f32_, f32_, 
               f32_, f32_, f32_>;
               // x, y, z, mass, Mx, My, Mz, fx, fy, fz

struct GridBuffer : Instance<grid_buffer_> {
  using base_t = Instance<grid_buffer_>;

  template <typename Allocator>
  GridBuffer(Allocator allocator)
      : base_t{spawn<grid_buffer_, orphan_signature>(allocator)} {}
  template <typename Allocator>
  void checkCapacity(Allocator allocator, std::size_t capacity) {
    if (capacity > _capacity)
      resize(capacity, capacity);
  }
  template <typename CudaContext> void reset(int blockCnt, CudaContext &cuDev) {
    using namespace placeholder;
#if 0
    checkCudaErrors(cudaMemsetAsync((void *)&this->val_1d(_0, 0), 0,
                                    grid_block_::size * blockCnt, cuDev.stream_compute()));
#else
    cuDev.compute_launch({blockCnt, config::g_blockvolume}, clear_grid, *this);
#endif
  }
};

/// 1D GridArray structure for device instantiation (JB)
struct GridArray : Instance<grid_array_> {
  using base_t = Instance<grid_array_>;
  GridArray &operator=(base_t &&instance) {
    static_cast<base_t &>(*this) = instance;
    return *this;
  }
  GridArray(base_t &&instance) { static_cast<base_t &>(*this) = instance; }
};

/// 1D GridTarget structure for device instantiation (JB)
struct GridTarget : Instance<grid_target_> {
  using base_t = Instance<grid_target_>;
  GridTarget &operator=(base_t &&instance) {
    static_cast<base_t &>(*this) = instance;
    return *this;
  }
  GridTarget(base_t &&instance) { static_cast<base_t &>(*this) = instance; }
};

} // namespace mn

#endif