#ifndef __GRID_BUFFER_CUH_
#define __GRID_BUFFER_CUH_
#include "mgmpm_kernels.cuh"
#include "settings.h"
#include <MnSystem/Cuda/HostUtils.hpp>
#include <MnBase/Meta/Polymorphism.h>

namespace mn {

using BlockDomain = compact_domain<char, config::g_blocksize,
                                   config::g_blocksize, config::g_blocksize>;
using GridDomain = compact_domain<int, config::g_grid_size_x, config::g_grid_size_y,
                                  config::g_grid_size_z>;
using GridBufferDomain = compact_domain<int, config::g_max_active_block>;
using GridArrayDomain = compact_domain<int, config::g_max_active_block>;
using GridTargetDomain = compact_domain<int, config::g_target_cells>;

using grid_block_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               BlockDomain, attrib_layout::soa, f32_, f32_, f32_, f32_,
               f32_, f32_, f32_, f32_, f32_>; //< mass, vel + dt*fint, 
                                              //< vel, 
                                              //< Vol, J
using grid_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               GridDomain, attrib_layout::aos, grid_block_>;
using grid_buffer_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               GridBufferDomain, attrib_layout::aos, grid_block_>;

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
  // template <typename Allocator>
  // GridTarget(Allocator allocator) : base_t{allocator} {}
};

} // namespace mn

#endif