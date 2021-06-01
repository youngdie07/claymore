#ifndef __SETTINGS_H_
#define __SETTINGS_H_
#include "partition_domain.h"
#include <MnBase/Math/Vec.h>
#include <MnBase/Object/Structural.h>
#include <array>

namespace mn {

using ivec3 = vec<int, 3>;
using vec3 = vec<float, 3>;
using vec9 = vec<float, 9>;
using vec3x3 = vec<float, 3, 3>;
using vec3x4 = vec<float, 3, 4>;
using vec3x3x3 = vec<float, 3, 3, 3>;

/// sand = Drucker Prager Plasticity, StvkHencky Elasticity
enum class material_e { JFluid = 0, FixedCorotated, Sand, NACC, Total };

/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html, F.3.16.5
/// benchmark setup
namespace config {
constexpr int g_device_cnt = 1;
constexpr material_e get_material_type(int did) noexcept {
  material_e type{material_e::JFluid};
  return type;
}
constexpr int g_total_frame_cnt = 60;

#define GBPCB 16
constexpr int g_num_grid_blocks_per_cuda_block = GBPCB;
constexpr int g_num_warps_per_grid_block = 1;
constexpr int g_num_warps_per_cuda_block = GBPCB;
constexpr int g_particle_batch_capacity = 128;

#define MODEL_PPC 4.f
constexpr float g_model_ppc = MODEL_PPC;
constexpr float cfl = 0.5f;

// background_grid
#define BLOCK_BITS 2
#define DOMAIN_BITS 10
#define DXINV (1.f * (1 << DOMAIN_BITS))
constexpr int g_domain_bits = DOMAIN_BITS;
constexpr int g_domain_size = (1 << DOMAIN_BITS);
constexpr float g_bc = 2;
constexpr float g_dx = 1.f / DXINV;
constexpr float g_dx_inv = DXINV;
constexpr float g_D_inv = 4.f * DXINV * DXINV;
constexpr int g_blockbits = BLOCK_BITS;
constexpr int g_blocksize = (1 << BLOCK_BITS);
constexpr int g_blockmask = ((1 << BLOCK_BITS) - 1);
constexpr int g_blockvolume = (1 << (BLOCK_BITS * 3));
constexpr int g_grid_bits = (DOMAIN_BITS - BLOCK_BITS);
constexpr int g_grid_size = (1 << (DOMAIN_BITS - BLOCK_BITS));
// constexpr int g_grid_size_x = g_grid_size;
// constexpr int g_grid_size_y = g_grid_size / 16;
// constexpr int g_grid_size_z = g_grid_size / 16;

// Domain size
#define DOMAIN_VOLUME 8192.f
constexpr float g_length   = 128.0f; //< Domain full length (m)
constexpr float g_length_x = 128.0f; //< Domain x length (m)
constexpr float g_length_y = 8.f;   //< Domain y length (m)
constexpr float g_length_z = 8.f;   //< Domain z length (m)
constexpr float g_grid_ratio_x = g_length_x / g_length; //< Domain x ratio
constexpr float g_grid_ratio_y = g_length_y / g_length; //< Domain y ratio
constexpr float g_grid_ratio_z = g_length_z / g_length; //< Domain z ratio
constexpr int g_grid_size_x = g_grid_size * g_grid_ratio_x; //< Domain x grid-blocks
constexpr int g_grid_size_y = g_grid_size * g_grid_ratio_y; //< Domain y grid-blocks
constexpr int g_grid_size_z = g_grid_size * g_grid_ratio_z; //< Domain z grid-blocks


// partition domains
constexpr box_domain<int, 3> get_domain(int did) noexcept {
  constexpr int len = g_grid_size / 2;
  box_domain<int, 3> domain{};
  for (int d = 0; d < 3; ++d) {
    domain._min[d] = 0;
    if (d == 0) domain._max[d] = g_grid_size_x - 1;
    else if (d == 1) domain._max[d] = g_grid_size_y - 1;
    else if (d == 2) domain._max[d] = g_grid_size_z - 1;
    //domain._max[d] = g_grid_size - 1;
  }
  if constexpr (g_device_cnt == 1) {
    /// default
  } else if (g_device_cnt == 2) {
    if (did == 0)
      //domain._max[0] = len;
      domain._max[0] = g_grid_size_x / 2;
    else if (did == 1)
      //domain._min[0] = len + 1;
      domain._min[0] = g_grid_size_x / 2 + 1;
  } else if (g_device_cnt == 4) {
    // domain._min[0] = (did & 2) ? len + 1 : 0;
    // domain._min[2] = (did & 1) ? len + 1 : 0;
    // domain._max[0] = (did & 2) ? g_grid_size - 1 : len;
    // domain._max[2] = (did & 1) ? g_grid_size - 1 : len;
    // if (did == 0) {
    //   domain._min[0] = 0;
    //   domain._max[0] = g_grid_size_x / 4;
    // }
    // else if (did == 1) {
    //   domain._min[0] = g_grid_size_x / 4 + 1;
    //   domain._max[0] = 2 * g_grid_size_x / 4;
    // }
    // else if (did == 2) {
    //   domain._min[0] = 2 * g_grid_size_x / 4 + 1;
    //   domain._max[0] = 3 * g_grid_size_x / 4;
    // }
    // else if (did == 3) {
    //   domain._min[0] = 3 * g_grid_size_x / 4 + 1;
    //   domain._max[0] = g_grid_size_x - 1;
    // }    
    if (did == 0) {
      domain._min[1] = 0;
      domain._max[1] = g_grid_size_z / 4;
    }
    else if (did == 1) {
      domain._min[1] = g_grid_size_z / 4 + 1;
      domain._max[1] = 2 * g_grid_size_z / 4;
    }
    else if (did == 2) {
      domain._min[1] = 2 * g_grid_size_z / 4 + 1;
      domain._max[1] = 3 * g_grid_size_z / 4;
    }
    else if (did == 3) {
      domain._min[1] = 3 * g_grid_size_z / 4 + 1;
      domain._max[1] = g_grid_size_z - 1;
    }    
    // domain._min[0] = (did & 2) ? g_grid_size_x / 2 + 1 : 0;
    // domain._min[2] = (did & 1) ? g_grid_size_z / 2 + 1 : 0;
    // domain._max[0] = (did & 2) ? g_grid_size_x - 1 : g_grid_size_x;
    // domain._max[2] = (did & 1) ? g_grid_size_z - 1 : g_grid_size_z;
  } else
    domain._max[0] = domain._max[1] = domain._max[2] = -3;
  return domain;
}

// particle
#define MAX_PPC 128
constexpr int g_max_ppc = MAX_PPC;
constexpr int g_bin_capacity = 32;
constexpr int g_particle_num_per_block = (MAX_PPC * (1 << (BLOCK_BITS * 3)));

// material parameters
#define DENSITY 1e3
#define YOUNGS_MODULUS 5e3
#define POISSON_RATIO 0.4f

//
constexpr float g_gravity = -9.81f;

/// only used on host
constexpr int g_max_particle_num = 2000000;
constexpr int g_max_active_block = 15000; /// 62500 bytes for active mask
constexpr std::size_t
calc_particle_bin_count(std::size_t numActiveBlocks) noexcept {
  return numActiveBlocks * (g_max_ppc * g_blockvolume / g_bin_capacity);
}
constexpr std::size_t g_max_particle_bin = g_max_particle_num / g_bin_capacity;
constexpr std::size_t g_max_halo_block = 4000;

} // namespace config

} // namespace mn

#endif