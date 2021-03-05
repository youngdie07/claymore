#ifndef __SETTINGS_H_
#define __SETTINGS_H_
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
enum class material_e { JFluid = 0, FixedCorotated, Sand, NACC, Rigid, Piston, IFluid, Total };

/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html, F.3.16.5
/// benchmark setup
namespace config {
constexpr int g_device_cnt = 1; //< Number of GPUs available
constexpr material_e get_material_type(int did) noexcept {
  material_e type{material_e::FixedCorotated};
  return type;
}
constexpr int g_total_frame_cnt = 60;

#define GBPCB 16
constexpr int g_num_grid_blocks_per_cuda_block = GBPCB;
constexpr int g_num_warps_per_grid_block = 1;
constexpr int g_num_warps_per_cuda_block = GBPCB;
constexpr int g_particle_batch_capacity = 128;

#define MODEL_PPC 32.f
constexpr float g_model_ppc = MODEL_PPC; //< Model particles-per-cell

// background_grid
#define BLOCK_BITS 2
#define DOMAIN_BITS 9
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
//constexpr int g_num_nodes = g_domain_size * g_domain_size * g_domain_size;

// particle
#define MAX_PPC 128
constexpr int g_max_ppc = MAX_PPC; //< Max particles per cell
constexpr int g_bin_capacity = 32; //< Max particles per bin
constexpr int g_particle_num_per_block = (MAX_PPC * (1 << (BLOCK_BITS * 3))); //< Max particles per block

// Material parameters
#define DENSITY 1e3
#define YOUNGS_MODULUS 5e7
#define POISSON_RATIO 0.4f
constexpr float g_cfl = 0.5f; //< CFL condition

// Unscaled ambient settings
constexpr float g_gravity = -9.81f; //< Gravity (m/s2)
constexpr float g_atm = 101.325e3;  //< Atm. Pressure (Pa)

// Domain size (meters)
#define DOMAIN_VOLUME 0.4f
constexpr float g_length   = 128.0f; //< Domain length(m)
constexpr float g_length_x = 128.0f; //< Domain x length (m)
constexpr float g_length_y = 8.0f;  //< Domain y length (m)
constexpr float g_length_z = 8.0f;  //< Domain z length (m)

// Domain ratio ( )
constexpr float g_grid_ratio_x = 1.f;     //< Domain x ratio
constexpr float g_grid_ratio_y = 1.f/16.f;     //< Domain y ratio
constexpr float g_grid_ratio_z = 1.f/16.f; //< Domain z ratio

// Domain grid blocks (#)
constexpr int g_grid_size_x = g_grid_size * g_grid_ratio_x; //< Domain x grid-blocks
constexpr int g_grid_size_y = g_grid_size * g_grid_ratio_y; //< Domain y grid-blocks
constexpr int g_grid_size_z = g_grid_size * g_grid_ratio_z; //< Domain z grid-blocks

/// only used on host
constexpr int g_max_particle_num = 500000;
constexpr int g_max_active_block = 3500; /// 62500 bytes for active mask
constexpr std::array<size_t, 5> g_max_active_block_arr = {3500, 3500, 3500, 3500, 3500};
constexpr std::size_t
calc_particle_bin_count(std::size_t numActiveBlocks) noexcept {
  return numActiveBlocks * (g_max_ppc * g_blockvolume / g_bin_capacity);
}
constexpr std::size_t g_max_particle_bin = g_max_particle_num / g_bin_capacity; //< Max particle bins (#)
constexpr std::size_t g_max_halo_block = 4000; //< Max halo blocks (#)

} // namespace config

using BlockDomain = compact_domain<char, config::g_blocksize,
                                   config::g_blocksize, config::g_blocksize>;
using GridDomain = compact_domain<int, config::g_grid_size_x, config::g_grid_size_y, config::g_grid_size_z>;
using GridBufferDomain = compact_domain<int, config::g_max_active_block>;

// Down-sampled output grid-block domain, used in grid_buffer.cuh (JB)
using GridArrayDomain = compact_domain<int, config::g_max_active_block>;

} // namespace mn

#endif
