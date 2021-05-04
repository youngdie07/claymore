#ifndef __SETTINGS_H_
#define __SETTINGS_H_
#include <MnBase/Math/Vec.h>
#include <MnBase/Object/Structural.h>
#include <array>

namespace mn {

// Vectors/tensors for convenience
using ivec3 = vec<int, 3>;
using vec3 = vec<float, 3>;
using vec9 = vec<float, 9>;
using vec3x3 = vec<float, 3, 3>;
using vec3x4 = vec<float, 3, 4>;
using vec3x3x3 = vec<float, 3, 3, 3>;

// Material models available
enum class material_e { JFluid = 0, FixedCorotated, Sand, NACC, Rigid, Piston, IFluid, Total };

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html, F.3.16.5
// benchmark setup
namespace config {
constexpr int g_device_cnt = 1; //< Number of GPUs available
constexpr material_e get_material_type(int did) noexcept {
  material_e type{material_e::FixedCorotated};
  return type;
}
constexpr int g_total_frame_cnt = 60;

// CUDA settings
#define GBPCB 16
constexpr int g_num_grid_blocks_per_cuda_block = GBPCB;
constexpr int g_num_warps_per_grid_block = 1;
constexpr int g_num_warps_per_cuda_block = GBPCB;
constexpr int g_particle_batch_capacity = 128;

#define MODEL_PPC 64.f
constexpr float g_model_ppc = MODEL_PPC; //< Model particles-per-cell

// Background_grid
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


// Particles
#define MAX_PPC 128
constexpr int g_max_ppc = MAX_PPC; //< Max particles per cell
constexpr int g_bin_capacity = 32; //< Max particles per bin, keep a multiple of 32
constexpr int g_particle_num_per_block = (MAX_PPC * (1 << (BLOCK_BITS * 3))); //< Max particles per block

// Material parameters
#define DENSITY 1e3
#define YOUNGS_MODULUS 5e7
#define POISSON_RATIO 0.4f
constexpr float g_cfl = 0.5f; //< CFL condition

// Ambient external field settings
constexpr float g_gravity = 0.f; //< Gravity (m/s2)
constexpr float g_atm = 101.325e3;  //< Atm. Pressure (Pa)

// Grid layers
//constexpr 

// Domain size
#define DOMAIN_VOLUME 0.4f
constexpr float g_length   = 1.0f; //< Domain full length (m)
constexpr float g_length_x = 1.0f; //< Domain x length (m)
constexpr float g_length_y = 1.f;   //< Domain y length (m)
constexpr float g_length_z = 1.f;   //< Domain z length (m)
constexpr float g_grid_ratio_x = g_length_x / g_length; //< Domain x ratio
constexpr float g_grid_ratio_y = g_length_y / g_length; //< Domain y ratio
constexpr float g_grid_ratio_z = g_length_z / g_length; //< Domain z ratio
constexpr int g_grid_size_x = g_grid_size * g_grid_ratio_x; //< Domain x grid-blocks
constexpr int g_grid_size_y = g_grid_size * g_grid_ratio_y; //< Domain y grid-blocks
constexpr int g_grid_size_z = g_grid_size * g_grid_ratio_z; //< Domain z grid-blocks

// Only used on host, preallocates memory for the data-structures
// Common source of crashes, memory is constrained by specific GPU
// Playing with these values can improve program memory 
constexpr int g_max_particle_num = 1000000;
constexpr int g_max_active_block = 15000; /// 62500 bytes for active mask
constexpr std::array<size_t, 5> g_max_active_block_arr = {15000, 10000, 25000, 3000, 2000};
constexpr std::size_t
calc_particle_bin_count(std::size_t numActiveBlocks) noexcept {
  return numActiveBlocks * (g_max_ppc * g_blockvolume / g_bin_capacity);
}
constexpr std::size_t g_max_particle_bin = g_max_particle_num / g_bin_capacity; //< Max particle bins (#)
constexpr std::size_t g_max_halo_block = 4000; //< Max halo blocks (#)
constexpr int g_target_cells = 50000; //< Max nodes in grid-cell target
} // namespace config

// Domain descriptions for different grid data-structures
// Used primarily in grid_buffer.cuh
using BlockDomain = compact_domain<char, config::g_blocksize,
                                   config::g_blocksize, config::g_blocksize>;
using BlockDomainLayer0 = compact_domain<char, config::g_blocksize << 0,
                                   config::g_blocksize << 0, config::g_blocksize << 0>;
using BlockDomainLayer1 = compact_domain<char, config::g_blocksize << 1,
                                   config::g_blocksize << 1, config::g_blocksize << 1>;
using BlockDomainLayer2 = compact_domain<char, config::g_blocksize << 2,
                                   config::g_blocksize << 2, config::g_blocksize << 2>;

using GridDomain = compact_domain<int, config::g_grid_size_x, config::g_grid_size_y, config::g_grid_size_z>;
using GridBufferDomain = compact_domain<int, config::g_max_active_block>;

// Down-sampled output grid-block domain, used in grid_buffer.cuh (JB)
using GridArrayDomain = compact_domain<int, config::g_max_active_block>;
using GridTargetDomain = compact_domain<int, config::g_target_cells>;
} // namespace mn

#endif
