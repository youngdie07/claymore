#ifndef __SETTINGS_H_
#define __SETTINGS_H_
#include "partition_domain.h"
#include <MnBase/Math/Vec.h>
#include <MnBase/Object/Structural.h>
#include <array>

namespace mn {

/// Vector type shorthand
using ivec3 = vec<int, 3>;
using vec3 = vec<float, 3>;
using vec9 = vec<float, 9>;
using vec3x3 = vec<float, 3, 3>;
using vec3x4 = vec<float, 3, 4>;
using vec3x3x3 = vec<float, 3, 3, 3>;

/// Available material models
enum class material_e { JFluid = 0, 
                        JFluid_ASFLIP, 
                        FixedCorotated, 
                        FixedCorotated_ASFLIP,
                        Sand, 
                        NACC, 
                        Meshed,
                        Total };

/// Available FEM element types
enum class fem_e { Tetrahedron = 0, 
                        Brick, 
                        Total };

/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html, F.3.16.5
/// benchmark setup
namespace config {
constexpr int g_device_cnt = 2;
constexpr int g_total_frame_cnt = 30;
constexpr material_e get_material_type(int did) noexcept {
  material_e type{material_e::JFluid};
  return type;
}

constexpr std::array<material_e, 5> g_material_list = {
                      material_e::Meshed, material_e::JFluid_ASFLIP, 
                      material_e::JFluid, material_e::JFluid_ASFLIP, 
                      material_e::FixedCorotated_ASFLIP};

constexpr std::array<fem_e, 5> g_fem_element_list = {
                      fem_e::Tetrahedron, fem_e::Tetrahedron, 
                      fem_e::Tetrahedron, fem_e::Tetrahedron, 
                      fem_e::Tetrahedron};

constexpr std::array<int, 5> g_fem_gpu = {1, 0, 0, 0, 0};

#define GBPCB 16
constexpr int g_num_grid_blocks_per_cuda_block = GBPCB;
constexpr int g_num_warps_per_grid_block = 1;
constexpr int g_num_warps_per_cuda_block = GBPCB;
constexpr int g_particle_batch_capacity = 128;

#define MODEL_PPC 1.f
#define MODEL_PPC_FC 1.f
constexpr float g_model_ppc = MODEL_PPC;
constexpr float cfl = 0.5f;

// background_grid
#define BLOCK_BITS 2
#define DOMAIN_BITS 7
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

// Domain size
#define DOMAIN_VOLUME 2097.152f //< g_length^3, IMPORTANT, scales mass-volume
constexpr float g_length   = 12.8f;//96.f; //< Domain full length (m)
constexpr float g_length_x = 12.8f;//96.f; //< Domain x length (m)
constexpr float g_length_y = 12.8f;   //< Domain y length (m)
constexpr float g_length_z = 3.2f;   //< Domain z length (m)
constexpr float g_grid_ratio_x = g_length_x / g_length; //< Domain x ratio
constexpr float g_grid_ratio_y = g_length_y / g_length; //< Domain y ratio
constexpr float g_grid_ratio_z = g_length_z / g_length; //< Domain z ratio
constexpr int g_grid_size_x = g_grid_size * g_grid_ratio_x; //< Domain x grid-blocks
constexpr int g_grid_size_y = g_grid_size * g_grid_ratio_y; //< Domain y grid-blocks
constexpr int g_grid_size_z = g_grid_size * g_grid_ratio_z; //< Domain z grid-blocks



// Particle
#define MAX_PPC 32
constexpr int g_max_ppc = MAX_PPC;
constexpr int g_bin_capacity = 32;
constexpr int g_particle_num_per_block = (MAX_PPC * (1 << (BLOCK_BITS * 3)));

// Material parameters
#define DENSITY 10       // kg/m3
#define YOUNGS_MODULUS 1e4 // Pascals
#define POISSON_RATIO 0.3f // rad

// Ambient parameters
constexpr float g_gravity = -9.81f; // m/s2

/// only used on host, reserves memory
constexpr int g_max_particle_num = 200000; // Particle upperbound
constexpr int g_max_active_block = 4000;  /// 62500 bytes for active mask
constexpr std::size_t
calc_particle_bin_count(std::size_t numActiveBlocks) noexcept {
  return numActiveBlocks * (g_max_ppc * g_blockvolume / g_bin_capacity);
}
constexpr std::size_t g_max_particle_bin = g_max_particle_num / g_bin_capacity;
constexpr std::size_t g_max_halo_block = 500;  //< Max halo blocks (#)
constexpr int g_target_cells = 1000; //< Max nodes in grid-cell target

/// FEM vertice and element settings (for Lagrangian forces) (JB)
constexpr int g_max_fem_vertice_num = 100000; //3636;  // Max no. of vertice on FEM mesh
constexpr int g_max_fem_element_num = 100000; //12500; // Max no. of element in FEM mesh
constexpr int g_max_fem_element_bin = 100000; //12500; // Max no. of element in FEM mesh
constexpr int g_fem_element_bin_capacity = 1;

constexpr int g_track_ID = 6301; //6554;

} // namespace config

} // namespace mn

#endif