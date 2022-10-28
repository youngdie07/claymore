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
                        JBarFluid,
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
constexpr int g_device_cnt = 1;
constexpr int g_total_frame_cnt = 30;
constexpr material_e get_material_type(int did) noexcept {
  material_e type{material_e::JFluid};
  return type;
}

constexpr std::array<material_e, 8> g_material_list = {
                      material_e::JBarFluid, material_e::JFluid_ASFLIP, 
                      material_e::JFluid, material_e::JFluid_ASFLIP, 
                      material_e::FixedCorotated_ASFLIP, material_e::JFluid, 
                      material_e::JFluid, material_e::JFluid};

constexpr std::array<fem_e, 8> g_fem_element_list = {
                      fem_e::Tetrahedron, fem_e::Tetrahedron, 
                      fem_e::Tetrahedron, fem_e::Tetrahedron, 
                      fem_e::Tetrahedron, fem_e::Tetrahedron,
                      fem_e::Tetrahedron, fem_e::Tetrahedron};

constexpr std::array<int, 8> g_ASFLIP_gpu = {1, 1, 1, 1, 1, 1, 1, 1};
constexpr std::array<int, 8> g_fem_gpu = {0, 0, 0, 0, 0, 0, 0, 0};
constexpr std::array<int, 8> g_FBar_gpu = {1, 0, 0, 0, 0, 0, 0, 0};


#define GBPCB 16
constexpr int g_num_grid_blocks_per_cuda_block = GBPCB;
constexpr int g_num_warps_per_grid_block = 1;
constexpr int g_num_warps_per_cuda_block = GBPCB;
constexpr int g_particle_batch_capacity = 128;

#define MODEL_PPC 1.f //< Default particles-per-cell
#define MODEL_PPC_FC 1.f //< 
constexpr float g_model_ppc = MODEL_PPC; //< Default particles-per-cell
constexpr float cfl = 0.5f; //< CFL Condition Coefficient

// background_grid
#define BLOCK_BITS 2
#define DOMAIN_BITS 8
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
#define DOMAIN_LENGTH 10.24f // Domain default length [m]
#define DOMAIN_VOLUME DOMAIN_LENGTH * DOMAIN_LENGTH * DOMAIN_LENGTH //< g_length^3, scales mass-volume at compilation
constexpr float g_length   = 10.24f; //< Domain full length (m)
constexpr float g_volume   = g_length * g_length * g_length; //< Domain max volume [m^3]
constexpr float g_length_x = 6.0f; //< Domain x length (m)
constexpr float g_length_y = 6.0f; //< Domain y length (m)
constexpr float g_length_z = 0.08f; //< Domain z length (m)
constexpr float g_domain_volume = g_length * g_length * g_length;
constexpr float g_grid_ratio_x = g_length_x / g_length + 16.0f * g_dx; //< Domain x ratio
constexpr float g_grid_ratio_y = g_length_y / g_length + 16.0f * g_dx; //< Domain y ratio
constexpr float g_grid_ratio_z = g_length_z / g_length + 16.0f * g_dx; //< Domain z ratio
constexpr int g_grid_size_x = (g_grid_size * g_grid_ratio_x ) + 1; //< Domain x grid-blocks
constexpr int g_grid_size_y = (g_grid_size * g_grid_ratio_y ) + 1; //< Domain y grid-blocks
constexpr int g_grid_size_z = (g_grid_size * g_grid_ratio_z ) + 1; //< Domain z grid-blocks

// Particle
#define MAX_PPC 32
constexpr int g_max_ppc = MAX_PPC;
constexpr int g_bin_capacity = 32;
constexpr int g_particle_num_per_block = (MAX_PPC * (1 << (BLOCK_BITS * 3)));

// Material parameters
#define DENSITY 1000       // kg/m3
#define YOUNGS_MODULUS 1e6 // Pascals
#define POISSON_RATIO 0.0f // rad

// Ambient parameters
constexpr float g_gravity = -9.81f; // m/s2

/// only used on host, reserves memory
constexpr int g_max_particle_num = 125000; // Particle upperbound
constexpr int g_max_active_block = 5500;  /// 62500 bytes for active mask
constexpr std::size_t
calc_particle_bin_count(std::size_t numActiveBlocks) noexcept {
  return numActiveBlocks * (g_max_ppc * g_blockvolume / g_bin_capacity);
}
constexpr std::size_t g_max_particle_bin = g_max_particle_num / g_bin_capacity;
constexpr std::size_t g_max_halo_block = 0;  //< Max halo blocks (#)
constexpr int g_target_cells = 8000; //< Max nodes in grid-cell target
constexpr int g_target_attribs = 10; //< Attributes on output grid-node (xyz, mass, momentum, force)

/// FEM vertice and element settings (for Lagrangian forces) (JB)
constexpr int g_max_fem_vertice_num = 9261; // Max no. of vertice on FEM mesh
constexpr int g_max_fem_element_num = 40000; // Max no. of element in FEM mesh
constexpr int g_max_fem_element_bin = 40000; // Max no. of element in FEM mesh
constexpr int g_fem_element_bin_capacity = 1;
constexpr int g_track_ID = 9260; // Index from [0, g_max_fem_vertice_num)
} // namespace config

} // namespace mn

#endif