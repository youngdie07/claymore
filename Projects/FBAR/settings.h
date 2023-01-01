#ifndef __SETTINGS_H_
#define __SETTINGS_H_
#include "partition_domain.h"
#include <MnBase/Math/Vec.h>
#include <MnBase/Object/Structural.h>
#include <string>
#include <iostream>
#include <array>
#include <vector>

namespace mn {


#define PREC double //< Set floating-point precision, i.e. double or float
#define PREC_P double //< Set particle floating-point precision, i.e. double or float
#define PREC_G float //< Set grid floating-point precision, i.e. double or float

using f_ = structural_entity<PREC>; //< Set precision of particle attributes
using fp_ = structural_entity<PREC_P>; //< Set precision of particle attributes
using fg_ = structural_entity<PREC_G>; //< Set precision of particle attributes

/// Vector type shorthand
using ivec3 = vec<int, 3>;
using pvec3 = vec<PREC, 3>;
using pvec6 = vec<PREC, 6>;
using pvec7 = vec<PREC, 7>;
using pvec9 = vec<PREC, 9>;
using pvec3x3 = vec<PREC, 3, 3>;
using pvec3x4 = vec<PREC, 3, 4>;
using pvec3x3x3 = vec<PREC, 3, 3, 3>;
using gvec3 = vec<PREC_G, 3>;
using gvec6 = vec<PREC_G, 6>;
using gvec7 = vec<PREC_G, 7>;
using gvec9 = vec<PREC_G, 9>;
using gvec3x3 = vec<PREC_G, 3, 3>;
using gvec3x4 = vec<PREC_G, 3, 4>;
using gvec3x3x3 = vec<PREC_G, 3, 3, 3>;
using vec3 = vec<float, 3>;
using vec6 = vec<float, 6>;
using vec7 = vec<float, 7>;
using vec9 = vec<float, 9>;
using vec3x3 = vec<float, 3, 3>;
using vec3x4 = vec<float, 3, 4>;
using vec3x3x3 = vec<float, 3, 3, 3>;
using dvec3 = vec<double, 3>;
using dvec6 = vec<double, 6>;
using dvec7 = vec<double, 7>;
using dvec9 = vec<double, 9>;
using dvec3x3 = vec<double, 3, 3>;
using dvec3x4 = vec<double, 3, 4>;
using dvec3x3x3 = vec<double, 3, 3, 3>;

/// Available material models
enum class material_e { JFluid = 0, 
                        JFluid_ASFLIP, 
                        JBarFluid,
                        FixedCorotated, 
                        FixedCorotated_ASFLIP,
                        FixedCorotated_ASFLIP_FBAR,
                        Sand, 
                        NACC, 
                        Meshed,
                        Total };

/// Available FEM element types
enum class fem_e {  Tetrahedron = 0, 
                    Tetrahedron_FBar,
                    Brick, 
                    Total };

/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html, F.3.16.5
/// benchmark setup
namespace config 
{
// GPU devices for simulation
constexpr int g_device_cnt = 1; //< IMPORTANT. Number of GPUs to compile for.

// Run-time animation default settings
constexpr int g_total_frame_cnt = 30; //< Default simulation frames
constexpr int g_fps = 60; //< Default frames-per-second
constexpr int g_log_level = 2; //< 0 = None, 1 = Basic info, 2 = Verbose info, 3 = Everything.

constexpr material_e get_material_type(int did) noexcept {
  material_e type{material_e::JFluid}; return type; }

constexpr fem_e get_element_type(int did) noexcept {
  fem_e type{fem_e::Tetrahedron}; return type;}



// Default material parameters
#define DENSITY 1000       //< Default density [kg/m^3]
#define YOUNGS_MODULUS 1e6 //< Default Young's Modulus [Pa]
#define POISSON_RATIO 0.3 // Default Poisson Ratio
#define MODEL_PPC 1.0 //< Default particles-per-cell
constexpr float g_model_ppc = MODEL_PPC; //< Default particles-per-cell

// Ambient parameters
#define PI_AS_A_DOUBLE 3.14159265358979323846
#define PI_AS_A_FLOAT  3.14159265358979323846f
#define GRAVITY -9.81 //< Default gravity (m/s^2)
#define CFL 0.5
constexpr float g_gravity = GRAVITY; //< Default gravity (m/s^2)
constexpr float cfl = CFL; //< Default CFL Condition Coefficient

// Grid set-up
#define BLOCK_BITS 2 //< Bits for grid block resolution. 2 -> 4x4x4 grid-nodes. 
#define DOMAIN_BITS 9 //< Bits for domain resolution. 8 -> 2^8x2^8x2^8 grid-nodes. Increase for more grid-nodes.
#define DXINV (1.f * (1 << DOMAIN_BITS))
constexpr int g_domain_bits = DOMAIN_BITS; //< Bits for grid block resolution.
constexpr int g_domain_size = (1 << DOMAIN_BITS); //< Max grid-nodes in domain direction.
constexpr float g_bc = 2; //< Offset for Off-by-2 (Claymore, Wang 2020)
constexpr float g_dx = 1.0 / DXINV; //< Grid-cell spacing
constexpr float g_dx_inv = DXINV; //< Max grid-nodes in a direction, inverse of  grid-spacing.
constexpr float g_D_inv = 4.0 * DXINV * DXINV; //< D_p^-1 for APIC (Jiang 2015). Default for quadratic B-Spline
constexpr double g_dx_inv_d = (1.0 * (1 << DOMAIN_BITS));
constexpr double g_dx_d = 1.0 / g_dx_inv_d;
constexpr int g_blockbits = BLOCK_BITS; //< Default block bits.
constexpr int g_blocksize = (1 << BLOCK_BITS); //< Grid-block size in direction. Default 4
constexpr int g_blockmask = ((1 << BLOCK_BITS) - 1); //< Mask for block. Default 3
constexpr int g_blockvolume = (1 << (BLOCK_BITS * 3)); //< No. grid-nodes in a block. Default 64.
constexpr int g_grid_bits = (DOMAIN_BITS - BLOCK_BITS); //< 
constexpr int g_grid_size = (1 << (DOMAIN_BITS - BLOCK_BITS)); //< Max blocks in domain direction
constexpr float g_offset = g_dx * 8; //< 

// Domain size
#define DOMAIN_LENGTH 12.8 //< Default max domain length [m]
#define DOMAIN_VOLUME DOMAIN_LENGTH * DOMAIN_LENGTH * DOMAIN_LENGTH //< g_length^3, scales mass-volume at compilation
constexpr double g_length   = 12.8; // 10.24f; //< Default domain full length (m)
constexpr double g_volume   = g_length * g_length * g_length; //< Default domain max volume [m^3]
constexpr double g_length_x = g_length; //< Default domain x length (m)
constexpr double g_length_y = g_length; //< Default domain y length (m)
constexpr double g_length_z = g_length; //< Default domain z length (m)
constexpr double g_domain_volume = g_length * g_length * g_length;
constexpr double g_grid_ratio_x = g_length_x / g_length + 0.0 * g_dx; //< Domain x ratio
constexpr double g_grid_ratio_y = g_length_y / g_length + 0.0 * g_dx; //< Domain y ratio
constexpr double g_grid_ratio_z = g_length_z / g_length + 0.0 * g_dx; //< Domain z ratio
// constexpr int g_grid_size_x = (g_grid_size * g_grid_ratio_x + 0.5) + 4; //< Domain x grid-blocks
// constexpr int g_grid_size_y = (g_grid_size * g_grid_ratio_y + 0.5) + 4; //< Domain y grid-blocks
// constexpr int g_grid_size_z = (g_grid_size * g_grid_ratio_z + 0.5) + 4; //< Domain z grid-blocks
constexpr int g_grid_size_x = g_grid_size ; //< Domain x grid-blocks
constexpr int g_grid_size_y = g_grid_size ; //< Domain y grid-blocks
constexpr int g_grid_size_z = g_grid_size ; //< Domain z grid-blocks


/// ------------------
/// Preallocate GPU memory for particles, grids, finite elements, grid-targets, etc
/// ------------------
// Grids
#define GBPCB 16
constexpr int g_num_grid_blocks_per_cuda_block = GBPCB;
constexpr int g_num_warps_per_grid_block = 1;
constexpr int g_num_warps_per_cuda_block = GBPCB;
constexpr int g_max_active_block = 4000; //< Max active blocks in gridBlocks. Preallocated, can resize. Lower = less memory used.
/// 62500 bytes for active mask

// Particles
#define MAX_PPC 64 //< VERY important. Max particles-per-cell. Substantially effects memory/performance, exceeding MAX_PPC deletes particles. Generally, use MAX_PPC = 8*(Actualy PPC) to account for compression.
constexpr int g_max_particle_num = 250000; //< Max no. particles. Preallocated, can resize.
constexpr int g_max_ppc = MAX_PPC; //< Default max_ppc
constexpr int g_bin_capacity = 32; //< Particles per particle bin. Multiple of 32
constexpr int g_particle_batch_capacity = 128;
constexpr int g_particle_num_per_block = 
    (MAX_PPC * (1 << (BLOCK_BITS * 3))); //< Max no. particles in a block
constexpr std::size_t g_max_particle_bin = 
    g_max_particle_num / g_bin_capacity; //< Max no. particle bins. Preallocated
constexpr std::size_t calc_particle_bin_count(std::size_t numActiveBlocks) noexcept {
    return numActiveBlocks * (g_max_ppc * g_blockvolume / g_bin_capacity); } //< Return max particle bins that fit in the active blocks 
constexpr int g_particle_attribs = 3; //< No. attribute values to output per particle 

// Finite Elements
constexpr int g_max_fem_vertice_num = 5445;  // Max no. of vertice on FEM mesh
constexpr int g_max_fem_element_num = 20480; // Max no. of element in FEM mesh
constexpr int g_fem_element_bin_capacity = 1; //< Finite elements per bin in elementBins
constexpr int g_max_fem_element_bin = 
    g_max_fem_element_num / g_fem_element_bin_capacity; // Max no. of finite element bins

// Grid-Targets
constexpr int g_target_cells = 10000; //< Max grid-nodes per gridTarget
constexpr int g_max_grid_target_nodes = 10000; //< Max grid-nodes per gridTarget
constexpr int g_target_attribs = 10; //< No. of values per gridTarget node

// Particle-Targets
constexpr int g_particle_target_cells = 10000; //< Max grid-nodes per gridTarget
constexpr int g_max_particle_target_nodes = 10000; //< Max particless per particleTarget
constexpr int g_particle_target_attribs = 10; //< No. of values per gridTarget node

// Halo
constexpr std::size_t g_max_halo_block = 0;  //< Max active halo blocks. Preallocated, can resize.


constexpr int g_track_ID = 0; // Index from [0, g_max_fem_vertice_num)
std::vector<int> g_track_IDs = {g_track_ID};

} // namespace config

} // namespace mn

#endif