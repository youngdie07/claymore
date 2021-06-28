#ifndef __MULTI_GMPM_KERNELS_CUH_
#define __MULTI_GMPM_KERNELS_CUH_

#include "boundary_condition.cuh"
#include "constitutive_models.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include "utility_funcs.hpp"
#include <MnBase/Algorithm/MappingKernels.cuh>
#include <MnBase/Math/Matrix/MatrixUtils.h>
#include <MnSystem/Cuda/DeviceUtils.cuh>

namespace mn {

using namespace config;
using namespace placeholder;

template <typename ParticleArray, typename Partition>
__global__ void activate_blocks(uint32_t particleCount, ParticleArray parray,
                                Partition partition) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount)
    return;
  ivec3 blockid{
      int(std::lround(parray.val(_0, parid) / g_dx) - 2) / g_blocksize,
      int(std::lround(parray.val(_1, parid) / g_dx) - 2) / g_blocksize,
      int(std::lround(parray.val(_2, parid) / g_dx) - 2) / g_blocksize};
  partition.insert(blockid);
}
template <typename ParticleArray, typename Partition>
__global__ void build_particle_cell_buckets(uint32_t particleCount,
                                            ParticleArray parray,
                                            Partition partition) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount)
    return;
  ivec3 coord{int(std::lround(parray.val(_0, parid) / g_dx) - 2),
              int(std::lround(parray.val(_1, parid) / g_dx) - 2),
              int(std::lround(parray.val(_2, parid) / g_dx) - 2)};
  int cellno = (coord[0] & g_blockmask) * g_blocksize * g_blocksize +
               (coord[1] & g_blockmask) * g_blocksize +
               (coord[2] & g_blockmask);
  coord = coord / g_blocksize;
  auto blockno = partition.query(coord);
  auto pidic = atomicAdd(partition._ppcs + blockno * g_blockvolume + cellno, 1);
  partition._cellbuckets[blockno * g_particle_num_per_block +
                         cellno * g_max_ppc + pidic] = parid;
}
__global__ void cell_bucket_to_block(int *_ppcs, int *_cellbuckets, int *_ppbs,
                                     int *_buckets) {
  int cellno = threadIdx.x & (g_blockvolume - 1);
  int pcnt = _ppcs[blockIdx.x * g_blockvolume + cellno];
  for (int pidic = 0; pidic < g_max_ppc; pidic++) {
    if (pidic < pcnt) {
      auto pidib = atomicAggInc<int>(_ppbs + blockIdx.x);
      _buckets[blockIdx.x * g_particle_num_per_block + pidib] =
          _cellbuckets[blockIdx.x * g_particle_num_per_block +
                       cellno * g_max_ppc + pidic];
    }
    __syncthreads();
  }
}
__global__ void compute_bin_capacity(uint32_t blockCount, int const *_ppbs,
                                     int *_bincaps) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  _bincaps[blockno] = (_ppbs[blockno] + g_bin_capacity - 1) / g_bin_capacity;
}
__global__ void init_adv_bucket(const int *_ppbs, int *_buckets) {
  auto pcnt = _ppbs[blockIdx.x];
  auto bucket = _buckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    bucket[pidib] =
        (dir_offset(ivec3{0, 0, 0}) * g_particle_num_per_block) | pidib;
  }
}
template <typename Grid> __global__ void clear_grid(Grid grid) {
  auto gridblock = grid.ch(_0, blockIdx.x);
  for (int cidib = threadIdx.x; cidib < g_blockvolume; cidib += blockDim.x) {
    gridblock.val_1d(_0, cidib) = 0.f;
    gridblock.val_1d(_1, cidib) = 0.f;
    gridblock.val_1d(_2, cidib) = 0.f;
    gridblock.val_1d(_3, cidib) = 0.f;
  }
}
template <typename Partition>
__global__ void register_neighbor_blocks(uint32_t blockCount,
                                         Partition partition) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  auto blockid = partition._activeKeys[blockno];
  for (char i = 0; i < 2; ++i)
    for (char j = 0; j < 2; ++j)
      for (char k = 0; k < 2; ++k)
        partition.insert(ivec3{blockid[0] + i, blockid[1] + j, blockid[2] + k});
}
template <typename Partition>
__global__ void register_exterior_blocks(uint32_t blockCount,
                                         Partition partition) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  auto blockid = partition._activeKeys[blockno];
  for (char i = -1; i < 2; ++i)
    for (char j = -1; j < 2; ++j)
      for (char k = -1; k < 2; ++k)
        partition.insert(ivec3{blockid[0] + i, blockid[1] + j, blockid[2] + k});
}
template <typename Grid, typename Partition>
__global__ void rasterize(uint32_t particleCount, const ParticleArray parray,
                          Grid grid, const Partition partition, float dt,
                          float mass) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount)
    return;

  vec3 local_pos{parray.val(_0, parid), parray.val(_1, parid),
                 parray.val(_2, parid)};
  vec3 vel;
  vec9 contrib, C;
  vel.set(0.f), contrib.set(0.f), C.set(0.f);
  // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
  float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
  float scale = g_length * g_length; //< Area scale (m^2)
  Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline
  contrib = (C * mass - contrib * dt) * Dp_inv;
  ivec3 global_base_index{int(std::lround(local_pos[0] * g_dx_inv) - 1),
                          int(std::lround(local_pos[1] * g_dx_inv) - 1),
                          int(std::lround(local_pos[2] * g_dx_inv) - 1)};
  local_pos = local_pos - global_base_index * g_dx;
  vec<vec3, 3> dws;
  for (int d = 0; d < 3; ++d)
    dws[d] = bspline_weight(local_pos[d]);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        ivec3 offset{i, j, k};
        vec3 xixp = offset * g_dx - local_pos;
        float W = dws[0][i] * dws[1][j] * dws[2][k];
        ivec3 local_index = global_base_index + offset;
        float wm = mass * W;
        int blockno = partition.query(ivec3{local_index[0] >> g_blockbits,
                                            local_index[1] >> g_blockbits,
                                            local_index[2] >> g_blockbits});
        auto grid_block = grid.ch(_0, blockno);
        for (int d = 0; d < 3; ++d)
          local_index[d] &= g_blockmask;
        atomicAdd(
            &grid_block.val(_0, local_index[0], local_index[1], local_index[2]),
            wm);
        atomicAdd(
            &grid_block.val(_1, local_index[0], local_index[1], local_index[2]),
            wm * vel[0] + (contrib[0] * xixp[0] + contrib[3] * xixp[1] +
                           contrib[6] * xixp[2]) *
                              W);
        atomicAdd(
            &grid_block.val(_2, local_index[0], local_index[1], local_index[2]),
            wm * vel[1] + (contrib[1] * xixp[0] + contrib[4] * xixp[1] +
                           contrib[7] * xixp[2]) *
                              W);
        atomicAdd(
            &grid_block.val(_3, local_index[0], local_index[1], local_index[2]),
            wm * vel[2] + (contrib[2] * xixp[0] + contrib[5] * xixp[1] +
                           contrib[8] * xixp[2]) *
                              W);
      }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::JFluid> pbuffer,
                                Partition partition) {
  uint32_t blockno = blockIdx.x;
  int pcnt = partition._ppbs[blockno];
  auto bucket = partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// J
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
  }
}

template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::FixedCorotated> pbuffer,
                Partition partition) {
  uint32_t blockno = blockIdx.x;
  int pcnt = partition._ppbs[blockno];
  auto bucket = partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
    pbin.val(_4, pidib % g_bin_capacity) = 0.f;
    pbin.val(_5, pidib % g_bin_capacity) = 0.f;
    pbin.val(_6, pidib % g_bin_capacity) = 0.f;
    pbin.val(_7, pidib % g_bin_capacity) = 1.f;
    pbin.val(_8, pidib % g_bin_capacity) = 0.f;
    pbin.val(_9, pidib % g_bin_capacity) = 0.f;
    pbin.val(_10, pidib % g_bin_capacity) = 0.f;
    pbin.val(_11, pidib % g_bin_capacity) = 1.f;
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::Sand> pbuffer,
                                Partition partition) {
  uint32_t blockno = blockIdx.x;
  int pcnt = partition._ppbs[blockno];
  auto bucket = partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
    pbin.val(_4, pidib % g_bin_capacity) = 0.f;
    pbin.val(_5, pidib % g_bin_capacity) = 0.f;
    pbin.val(_6, pidib % g_bin_capacity) = 0.f;
    pbin.val(_7, pidib % g_bin_capacity) = 1.f;
    pbin.val(_8, pidib % g_bin_capacity) = 0.f;
    pbin.val(_9, pidib % g_bin_capacity) = 0.f;
    pbin.val(_10, pidib % g_bin_capacity) = 0.f;
    pbin.val(_11, pidib % g_bin_capacity) = 1.f;
    /// logJp
    pbin.val(_12, pidib % g_bin_capacity) =
        ParticleBuffer<material_e::Sand>::logJp0;
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::NACC> pbuffer,
                                Partition partition) {
  uint32_t blockno = blockIdx.x;
  int pcnt = partition._ppbs[blockno];
  auto bucket = partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
    pbin.val(_4, pidib % g_bin_capacity) = 0.f;
    pbin.val(_5, pidib % g_bin_capacity) = 0.f;
    pbin.val(_6, pidib % g_bin_capacity) = 0.f;
    pbin.val(_7, pidib % g_bin_capacity) = 1.f;
    pbin.val(_8, pidib % g_bin_capacity) = 0.f;
    pbin.val(_9, pidib % g_bin_capacity) = 0.f;
    pbin.val(_10, pidib % g_bin_capacity) = 0.f;
    pbin.val(_11, pidib % g_bin_capacity) = 1.f;
    /// logJp
    pbin.val(_12, pidib % g_bin_capacity) =
        ParticleBuffer<material_e::NACC>::logJp0;
  }
}

template <typename Grid, typename Partition>
__global__ void update_grid_velocity_query_max(uint32_t blockCount, Grid grid,
                                               Partition partition, float dt,
                                               float *maxVel, float curTime, vec3 waveMaker) {
  constexpr int bc = g_bc;
  constexpr int numWarps =
      g_num_grid_blocks_per_cuda_block * g_num_warps_per_grid_block;
  constexpr unsigned activeMask = 0xffffffff;
  //__shared__ float sh_maxvels[g_blockvolume * g_num_grid_blocks_per_cuda_block
  /// 32];
  extern __shared__ float sh_maxvels[];
  std::size_t blockno = blockIdx.x * g_num_grid_blocks_per_cuda_block +
                        threadIdx.x / 32 / g_num_warps_per_grid_block;
  auto blockid = partition._activeKeys[blockno];
  int isInBound = ((blockid[0] < bc || blockid[0] >= g_grid_size_x - bc) << 2) |
                  ((blockid[1] < bc || blockid[1] >= g_grid_size_y - bc) << 1) |
                  (blockid[2] < bc || blockid[2] >= g_grid_size_z - bc);
  if (threadIdx.x < numWarps)
    sh_maxvels[threadIdx.x] = 0.0f;
  __syncthreads();

  /// within-warp computations
  if (blockno < blockCount) {
    auto grid_block = grid.ch(_0, blockno);
    for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {
      float mass = grid_block.val_1d(_0, cidib), velSqr = 0.f, vel[3];
      if (mass > 0.f) {
        mass = 1.f / mass;

        // Grid node coordinate [i,j,k] in grid-block
        int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
        int j = (cidib >> g_blockbits) & g_blockmask;
        int k = cidib & g_blockmask;
        // Grid node position [x,y,z] in entire domain
        float xc = (4*blockid[0]*g_dx) + (i*g_dx); // + (g_dx/2.f);
        float yc = (4*blockid[1]*g_dx) + (j*g_dx); // + (g_dx/2.f);
        float zc = (4*blockid[2]*g_dx) + (k*g_dx); // + (g_dx/2.f);

        // Offset condition for Off-by-2 (see Xinlei & Fang et al.)
        // Note, subtract 16 nodes from total
        // (4 grid blocks) to have available scene length
        float offset = (8.f*g_dx);

        // Retrieve grid momentums (kg*m/s2)
        vel[0] = grid_block.val_1d(_1, cidib); //< mvx
        vel[1] = grid_block.val_1d(_2, cidib); //< mvy
        vel[2] = grid_block.val_1d(_3, cidib); //< mvz


        // WASIRF Harris Flume (Slip)
        // Acts on individual grid-cell velocities
        // https://teamer-us.org/product/university-of-washington-harris-hydraulics-wasirf/
        float flumex = 87.4268f / g_length - g_dx; // Actually 12m, added run-in/out
        float flumey = 4.572f / g_length - g_dx; // 1.22m Depth
        float flumez = 3.6576f / g_length - g_dx; // 0.91m Width
        int isInFlume =  ((xc < offset || xc >= flumex + offset) << 2) |
                         ((yc <= offset || yc >= flumey + offset) << 1) |
                          (zc <= offset || zc >= flumez + offset);
        isInBound |= isInFlume; // Update with regular boundary for efficiency


        // Add grid-cell boundary for structural block, WASIRF flume
        vec3 struct_dim; //< Dimensions of structure in [1,1,1] pseudo-dimension
        struct_dim[0] = (0.7871f) / g_length;
        struct_dim[1] = (0.3935f) / g_length;
        struct_dim[2] = (0.7871f) / g_length;
        vec3 struct_pos; //< Position of structures in [1,1,1] pseudo-dimension
        struct_pos[0] = ((46 + 12 + 36 + 48 + (10.f/12.f))*0.3048f) / g_length + offset;
        struct_pos[1] = (2.f) / g_length + (0.5f * g_dx) + offset;
        struct_pos[2] = (flumez - struct_dim[2]) / 2.f + offset;
        float t = 1.0f * g_dx;

        // Check if grid-cell is within sticky interior of structural box
        // Subtract slip-layer thickness from structural box dimension for geometry
        int isOutStruct  = ((xc >= struct_pos[0] + t && xc < struct_pos[0] + struct_dim[0] - t) << 2) | 
                           ((yc >= struct_pos[1] + t && yc < struct_pos[1] + struct_dim[1] - t) << 1) |
                            (zc >= struct_pos[2] + t && zc < struct_pos[2] + struct_dim[2] - t);
        if (isOutStruct != 7) isOutStruct = 0; // Check if 111, reset otherwise
        isInBound |= isOutStruct; // Update with regular boundary for efficiency
        
        // Check exterior slip-layer of structural block(OSU Flume)
        // One-cell depth, six-faces, order matters! (over-writes on edges, favors front) 
        int isOnStructFace[6];
        // Right (z+)
        isOnStructFace[0] = ((xc >= struct_pos[0] && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] + struct_dim[2] - t && zc < struct_pos[2] + struct_dim[2]);
        // Left (z-)
        isOnStructFace[1] = ((xc >= struct_pos[0] && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + t);        
        // Top (y+)
        isOnStructFace[2] = ((xc >= struct_pos[0] && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                            ((yc >= struct_pos[1] + struct_dim[1] - t && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + struct_dim[2]);
        // Bottom (y-)
        isOnStructFace[3] = ((xc >= struct_pos[0] && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + t) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + struct_dim[2]);
        // Back (x+)
        isOnStructFace[4] = ((xc >= struct_pos[0] + struct_dim[0] - t && xc < struct_pos[0] + struct_dim[1]) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + struct_dim[2]);
        // Front (x-)
        isOnStructFace[5] = ((xc >= struct_pos[0] && xc < struct_pos[0] + t) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + struct_dim[2]);                             
        // Reduce results from box faces to single result
        int isOnStruct = 0; // Collision reduction variable
        for (int iter=0; iter<6; iter++) {
          if (isOnStructFace[iter] != 7) {
            // Check if 111 (7), set 000 (0) otherwise
            isOnStructFace[iter] = 0;
          } else {
            isOnStructFace[iter] = (1 << iter / 2);
          }
          isOnStruct |= isOnStructFace[iter]; // OR operator to reduce results
        }
        if (isOnStruct == 6 || isOnStruct == 5) isOnStruct = 4; // Overlaps on front
        else if (isOnStruct == 3 || isOnStruct == 7) isOnStruct = 0; // Overlaps on sides
        isInBound |= isOnStruct; // Update with regular boundary for efficiency


#if 1
        ///< Slip contact        
        // Set cell velocity after grid-block/cell boundary check
        vel[0] = isInBound & 4 ? 0.f : vel[0] * mass; //< vx = mvx / m
        vel[1] = isInBound & 2 ? 0.f : vel[1] * mass; //< vy = mvy / m
        vel[1] += isInBound & 2 ? 0.f : (g_gravity / g_length) * dt;  //< Grav. effect
        vel[2] = isInBound & 1 ? 0.f : vel[2] * mass; //< vz = mvz / m
#endif        

#if 0
        ///< Sticky contact
        if (isInBound) ///< sticky
          vel.set(0.f);
#endif


        vec3 ns; //< Ramp boundary surface normal
        float ys;
        float xs;
        float xo;

        // Start ramp segment definition for OSU flume
        // Based on bathymetry diagram, February
        if (xc < (14.2748/g_length)+offset) {
          // Flat, 0' elev., 0' - 46'10
          ns[0] = 0.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = offset;
          float yo = offset;
          xs = xc;
          ys = yo;

        } else if (xc > (14.2748/g_length)+offset && xc < (17.9324/g_length)+offset){
          // Flat (adjustable), 0' elev., 46'10 - 58'10
          ns[0] = 0.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (14.2748 / g_length) + offset;
          float yo = offset;
          xs = xc;
          ys = yo;

        } else if (xc > (17.9324/g_length)+offset && xc < (28.905/g_length)+offset) {
          // 1:12, 0' elev., 58'10 - 94'10
          ns[0] = -1.f/12.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (17.9324 / g_length) + offset;
          float yo = offset;
          xs = xc;
          ys = 1.f/12.f * (xc - xo) + yo;

        } else if (xc > (28.905/g_length)+offset && xc < (43.5356/g_length)+offset) {
          // 1:24, 3' elev., 94'10 - 142'10
          ns[0] = -1.f/24.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (28.905 / g_length) + offset;
          float yo = (0.9144 / g_length) + offset;
          xs = xc;
          ys = 1.f/24.f * (xc - xo) + yo;

        } else if (xc > (43.5356/g_length)+offset && xc < (80.1116/g_length)+offset) {
          // Flat, 5' elev., 142'10 - 262'10
          ns[0] = 0.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (43.5356 / g_length) + offset;
          float yo = (1.524 / g_length) + offset;
          xs = xc;
          ys = yo;

        } else if (xc > (80.1116/g_length)+offset && xc < (87.4268/g_length)+offset) {
          // 1:12, 5' elev., 262'10 - 286'10
          ns[0] = -1.f/12.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (80.1116 / g_length) + offset;
          float yo = (1.524 / g_length) + offset;
          xs = xc;
          ys = 1.f/12.f * (xc - xo) + yo;

        } else {
          // Flat, 7' elev., 286'10 onward
          ns[0]=0.f;
          ns[1]=1.f;
          ns[2]=0.f;
          float yo = (2.1336 / g_length) + offset;
          ys = yo;        
        }

        float ns_mag = sqrt(ns[0]*ns[0] + ns[1]*ns[1] + ns[2]*ns[2]);
        ns = ns / ns_mag;
        float vdotns = vel[0]*ns[0] + vel[1]*ns[1] + vel[2]*ns[2];

        // Boundary thickness and cell distance
        //h  = sqrt((g_dx*g_dx*ns[0]) + (g_dx*g_dx*ns[1]) + (g_dx*g_dx*ns[2]));
        //float r  = sqrt((xc - xs)*(xc - xs) + (yc - ys)*(yc - ys));

        // Decay coefficient
        float ySF;
        if (yc > ys + g_dx) {
          ySF = 0.f;
        } else if (yc <= ys){
          ySF = 1.f;
        } else {
          ySF = ((g_dx - (yc - ys)) / g_dx) * ((g_dx - (yc - ys)) / g_dx);
        }

        // fbc = -fint - fext - (1/dt)*p
        // a = (1 / mass) * (fint + fext + ySf*fbc) 

        // Adjust velocity relative to surface
        if (0) {
          // Normal adjustment in decay layer, fix below
          if (ySF == 1.f) {
            vel[0] = vel[1] = vel[2] = 0.f;
          } else if (ySF > 0.f && ySF < 1.f) {
            vel[0] = vel[0] - ySF * (vel[0] - vdotns * ns[0]);
            vel[1] = vel[1] - ySF * (vel[1] - vdotns * ns[1]);
            vel[2] = vel[2] - ySF * (vel[2] - vdotns * ns[2]);  
          }
        }
        if (1) {
          // Free above surface, normal adjusted below
          vel[0] = vel[0] - ySF * (vdotns * ns[0]);
          vel[1] = vel[1] - ySF * (vdotns * ns[1]);
          vel[2] = vel[2] - ySF * (vdotns * ns[2]);
        }

        if (0) {
          // OSU Wave-Maker - Manual Control
          float wm_pos;
          float wm_vel;
          if (curTime >= 2.f && curTime < 4.f){
            wm_vel = 2.f / g_length;
            wm_pos = (curTime - 2.f) * 2.f / g_length + offset;
          } else if (curTime >= 4.f) {
            wm_vel = 0.f;
            wm_pos = (4.f - 2.f) * 2.f / g_length + offset;
          } else {
            wm_vel = 0.f;
            wm_pos = offset;
          }
          if (xc < wm_pos) {
            vel[0] = wm_vel;
          }
        }
        if (1) {
          // OSU Wave-Maker - CSV Control
          if (xc <= (waveMaker[1] / g_length + offset)) {
             //vel[0] = fmax(waveMaker[2] / g_length, abs(vel[0]));
             vel[0] = waveMaker[2] / g_length;

          }
        }

        grid_block.val_1d(_1, cidib) = vel[0];
        velSqr += vel[0] * vel[0];
        grid_block.val_1d(_2, cidib) = vel[1];
        velSqr += vel[1] * vel[1];
        grid_block.val_1d(_3, cidib) = vel[2];
        velSqr += vel[2] * vel[2];
      }
      // unsigned activeMask = __ballot_sync(0xffffffff, mv[0] != 0.0f);
      for (int iter = 1; iter % 32; iter <<= 1) {
        float tmp = __shfl_down_sync(activeMask, velSqr, iter, 32);
        if ((threadIdx.x % 32) + iter < 32)
          velSqr = tmp > velSqr ? tmp : velSqr;
      }
      if (velSqr > sh_maxvels[threadIdx.x / 32] && (threadIdx.x % 32) == 0)
        sh_maxvels[threadIdx.x / 32] = velSqr;
    }
  }
  __syncthreads();
  /// various assumptions
  for (int interval = numWarps >> 1; interval > 0; interval >>= 1) {
    if (threadIdx.x < interval) {
      if (sh_maxvels[threadIdx.x + interval] > sh_maxvels[threadIdx.x])
        sh_maxvels[threadIdx.x] = sh_maxvels[threadIdx.x + interval];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicMax(maxVel, sh_maxvels[0]);
}

template <typename Grid, typename Partition, typename Boundary>
__global__ void update_grid_velocity_query_max(uint32_t blockCount, Grid grid,
                                               Partition partition, float dt,
                                               Boundary boundary,
                                               float *maxVel, float curTime) {
  constexpr int bc = 2;
  constexpr int numWarps =
      g_num_grid_blocks_per_cuda_block * g_num_warps_per_grid_block;
  constexpr unsigned activeMask = 0xffffffff;
  //__shared__ float sh_maxvels[g_blockvolume * g_num_grid_blocks_per_cuda_block
  /// 32];
  extern __shared__ float sh_maxvels[];
  std::size_t blockno = blockIdx.x * g_num_grid_blocks_per_cuda_block +
                        threadIdx.x / 32 / g_num_warps_per_grid_block;
  auto blockid = partition._activeKeys[blockno];
  int isInBound = ((blockid[0] < bc || blockid[0] >= g_grid_size_x - bc) << 2) |
                  ((blockid[1] < bc || blockid[1] >= g_grid_size_y - bc) << 1) |
                  (blockid[2] < bc || blockid[2] >= g_grid_size_z - bc);
  if (threadIdx.x < numWarps)
    sh_maxvels[threadIdx.x] = 0.0f;
  __syncthreads();

  /// within-warp computations
  if (blockno < blockCount) {
    auto grid_block = grid.ch(_0, blockno);
    for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {
      float mass = grid_block.val_1d(_0, cidib), velSqr = 0.f;
      vec3 vel;
      if (mass > 0.f) {
        mass = 1.f / mass;

        // Grid node coordinate [i,j,k] in grid-block
        int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
        int j = (cidib >> g_blockbits) & g_blockmask;
        int k = cidib & g_blockmask;
        // Grid node position [x,y,z] in entire domain
        float xc = (4*blockid[0]*g_dx) + (i*g_dx); // + (g_dx/2.f);
        float yc = (4*blockid[1]*g_dx) + (j*g_dx); // + (g_dx/2.f);
        float zc = (4*blockid[2]*g_dx) + (k*g_dx); // + (g_dx/2.f);

        // Offset condition for Off-by-2 (see Xinlei & Fang et al.)
        // Note you should subtract 16 nodes from total
        // (or 4 grid blocks) to have total available length
        float offset = (8.f*g_dx);

        // Retrieve grid momentums (kg*m/s2)
        vel[0] = grid_block.val_1d(_1, cidib); //< mvx
        vel[1] = grid_block.val_1d(_2, cidib); //< mvy
        vel[2] = grid_block.val_1d(_3, cidib); //< mvz


        // WASIRF Harris Flume (Slip)
        // Acts on individual grid-cell velocities
        // https://teamer-us.org/product/university-of-washington-harris-hydraulics-wasirf/
        float flumex = 104.f / g_length; // Actually 12m, added run-in/out
        float flumey = 4.6f / g_length; // 1.22m Depth
        float flumez = 3.67f / g_length; // 0.91m Width
        int isInFlume =  ((xc < offset || xc >= flumex + offset) << 2) |
                         ((yc < offset || yc >= flumey + offset) << 1) |
                          (zc < offset || zc >= flumez + offset);
        isInBound |= isInFlume; // Update with regular boundary for efficiency


        // Add grid-cell boundary for structural block, WASIRF flume
        vec3 struct_dim; //< Dimensions of structure in [1,1,1] pseudo-dimension
        struct_dim[0] = (0.7871f) / g_length;
        struct_dim[1] = (0.3935f) / g_length;
        struct_dim[2] = (0.7871f) / g_length;
        vec3 struct_pos; //< Position of structures in [1,1,1] pseudo-dimension
        struct_pos[0] = ((46 + 12 + 36 + 48 + (10.f/12.f))*0.3048f) / g_length + offset;
        struct_pos[1] = ((69.f/12.f)*0.3048f) / g_length + offset;
        struct_pos[2] = (flumez - struct_dim[2]) / 2.f + offset;
        float t = 1.0f * g_dx;

        // Check if grid-cell is within sticky interior of structural box
        // Subtract slip-layer thickness from structural box dimension for geometry
        int isOutStruct  = ((xc >= struct_pos[0] + t && xc < struct_pos[0] + struct_dim[0] - t) << 2) | 
                           ((yc >= struct_pos[1] + t && yc < struct_pos[1] + struct_dim[1] - t) << 1) |
                            (zc >= struct_pos[2] + t && zc < struct_pos[2] + struct_dim[2] - t);
        if (isOutStruct != 7) isOutStruct = 0; // Check if 111, reset otherwise
        isInBound |= isOutStruct; // Update with regular boundary for efficiency
        
        // Check exterior slip-layer of structural block(OSU Flume)
        // One-cell depth, six-faces, order matters! (over-writes on edges, favors front) 
        int isOnStructFace[6];
        // Right (z+)
        isOnStructFace[0] = ((xc >= struct_pos[0] && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] + struct_dim[2] - t && zc < struct_pos[2] + struct_dim[2]);
        // Left (z-)
        isOnStructFace[1] = ((xc >= struct_pos[0] && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + t);        
        // Top (y+)
        isOnStructFace[2] = ((xc >= struct_pos[0] && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                            ((yc >= struct_pos[1] + struct_dim[1] - t && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + struct_dim[2]);
        // Bottom (y-)
        isOnStructFace[3] = ((xc >= struct_pos[0] && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + t) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + struct_dim[2]);
        // Back (x+)
        isOnStructFace[4] = ((xc >= struct_pos[0] + struct_dim[0] - t && xc < struct_pos[0] + struct_dim[1]) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + struct_dim[2]);
        // Front (x-)
        isOnStructFace[5] = ((xc >= struct_pos[0] && xc < struct_pos[0] + t) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + struct_dim[2]);                             
        // Reduce results from box faces to single result
        int isOnStruct = 0; // Collision reduction variable
        for (int iter=0; iter<6; iter++) {
          if (isOnStructFace[iter] != 7) {
            // Check if 111 (7), set 000 (0) otherwise
            isOnStructFace[iter] = 0;
          } else {
            isOnStructFace[iter] = (1 << iter / 2);
          }
          isOnStruct |= isOnStructFace[iter]; // OR operator to reduce results
        }
        if (isOnStruct == 6 || isOnStruct == 5) isOnStruct = 4; // Overlaps on front
        else if (isOnStruct == 3 || isOnStruct == 7) isOnStruct = 0; // Overlaps on sides
        isInBound |= isOnStruct; // Update with regular boundary for efficiency


#if 1
        ///< Slip contact        
        // Set cell velocity after grid-block/cell boundary check
        vel[0] = isInBound & 4 ? 0.f : vel[0] * mass; //< vx = mvx / m
        vel[1] = isInBound & 2 ? 0.f : vel[1] * mass; //< vy = mvy / m
        vel[1] += isInBound & 2 ? 0.f : (g_gravity / g_length) * dt;  //< Grav. effect
        vel[2] = isInBound & 1 ? 0.f : vel[2] * mass; //< vz = mvz / m
#endif        

#if 0
        ///< Sticky contact
        if (isInBound) ///< sticky
          vel.set(0.f);
#endif


        vec3 ns; //< Ramp boundary surface normal
        float ys;
        float xs;
        float xo;

        // Start ramp segment definition for OSU flume
        // Based on bathymetry diagram, February
        if (xc < (14.2748/g_length)+offset) {
          // Flat, 0' elev., 0' - 46'10
          ns[0] = 0.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = offset;
          float yo = offset;
          xs = xc;
          ys = yo;

        } else if (xc > (14.2748/g_length)+offset && xc < (17.9324/g_length)+offset){
          // Flat (adjustable), 0' elev., 46'10 - 58'10
          ns[0] = 0.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (14.2748 / g_length) + offset;
          float yo = offset;
          xs = xc;
          ys = yo;

        } else if (xc > (17.9324/g_length)+offset && xc < (28.905/g_length)+offset) {
          // 1:12, 0' elev., 58'10 - 94'10
          ns[0] = -1.f/12.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (17.9324 / g_length) + offset;
          float yo = offset;
          xs = xc;
          ys = 1.f/12.f * (xc - xo) + yo;

        } else if (xc > (28.905/g_length)+offset && xc < (43.5356/g_length)+offset) {
          // 1:24, 3' elev., 94'10 - 142'10
          ns[0] = -1.f/24.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (28.905 / g_length) + offset;
          float yo = (0.9144 / g_length) + offset;
          xs = xc;
          ys = 1.f/24.f * (xc - xo) + yo;

        } else if (xc > (43.5356/g_length)+offset && xc < (80.1116/g_length)+offset) {
          // Flat, 5' elev., 142'10 - 262'10
          ns[0] = 0.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (43.5356 / g_length) + offset;
          float yo = (1.524 / g_length) + offset;
          xs = xc;
          ys = yo;

        } else if (xc > (80.1116/g_length)+offset && xc < (87.4268/g_length)+offset) {
          // 1:12, 5' elev., 262'10 - 286'10
          ns[0] = -1.f/12.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (80.1116 / g_length) + offset;
          float yo = (1.524 / g_length) + offset;
          xs = xc;
          ys = 1.f/12.f * (xc - xo) + yo;

        } else {
          // Flat, 7' elev., 286'10 onward
          ns[0]=0.f;
          ns[1]=1.f;
          ns[2]=0.f;
          float yo = (2.1336 / g_length) + offset;
          ys = yo;        
        }

        float ns_mag = sqrt(ns[0]*ns[0] + ns[1]*ns[1] + ns[2]*ns[2]);
        ns = ns / ns_mag;
        float vdotns = vel[0]*ns[0] + vel[1]*ns[1] + vel[2]*ns[2];

        // Boundary thickness and cell distance
        //h  = sqrt((g_dx*g_dx*ns[0]) + (g_dx*g_dx*ns[1]) + (g_dx*g_dx*ns[2]));
        //float r  = sqrt((xc - xs)*(xc - xs) + (yc - ys)*(yc - ys));

        // Decay coefficient
        float ySF;
        if (yc > ys) {
          ySF = 0.f;
        } else if (yc <= ys){
          ySF = 1.f;
        }

        // fbc = -fint - fext - (1/dt)*p
        // a = (1 / mass) * (fint + fext + ySf*fbc) 

        // Adjust velocity relative to surface
        if (0) {
          // Normal adjustment in decay layer, fix below
          if (ySF == 1.f) {
            vel.set(0.f);
          } else if (ySF > 0.f && ySF < 1.f) {
            vel[0] = vel[0] - ySF * (vel[0] - vdotns * ns[0]);
            vel[1] = vel[1] - ySF * (vel[1] - vdotns * ns[1]);
            vel[2] = vel[2] - ySF * (vel[2] - vdotns * ns[2]);  
          }
        }
        if (1) {
          // Free above surface, normal adjusted below
          vel[0] = vel[0] - ySF * (vdotns * ns[0]);
          vel[1] = vel[1] - ySF * (vdotns * ns[1]);
          vel[2] = vel[2] - ySF * (vdotns * ns[2]);
        }


        ivec3 cellid{(cidib & 0x30) >> 4, (cidib & 0xc) >> 2, cidib & 0x3};
        boundary.detect_and_resolve_collision(blockid, cellid, 0.f, vel);
        velSqr = vel.dot(vel);
        grid_block.val_1d(_1, cidib) = vel[0];
        grid_block.val_1d(_2, cidib) = vel[1];
        grid_block.val_1d(_3, cidib) = vel[2];
      }
      // unsigned activeMask = __ballot_sync(0xffffffff, mv[0] != 0.0f);
      for (int iter = 1; iter % 32; iter <<= 1) {
        float tmp = __shfl_down_sync(activeMask, velSqr, iter, 32);
        if ((threadIdx.x % 32) + iter < 32)
          velSqr = tmp > velSqr ? tmp : velSqr;
      }
      if (velSqr > sh_maxvels[threadIdx.x / 32] && (threadIdx.x % 32) == 0)
        sh_maxvels[threadIdx.x / 32] = velSqr;
    }
  }
  __syncthreads();
  /// various assumptions
  for (int interval = numWarps >> 1; interval > 0; interval >>= 1) {
    if (threadIdx.x < interval) {
      if (sh_maxvels[threadIdx.x + interval] > sh_maxvels[threadIdx.x])
        sh_maxvels[threadIdx.x] = sh_maxvels[threadIdx.x + interval];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicMax(maxVel, sh_maxvels[0]);
}

template <typename Partition, typename Grid>
__global__ void g2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JFluid> pbuffer,
                      ParticleBuffer<material_e::JFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) {
    blockid = blocks[blockIdx.x];
    src_blockno = partition.query(blockid);
  } else {
    if (partition._haloMarks[blockIdx.x])
      return;
    blockid = partition._activeKeys[blockIdx.x];
    src_blockno = blockIdx.x;
  }
  // auto blockid = partition._activeKeys[blockIdx.x];

  for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
    char local_block_id = base / numViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    auto grid_block = grid.ch(_0, blockno);
    int channelid = base % numViPerBlock;
    char c = channelid & 0x3f;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val;
    if (channelid == 0)
      val = grid_block.val_1d(_1, c);
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c);
    else
      val = grid_block.val_1d(_3, c);
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.f;
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          partition
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno = prev_partition._binsts[source_blockno] +
                       source_pidib / g_bin_capacity;
    }
    vec3 pos;
    float J;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
      J = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    vec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    vec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    vec3 vel;
    vel.set(0.f);
    vec9 C;
    C.set(0.f);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          vel += vi * W;
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
        }
    pos += vel * dt;

    J = (1 + (C[0] + C[4] + C[8]) * dt * Dp_inv) * J;
    if (J < 0.1)
      J = 0.1;
    vec9 contrib;
    {
      float voln = J * pbuffer.volume;
      float pressure = (pbuffer.bulk / pbuffer.gamma) * (powf(J, -pbuffer.gamma) - 1.f);
      {
        contrib[0] =
            ((C[0] + C[0]) * Dp_inv * pbuffer.visco - pressure) * voln;
        contrib[1] = (C[1] + C[3]) * Dp_inv * pbuffer.visco * voln;
        contrib[2] = (C[2] + C[6]) * Dp_inv * pbuffer.visco * voln;

        contrib[3] = (C[3] + C[1]) * Dp_inv * pbuffer.visco * voln;
        contrib[4] =
            ((C[4] + C[4]) * Dp_inv * pbuffer.visco - pressure) * voln;
        contrib[5] = (C[5] + C[7]) * Dp_inv * pbuffer.visco * voln;

        contrib[6] = (C[6] + C[2]) * Dp_inv * pbuffer.visco * voln;
        contrib[7] = (C[7] + C[5]) * Dp_inv * pbuffer.visco * voln;
        contrib[8] =
            ((C[8] + C[8]) * Dp_inv * pbuffer.visco - pressure) * voln;
      }
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      {
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = J;
      }
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      partition.add_advection(local_base_index - 1, dirtag, pidib);
    }
    // dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          auto wm = pbuffer.mass * W;
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W);
        }
  }
  __syncthreads();
  /// arena no, channel no, cell no
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    char local_block_id = base / numMViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    // auto grid_block = next_grid.template ch<0>(blockno);
    int channelid = base & (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 0)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
    else if (channelid == 1)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
    else if (channelid == 2)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
    else
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
  }
}

template <typename Partition, typename Grid>
__global__ void g2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::FixedCorotated> pbuffer,
                      ParticleBuffer<material_e::FixedCorotated> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) {
    blockid = blocks[blockIdx.x];
    src_blockno = partition.query(blockid);
  } else {
    if (partition._haloMarks[blockIdx.x])
      return;
    blockid = partition._activeKeys[blockIdx.x];
    src_blockno = blockIdx.x;
  }
  // auto blockid = partition._activeKeys[blockIdx.x];

  for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
    char local_block_id = base / numViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    auto grid_block = grid.ch(_0, blockno);
    int channelid = base % numViPerBlock;
    char c = channelid & 0x3f;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val;
    if (channelid == 0)
      val = grid_block.val_1d(_1, c);
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c);
    else
      val = grid_block.val_1d(_3, c);
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.f;
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          partition
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno = prev_partition._binsts[source_blockno] +
                       source_pidib / g_bin_capacity;
    }
    vec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    vec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    vec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    vec3 vel;
    vel.set(0.f);
    vec9 C;
    C.set(0.f);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          vel += vi * W;
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
        }
    pos += vel * dt;

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.f : 1.f);

    vec9 contrib;
    {
      vec9 F;
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      contrib[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      contrib[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      contrib[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      contrib[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      contrib[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      contrib[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      contrib[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      contrib[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      contrib[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      {
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0];
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1];
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2];
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3];
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4];
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5];
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6];
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7];
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8];
      }
      compute_stress_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                                    F, contrib);
      contrib = (C * pbuffer.mass - contrib * newDt) *Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      partition.add_advection(local_base_index - 1, dirtag, pidib);
    }
    // dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          auto wm = pbuffer.mass * W;
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W);
        }
  }
  __syncthreads();
  /// arena no, channel no, cell no
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    char local_block_id = base / numMViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    // auto grid_block = next_grid.template ch<0>(blockno);
    int channelid = base & (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 0)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
    else if (channelid == 1)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
    else if (channelid == 2)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
    else
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
  }
}

template <typename Partition, typename Grid>
__global__ void g2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Sand> pbuffer,
                      ParticleBuffer<material_e::Sand> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) {
    blockid = blocks[blockIdx.x];
    src_blockno = partition.query(blockid);
  } else {
    if (partition._haloMarks[blockIdx.x])
      return;
    blockid = partition._activeKeys[blockIdx.x];
    src_blockno = blockIdx.x;
  }
  // auto blockid = partition._activeKeys[blockIdx.x];

  for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
    char local_block_id = base / numViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    auto grid_block = grid.ch(_0, blockno);
    int channelid = base % numViPerBlock;
    char c = channelid & 0x3f;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val;
    if (channelid == 0)
      val = grid_block.val_1d(_1, c);
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c);
    else
      val = grid_block.val_1d(_3, c);
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.f;
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          partition
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno = prev_partition._binsts[source_blockno] +
                       source_pidib / g_bin_capacity;
    }
    vec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    vec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    vec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    vec3 vel;
    vel.set(0.f);
    vec9 C;
    C.set(0.f);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline
    
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          vel += vi * W;
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
        }
    pos += vel * dt;

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.f : 1.f);

    vec9 contrib;
    {
      vec9 F;
      float logJp;
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      contrib[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      contrib[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      contrib[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      contrib[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      contrib[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      contrib[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      contrib[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      contrib[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      contrib[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      logJp = source_particle_bin.val(_12, source_pidib % g_bin_capacity);

      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      compute_stress_sand(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                          pbuffer.cohesion, pbuffer.beta, pbuffer.yieldSurface,
                          pbuffer.volumeCorrection, logJp, F, contrib);
      {
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0];
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1];
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2];
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3];
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4];
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5];
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6];
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7];
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8];
        particle_bin.val(_12, pidib % g_bin_capacity) = logJp;
      }

      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      partition.add_advection(local_base_index - 1, dirtag, pidib);
    }
    // dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          auto wm = pbuffer.mass * W;
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W);
        }
  }
  __syncthreads();
  /// arena no, channel no, cell no
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    char local_block_id = base / numMViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    // auto grid_block = next_grid.template ch<0>(blockno);
    int channelid = base & (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 0)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
    else if (channelid == 1)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
    else if (channelid == 2)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
    else
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
  }
}

template <typename Partition, typename Grid>
__global__ void g2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NACC> pbuffer,
                      ParticleBuffer<material_e::NACC> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) {
    blockid = blocks[blockIdx.x];
    src_blockno = partition.query(blockid);
  } else {
    if (partition._haloMarks[blockIdx.x])
      return;
    blockid = partition._activeKeys[blockIdx.x];
    src_blockno = blockIdx.x;
  }
  // auto blockid = partition._activeKeys[blockIdx.x];

  for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
    char local_block_id = base / numViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    auto grid_block = grid.ch(_0, blockno);
    int channelid = base % numViPerBlock;
    char c = channelid & 0x3f;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val;
    if (channelid == 0)
      val = grid_block.val_1d(_1, c);
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c);
    else
      val = grid_block.val_1d(_3, c);
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.f;
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          partition
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno = prev_partition._binsts[source_blockno] +
                       source_pidib / g_bin_capacity;
    }
    vec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    vec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    vec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    vec3 vel;
    vel.set(0.f);
    vec9 C;
    C.set(0.f);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          vel += vi * W;
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
        }
    pos += vel * dt;

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.f : 1.f);

    vec9 contrib;
    {
      vec9 F;
      float logJp;
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      contrib[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      contrib[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      contrib[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      contrib[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      contrib[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      contrib[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      contrib[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      contrib[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      contrib[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      logJp = source_particle_bin.val(_12, source_pidib % g_bin_capacity);

      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      compute_stress_nacc(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                          pbuffer.bm, pbuffer.xi, pbuffer.beta, pbuffer.Msqr,
                          pbuffer.hardeningOn, logJp, F, contrib);
      {
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0];
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1];
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2];
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3];
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4];
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5];
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6];
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7];
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8];
        particle_bin.val(_12, pidib % g_bin_capacity) = logJp;
      }

      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      partition.add_advection(local_base_index - 1, dirtag, pidib);
    }
    // dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          auto wm = pbuffer.mass * W;
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W);
        }
  }
  __syncthreads();
  /// arena no, channel no, cell no
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    char local_block_id = base / numMViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    // auto grid_block = next_grid.template ch<0>(blockno);
    int channelid = base & (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 0)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
    else if (channelid == 1)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
    else if (channelid == 2)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
    else
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
  }
}

template <typename Grid>
__global__ void mark_active_grid_blocks(uint32_t blockCount, const Grid grid,
                                        int *_marks) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  int blockno = idx / g_blockvolume, cellno = idx % g_blockvolume;
  if (blockno >= blockCount)
    return;
  if (grid.ch(_0, blockno).val_1d(_0, cellno) != 0.f)
    _marks[blockno] = 1;
}

__global__ void mark_active_particle_blocks(uint32_t blockCount,
                                            const int *__restrict__ _ppbs,
                                            int *_marks) {
  std::size_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  _marks[blockno] = _ppbs[blockno] > 0 ? 1 : 0;
}

template <typename Partition>
__global__ void
update_partition(uint32_t blockCount, const int *__restrict__ _sourceNos,
                 const Partition partition, Partition next_partition) {
  __shared__ std::size_t sourceNo[1];
  std::size_t blockno = blockIdx.x;
  if (blockno >= blockCount)
    return;
  if (threadIdx.x == 0) {
    sourceNo[0] = _sourceNos[blockno];
    auto sourceBlockid = partition._activeKeys[sourceNo[0]];
    next_partition._activeKeys[blockno] = sourceBlockid;
    next_partition.reinsert(blockno);
    next_partition._ppbs[blockno] = partition._ppbs[sourceNo[0]];
  }
  __syncthreads();

  auto pcnt = next_partition._ppbs[blockno];
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x)
    next_partition._blockbuckets[blockno * g_particle_num_per_block + pidib] =
        partition._blockbuckets[sourceNo[0] * g_particle_num_per_block + pidib];
}

template <typename Partition, typename Grid>
__global__ void copy_selected_grid_blocks(
    const ivec3 *__restrict__ prev_blockids, const Partition partition,
    const int *__restrict__ _marks, Grid prev_grid, Grid grid) {
  auto blockid = prev_blockids[blockIdx.x];
  if (_marks[blockIdx.x]) {
    auto blockno = partition.query(blockid);
    if (blockno == -1)
      return;
    auto sourceblock = prev_grid.ch(_0, blockIdx.x);
    auto targetblock = grid.ch(_0, blockno);
    targetblock.val_1d(_0, threadIdx.x) = sourceblock.val_1d(_0, threadIdx.x);
    targetblock.val_1d(_1, threadIdx.x) = sourceblock.val_1d(_1, threadIdx.x);
    targetblock.val_1d(_2, threadIdx.x) = sourceblock.val_1d(_2, threadIdx.x);
    targetblock.val_1d(_3, threadIdx.x) = sourceblock.val_1d(_3, threadIdx.x);
  }
}

template <typename Partition>
__global__ void check_table(uint32_t blockCount, Partition partition) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  auto blockid = partition._activeKeys[blockno];
  if (partition.query(blockid) != blockno)
    printf("DAMN, partition table is wrong!\n");
}
template <typename Grid> __global__ void sum_grid_mass(Grid grid, float *sum) {
  atomicAdd(sum, grid.ch(_0, blockIdx.x).val_1d(_0, threadIdx.x));
}
__global__ void sum_particle_count(uint32_t count, int *__restrict__ _cnts,
                                   int *sum) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;
  atomicAdd(sum, _cnts[idx]);
}

template <typename Partition>
__global__ void check_partition(uint32_t blockCount, Partition partition) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= blockCount)
    return;
  ivec3 blockid = partition._activeKeys[idx];
  if (blockid[0] == 0 || blockid[1] == 0 || blockid[2] == 0)
    printf("\tDAMN, encountered zero block record\n");
  if (partition.query(blockid) != idx) {
    int id = partition.query(blockid);
    ivec3 bid = partition._activeKeys[id];
    printf("\t\tcheck partition %d, (%d, %d, %d), feedback index %d, (%d, %d, "
           "%d)\n",
           idx, (int)blockid[0], (int)blockid[1], (int)blockid[2], id, bid[0],
           bid[1], bid[2]);
  }
}

template <typename Partition, typename Domain>
__global__ void check_partition_domain(uint32_t blockCount, int did,
                                       Domain const domain,
                                       Partition partition) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= blockCount)
    return;
  ivec3 blockid = partition._activeKeys[idx];
  if (domain.inside(blockid)) {
    printf(
        "%d-th block (%d, %d, %d) is in domain[%d] (%d, %d, %d)-(%d, %d, %d)\n",
        idx, blockid[0], blockid[1], blockid[2], did, domain._min[0],
        domain._min[1], domain._min[2], domain._max[0], domain._max[1],
        domain._max[2]);
  }
}

template <typename Partition, typename ParticleBuffer, typename ParticleArray>
__global__ void retrieve_particle_buffer(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer pbuffer,
                                         ParticleArray parray, int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    auto parid = atomicAdd(_parcnt, 1);
    /// pos
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);
  }
}


/// Functions to retrieve particle positions and attributes (JB)
/// Copies from particle buffer to two particle arrays (device --> device)
/// Depends on material model (JFluid, FixedCorotated, NACC, Sand) 
/// Copy/paste/modify function for new material
template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::JFluid> pbuffer,
                                         ParticleArray parray, 
                                         ParticleArray pattrib,
                                         int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);

    if (1) {
      /// Send attributes (J, P, P - Patm) to pattribs (device --> device)
      float J = source_bin.val(_3, _source_pidib);
      float pressure = (pbuffer.bulk / pbuffer.gamma) * 
        (powf(J, -pbuffer.gamma) - 1.f);       //< Tait-Murnaghan Pressure (Pa)
      pattrib.val(_0, parid) = J;              //< J (V/Vo)
      pattrib.val(_1, parid) = pressure;       //< Pressure (Pa)
      pattrib.val(_2, parid) = (float)pcnt;    //< Particle count for block (#)
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::FixedCorotated> pbuffer,
                                         ParticleArray parray, 
                                         ParticleArray pattrib,
                                         int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);

    if (1) {
      /// Send attributes to pattribs (device --> device) 
      pattrib.val(_0, parid) = 0.f;  
      pattrib.val(_1, parid) = 0.f; 
      pattrib.val(_2, parid) = (float)pcnt;    //< Particle count for block (#)
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::NACC> pbuffer,
                                         ParticleArray parray, 
                                         ParticleArray pattrib,
                                         int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);

    if (1) {
      /// Send attributes to pattribs (device --> device)
      pattrib.val(_0, parid) = 0.f; 
      pattrib.val(_1, parid) = 0.f; 
      pattrib.val(_2, parid) = (float)pcnt;    //< Particle count for block (#)
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::Sand> pbuffer,
                                         ParticleArray parray, 
                                         ParticleArray pattrib,
                                         int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);

    if (1) {
      /// Send attributes to pattribs (device --> device)
      pattrib.val(_0, parid) = 0.f;    
      pattrib.val(_1, parid) = 0.f; 
      pattrib.val(_2, parid) = (float)pcnt;    //< Particle count for block (#)
    }
  }
}


/// Retrieve grid-cells between points a & b from grid-buffer to gridTarget (JB)
template <typename Partition, typename Grid, typename GridTarget>
__global__ void retrieve_selected_grid_cells(
    uint32_t blockCount, const ivec3 *__restrict__ prev_blockids, const Partition partition,
    Grid prev_grid, GridTarget garray,
    float dt, float *forceSum, vec3 point_a, vec3 point_b) {

  auto blockno = blockIdx.x;  //< Block number in partition
  if (1) {
    //auto blockid = prev_blockids[blockno]; //< 3D grid-block index
    auto blockid = partition._activeKeys[blockno];
    if (blockno < blockCount) {
      if (blockno == -1)
        return;

      auto sourceblock = prev_grid.ch(_0, blockno); //< Set grid-block by block index

      // Tolerance layer thickness around designated target space
      float tol = g_dx * 0.0f;

      // Add +1 to each? For point_b ~= point_a...
      ivec3 maxNodes_coord;
      maxNodes_coord[0] = (int)((point_b[0] + tol - point_a[0] + tol) * g_dx_inv + 0.5);
      maxNodes_coord[1] = (int)((point_b[1] + tol - point_a[1] + tol) * g_dx_inv + 0.5);
      maxNodes_coord[2] = (int)((point_b[2] + tol - point_a[2] + tol) * g_dx_inv + 0.5);
      int maxNodes = maxNodes_coord[0] * maxNodes_coord[1] * maxNodes_coord[2];
      if (maxNodes >= g_target_cells) printf("Allocate more space for gridTarget!\n");

      // Loop through cells in grid-block, stride by 32 to avoid thread conflicts
      for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {

        // Grid node coordinate [i,j,k] in grid-block
        int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
        int j = (cidib >> g_blockbits) & g_blockmask;
        int k = cidib & g_blockmask;

        // Grid node position [x,y,z] in entire domain 
        float xc = (4*blockid[0]*g_dx) + (i*g_dx); // + (g_dx/2.f);
        float yc = (4*blockid[1]*g_dx) + (j*g_dx); // + (g_dx/2.f);
        float zc = (4*blockid[2]*g_dx) + (k*g_dx); // + (g_dx/2.f);

        // Exit thread if cell is not inside grid-target +/- tol
        if (xc < point_a[0] - tol || xc > point_b[0] + tol) {
          continue;
        }
        if (yc < point_a[1] - tol || yc > point_b[1] + tol) {
          continue;
        }
        if (zc < point_a[2] - tol || zc > point_b[2] + tol) {
          continue;
        }

        // Unique ID by spatial position of cell in target [0 to g_target_cells-1]
        int node_id;
        node_id = ((int)((xc - point_a[0] + tol) * g_dx_inv + 0.5f) * maxNodes_coord[1] * maxNodes_coord[2]) +
                  ((int)((yc - point_a[1] + tol) * g_dx_inv + 0.5f) * maxNodes_coord[2]) +
                  ((int)((zc - point_a[2] + tol) * g_dx_inv + 0.5f));
        while (garray.val(_3, node_id) != 0.f) {
          node_id += 1;
          if (node_id > g_target_cells) {
            printf("node_id bigger than g_target_cells!");
            break;
          }
        }
        __syncthreads(); // Sync threads in block

        /// Set values in grid-array to specific cell from grid-buffer
        garray.val(_0, node_id) = xc;
        garray.val(_1, node_id) = yc;
        garray.val(_2, node_id) = zc;
        garray.val(_3, node_id) = sourceblock.val(_0, i, j, k);
        garray.val(_4, node_id) = sourceblock.val(_1, i, j, k);
        garray.val(_5, node_id) = sourceblock.val(_2, i, j, k);
        garray.val(_6, node_id) = sourceblock.val(_3, i, j, k);

        __syncthreads(); // Sync threads in block

        /// Set values in grid-array to specific cell from grid-buffer
        float m1  = garray.val(_3, node_id);
        float m2  = m1;
        float m = m1;
        if (m1 > 0.f) {
          m1 = 1.f / m1; //< Invert mass, avoids division operator
        }
        if (m2 > 0.f) {
          m2 = 1.f / m2; //< Invert mass, avoids division operator
        }

        float vx1 = garray.val(_4, node_id) * m1 * g_length;
        float vy1 = garray.val(_5, node_id) * m1 * g_length;
        float vz1 = garray.val(_6, node_id) * m1 * g_length;
        float vx2 = 0.f;
        float vy2 = 0.f;
        float vz2 = 0.f;

        float fx = m * (vx1 - vx2) / dt;
        float fy = m * (vy1 - vy2) / dt;
        float fz = m * (vz1 - vz2) / dt;

        garray.val(_7, node_id) = fx;
        garray.val(_8, node_id) = fy;
        garray.val(_9, node_id) = fz;
        __syncthreads(); // Sync threads in block
        atomicAdd(forceSum, fx);
        __syncthreads(); // Sync threads in block
      }
    }
  }
}

} // namespace mn

#endif