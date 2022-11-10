#ifndef __MGSP_BENCHMARK_CUH_
#define __MGSP_BENCHMARK_CUH_
#include "boundary_condition.cuh"
#include "grid_buffer.cuh"
#include "halo_buffer.cuh"
#include "halo_kernels.cuh"
#include "hash_table.cuh"
#include "mgmpm_kernels.cuh"
#include "particle_buffer.cuh"
#include "fem_buffer.cuh"
#include "settings.h"
#include <MnBase/Concurrency/Concurrency.h>
#include <MnBase/Meta/ControlFlow.h>
#include <MnBase/Meta/TupleMeta.h>
#include <MnBase/Profile/CppTimers.hpp>
#include <MnBase/Profile/CudaTimers.cuh>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <MnSystem/IO/ParticleIO.hpp>
#include <array>
#include <fmt/color.h>
#include <fmt/core.h>
#include <vector>

namespace mn {

struct mgsp_benchmark {
  using streamIdx = Cuda::StreamIndex;
  using eventIdx = Cuda::EventIndex;
  using host_allocator = heap_allocator;
  struct device_allocator { // hide the global one
    void *allocate(std::size_t bytes) {
      void *ret;
      checkCudaErrors(cudaMalloc(&ret, bytes));
      return ret;
    }
    void deallocate(void *p, std::size_t) { checkCudaErrors(cudaFree(p)); }
  };
  struct temp_allocator {
    explicit temp_allocator(int did) : did{did} {}
    void *allocate(std::size_t bytes) {
      return Cuda::ref_cuda_context(did).borrow(bytes);
    }
    void deallocate(void *p, std::size_t) {}
    int did;
  };
  template <std::size_t I> void initParticles() {
    auto &cuDev = Cuda::ref_cuda_context(I);
    cuDev.setContext();
    tmps[I].alloc(config::g_max_active_block);
    for (int copyid = 0; copyid < 2; copyid++) {
      gridBlocks[copyid].emplace_back(device_allocator{});
      partitions[copyid].emplace_back(device_allocator{},
                                      config::g_max_active_block);
    }
    cuDev.syncStream<streamIdx::Compute>();

    elementBins.emplace_back(
          ElementBuffer<config::g_fem_element_list[I]>{device_allocator{}});
    cuDev.syncStream<streamIdx::Compute>();

    inputHaloGridBlocks.emplace_back(g_device_cnt);
    outputHaloGridBlocks.emplace_back(g_device_cnt);
    
    d_vertices[I] = spawn<vertice_array_13_, orphan_signature>(device_allocator{}); //< FEM vertices
    d_elements[I] = spawn<element_array_, orphan_signature>(device_allocator{}); //< FEM elements

    d_gridTarget.emplace_back(
        std::move(GridTarget{spawn<grid_target_, orphan_signature>(
            device_allocator{}, sizeof(std::array<float, config::g_target_attribs>) * config::g_target_cells)}));
    vel0[I][0] = 0.;
    vel0[I][1] = 0.;
    vel0[I][2] = 0.;

    checkedCnts[I][0] = 0;
    checkedCnts[I][1] = 0;
    curNumActiveBlocks[I] = config::g_max_active_block;
    curNumActiveBins[I] = config::g_max_particle_bin;
    element_cnt[I] = config::g_max_fem_element_num;
    vertice_cnt[I] = config::g_max_fem_vertice_num;

    /// Loop through global device count, tail-recursion optimization
    if constexpr (I + 1 < config::g_device_cnt)
      initParticles<I + 1>();
  }
  mgsp_benchmark(PREC l = config::g_length, float dt = 1e-4, int fp = 24, int frames = 60, float g = -9.81f)
      : length{l}, dtDefault{dt}, curTime{0.f}, rollid{0}, curFrame{0}, curStep{0},
        fps{fp}, nframes{frames}, grav{g}, bRunning{true} {
    // data
    // _hostData =
    //     spawn<signed_distance_field_, orphan_signature>(host_allocator{});
    // printf("length %f \n", length);
    // cudaMemcpyToSymbol("length", &length, sizeof(PREC), cudaMemcpyHostToDevice);

    collisionObjs.resize(config::g_device_cnt);
    initParticles<0>();
    fmt::print("GPU[{}] Grid Blocks, Double-Buffered with Padding: gridBlocks[0] size {} bytes -vs- gridBlocks[1] size {} bytes.\n", 0,
               std::decay<decltype(*gridBlocks[0].begin())>::type::size,
               std::decay<decltype(*gridBlocks[1].begin())>::type::size);   
    fmt::print("GPU[{}] Partitions, Double-Buffered with Padding: Partitions[0] size {} bytes -vs- Paritions[1] size {} bytes.\n", 0,
               sizeof(partitions[0]),
               std::decay<decltype(*partitions[1].begin())>::type::base_t::size); 
    // Tasks
    for (int did = 0; did < config::g_device_cnt; ++did) {
      ths[did] = std::thread([this](int did) { this->gpu_worker(did); }, did);
    }
  }
  ~mgsp_benchmark() {
    auto is_empty = [this]() {
      for (int did = 0; did < config::g_device_cnt; ++did)
        if (!jobs[did].empty())
          return false;
      return true;
    };
    do {
      cv_slave.notify_all();
    } while (!is_empty());
    bRunning = false;
    for (auto &th : ths)
      th.join();
  }

  // Initialize particle models. Allow for varied materials on GPUs.
  template <material_e m>
  void initModel(int devid, const std::vector<std::array<PREC, 3>> &model,
                  const mn::vec<PREC, 3> &v0) {
    auto &cuDev = Cuda::ref_cuda_context(devid);
    cuDev.setContext();
    for (int I = 0; I < config::g_device_cnt; I++){
      // Double-buffering for G2P2G
      for (int copyid = 0; copyid < 2; copyid++) {
        particleBins[copyid].emplace_back(ParticleBuffer<m>(
            device_allocator{}));
      }
    }
    fmt::print("GPU[{}] Particle Bins, Double-Buffered with Padding: ParticleBin[0] size {} bytes -vs- ParticleBin[1] size {} bytes.\n", devid,
               match(particleBins[0][devid])([&](auto &pb) { return pb.size; }),
               match(particleBins[1][devid])([&](auto &pb) { return pb.size; }));    
    for (int i = 0; i < 3; ++i) vel0[devid][i] = v0[i]; // Initial velocity for paritcles on GPU

    particles[devid] = spawn<particle_array_, orphan_signature>(device_allocator{});
    pattribs[devid]  = spawn<particle_array_, orphan_signature>(device_allocator{}); //< Particle attributes on device
    cuDev.syncStream<streamIdx::Compute>();

    pcnt[devid] = model.size(); // Number of particles initially
    fmt::print("GPU[{}] Initialize device array with {} particles.\n", devid, pcnt[devid]);
    cudaMemcpyAsync((void *)&particles[devid].val_1d(_0, 0), model.data(),
                    sizeof(std::array<PREC, 3>) * model.size(),
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    //cuDev.syncStream<streamIdx::Compute>();
    cudaMemcpyAsync((void *)&pattribs[devid].val_1d(_0, 0), model.data(),
                    sizeof(std::array<PREC, 3>) * model.size(),
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    // Clear any existing CSV file, set header.
    std::string fn_disp = std::string{"disp_time_series"} + "_target[0]_dev[" + std::to_string(devid) + "].csv";
    dispFile[devid].open (fn_disp, std::ios::out | std::ios::trunc); // Initialize *.csv
    dispFile[devid] << "Time [s]" << "," << "Displacement [m]" << ",\n";
    dispFile[devid].close();
    // Output initial particle positions as BGEO file on disk.
    std::string fn = std::string{"model"} + "_dev[" + std::to_string(devid) +
                     "]_frame[-1].bgeo";
    IO::insert_job([fn, model]() { write_partio<PREC, 3>(fn, model); });
    IO::flush();
  }
  
  // Initialize FEM vertices and elements
  template<std::size_t VERTICE_SIZE, std::size_t ELEMENT_SIZE>
  void initFEM(int devid, const std::vector<std::array<PREC, VERTICE_SIZE>> &h_vertices,
                std::vector<std::array<int, ELEMENT_SIZE>> &h_elements) {
    auto &cuDev = Cuda::ref_cuda_context(devid);
    cuDev.setContext();
    
    vertice_cnt[devid] = h_vertices.size(); // Vertice count
    element_cnt[devid] = h_elements.size(); // Element count
    cuDev.syncStream<streamIdx::Compute>();

    // Set FEM vertices in GPU array
    fmt::print("GPU[{}] Initialize device array with {} vertices.\n", devid, vertice_cnt[devid]);
    cudaMemcpyAsync((void *)&d_vertices[devid].val_1d(_0, 0), h_vertices.data(),
                    sizeof(std::array<PREC, VERTICE_SIZE>) * vertice_cnt[devid],
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    // Set FEM elements in GPU array
    fmt::print("GPU[{}] Initialize FEM elements with {} ID arrays\n", devid, element_cnt[devid]);
    cudaMemcpyAsync((void *)&d_elements[devid].val_1d(_0, 0), h_elements.data(),
                    sizeof(std::array<int, ELEMENT_SIZE>) * element_cnt[devid],
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();
  }

  void initBoundary(std::string fn) {
    initFromSignedDistanceFile(fn,
                               vec<std::size_t, 3>{(std::size_t)1024,
                                                   (std::size_t)1024,
                                                   (std::size_t)512},
                               _hostData);
    for (int did = 0; did < config::g_device_cnt; ++did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      cuDev.setContext();
      collisionObjs[did] = SignedDistanceGrid{device_allocator{}};
      collisionObjs[did]->init(_hostData, cuDev.stream_compute());
      cuDev.syncStream<streamIdx::Compute>();
    }
  }

  /// Initialize target from host setting (&h_gridTarget), output as *.bgeo (JB)
  void initGridTarget(int devid,
                      const std::vector<std::array<float, config::g_target_attribs>> &h_gridTarget, 
                      const mn::vec<float, 3> &h_point_a, 
                      const mn::vec<float, 3> &h_point_b, 
                      float freq) {
    auto &cuDev = Cuda::ref_cuda_context(devid);
    cuDev.setContext();
    fmt::print("Entered initGridTarget in mgsp_benchmark.cuh!\n");
    h_target_freq = freq; // Set output frequency [Hz] for target (host)
    flag_gt = 1;
    std::string fn_force = std::string{"force_time_series"} + "_target[0]_dev[" + std::to_string(devid) + "].csv";
    forceFile[devid].open (fn_force, std::ios::out | std::ios::trunc); // Initialize *.csv
    forceFile[devid] << "Time [s]" << "," << "Force [n]" << ",\n";
    forceFile[devid].close();
    // Direction of load-cell measurement
    // {0,1,2,3,4,5,6,7,8,9} <- {x,x-,x+,y,y-,y+,z,z-,z+}
    //h_target_dir = target_dir;

    // Set points a/b (device) for grid-target volume using (host, from JSON)
    for (int d = 0; d < 3; d++){
      d_point_a[d] = h_point_a[d];
      d_point_b[d] = h_point_b[d];
    }
    target_cnt[devid] = h_gridTarget.size(); // Set size

    /// Populate target (device) with data from target (host) (JB)
    cudaMemcpyAsync((void *)&d_gridTarget[devid].val_1d(_0, 0), h_gridTarget.data(),
                    sizeof(std::array<float, config::g_target_attribs>) * h_gridTarget.size(),
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();
    fmt::print("Populated target in initGridTarget mgsp_benchmark.cuh!\n");

    /// Write target data to a *.bgeo output file using Partio  
    std::string fn = std::string{"gridTarget"}  + "_dev[" + std::to_string(devid) + "]_frame[-1].bgeo";
    IO::insert_job([fn, h_gridTarget]() { write_partio_gridTarget<float, config::g_target_attribs>(fn, h_gridTarget); });
    IO::flush();

    fmt::print("Finished initGridTarget in mgsp_benchmark.cuh!\n");
  }

  /// Initialize elevation gauge from host setting (&h_wg_point_a/b)
  void initWaveGauge(int devid,
                      const mn::vec<float, 3> &h_wg_point_a, 
                      const mn::vec<float, 3> &h_wg_point_b, 
                      float wg_freq) {
    auto &cuDev = Cuda::ref_cuda_context(devid);
    cuDev.setContext();
    fmt::print("Entered initWaveGauge in mgsp_benchmark.cuh!\n");
    h_wg_freq = wg_freq; // Set output frequency for wave-gauge (host)
    d_wg_point_a.emplace_back();
    d_wg_point_b.emplace_back();
    // Set points a/b (device) for wave-gauge (host, from JSON)
    for (int d = 0; d < 3; d++){
      d_wg_point_a.back()[d] = h_wg_point_a[d];
      d_wg_point_b.back()[d] = h_wg_point_b[d];
    }
    cuDev.syncStream<streamIdx::Compute>();

    fmt::print("Exiting initWaveGauge in mgsp_benchmark.cuh!\n");
  }

  /// Initialize basic boundaries on the grid
  void initGridBoundaries(int devid,
                      const mn::vec<float, 7> &h_walls, 
                      const mn::vec<float, 7> &h_boxes) {
    auto &cuDev = Cuda::ref_cuda_context(devid);
    cuDev.setContext();
    fmt::print("Entered initGridBoundaries in mgsp_benchmark.cuh!\n");
    walls.emplace_back(h_walls);
    boxes.emplace_back(h_boxes);
    cuDev.syncStream<streamIdx::Compute>();

    fmt::print("Exiting initGridBoundaries!\n");
  }

  /// Init OSU wave-maker on device (d_waveMaker) from host (&h_waveMaker) (JB)
  void initWaveMaker(int devid,
                     const std::vector<std::array<float, 3>> &waveMaker) {
    auto &cuDev = Cuda::ref_cuda_context(devid);
    cuDev.setContext();
    fmt::print("Just entered initWaveMaker!\n");
    h_waveMaker = waveMaker;
    /// Populate wave-maker (device) with data from wave-maker (host) (JB)
    for (int d = 0; d < 3; d++) d_waveMaker[d] = (float)h_waveMaker[0][d]; //< Set vals
    fmt::print("Init waveMaker with time {}s, disp {}m, vel {}m/s\n", d_waveMaker[0], d_waveMaker[1], d_waveMaker[2]);
  }  
  
  /// Set OSU wave-maker on device (d_waveMaker) by host (&h_waveMaker) (JB)
  void setWaveMaker(int devid,
                    std::vector<std::array<float, 3>> &h_waveMaker,
                    float curTime) {
    auto &cuDev = Cuda::ref_cuda_context(devid);
    cuDev.setContext();
    //float wm_dt = (float)h_waveMaker[1][0] - (float)h_waveMaker[0][0]; //< Wave-maker time-step
    float wm_dt = 1.f/120.f;
    int step = (int)(curTime / wm_dt); //< Index for time
    if (step >= h_waveMaker.size()) step = h_waveMaker.size() - 1; //< Index-limit
    for (int d = 0; d < 3; d++) d_waveMaker[d] = (float)h_waveMaker[step][d]; //< Set vals
    fmt::print("Set waveMaker with step {}, dt {}s, time {}s, disp {}m, vel {}m/s\n", step, wm_dt, d_waveMaker[0], d_waveMaker[1], d_waveMaker[2]);
  }

  // vol is ppc here
  void updateJFluidParameters(int did, PREC rho, PREC ppc, PREC bulk, PREC gamma,
                              PREC visco,
                              bool ASFLIP, bool FEM, bool FBAR, std::vector<std::string> names) {
    match(particleBins[0][did])([&](auto &pb) {},
                                  [&](ParticleBuffer<material_e::JFluid> &pb) {
                                    pb.updateParameters(length, rho, ppc, bulk, gamma,
                                                        visco,
                                                        ASFLIP, FEM, FBAR);
                                    pb.updateOutputs(names);
                                  });
    match(particleBins[1][did])([&](auto &pb) {},
                                  [&](ParticleBuffer<material_e::JFluid> &pb) {
                                    pb.updateParameters(length, rho, ppc, bulk, gamma,
                                                        visco,
                                                        ASFLIP, FEM, FBAR);
                                    pb.updateOutputs(names);
                                  });
  }

  void updateJFluidASFLIPParameters(int did, PREC rho, PREC ppc, PREC bulk, PREC gamma,
                              PREC visco,
                              PREC a, PREC bmin, PREC bmax,
                              bool ASFLIP, bool FEM, bool FBAR, std::vector<std::string> names) {
    match(particleBins[0][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::JFluid_ASFLIP> &pb) {
          pb.updateParameters(length, rho, ppc, bulk, gamma,
                              visco, a, bmin, bmax,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
    match(particleBins[1][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::JFluid_ASFLIP> &pb) {
          pb.updateParameters(length, rho, ppc, bulk, gamma,
                              visco, a, bmin, bmax,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
  }

  void updateJBarFluidParameters(int did, PREC rho, PREC ppc, PREC bulk, PREC gamma,
                              PREC visco,
                              PREC a, PREC bmin, PREC bmax,
                              bool ASFLIP, bool FEM, bool FBAR, std::vector<std::string> names) {
    match(particleBins[0][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::JBarFluid> &pb) {
          pb.updateParameters(length, rho, ppc, bulk, gamma,
                              visco, a, bmin, bmax,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
    match(particleBins[1][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::JBarFluid> &pb) {
          pb.updateParameters(length, rho, ppc, bulk, gamma,
                              visco, a, bmin, bmax,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
  }
  // void updateJBarFluidOutput(int did, std::vector<std::string> names) {
  //   output_attribs[did] = names;
  //   match(particleBins[0][did])(
  //       [&](auto &pb) {},
  //       [&](ParticleBuffer<material_e::JBarFluid> &pb) {
  //         pb.updateOutputs(names);
  //       });
  //   match(particleBins[1][did])(
  //       [&](auto &pb) {},
  //       [&](ParticleBuffer<material_e::JBarFluid> &pb) {
  //         pb.updateOutputs(names);
  //       });
  // }

  void updateFRParameters(int did, PREC rho, PREC ppc, PREC ym, PREC pr,
                          bool ASFLIP, bool FEM, bool FBAR, std::vector<std::string> names) {
    match(particleBins[0][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::FixedCorotated> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
    match(particleBins[1][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::FixedCorotated> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
  }

  void updateFRASFLIPParameters(int did, PREC rho, PREC ppc, PREC ym, PREC pr,
                                PREC a, PREC bmin, PREC bmax,
                                bool ASFLIP, bool FEM, bool FBAR, std::vector<std::string> names) {
    match(particleBins[0][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::FixedCorotated_ASFLIP> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr, a, bmin, bmax,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
    match(particleBins[1][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::FixedCorotated_ASFLIP> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr, a, bmin, bmax,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
  }
  void updateSandParameters(int did, PREC rho, PREC ppc, PREC ym, PREC pr,
                            bool ASFLIP, bool FEM, bool FBAR, std::vector<std::string> names) {
    match(particleBins[0][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::Sand> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
    match(particleBins[1][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::Sand> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
  }
  void updateNACCParameters(int did, PREC rho, PREC ppc, PREC ym, PREC pr,
                            PREC beta, PREC xi,
                            bool ASFLIP, bool FEM, bool FBAR, std::vector<std::string> names) {
    match(particleBins[0][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::NACC> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr, beta,
                              xi,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
    match(particleBins[1][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::NACC> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr, beta,
                              xi,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
  }
  void updateMeshedParameters(int did, PREC rho, PREC ppc, PREC ym, PREC pr,
                              PREC a, PREC bmin, PREC bmax,
                              bool ASFLIP, bool FEM, bool FBAR, std::vector<std::string> names) {
    match(particleBins[0][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::Meshed> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr, a, bmin, bmax,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
    match(particleBins[1][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::Meshed> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr, a, bmin, bmax,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
    match(elementBins[did])(
        [&](auto &eb) {},
        [&](ElementBuffer<fem_e::Tetrahedron_FBar> &eb) {
          eb.updateParameters(length, rho, ppc, ym, pr);
        });    
  }



  template <typename CudaContext>
  void exclScan(std::size_t cnt, int const *const in, int *out,
                CudaContext &cuDev) {
#if 1
    auto policy = thrust::cuda::par.on((cudaStream_t)cuDev.stream_compute());
    thrust::exclusive_scan(policy, getDevicePtr(in), getDevicePtr(in) + cnt,
                           getDevicePtr(out));
#else
    std::size_t temp_storage_bytes = 0;
    auto plus_op = [] __device__(const int &a, const int &b) { return a + b; };
    checkCudaErrors(cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes,
                                                   in, out, plus_op, 0, cnt,
                                                   cuDev.stream_compute()));
    void *d_tmp = tmps[cuDev.getDevId()].d_tmp;
    checkCudaErrors(cub::DeviceScan::ExclusiveScan(d_tmp, temp_storage_bytes,
                                                   in, out, plus_op, 0, cnt,
                                                   cuDev.stream_compute()));
#endif
  }
  // Used to set initial mass, momentum on grid
  PREC getMass(int did) {
    return match(particleBins[rollid][did])(
        [&](const auto &particleBuffer) { return (PREC)particleBuffer.mass; });
  }
  // Used to set initial volume on grid (Simple FBar Method)
  PREC getVolume(int did) {
    return match(particleBins[rollid][did])(
        [&](const auto &particleBuffer) { return (PREC)particleBuffer.volume; });
  }
  int getWaveGaugeCnt() const noexcept {return (int)d_wg_point_a.size();}

  void checkCapacity(int did) {
    if (ebcnt[did] > curNumActiveBlocks[did] * 4 / 5 &&
        checkedCnts[did][0] == 0) {
      curNumActiveBlocks[did] = curNumActiveBlocks[did] * 5 / 4;
      checkedCnts[did][0] = 2;
      fmt::print(fmt::emphasis::bold, "resizing blocks {} -> {}\n", ebcnt[did],
                 curNumActiveBlocks[did]);
    }
    if (bincnt[did] > curNumActiveBins[did] * 5 / 6 &&
        checkedCnts[did][1] == 0) {
      curNumActiveBins[did] = curNumActiveBins[did] * 6 / 5;
      checkedCnts[did][1] = 2;
      fmt::print(fmt::emphasis::bold, "resizing bins {} -> {}\n", bincnt[did],
                 curNumActiveBins[did]);
    }
  }
  /// thread local ctrl flow
  void gpu_worker(int did) {
    auto wait = [did, this]() {
      std::unique_lock<std::mutex> lk{this->mut_slave};
      this->cv_slave.wait(lk, [did, this]() {
        return !this->bRunning || !this->jobs[did].empty();
      });
    };
    auto signal = [this]() {
      std::unique_lock<std::mutex> lk{this->mut_ctrl};
      this->idleCnt.fetch_add(1);
      lk.unlock();
      this->cv_ctrl.notify_one();
    };
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    fmt::print(fg(fmt::color::light_blue),
               "{}-th gpu worker operates on GPU {}\n", did, cuDev.getDevId());
    while (this->bRunning) {
      wait();
      auto job = this->jobs[did].try_pop();
      if (job)
        (*job)(did);
      signal();
    }
    fmt::print(fg(fmt::color::light_blue), "{}-th gpu worker exits\n", did);
  }
  void sync() {
    std::unique_lock<std::mutex> lk{mut_ctrl};
    cv_ctrl.wait(lk,
                 [this]() { return this->idleCnt == config::g_device_cnt; });
    fmt::print(fmt::emphasis::bold,
               "-----------------------------------------------------------"
               "-----\n");
  }
  void issue(std::function<void(int)> job) {
    std::unique_lock<std::mutex> lk{mut_slave};
    for (int did = 0; did < config::g_device_cnt; ++did)
      jobs[did].push(job);
    idleCnt = 0;
    lk.unlock();
    cv_slave.notify_all();
  }
  void main_loop() {
    /// initial
    nextTime = 1.f / fps;
    dt = compute_dt(0.f, curTime, nextTime, dtDefault);
    fmt::print(fmt::emphasis::bold, "{} --{}--> {}, defaultDt: {}\n", curTime,
               dt, nextTime, dtDefault);
    gt_freq_step = 0;
    wg_freq_step = 0;
    curFrame = 0;
    initial_setup();
    curTime = dt;
    for (curFrame = 1; curFrame <= nframes; ++curFrame) {
      for (; curTime < nextTime; curTime += dt, curStep++) {
        /// max grid vel
        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          /// check capacity
          checkCapacity(did);
          float *d_maxVel = tmps[did].d_maxVel;
          CudaTimer timer{cuDev.stream_compute()};
          timer.tick();
          checkCudaErrors(cudaMemsetAsync(d_maxVel, 0, sizeof(float),
                                          cuDev.stream_compute()));
          //setWaveMaker(did, h_waveMaker, curTime); //< Update d_waveMaker for time
          if (collisionObjs[did])
            cuDev.compute_launch(
                {(nbcnt[did] + g_num_grid_blocks_per_cuda_block - 1) /
                     g_num_grid_blocks_per_cuda_block,
                 g_num_warps_per_cuda_block * 32, g_num_warps_per_cuda_block},
                update_grid_velocity_query_max, (uint32_t)nbcnt[did],
                gridBlocks[0][did], partitions[rollid][did], dt,
                (const SignedDistanceGrid)(*collisionObjs[did]), d_maxVel, curTime, grav);
          else
            cuDev.compute_launch(
                {(nbcnt[did] + g_num_grid_blocks_per_cuda_block - 1) /
                     g_num_grid_blocks_per_cuda_block,
                 g_num_warps_per_cuda_block * 32, g_num_warps_per_cuda_block},
                update_grid_velocity_query_max, (uint32_t)nbcnt[did],
                gridBlocks[0][did], partitions[rollid][did], dt, d_maxVel, curTime, grav, walls[0], boxes[0], length);
          checkCudaErrors(cudaMemcpyAsync(&maxVels[did], d_maxVel,
                                          sizeof(float), cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          timer.tock(fmt::format("GPU[{}] frame {} step {} grid_update_query",
                                 did, curFrame, curStep));
        });
        sync();

        /// host: compute maxvel & next dt
        float maxVel = 0.f;
        for (int did = 0; did < g_device_cnt; ++did)
          if (maxVels[did] > maxVel)
            maxVel = maxVels[did];
        maxVel = std::sqrt(maxVel);
        //nextDt = compute_dt(maxVel, curTime, nextTime, dtDefault);
        nextDt = dtDefault;
        fmt::print(fmt::emphasis::bold,
                   "{} [s] --{}--> {} [s], defaultDt: {} [s], maxVel: {} [m/s]\n", curTime,
                   nextDt, nextTime, dtDefault, (maxVel*length));

        /// g2p2g
        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          CudaTimer timer{cuDev.stream_compute()};

          /// check capacity
          if (checkedCnts[did][1] > 0) {
            match(particleBins[rollid ^ 1][did])([&](auto &pb) {
              pb.resize(device_allocator{}, curNumActiveBins[did]);
            });
            checkedCnts[did][1]--;
          }

          timer.tick();
          // grid
          gridBlocks[1][did].reset(nbcnt[did], cuDev);
          // adv map
          checkCudaErrors(
              cudaMemsetAsync(partitions[rollid][did]._ppcs, 0,
                              sizeof(int) * ebcnt[did] * g_blockvolume,
                              cuDev.stream_compute()));
          // g2p2g 
          {
            match(particleBins[rollid][did])([&](const auto &pb) {
              if (pb.use_ASFLIP == true  &&
                  pb.use_FEM    == false && 
                  pb.use_FBAR   == false) {
                if (partitions[rollid][did].h_count)
                  cuDev.compute_launch(
                      {partitions[rollid][did].h_count, 128,
                      (512 * 6 * 4) + (512 * 7 * 4)},
                      g2p2g, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(
                          particleBins[rollid ^ 1][did]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did],
                      d_vertices[did]);
                }
            });
            cuDev.syncStream<streamIdx::Compute>();
          }
                                 
          // Grid-to-Particle - F-Bar Update - Particle-to-Grid
          {
            timer.tick();
            match(particleBins[rollid][did])([&](const auto &pb) {
            if (pb.use_ASFLIP == true  &&
                pb.use_FEM    == false && 
                pb.use_FBAR   == true) {
                  if (partitions[rollid][did].h_count) 
                  cuDev.compute_launch(
                      {partitions[rollid][did].h_count, 128,
                      (512 * 3 * sizeof(PREC_G)) + (512 * 2 * sizeof(PREC_G))},
                      g2p_FBar, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(
                          particleBins[rollid ^ 1][did]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did], length);
                  cuDev.syncStream<streamIdx::Compute>();
                  timer.tock(fmt::format("GPU[{}] frame {} step {} g2p_FBar", did,
                                        curFrame, curStep));
                

                  // timer.tick();
                  // if (partitions[rollid][did].h_count) {
                  //   fmt::print("F-Bar h_count {}", partitions[rollid][did].h_count);
                  //   cuDev.compute_launch(
                  //         {nbcnt[did], config::g_blockvolume},
                  //         update_grid_FBar, (uint32_t)nbcnt[did],
                  //         gridBlocks[0][did], partitions[rollid][did], dt, curTime);
                  // }
                  // cuDev.syncStream<streamIdx::Compute>();
                  // timer.tock(fmt::format("GPU[{}] frame {} step {} update_grid_FBar",
                  //                       did, curFrame, curStep));

                  // p2g_FBar
                  timer.tick();

                  if (partitions[rollid][did].h_count)
                  cuDev.compute_launch(
                      {partitions[rollid][did].h_count, 128,
                      (512 * 8 * sizeof(PREC_G)) + (512 * 7 * sizeof(PREC_G))},
                      p2g_FBar, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(
                          particleBins[rollid ^ 1][did]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did], length);
                  cuDev.syncStream<streamIdx::Compute>();
                  timer.tock(fmt::format("GPU[{}] frame {} step {} p2g_FBar", did,
                                        curFrame, curStep));
              }
            });
          } //< End F-Bar

          // Grid-to-Vertices - Update FEM - Vertices-to-Grid
          {
            match(particleBins[rollid][did])([&](const auto &pb) {
            if (pb.use_ASFLIP == true  &&
                pb.use_FEM    == true && 
                pb.use_FBAR   == false) {
              
                // g2p2v - Halo
                timer.tick();
                if (partitions[rollid][did].h_count)
                  cuDev.compute_launch(
                      {partitions[rollid][did].h_count, 128,
                      (512 * 6 * 4) + (512 * 7 * 4)},
                      g2p2v, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(
                          particleBins[rollid ^ 1][did]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did],
                      d_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} halo_g2p2v", did,
                                      curFrame, curStep));
            
                // g2p2v - Non-Halo
                timer.tick();
                cuDev.compute_launch(
                    {pbcnt[did], 128, (512 * 6 * 4) + (512 * 7 * 4)}, 
                    g2p2v, dt, nextDt, 
                    (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(
                        particleBins[rollid ^ 1][did]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did],
                    d_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non-halo_g2p2g", did,
                                      curFrame, curStep));

                // v2fem2v - Halo and Non-Halo
                timer.tick();
                match(elementBins[did])([&](const auto &eb) {
                  //if (partitions[rollid][did].h_count)
                    cuDev.compute_launch(
                        {element_cnt[did], 4},
                        v2fem2v, dt, nextDt,
                        d_vertices[did], d_elements[did], eb);
                });
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} v2fem2v", did,
                                      curFrame, curStep));
                // v2p2g - Halo
                timer.tick();
                if (partitions[rollid][did].h_count)
                  cuDev.compute_launch(
                      {partitions[rollid][did].h_count, 128,
                      (512 * 6 * 4) + (512 * 7 * 4)},
                      v2p2g, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(
                          particleBins[rollid ^ 1][did]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did],
                      d_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} halo_v2p2g", did,
                                      curFrame, curStep));
                
              }
            });
          } //< End FEM

          // Grid-to-Vertices - Update FEM + Simple F-Bar - Vertices-to-Grid
          {
            match(particleBins[rollid][did])([&](const auto &pb) {
            if (pb.use_ASFLIP == true  &&
                pb.use_FEM    == true && 
                pb.use_FBAR   == true) {
              
                // g2p2v - F-Bar - Halo
                timer.tick();
                if (partitions[rollid][did].h_count)
                  cuDev.compute_launch(
                      {partitions[rollid][did].h_count, 128, (512 * 6 * 4)},
                      g2p2v_FBar, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(
                          particleBins[rollid ^ 1][did]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did],
                      d_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} halo_g2p2v_FBar", did,
                                      curFrame, curStep));
            
                // g2p2v - F-Bar - Non-Halo
                timer.tick();
                cuDev.compute_launch(
                    {pbcnt[did], 128, (512 * 6 * 4)}, 
                    g2p2v_FBar, dt, nextDt, 
                    (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(
                        particleBins[rollid ^ 1][did]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did],
                    d_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non-halo_g2p2v_FBar", did,
                                      curFrame, curStep));

                // v2fem2v - Halo and Non-Halo
                timer.tick();
                match(elementBins[did])([&](const auto &eb) {
                  //if (partitions[rollid][did].h_count)
                    cuDev.compute_launch(
                        {element_cnt[did], 4},
                        v2fem_FBar, dt, nextDt,
                        d_vertices[did], d_elements[did], eb);
                });
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} v2fem_FBar", did,
                                      curFrame, curStep));

                timer.tick();
                match(elementBins[did])([&](const auto &eb) {
                  //if (partitions[rollid][did].h_count)
                    cuDev.compute_launch(
                        {element_cnt[did], 4},
                        fem2v_FBar, dt, nextDt,
                        d_vertices[did], d_elements[did], eb);
                });
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} fem2v_FBar", did,
                                      curFrame, curStep));
                // v2p2g_FBar - Halo
                timer.tick();
                if (partitions[rollid][did].h_count)
                  cuDev.compute_launch(
                      {partitions[rollid][did].h_count, 128,
                      (512 * 6 * 4) + (512 * 7 * 4)},
                      v2p2g_FBar, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(
                          particleBins[rollid ^ 1][did]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did],
                      d_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} halo_v2p2g_FBar", did,
                                      curFrame, curStep));
                
              }
            });
          } //< End FEM + F-Bar

        });
        sync();

        collect_halo_grid_blocks();

        /// g2p2g - non-Halo
        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          CudaTimer timer{cuDev.stream_compute()};

          timer.tick();
          // Grid-to-Particle-to-Grid - No ASFLIP
          if (0) { 
            match(particleBins[rollid][did])([&](const auto &pb) {
              cuDev.compute_launch(
                  {pbcnt[did], 128, (512 * 3 * 4) + (512 * 4 * 4)}, g2p2g, dt,
                  nextDt, (const ivec3 *)nullptr, pb,
                  get<typename std::decay_t<decltype(pb)>>(
                      particleBins[rollid ^ 1][did]),
                  partitions[rollid ^ 1][did], partitions[rollid][did],
                  gridBlocks[0][did], gridBlocks[1][did],
                  d_vertices[did]);
            });
          }
          // Grid-to-Particle-to-Grid - ASFLIP
          {
            match(particleBins[rollid][did])([&](const auto &pb) {
            if (pb.use_ASFLIP == true  &&
                pb.use_FEM    == false && 
                pb.use_FBAR   == false) {
                // g2p2g - Non-Halo
                timer.tick();
                cuDev.compute_launch(
                    {pbcnt[did], 128, (512 * 6 * 4) + (512 * 7 * 4)}, g2p2g, dt,
                    nextDt, (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(
                        particleBins[rollid ^ 1][did]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did],
                    d_vertices[did]);
                timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_g2p2g", did,
                                      curFrame, curStep));
              }
            });
          } //< End G2P2G (MPM + ASFLIP)
          // Grid-to-Particle - Update F-Bar - Particle-to-Grid 
          {
            match(particleBins[rollid][did])([&](const auto &pb) {
            if (pb.use_ASFLIP == true  &&
                pb.use_FEM    == false && 
                pb.use_FBAR   == true) {
                // g2p F-Bar - Non-Halo
                timer.tick();
                cuDev.compute_launch(
                    {pbcnt[did], 128, (512 * 3 * sizeof(PREC_G)) + (512 * 2 * sizeof(PREC_G))}, 
                    g2p_FBar, dt, nextDt, 
                    (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(
                        particleBins[rollid ^ 1][did]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did], length);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_g2p_FBar", did,
                                      curFrame, curStep));          

                // p2g F-Bar - Non-Halo
                timer.tick();

                cuDev.compute_launch(
                    {pbcnt[did], 128, (512 * 8 * sizeof(PREC_G)) + (512 * 7 * sizeof(PREC_G))}, 
                    p2g_FBar, dt, nextDt, 
                    (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(
                        particleBins[rollid ^ 1][did]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did], length);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_p2g_FBar", did,
                                      curFrame, curStep));
                }
            });
          } //< End Non-Halo F-Bar + ASFLIP

          // Grid-to-Vertices - Update FEM - Vertices-to-Grid
          {
            match(particleBins[rollid][did])([&](const auto &pb) {
            if (pb.use_ASFLIP == true  &&
                pb.use_FEM    == true && 
                pb.use_FBAR   == false) {
                timer.tick();

                cuDev.compute_launch(
                    {pbcnt[did], 128, (512 * 6 * 4) + (512 * 7 * 4)}, v2p2g, dt,
                    nextDt, (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(
                        particleBins[rollid ^ 1][did]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did],
                    d_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_v2p2g", did,
                                      curFrame, curStep));
              }
            });
          } //< End Non-Halo FEM + ASFLIP

          // Grid-to-Vertices - Update FEM - Vertices-to-Grid
          {
            match(particleBins[rollid][did])([&](const auto &pb) {
            if (pb.use_ASFLIP == true  &&
                pb.use_FEM    == true  && 
                pb.use_FBAR   == true) {
                timer.tick();

                cuDev.compute_launch(
                    {pbcnt[did], 128, (512 * 6 * 4) + (512 * 7 * 4)}, 
                    v2p2g_FBar, dt, nextDt, (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(
                        particleBins[rollid ^ 1][did]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did],
                    d_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_v2p2g_FBar", did,
                                      curFrame, curStep));
              }
            });
          } //< End Non-Halo FEM + ASFLIP

          // Resize partition because particles have moved in the domain.
          timer.tick();
          if (checkedCnts[did][0] > 0) {
            partitions[rollid ^ 1][did].resizePartition(
                device_allocator{}, curNumActiveBlocks[did]);
            checkedCnts[did][0]--;
          }
          timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_checkedCnts", did,
                                 curFrame, curStep));
        });


        sync();

        // Each GPU adds all other GPU Halo Grid-Blocks to its own Grid
        reduce_halo_grid_blocks();

        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          CudaTimer timer{cuDev.stream_compute()};
          timer.tick();
          /// Mark particle blocks
          partitions[rollid][did].buildParticleBuckets(cuDev, ebcnt[did]);

          int *activeBlockMarks = tmps[did].activeBlockMarks,
              *destinations = tmps[did].destinations,
              *sources = tmps[did].sources;
          checkCudaErrors(cudaMemsetAsync(activeBlockMarks, 0,
                                          sizeof(int) * nbcnt[did],
                                          cuDev.stream_compute()));
          /// Mark grid blocks
          cuDev.compute_launch({(nbcnt[did] * g_blockvolume + 127) / 128, 128},
                               mark_active_grid_blocks, (uint32_t)nbcnt[did],
                               gridBlocks[1][did], activeBlockMarks);
          cuDev.compute_launch({(ebcnt[did] + 1 + 127) / 128, 128},
                               mark_active_particle_blocks, ebcnt[did] + 1,
                               partitions[rollid][did]._ppbs, sources);
          exclScan(ebcnt[did] + 1, sources, destinations, cuDev);
          /// Building new partition
          // Block count
          checkCudaErrors(cudaMemcpyAsync(
              partitions[rollid ^ 1][did]._cnt, destinations + ebcnt[did],
              sizeof(int), cudaMemcpyDefault, cuDev.stream_compute()));
          checkCudaErrors(cudaMemcpyAsync(
              &pbcnt[did], destinations + ebcnt[did], sizeof(int),
              cudaMemcpyDefault, cuDev.stream_compute()));
          cuDev.compute_launch({(ebcnt[did] + 255) / 256, 256},
                               exclusive_scan_inverse, ebcnt[did],
                               (const int *)destinations, sources);
          // Partition index-table, activeKeys, ppb, buckets
          partitions[rollid ^ 1][did].resetTable(cuDev.stream_compute());
          cuDev.syncStream<streamIdx::Compute>();
          cuDev.compute_launch({pbcnt[did], 128}, update_partition,
                               (uint32_t)pbcnt[did], (const int *)sources,
                               partitions[rollid][did],
                               partitions[rollid ^ 1][did]);
          // Partition binsts
          {
            int *binpbs = tmps[did].binpbs;
            cuDev.compute_launch({(pbcnt[did] + 1 + 127) / 128, 128},
                                 compute_bin_capacity, pbcnt[did] + 1,
                                 (const int *)partitions[rollid ^ 1][did]._ppbs,
                                 binpbs);
            exclScan(pbcnt[did] + 1, binpbs,
                     partitions[rollid ^ 1][did]._binsts, cuDev);
            checkCudaErrors(cudaMemcpyAsync(
                &bincnt[did], partitions[rollid ^ 1][did]._binsts + pbcnt[did],
                sizeof(int), cudaMemcpyDefault, cuDev.stream_compute()));
            cuDev.syncStream<streamIdx::Compute>();
          }
          timer.tock(fmt::format("GPU[{}] frame {} step {} update_partition",
                                 did, curFrame, curStep));

          /// Register neighboring blocks
          timer.tick();
          cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
                               register_neighbor_blocks, (uint32_t)pbcnt[did],
                               partitions[rollid ^ 1][did]);
          auto prev_nbcnt = nbcnt[did]; //< Previous neighbor block count 
          checkCudaErrors(cudaMemcpyAsync(
              &nbcnt[did], partitions[rollid ^ 1][did]._cnt, sizeof(int),
              cudaMemcpyDefault, cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();
          timer.tock(
              fmt::format("GPU[{}] frame {} step {} build_partition_for_grid",
                          did, curFrame, curStep));

          /// Grid check capacity, resize if needed
          if (checkedCnts[did][0] > 0) {
            gridBlocks[0][did].resize(device_allocator{},
                                      curNumActiveBlocks[did]);
          }
          /// Rearrange grid blocks
          timer.tick();
          gridBlocks[0][did].reset(ebcnt[did], cuDev);
          cuDev.compute_launch(
              {prev_nbcnt, g_blockvolume}, copy_selected_grid_blocks,
              (const ivec3 *)partitions[rollid][did]._activeKeys,
              partitions[rollid ^ 1][did], (const int *)activeBlockMarks,
              gridBlocks[1][did], gridBlocks[0][did]);
          cuDev.syncStream<streamIdx::Compute>();
          timer.tock(fmt::format("GPU[{}] frame {} step {} copy_grid_blocks",
                                 did, curFrame, curStep));
          /// Check capacity, resize if needed.
          if (checkedCnts[did][0] > 0) {
            gridBlocks[1][did].resize(device_allocator{},
                                      curNumActiveBlocks[did]);
            tmps[did].resize(curNumActiveBlocks[did]);
          }
        });
        sync();

        /// Tag grid-blocks as Halo if they overlap other GPUs appropiately
        halo_tagging();

        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          CudaTimer timer{cuDev.stream_compute()};

          timer.tick();
          /// Register exterior blocks (all blocks surrounding particle blocks)
          cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
                               register_exterior_blocks, (uint32_t)pbcnt[did],
                               partitions[rollid ^ 1][did]);
          checkCudaErrors(cudaMemcpyAsync(
              &ebcnt[did], partitions[rollid ^ 1][did]._cnt, sizeof(int),
              cudaMemcpyDefault, cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();
          fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
                     "Block count on device {}: Particle {}, Neighbor {}, Exterior {}, Active [{}]; Bins {}, Active Bins [{}]\n",
                     did, pbcnt[did], nbcnt[did], ebcnt[did],
                     curNumActiveBlocks[did], bincnt[did],
                     curNumActiveBins[did]);
          timer.tock(fmt::format(
              "GPU[{}] frame {} step {} build_partition_for_particles", did,
              curFrame, curStep));
        });
        sync();
        rollid ^= 1;
        dt = nextDt; 
        // Output gridTarget
        if (flag_gt) { 
          // Check if end of frame or output frequency
          if (fmod(curTime, (1.f/h_target_freq)) < dt || curTime + dt >= nextTime) {
            issue([this](int did) {
              IO::flush();    // Clear IO
              output_gridcell_target(did); // Output gridTarget as *.bgeo
            });
            sync();
            gt_freq_step += 1; // Iterate freq_step           
          }
        } //< End-of Grid-Target output

      } //< End of time-step, re-loop or go next line end of frame
      // Output particle models
      issue([this](int did) {
        IO::flush();
        output_model(did);
      });
      sync();
      nextTime = 1.f * (curFrame + 1) / fps;
      fmt::print(fmt::emphasis::bold | fg(fmt::color::red),
                 "-----------------------------------------------------------"
                 "-----\n");
    } //< End of frame, reloop or go next line if end of simulation
  } //< End of main simulation loop
  
  void output_model(int did) {
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    CudaTimer timer{cuDev.stream_compute()};
    timer.tick();
    

    int parcnt, *d_parcnt = (int *)cuDev.borrow(sizeof(int));
    checkCudaErrors(
        cudaMemsetAsync(d_parcnt, 0, sizeof(int), cuDev.stream_compute()));

    // Setup forceSum to sum all forces in grid-target in a kernel
    PREC dispVal, *d_dispVal = (PREC *)cuDev.borrow(sizeof(PREC));
    checkCudaErrors(
        cudaMemsetAsync(d_dispVal, 0.0, sizeof(PREC), cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();

    checkCudaErrors(
        cudaMemsetAsync(d_parcnt, 0, sizeof(int), cuDev.stream_compute()));
    match(particleBins[rollid][did])([&](const auto &pb) {
      cuDev.compute_launch({pbcnt[did], 128}, retrieve_particle_buffer_attributes,
                           partitions[rollid][did], partitions[rollid ^ 1][did],
                           pb, particles[did], pattribs[did], d_dispVal, d_parcnt, length);
    });
    checkCudaErrors(cudaMemcpyAsync(&parcnt, d_parcnt, sizeof(int),
                                    cudaMemcpyDefault, cuDev.stream_compute()));
    //cuDev.syncStream<streamIdx::Compute>();
  
    // Copy displacement to host
    checkCudaErrors(cudaMemcpyAsync(&dispVal, d_dispVal, sizeof(PREC),
                                    cudaMemcpyDefault,
                                    cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();
    fmt::print(fg(fmt::color::red), "GPU[{}] Total number of particles: {}\n", did, parcnt);
    fmt::print(fg(fmt::color::red), "GPU[{}] Tracked value of particle ID {} in model: {} \n", did, g_track_ID, dispVal);
  

    if (1) {
      std::string fn_disp = std::string{"disp_time_series"} + "_target[0]_dev[" + std::to_string(did) + "].csv";
      dispFile[did].open (fn_disp, std::ios::out | std::ios::app);
      dispFile[did] << curTime << "," << dispVal << ",\n";
      dispFile[did].close();
      if (verb) fmt::print(fg(fmt::color::red), "GPU[{}] CSV write finished.\n", did);
    }

    models[did].resize(parcnt);
    checkCudaErrors(cudaMemcpyAsync(models[did].data(),
                                    (void *)&particles[did].val_1d(_0, 0),
                                    sizeof(std::array<PREC, 3>) * (parcnt),
                                    cudaMemcpyDefault, cuDev.stream_compute()));
    //cuDev.syncStream<streamIdx::Compute>();

    attribs[did].resize(parcnt);
    checkCudaErrors(cudaMemcpyAsync(attribs[did].data(),
                                    (void *)&pattribs[did].val_1d(_0, 0),
                                    sizeof(std::array<PREC, 3>) * (parcnt),
                                    cudaMemcpyDefault, cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();

    std::string fn = std::string{"model"} + "_dev[" + std::to_string(did) +
                     "]_frame[" + std::to_string(curFrame) + "].bgeo";
    IO::insert_job(
        [fn, m = models[did], a = attribs[did]]() { write_partio_particles<PREC, 3>(fn, m, a); });
    
    timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_particles", did,
                           curFrame, curStep));
  }


  /// Output data from grid blocks (mass, momentum) to *.bgeo (JB)
  void output_gridcell_target(int did) {
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    CudaTimer timer{cuDev.stream_compute()};
    timer.tick();
    if (verb) fmt::print(fg(fmt::color::red), "GPU[{}] Entered output_gridcell_target\n", did);

    int target_cnt, *d_target_cnt = (int *)cuDev.borrow(sizeof(int));
    checkCudaErrors(
        cudaMemsetAsync(d_target_cnt, 0, sizeof(int), 
        cuDev.stream_compute())); /// Reset memory
    target_cnt = config::g_target_cells; // Max number of grid-nodes in target
    cuDev.syncStream<streamIdx::Compute>();

    // Setup forceSum to sum all forces in grid-target in a kernel
    float forceSum, *d_forceSum = (float *)cuDev.borrow(sizeof(float));
    checkCudaErrors(
        cudaMemsetAsync(d_forceSum, 0.f, sizeof(float), cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();

    // Reset gridTarget (device) to zeroes asynchronously
    checkCudaErrors(
        cudaMemsetAsync((void *)&d_gridTarget[did].val_1d(_0, 0), 0,
                        sizeof(std::array<float, config::g_target_attribs>) * (target_cnt),
                        cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();

    fmt::print(fg(fmt::color::red), "GPU[{}] Launch retrieve_selected_grid_cells\n", did);

    /// Copy down-sampled grid value from buffer (device) to target (device)
    cuDev.compute_launch(
              {curNumActiveBlocks[did], 32}, retrieve_selected_grid_cells, 
              (uint32_t)nbcnt[did], partitions[rollid][did], 
              gridBlocks[0][did], d_gridTarget[did],
              nextDt, d_forceSum, d_point_a, d_point_b, length);
    cudaGetLastError();
    cuDev.syncStream<streamIdx::Compute>();

    // Copy force summation to host
    checkCudaErrors(cudaMemcpyAsync(&forceSum, d_forceSum, sizeof(float),
                                    cudaMemcpyDefault,
                                    cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();
    fmt::print(fg(fmt::color::red), "GPU[{}] Force summation in gridTarget: {} N\n", did, forceSum);

    // Output gridTarget binary geometry file with x,y,z, forces, etc.
    // Check if it is last step of frame, or initial frame
    if (curTime + dt >= nextTime || curTime == 0) {

      // Asynchronously copy data from target (device) to target (host)
      h_gridTarget[did].resize(target_cnt);
      checkCudaErrors(
          cudaMemcpyAsync(h_gridTarget[did].data(), (void *)&d_gridTarget[did].val_1d(_0, 0),
                          sizeof(std::array<float, config::g_target_attribs>) * (target_cnt),
                          cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
      /// Output to Partio as 'gridTarget_dev[ ]_frame[ ].bgeo'
      std::string fn = std::string{"gridTarget"} + "_dev[" + std::to_string(did) + "]_frame[" + std::to_string(curFrame) + "].bgeo";
      IO::insert_job([fn, m = h_gridTarget[did]]() { write_partio_gridTarget<float, config::g_target_attribs>(fn, m); });
      if (verb) fmt::print(fg(fmt::color::red), "GPU[{}] gridTarget outputted.\n", did);
    }

    if (fmod(curTime, (1.f/h_target_freq)) < dt) {
      std::string fn = std::string{"force_time_series"} + "_target[0]_dev[" + std::to_string(did) + "].csv";
      forceFile[did].open (fn, std::ios::out | std::ios::app);
      forceFile[did] << curTime << "," << forceSum << ",\n";
      forceFile[did].close();
    }
    timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_cells", did,
                           curFrame, curStep));
  }

  /// Output data from grid blocks (mass, momentum) to *.bgeo (JB)
  void output_wave_gauge(int did) {
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    CudaTimer timer{cuDev.stream_compute()};
    timer.tick();

    // Iterate through all wave-gauges
    for (int i = 0; i < getWaveGaugeCnt(); ++i){

      // Setup forceSum to sum all forces in grid-target in a kernel
      float waveMax, *d_waveMax = (float *)cuDev.borrow(sizeof(float));
      checkCudaErrors(
          cudaMemsetAsync(d_waveMax, 0.f, sizeof(float), cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();

      /// Copy down-sampled grid value from buffer (device) to target (device)
      cuDev.compute_launch(
                {curNumActiveBlocks[did], 32}, retrieve_wave_gauge, 
                (uint32_t)nbcnt[did], partitions[rollid][did], 
                gridBlocks[0][did],
                nextDt, d_waveMax, d_wg_point_a[i], d_wg_point_b[i], length);
      cuDev.syncStream<streamIdx::Compute>();

      // Copy force summation to host
      checkCudaErrors(cudaMemcpyAsync(&waveMax, d_waveMax, sizeof(float),
                                      cudaMemcpyDefault,
                                      cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
      if (verb) fmt::print(fg(fmt::color::red), "GPU[{}] Surface elevation in Wave-Gauge[{}]: {} meters\n", did, i, waveMax);

      {
        std::string fn = "wg[" + std::to_string(i) + "]_dev[" + std::to_string(did) + "].csv";
        std::ofstream waveMaxFile;
        waveMaxFile.open (fn, std::ios::out | std::ios::app);
        waveMaxFile << curTime << "," << waveMax << ",\n";
        waveMaxFile.close();
        if (verb) fmt::print(fg(fmt::color::red), "GPU[{}] Wave-Gauge[{}] CSV write finished.\n", did, i);
      }
    }
    timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_wave_gauge", did,
                           curFrame, curStep));
  }

  void initial_setup() {
    issue([this](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      cuDev.setContext();
      CudaTimer timer{cuDev.stream_compute()};
      timer.tick();
      // Partition activate particle blocks
      cuDev.compute_launch({(pcnt[did] + 255) / 256, 256}, activate_blocks,
                           pcnt[did], particles[did],
                           partitions[rollid ^ 1][did]);
      checkCudaErrors(cudaMemcpyAsync(
          &pbcnt[did], partitions[rollid ^ 1][did]._cnt, sizeof(int),
          cudaMemcpyDefault, cuDev.stream_compute()));
      timer.tock(fmt::format("GPU[{}] step {} init_table", did, curStep));

      timer.tick();
      cuDev.resetMem();
      // Partition particle buckets
      cuDev.compute_launch({(pcnt[did] + 255) / 256, 256},
                           build_particle_cell_buckets, pcnt[did],
                           particles[did], partitions[rollid ^ 1][did]);
      // Partition buckets, binsts
      cuDev.syncStream<streamIdx::Compute>();
      partitions[rollid ^ 1][did].buildParticleBuckets(cuDev, pbcnt[did]);
      {
        int *binpbs = tmps[did].binpbs;
        cuDev.compute_launch({(pbcnt[did] + 1 + 127) / 128, 128},
                             compute_bin_capacity, pbcnt[did] + 1,
                             (const int *)partitions[rollid ^ 1][did]._ppbs,
                             binpbs);
        exclScan(pbcnt[did] + 1, binpbs, partitions[rollid ^ 1][did]._binsts,
                 cuDev);
        checkCudaErrors(cudaMemcpyAsync(
            &bincnt[did], partitions[rollid ^ 1][did]._binsts + pbcnt[did],
            sizeof(int), cudaMemcpyDefault, cuDev.stream_compute()));
        cuDev.syncStream<streamIdx::Compute>();
      }
      // Copy GPU particle data from basic device arrays to Particle Buffer
      match(particleBins[rollid][did])([&](const auto &pb) {
        cuDev.compute_launch({pbcnt[did], 128}, array_to_buffer, particles[did],
                             pb, partitions[rollid ^ 1][did], vel0[did]);
      });
      // FEM Precompute
      match(particleBins[rollid][did])([&](const auto &pb) {
        if (pb.use_FEM == true) {
          // Resize elementBins
          match(elementBins[did])([&](auto &eb) {
            eb.resize(device_allocator{}, element_cnt[did]);
          });
          cuDev.syncStream<streamIdx::Compute>();
          // Precomputation of element variables (e.g. volume)
          match(elementBins[did])([&](auto &eb) {
            cuDev.compute_launch({element_cnt[did], g_fem_element_bin_capacity},
                                fem_precompute, d_vertices[did], d_elements[did], eb);
          });
          cuDev.syncStream<streamIdx::Compute>();
        }
      }); //< End FEM Precompute

      // Register neighbor Grid-Blocks
      cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
                           register_neighbor_blocks, (uint32_t)pbcnt[did],
                           partitions[rollid ^ 1][did]);
      checkCudaErrors(cudaMemcpyAsync(
          &nbcnt[did], partitions[rollid ^ 1][did]._cnt, sizeof(int),
          cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
      // Register exterior Grid-Blocks
      cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
                           register_exterior_blocks, (uint32_t)pbcnt[did],
                           partitions[rollid ^ 1][did]);
      checkCudaErrors(cudaMemcpyAsync(
          &ebcnt[did], partitions[rollid ^ 1][did]._cnt, sizeof(int),
          cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
      timer.tock(fmt::format("GPU[{}] step {} init_partition", did, curStep));

      fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
                 "block count on device {}: {}, {}, {} [{}]; {} [{}]\n", did,
                 pbcnt[did], nbcnt[did], ebcnt[did], curNumActiveBlocks[did],
                 bincnt[did], curNumActiveBins[did]);
    });
    sync();

    halo_tagging(); //< Tag Halo Grid-Blocks (i.e. blocks to send to other GPUs)

    issue([this](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      CudaTimer timer{cuDev.stream_compute()};

      /// Need to copy halo tag info to next partition
      partitions[rollid ^ 1][did].copy_to(partitions[rollid][did], ebcnt[did],
                                          cuDev.stream_compute());
      checkCudaErrors(cudaMemcpyAsync(
          partitions[rollid][did]._activeKeys,
          partitions[rollid ^ 1][did]._activeKeys, sizeof(ivec3) * ebcnt[did],
          cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();

      timer.tick();
      gridBlocks[0][did].reset(nbcnt[did], cuDev); // Zero out blocks on all Grids 0
      // Send initial information from Particle Arrays to Grids 0 (e.g. mass, velocity)
      cuDev.compute_launch({(pcnt[did] + 255) / 256, 256}, rasterize, pcnt[did],
                           particles[did], gridBlocks[0][did],
                           partitions[rollid][did], dt, getMass(did), getVolume(did), vel0[did], length);
      // Initialize advection buckets on Partitions
      cuDev.compute_launch({pbcnt[did], 128}, init_adv_bucket,
                           (const int *)partitions[rollid][did]._ppbs,
                           partitions[rollid][did]._blockbuckets);
      cuDev.syncStream<streamIdx::Compute>();
      timer.tock(fmt::format("GPU[{}] step {} init_grid", did, curStep));
    });
    sync();

    collect_halo_grid_blocks(0); //< 
    reduce_halo_grid_blocks(0);

    // Output Grid-Target Frame 0
    if (flag_gt) {
      issue([this](int did) {
        IO::flush();    // Clear IO
        output_gridcell_target(did); // Output as *.bgeo and summed force *.csv
      });
      sync();
      gt_freq_step += 1; // Iterate counter for Grid-Target          
    }

    // Output particle models frame 0
    issue([this](int did) {
      IO::flush();
      output_model(did);
    });
    sync();
  } //< Finished initial simulation setup. Go back to main-loop.

  void halo_tagging() {
    issue([this](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      cuDev.resetMem();
      for (int otherdid = 0; otherdid < config::g_device_cnt; otherdid++)
        if (otherdid != did)
          haloBlockIds[did][otherdid] =
              (ivec3 *)cuDev.borrow(sizeof(ivec3) * nbcnt[otherdid]);
      /// init halo blockids
      outputHaloGridBlocks[did].initBlocks(temp_allocator{did}, nbcnt[did]);
      inputHaloGridBlocks[did].initBlocks(temp_allocator{did}, nbcnt[did]);
    });
    sync();
    issue([this](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      /// prepare counts
      outputHaloGridBlocks[did].resetCounts(cuDev.stream_compute());
      cuDev.syncStream<streamIdx::Compute>();
      /// sharing local active blocks
      for (int otherdid = 0; otherdid < config::g_device_cnt; otherdid++)
        if (otherdid != did) {
          checkCudaErrors(
              cudaMemcpyAsync(haloBlockIds[otherdid][did],
                              partitions[rollid ^ 1][did]._activeKeys,
                              sizeof(ivec3) * nbcnt[did], cudaMemcpyDefault,
                              cuDev.stream_spare(otherdid)));
          cuDev.spare_event_record(otherdid);
        }
    });
    sync();
    issue([this](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      CudaTimer timer{cuDev.stream_compute()};
      timer.tick();
      /// Initialize overlap marks
      partitions[rollid ^ 1][did].resetOverlapMarks(nbcnt[did],
                                                    cuDev.stream_compute());
      cuDev.syncStream<streamIdx::Compute>();
      /// Receiving active blocks from other GPU devices
      for (int otherdid = 0; otherdid < config::g_device_cnt; otherdid++)
        if (otherdid != did) {
          cuDev.spareStreamWaitForEvent(
              otherdid, Cuda::ref_cuda_context(otherdid).event_spare(did));
          cuDev.spare_launch(otherdid, {(nbcnt[otherdid] + 127) / 128, 128},
                             mark_overlapping_blocks, (uint32_t)nbcnt[otherdid],
                             otherdid,
                             (const ivec3 *)haloBlockIds[did][otherdid],
                             partitions[rollid ^ 1][did],
                             outputHaloGridBlocks[did]._counts + otherdid,
                             outputHaloGridBlocks[did]._buffers[otherdid]);
          cuDev.spare_event_record(otherdid);
          cuDev.computeStreamWaitForEvent(cuDev.event_spare(otherdid));
        }
      /// Self halo particle block
      partitions[rollid ^ 1][did].resetHaloCount(cuDev.stream_compute());
      cuDev.compute_launch(
          {(pbcnt[did] + 127) / 128, 128}, collect_blockids_for_halo_reduction,
          (uint32_t)pbcnt[did], did, partitions[rollid ^ 1][did]);
      /// Retrieve halo-tag counts
      partitions[rollid ^ 1][did].retrieveHaloCount(cuDev.stream_compute());
      outputHaloGridBlocks[did].retrieveCounts(cuDev.stream_compute());
      cuDev.syncStream<streamIdx::Compute>();
      timer.tock(fmt::format("GPU[{}] step {} halo_tagging", did, curStep));

      fmt::print(fg(fmt::color::green), "Halo particle blocks GPU[{}]: {}\n", did,
                 partitions[rollid ^ 1][did].h_count);
      for (int otherdid = 0; otherdid < config::g_device_cnt; otherdid++)
        fmt::print(fg(fmt::color::green), "Halo grid blocks GPU[{}] with GPU[{}]: {}\n", did,
                   otherdid, outputHaloGridBlocks[did].h_counts[otherdid]);
    });
    sync();
  }
  void collect_halo_grid_blocks(int gid = 1) {
    /// init halo grid blocks
    issue([this](int did) {
      std::vector<uint32_t> counts(config::g_device_cnt);
      outputHaloGridBlocks[did].initBuffer(temp_allocator{did},
                                           outputHaloGridBlocks[did].h_counts);
      for (int otherdid = 0; otherdid < config::g_device_cnt; otherdid++)
        counts[otherdid] = (otherdid != did)
                               ? outputHaloGridBlocks[otherdid].h_counts[did]
                               : 0; //< Set count of GPU halo grid-blocks to output, set zero if no other GPUs
      inputHaloGridBlocks[did].initBuffer(temp_allocator{did}, counts);
    });
    sync();
    issue([this, gid](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      CppTimer timer{};
      timer.tick();
      /// Share local GPU active Halo Blocks with neighboring GPUs
      for (int otherdid = 0; otherdid < config::g_device_cnt; otherdid++)
        if (otherdid != did) {
          if (outputHaloGridBlocks[did].h_counts[otherdid] > 0) {
            auto &cnt = outputHaloGridBlocks[did].h_counts[otherdid];
            cuDev.spare_launch(otherdid, {cnt, config::g_blockvolume},
                               collect_grid_blocks, gridBlocks[gid][did],
                               partitions[rollid][did],
                               outputHaloGridBlocks[did]._buffers[otherdid]);
            outputHaloGridBlocks[did].send(inputHaloGridBlocks[otherdid], did,
                                           otherdid,
                                           cuDev.stream_spare(otherdid));
            cuDev.spare_event_record(otherdid);
          } else
            inputHaloGridBlocks[otherdid].h_counts[did] = 0; //< If no other GPU, set input count zero
        }
      timer.tock(
          fmt::format("GPU[{}] step {} collect_send_halo_grid", did, curStep));
    });
    sync();
  }
  void reduce_halo_grid_blocks(int gid = 1) {
    issue([this, gid](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      CppTimer timer{};
      timer.tick();
      /// Receiving active Halo Blocks from other GPU devices, merge with local Blocks
      for (int otherdid = 0; otherdid < config::g_device_cnt; otherdid++)
        if (otherdid != did) {
          if (inputHaloGridBlocks[did].h_counts[otherdid] > 0) {
            cuDev.spareStreamWaitForEvent(
                otherdid, Cuda::ref_cuda_context(otherdid).event_spare(did));
            cuDev.spare_launch(otherdid,
                               {inputHaloGridBlocks[did].h_counts[otherdid],
                                config::g_blockvolume},
                               reduce_grid_blocks, gridBlocks[gid][did],
                               partitions[rollid][did],
                               inputHaloGridBlocks[did]._buffers[otherdid]);
            cuDev.spare_event_record(otherdid);
            cuDev.computeStreamWaitForEvent(cuDev.event_spare(otherdid));
          }
        }
      cuDev.syncStream<streamIdx::Compute>();
      timer.tock(fmt::format("GPU[{}] step {} receive_reduce_halo_grid", did,
                             curStep));
    });
    sync();

  }

  /// Declare variables for 

  /// Simulation basic run-time settings
  float dt, nextDt, dtDefault, curTime, nextTime, maxVel, grav;
  PREC length;
  uint64_t curFrame, curStep, fps, nframes;
  /// Data-structures on GPU device, double-buffering for G2P2G
  std::vector<optional<SignedDistanceGrid>> collisionObjs;
  std::vector<GridBuffer> gridBlocks[2]; //< grid_buffer.cuh
  std::vector<particle_buffer_t> particleBins[2]; //< particle_buffer.cuh
  std::vector<Partition<1>> partitions[2]; ///< with halo info, halo_buffer.cuh
  std::vector<HaloGridBlocks> inputHaloGridBlocks, outputHaloGridBlocks;
  // std::vector<HaloParticleBlocks> inputHaloParticleBlocks,
  // outputHaloParticleBlocks;
  std::vector<element_buffer_t> elementBins;

  pvec3 vel0[config::g_device_cnt]; ///< Set initial velocities per gpu model (JB)
  std::vector<GridTarget> d_gridTarget; ///< Target node structure on device, 7+ f32 (x,y,z, mass, mx,my,mz, fx, fy, fz) (JB)
  vec3 d_point_a; ///< Point A of target (JB)
  vec3 d_point_b; ///< Point B of target (JB)
  std::vector<vec7> walls; ///< 
  std::vector<vec7> boxes; ///< 
  std::vector<vec3> d_wg_point_a; ///< Point A of target (JB)
  std::vector<vec3> d_wg_point_b; ///< Point B of target (JB)
  vec3 d_waveMaker; ///< OSU wm info (time, disp, vel) on device (JB)
  std::vector<std::string> output_attribs[config::g_device_cnt];
  //< Data-structures configured in particle_buffer.cuh
  vec<ParticleArray, config::g_device_cnt> particles; //< Basic GPU vector for Particle positions
  vec<ParticleArray, config::g_device_cnt> pattribs;  //< Basic GPU vector for Particle attributes
  
  //< Data-structures configured in fem_buffer.cuh
  vec<VerticeArray, config::g_device_cnt> d_vertices; //< Basic GOU vector for FEM Vertice positions
  vec<ElementArray, config::g_device_cnt> d_elements; //< Basic GPU vector for FEM Element node IDs

  struct {
    void *base;
    float *d_maxVel;
    int *d_tmp;
    int *activeBlockMarks;
    int *destinations;
    int *sources;
    int *binpbs;
    void alloc(int maxBlockCnt) {
      checkCudaErrors(cudaMalloc(&base, sizeof(int) * (maxBlockCnt * 5 + 1)));
      d_maxVel = (float *)((char *)base + sizeof(int) * maxBlockCnt * 5);
      d_tmp = (int *)((uintptr_t)base);
      activeBlockMarks = (int *)((char *)base + sizeof(int) * maxBlockCnt);
      destinations = (int *)((char *)base + sizeof(int) * maxBlockCnt * 2);
      sources = (int *)((char *)base + sizeof(int) * maxBlockCnt * 3);
      binpbs = (int *)((char *)base + sizeof(int) * maxBlockCnt * 4);
    }
    void dealloc() {
      cudaDeviceSynchronize();
      checkCudaErrors(cudaFree(base));
    }
    void resize(int maxBlockCnt) {
      dealloc();
      alloc(maxBlockCnt);
    }
  } tmps[config::g_device_cnt];
  // halo data
  vec<ivec3 *, config::g_device_cnt, config::g_device_cnt> haloBlockIds;

  /// data on host
  static_assert(std::is_same<GridBufferDomain::index_type, int>::value,
                "block index type is not int");
  char rollid;
  std::size_t curNumActiveBlocks[config::g_device_cnt],
      curNumActiveBins[config::g_device_cnt],
      checkedCnts[config::g_device_cnt][2];
  vec<float, config::g_device_cnt> maxVels;
  vec<int, config::g_device_cnt> pbcnt, nbcnt, ebcnt, bincnt; ///< num blocks
  vec<int, config::g_device_cnt> element_cnt;
  vec<int, config::g_device_cnt> vertice_cnt;
  vec<uint32_t, config::g_device_cnt> pcnt;                   ///< num particles
  vec<uint32_t, config::g_device_cnt> target_cnt; ///< Number of target grid nodes (JB)
  std::vector<float> durations[config::g_device_cnt + 1];
  std::vector<std::array<PREC, 3>> models[config::g_device_cnt];
  std::vector<std::array<PREC, 3>> attribs[config::g_device_cnt];
  std::vector<std::array<float, config::g_target_attribs>> h_gridTarget[config::g_device_cnt];   ///< Grid target info (x,y,z,m,mx,my,mz,fx,fy,fz) on host (JB)
  std::vector<std::array<PREC, 13>> h_vertices[config::g_device_cnt];
  std::vector<std::array<int, 4>> h_elements[config::g_device_cnt];
  std::vector<std::array<float, 3>> h_waveMaker;   ///< wave-maker (time, disp, vel) on host (JB)

  int flag_gt = 0; // 0 if no grid targets, 1 if grid targets to output
  int h_target_dir; // Direction of grid-target load-cell {0,1,2,3,...}<-{x,x-,x+,y,...}
  float h_target_freq = 1.f; // Frequency of grid-target output
  int flag_wg = 0; // 0 if no wave-gauges, 1 if wave-gauges to output
  float h_wg_freq = 1.f; // Frequency of wave-gauge output
  int gt_freq_step; ///< Frequency step for target output (JB)
  int wg_freq_step; ///< Frequency step for wave-gauge output (JB)

  std::ofstream dispFile[mn::config::g_device_cnt];
  std::ofstream forceFile[mn::config::g_device_cnt];
  char verb = 0; //< If true, print more information to terminal

  Instance<signed_distance_field_> _hostData;

  /// control
  bool bRunning;
  threadsafe_queue<std::function<void(int)>> jobs[config::g_device_cnt];
  std::thread ths[config::g_device_cnt]; ///< thread is not trivial
  std::mutex mut_slave, mut_ctrl;
  std::condition_variable cv_slave, cv_ctrl;
  std::atomic_uint idleCnt{0};

  /// computations per substep
  std::vector<std::function<void(int)>> init_tasks;
  std::vector<std::function<void(int)>> loop_tasks;
};

} // namespace mn

#endif