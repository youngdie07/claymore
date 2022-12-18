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
  template <std::size_t GPU_ID> void initParticles() {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    tmps[GPU_ID].alloc(config::g_max_active_block);
    for (int copyid = 0; copyid < 2; copyid++) {
      gridBlocks[copyid].emplace_back(device_allocator{});
      partitions[copyid].emplace_back(device_allocator{},
                                      config::g_max_active_block);
    }
    cuDev.syncStream<streamIdx::Compute>();

    inputHaloGridBlocks.emplace_back(g_device_cnt);
    outputHaloGridBlocks.emplace_back(g_device_cnt);
    
    flag_fem[GPU_ID] = 0;
    element_cnt[GPU_ID] = config::g_max_fem_element_num;
    vertice_cnt[GPU_ID] = config::g_max_fem_vertice_num;
    // device_vertices[GPU_ID] = spawn<vertice_array_13_, orphan_signature>(device_allocator{}); //< FEM vertices
    // device_element_IDs[GPU_ID] = spawn<element_array_, orphan_signature>(device_allocator{}); //< FEM elements

    device_element_attribs[GPU_ID]  = spawn<element_attrib_, orphan_signature>(device_allocator{}); //< Particle attributes on device

    // element_attribs.emplace_back(
    //     std::move(ElementAttrib{spawn<element_attrib_, orphan_signature>(
    //         device_allocator{}, sizeof(std::array<PREC, 6>) * config::g_max_fem_element_num)}));

    // Add/initialize a gridTarget data-structure per GPU within device_gridTarget vector. 
    // Preallocate memory using grid_target_ data-structure (grid_buffer.cuh). Zero-out values.
    device_gridTarget.emplace_back(
        std::move(GridTarget{spawn<grid_target_, orphan_signature>(
            device_allocator{}, sizeof(std::array<PREC_G, config::g_target_attribs>) * config::g_target_cells)}));
    checkCudaErrors(  cudaMemsetAsync((void *)&device_gridTarget[GPU_ID].val_1d(_0, 0), 0,
                        sizeof(std::array<PREC_G, config::g_target_attribs>) * config::g_target_cells,
                        cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();
    
    vel0[GPU_ID][0] = 0.;
    vel0[GPU_ID][1] = 0.;
    vel0[GPU_ID][2] = 0.;

    checkedCnts[GPU_ID][0] = 0;
    checkedCnts[GPU_ID][1] = 0;
    curNumActiveBlocks[GPU_ID] = config::g_max_active_block;
    curNumActiveBins[GPU_ID] = config::g_max_particle_bin;

    // Loop through global device count, tail-recursion optimization
    // "if constexpr" is a C++ 17 feature, can be changed if causing compilation issues.
    if constexpr (GPU_ID + 1 < config::g_device_cnt)
      initParticles<GPU_ID + 1>();
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


    {
      std::string fn_energy = std::string{"grid_energy_time_series.csv"};
      energyFile.open(fn_energy, std::ios::out | std::ios::trunc); // Initialize *.csv
      energyFile << "Time" << "," << "Kinetic_FLIP" <<  "," << "Kinetic_PIC" << ",\n";
      energyFile.close();
    }
    //IO::flush();

    {
      std::string fn_energy = std::string{"particle_energy_time_series.csv"};
      particleEnergyFile.open(fn_energy, std::ios::out | std::ios::trunc); // Initialize *.csv
      particleEnergyFile << "Time" << "," << "Kinetic" << "," << "Gravity" << "," << "Strain" << ",\n";
      particleEnergyFile.close();
    }
    //IO::flush();

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
  void initModel(int GPU_ID, const std::vector<std::array<PREC, 3>> &model,
                  const mn::vec<PREC, 3> &v0) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    for (int gpu = 0; gpu < config::g_device_cnt; gpu++){
      // Double-buffering for G2P2G
      for (int copyid = 0; copyid < 2; copyid++) {
        particleBins[copyid].emplace_back(ParticleBuffer<m>(
            device_allocator{}));
      }
    }
    fmt::print("GPU[{}] Particle Bins, Double-Buffered with Padding: ParticleBin[0] size {} bytes -vs- ParticleBin[1] size {} bytes.\n", GPU_ID,
               match(particleBins[0][GPU_ID])([&](auto &pb) { return pb.size; }),
               match(particleBins[1][GPU_ID])([&](auto &pb) { return pb.size; }));    
    for (int i = 0; i < 3; ++i) vel0[GPU_ID][i] = v0[i]; // Initial velocity for paritcles on GPU

    particles[GPU_ID] = spawn<particle_array_, orphan_signature>(device_allocator{});
    pattribs[GPU_ID]  = spawn<particle_array_, orphan_signature>(device_allocator{}); //< Particle attributes on device

    //particles.emplace_back(ParticleArray<3>(device_allocator{}));
    //pattribs.emplace_back(ParticleAttrib<3>(device_allocator{}));

    cuDev.syncStream<streamIdx::Compute>();

    pcnt[GPU_ID] = model.size(); // Number of particles initially
    fmt::print("GPU[{}] Initialize device array with {} particles.\n", GPU_ID, pcnt[GPU_ID]);
    cudaMemcpyAsync((void *)&particles[GPU_ID].val_1d(_0, 0), model.data(),
                    sizeof(std::array<PREC, 3>) * model.size(),
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    //cuDev.syncStream<streamIdx::Compute>();
    cudaMemcpyAsync((void *)&pattribs[GPU_ID].val_1d(_0, 0), model.data(),
                    sizeof(std::array<PREC, 3>) * model.size(),
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    // Clear any existing CSV file, set header.
    std::string fn_disp = std::string{"disp_time_series"} + "_target[0]_dev[" + std::to_string(GPU_ID) + "].csv";
    dispFile[GPU_ID].open (fn_disp, std::ios::out | std::ios::trunc); // Initialize *.csv
    dispFile[GPU_ID] << "Time [s]" << "," << "Displacement [m]" << ",\n";
    dispFile[GPU_ID].close();
    // Output initial particle positions as BGEO file on disk.
    std::string fn = std::string{"model"} + "_dev[" + std::to_string(GPU_ID) +
                     "]_frame[-1].bgeo";
    IO::insert_job([fn, model]() { write_partio<PREC, 3>(fn, model); });
    IO::flush();
  }
  
  // Initialize FEM vertices and elements
  template<fem_e f>
  void initFEM(int GPU_ID, const std::vector<std::array<PREC, 13>> &input_vertices,
                const std::vector<std::array<int, 4>> &input_elements,
                const std::vector<std::array<PREC, 6>> &input_element_attribs) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    
    flag_fem[GPU_ID] = 1;
    device_vertices[GPU_ID] = spawn<vertice_array_13_, orphan_signature>(device_allocator{}); //< FEM vertices
    device_element_IDs[GPU_ID] = spawn<element_array_, orphan_signature>(device_allocator{}); //< FEM elements
    cuDev.syncStream<streamIdx::Compute>();

    for (int gpu = 0; gpu < config::g_device_cnt; gpu++)
    {
      elementBins.emplace_back(ElementBuffer<f>(device_allocator{}));
    }

    vertice_cnt[GPU_ID] = input_vertices.size(); // Vertice count
    element_cnt[GPU_ID] = input_elements.size(); // Element count
    cuDev.syncStream<streamIdx::Compute>();

    // device_vertices[GPU_ID] = spawn<vertice_array_13_, orphan_signature>(device_allocator{}); //< FEM vertices
    // device_element_IDs[GPU_ID] = spawn<element_array_, orphan_signature>(device_allocator{}); //< FEM elements
    // element_attribs[GPU_ID]  = spawn<element_attrib_, orphan_signature>(device_allocator{}); //< Particle attributes on device
    // cuDev.syncStream<streamIdx::Compute>();

    // Set FEM vertices in GPU array
    fmt::print("GPU[{}] Initialize device array with {} vertices.\n", GPU_ID, vertice_cnt[GPU_ID]);
    cudaMemcpyAsync((void *)&device_vertices[GPU_ID].val_1d(_0, 0), input_vertices.data(),
                    sizeof(std::array<PREC, 13>) * vertice_cnt[GPU_ID],
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    // Set FEM elements in GPU array
    fmt::print("GPU[{}] Initialize FEM elements with {} ID arrays\n", GPU_ID, element_cnt[GPU_ID]);
    cudaMemcpyAsync((void *)&device_element_IDs[GPU_ID].val_1d(_0, 0), input_elements.data(),
                    sizeof(std::array<int, 4>) * element_cnt[GPU_ID],
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    // Set FEM elements in GPU array
    fmt::print("GPU[{}] Initialize FEM element attribs with {} arrays\n", GPU_ID, element_cnt[GPU_ID]);
    cudaMemcpyAsync((void *)&device_element_attribs[GPU_ID].val_1d(_0, 0), input_element_attribs.data(),
                    sizeof(std::array<PREC, 6>) * element_cnt[GPU_ID],
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    host_element_IDs[GPU_ID].resize(element_cnt[GPU_ID]);
    host_element_attribs[GPU_ID].resize(element_cnt[GPU_ID]);

    cuDev.syncStream<streamIdx::Compute>();
    IO::flush();

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

  /// Initialize target from host setting (&input_gridTarget), output as *.bgeo (JB)
  void initGridTarget(int GPU_ID,
                      const std::vector<std::array<float, config::g_target_attribs>> &input_gridTarget, 
                      const mn::vec<float, 7> &host_target,  
                      float freq) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    fmt::print("Entered initGridTarget in mgsp_benchmark.cuh!\n");
    host_target_freq = freq; // Set output frequency [Hz] for target (host)
    flag_gt = 1;
    std::string fn_force = std::string{"force_time_series"} + "_gridTarget["+ std::to_string(number_of_grid_targets)+"]_dev[" + std::to_string(GPU_ID) + "].csv";
    forceFile[GPU_ID].open (fn_force, std::ios::out | std::ios::trunc); // Initialize *.csv
    forceFile[GPU_ID] << "Time [s]" << "," << "Force [n]" << ",\n";
    forceFile[GPU_ID].close();
    // Direction of load-cell measurement
    // {0,1,2,3,4,5,6,7,8,9} <- {x,x-,x+,y,y-,y+,z,z-,z+}

    // Set points a/b (device) for grid-target volume using (host, from JSON)
    if (GPU_ID == 0) 
    {
      number_of_grid_targets += 1; 
      device_target.emplace_back();
      tarcnt.emplace_back();
      for (int d = 0; d < 7; d++)
        device_target.back()[d] = host_target[d];
    }
    tarcnt.back()[GPU_ID] = input_gridTarget.size(); // Set size
    int target_ID = number_of_grid_targets-1;
    //printf("GPU[%d] Number of targets: %d \n", GPU_ID, number_of_grid_targets);
    printf("GPU[%d] Target[%d] node count: %d \n", GPU_ID, target_ID, tarcnt[target_ID][GPU_ID]);


    /// Populate target (device) with data from target (host) (JB)
    cudaMemcpyAsync((void *)&device_gridTarget[GPU_ID].val_1d(_0, 0), input_gridTarget.data(),
                    sizeof(std::array<PREC_G, config::g_target_attribs>) *  input_gridTarget.size(),
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();
    fmt::print("Populated target in initGridTarget mgsp_benchmark.cuh!\n");

    /// Write target data to a *.bgeo output file using Partio  
    std::string fn = std::string{"gridTarget"}  +"[" + std::to_string(target_ID) + "]" + "_dev[" + std::to_string(GPU_ID) + "]_frame[-1].bgeo";
    IO::insert_job([fn, input_gridTarget]() { write_partio_gridTarget<float, config::g_target_attribs>(fn, input_gridTarget); });
    IO::flush();
  }

  /// Initialize elevation gauge from host setting (&host_wg_point_a/b)
  void initWaveGauge(int GPU_ID,
                      const mn::vec<float, 3> &host_wg_point_a, 
                      const mn::vec<float, 3> &host_wg_point_b, 
                      float wg_freq) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    fmt::print("Entered initWaveGauge in mgsp_benchmark.cuh!\n");
    host_wg_freq = wg_freq; // Set output frequency for wave-gauge (host)
    device_wg_point_a.emplace_back();
    device_wg_point_b.emplace_back();
    // Set points a/b (device) for wave-gauge (host, from JSON)
    for (int d = 0; d < 3; d++){
      device_wg_point_a.back()[d] = host_wg_point_a[d];
      device_wg_point_b.back()[d] = host_wg_point_b[d];
    }
    cuDev.syncStream<streamIdx::Compute>();

    fmt::print("Exiting initWaveGauge in mgsp_benchmark.cuh!\n");
  }

  /// Initialize basic boundaries on the grid
  void initGridBoundaries(int GPU_ID,
                      const mn::vec<float, 7> &host_walls, 
                      const mn::vec<float, 7> &host_boxes) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    fmt::print("Entered initGridBoundaries in mgsp_benchmark.cuh!\n");
    walls.emplace_back(host_walls);
    boxes.emplace_back(host_boxes);
    cuDev.syncStream<streamIdx::Compute>();

    fmt::print("Exiting initGridBoundaries!\n");
  }

  /// Init OSU wave-maker on device (device_waveMaker) from host (&host_waveMaker) (JB)
  void initWaveMaker(int GPU_ID,
                     const std::vector<std::array<float, 3>> &waveMaker) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    fmt::print("Just entered initWaveMaker!\n");
    host_waveMaker = waveMaker;
    /// Populate wave-maker (device) with data from wave-maker (host) (JB)
    for (int d = 0; d < 3; d++) device_waveMaker[d] = (float)host_waveMaker[0][d]; //< Set vals
    fmt::print("Init waveMaker with time {}s, disp {}m, vel {}m/s\n", device_waveMaker[0], device_waveMaker[1], device_waveMaker[2]);
  }  
  
  /// Set OSU wave-maker on device (device_waveMaker) by host (&host_waveMaker) (JB)
  void setWaveMaker(int GPU_ID,
                    std::vector<std::array<float, 3>> &host_waveMaker,
                    float curTime) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    //float wm_dt = (float)host_waveMaker[1][0] - (float)host_waveMaker[0][0]; //< Wave-maker time-step
    float wm_dt = 1.f/120.f;
    int step = (int)(curTime / wm_dt); //< Index for time
    if (step >= host_waveMaker.size()) step = host_waveMaker.size() - 1; //< Index-limit
    for (int d = 0; d < 3; d++) device_waveMaker[d] = (float)host_waveMaker[step][d]; //< Set vals
    fmt::print("Set waveMaker with step {}, dt {}s, time {}s, disp {}m, vel {}m/s\n", step, wm_dt, device_waveMaker[0], device_waveMaker[1], device_waveMaker[2]);
  }

  // vol is ppc here
  void updateJFluidParameters(int did, PREC rho, PREC ppc, PREC bulk, PREC gamma,
                              PREC visco,
                              bool ASFLIP, bool FEM, bool FBAR, 
                              std::vector<std::string> names) {
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
                              bool ASFLIP, bool FEM, bool FBAR, 
                              std::vector<std::string> names) {
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
                              bool ASFLIP, bool FEM, bool FBAR, 
                              std::vector<std::string> names) {
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

  void updateFRParameters(int did, PREC rho, PREC ppc, PREC ym, PREC pr,
                          bool ASFLIP, bool FEM, bool FBAR, 
                          std::vector<std::string> names) {
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
  void update_FR_ASFLIP_Parameters(int did, PREC rho, PREC ppc, PREC ym, PREC pr,
                                PREC a, PREC bmin, PREC bmax,
                                bool ASFLIP, bool FEM, bool FBAR, 
                                std::vector<std::string> names) {
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
  void update_FR_ASFLIP_FBAR_Parameters(int did, PREC rho, PREC ppc, PREC ym, PREC pr,
                                PREC a, PREC bmin, PREC bmax,
                                bool ASFLIP, bool FEM, bool FBAR, 
                                std::vector<std::string> names) {
    match(particleBins[0][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr, a, bmin, bmax,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
    match(particleBins[1][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr, a, bmin, bmax,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
  }
  void updateSandParameters(int did, PREC rho, PREC ppc, PREC ym, PREC pr,
                            PREC logJ, PREC friction_angle, PREC c, PREC b, bool volCorrection,
                            PREC a, PREC bmin, PREC bmax,
                            bool ASFLIP, bool FEM, bool FBAR, 
                            std::vector<std::string> names) {
    match(particleBins[0][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::Sand> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr,
                              logJ, friction_angle, c, b, volCorrection,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
    match(particleBins[1][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::Sand> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr,
                              logJ, friction_angle, c, b, volCorrection,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
  }
  void updateNACCParameters(int did, PREC rho, PREC ppc, PREC ym, PREC pr,
                            PREC beta, PREC xi,
                            bool ASFLIP, bool FEM, bool FBAR, 
                            std::vector<std::string> names) {
    match(particleBins[0][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::NACC> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr, beta, xi,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
    match(particleBins[1][did])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::NACC> &pb) {
          pb.updateParameters(length, rho, ppc, ym, pr, beta, xi,
                              ASFLIP, FEM, FBAR);
          pb.updateOutputs(names);
        });
  }

  void updateMeshedParameters(int did, PREC rho, PREC ppc, PREC ym, PREC pr,
                              PREC a, PREC bmin, PREC bmax,
                              bool ASFLIP, bool FEM, bool FBAR, std::vector<std::string> names) 
  {
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
        [&](ElementBuffer<fem_e::Tetrahedron> &eb) {
          eb.updateParameters(length, rho, ppc, ym, pr);
        });   
    std::cout << "Update Meshed parameters." << '\n';
 
  }
  void updateMeshedFBARParameters(int did, PREC rho, PREC ppc, PREC ym, PREC pr,
                              PREC a, PREC bmin, PREC bmax,
                              bool ASFLIP, bool FEM, bool FBAR, std::vector<std::string> names) 
  {    
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
    std::cout << "Update Meshed FBAR parameters." << '\n';

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
    void *device_tmp = tmps[cuDev.getDevId()].device_tmp;
    checkCudaErrors(cub::DeviceScan::ExclusiveScan(device_tmp, temp_storage_bytes,
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
  int getWaveGaugeCnt() const noexcept {return (int)device_wg_point_a.size();}

  void checkCapacity(int did) {
    if (ebcnt[did] > curNumActiveBlocks[did] * 7 / 8 &&
        checkedCnts[did][0] == 0) {
      curNumActiveBlocks[did] = std::max(curNumActiveBlocks[did] * 5 / 4, (std::size_t)ebcnt[did]);
      checkedCnts[did][0] = 2;
      fmt::print(fmt::emphasis::bold, "resizing blocks {} -> {}\n", ebcnt[did],
                 curNumActiveBlocks[did]);
    }
    if (bincnt[did] > curNumActiveBins[did] * 9 / 10 &&
        checkedCnts[did][1] == 0) {
      curNumActiveBins[did] = std::max(curNumActiveBins[did] * 9 / 8, (std::size_t)bincnt[did]);
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
    nextTime = 1.0 / fps; // Initial next time
    //dt = compute_dt(0.f, curTime, nextTime, dtDefault);
    dt = dtDefault;
    fmt::print(fmt::emphasis::bold, "{} --{}--> {}, defaultDt: {}\n", curTime,
               dt, nextTime, dtDefault);
    curFrame = 0;
    initial_setup();
    curTime = dt;
    for (curFrame = 1; curFrame <= nframes; ++curFrame) {
      for (; curTime < nextTime; curTime += dt, curStep++) {
        /// max grid vel
        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          checkCapacity(did);
          PREC_G *device_maxVel = tmps[did].device_maxVel;
          PREC_G *device_kinetic_energy_grid = tmps[did].device_kinetic_energy_grid;
          PREC_G *device_gravity_energy_grid = tmps[did].device_gravity_energy_grid;

          CudaTimer timer{cuDev.stream_compute()};
          timer.tick();
          checkCudaErrors(cudaMemsetAsync(device_maxVel, 0, sizeof(PREC_G),
                                          cuDev.stream_compute()));
          checkCudaErrors(cudaMemsetAsync(device_kinetic_energy_grid, 0, sizeof(PREC_G),
                                          cuDev.stream_compute()));
          checkCudaErrors(cudaMemsetAsync(device_gravity_energy_grid, 0, sizeof(PREC_G),
                                          cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();
          // Grid Update
          if (curStep == 0) dt = dt/2; //< Init. grid vel. update shifted 1/2 dt. Leap-frog time-integration, -> stability/accuracy
          if (collisionObjs[did]) // If using SDF boundaries
            cuDev.compute_launch(
                {(nbcnt[did] + g_num_grid_blocks_per_cuda_block - 1) /
                     g_num_grid_blocks_per_cuda_block,
                 g_num_warps_per_cuda_block * 32, g_num_warps_per_cuda_block * sizeof(PREC_G)},
                update_grid_velocity_query_max, (uint32_t)nbcnt[did],
                gridBlocks[0][did], partitions[rollid][did], dt,
                (const SignedDistanceGrid)(*collisionObjs[did]), device_maxVel, curTime, grav);
          else {// If only using basic geometry boundaries
            cuDev.compute_launch(
                {(nbcnt[did] + g_num_grid_blocks_per_cuda_block - 1) /
                     g_num_grid_blocks_per_cuda_block,
                 g_num_warps_per_cuda_block * 32, g_num_warps_per_cuda_block * sizeof(PREC_G)},
                update_grid_velocity_query_max, (uint32_t)nbcnt[did],
                gridBlocks[0][did], partitions[rollid][did], dt, device_maxVel, curTime, grav, walls[0], boxes[0], length);
            
            cuDev.syncStream<streamIdx::Compute>();

            cuDev.compute_launch(
                {(nbcnt[did] + g_num_grid_blocks_per_cuda_block - 1) /
                     g_num_grid_blocks_per_cuda_block,
                 g_num_warps_per_cuda_block * 32, g_num_warps_per_cuda_block * sizeof(PREC_G)},
                query_energy_grid, (uint32_t)nbcnt[did],
                gridBlocks[0][did], partitions[rollid][did], dt, device_kinetic_energy_grid, device_gravity_energy_grid, curTime, grav, walls[0], boxes[0], length);
            cuDev.syncStream<streamIdx::Compute>();

          }
          cuDev.syncStream<streamIdx::Compute>();

          checkCudaErrors(cudaMemcpyAsync(&maxVels[did], device_maxVel,
                                          sizeof(PREC_G), cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          checkCudaErrors(cudaMemcpyAsync(&kinetic_energy_grid_vals[did], device_kinetic_energy_grid,
                                          sizeof(PREC_G), cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          checkCudaErrors(cudaMemcpyAsync(&gravity_energy_grid_vals[did], device_gravity_energy_grid,
                                          sizeof(PREC_G), cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          timer.tock(fmt::format("GPU[{}] frame {} step {} grid_update_query",
                                 did, curFrame, curStep));
        });
        sync();
        if (curStep == 0) dt = dt*2; //< Use symplectic Euler dt after leap-frog init.

        /// host: compute maxvel & next dt
        PREC_G maxVel = 0.0;
        PREC_G sum_kinetic_energy_grid = 0.0; //< Kinetic energy summed across all GPU grids
        PREC_G sum_gravity_energy_grid = 0.0;

        for (int did = 0; did < g_device_cnt; ++did)
        {
          if (maxVels[did] > maxVel) maxVel = maxVels[did];
          sum_kinetic_energy_grid += kinetic_energy_grid_vals[did];
          sum_gravity_energy_grid += gravity_energy_grid_vals[did];
        }
        maxVel = std::sqrt(maxVel);

        //nextDt = compute_dt(maxVel, curTime, nextTime, dtDefault);
        nextDt = dtDefault;
        fmt::print(fmt::emphasis::bold,
                   "{} [s] --{}--> {} [s], defaultDt: {} [s], maxVel: {} [m/s], kinetic_energy: {} [kg m2/s2]\n", curTime,
                   nextDt, nextTime, dtDefault, (maxVel*length), sum_kinetic_energy_grid);

        /// Run g2p2g, g2p - p2g, or g2p2v-v2fem2v-v2p2g pipeline on each GPU
        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          CudaTimer timer{cuDev.stream_compute()};

          /// Check capacity of particle bins, resize if needed
          if (checkedCnts[did][1] > 0) {
            match(particleBins[rollid ^ 1][did])([&](auto &pb) {
              pb.resize(device_allocator{}, curNumActiveBins[did]);
            });
            checkedCnts[did][1]--;
          }

          timer.tick();
          // Reset values on gridBlocks[1]
          // gridBlocks[0][did].reset_FBar(nbcnt[did], cuDev);
          // cuDev.syncStream<streamIdx::Compute>();

          gridBlocks[1][did].reset(nbcnt[did], cuDev);

          // adv map
          checkCudaErrors(
              cudaMemsetAsync(partitions[rollid][did]._ppcs, 0,
                              sizeof(int) * ebcnt[did] * g_blockvolume,
                              cuDev.stream_compute()));
          // Grid-to-Particle-to-Grid - g2p2g 
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
                      gridBlocks[0][did], gridBlocks[1][did]);
                }
            });
            cuDev.syncStream<streamIdx::Compute>();
          }
                                 
          // Grid-to-Particle - F-Bar Update - Particle-to-Grid
          {
            match(particleBins[rollid][did])([&](const auto &pb) {
            if (pb.use_ASFLIP == true  &&
                pb.use_FEM    == false && 
                pb.use_FBAR   == true) {
                  timer.tick();
                  // g2g_FBar Halo
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
                
                  // p2g_FBar Halo
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
                pb.use_FEM    == true  && 
                pb.use_FBAR   == false) {
              
                // g2p2v - Halo
                timer.tick();
                if (partitions[rollid][did].h_count)
                  cuDev.compute_launch(
                      {partitions[rollid][did].h_count, 128,
                      (512 * 6 * 4)},
                      g2p2v, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(
                          particleBins[rollid ^ 1][did]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did],
                      device_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} halo_g2p2v", did,
                                      curFrame, curStep));
            
                // g2p2v - Non-Halo
                timer.tick();
                cuDev.compute_launch(
                    {pbcnt[did], 128, (512 * 6 * 4)}, 
                    g2p2v, dt, nextDt, 
                    (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(
                        particleBins[rollid ^ 1][did]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did],
                    device_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non-halo_g2p2g", did,
                                      curFrame, curStep));

                // v2fem2v - Halo and Non-Halo
                timer.tick();
                match(elementBins[did])([&](const auto &eb) {
                  cuDev.compute_launch(
                      {(element_cnt[did] - 1) / 32 + 1, 32},
                      v2fem2v, (uint32_t)element_cnt[did], dt, nextDt,
                      device_vertices[did], device_element_IDs[did], eb);
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
                      device_vertices[did]);
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
                pb.use_FEM    == true  && 
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
                      device_vertices[did]);
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
                    device_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non-halo_g2p2v_FBar", did,
                                      curFrame, curStep));

                // v2fem_FBar - Halo and Non-Halo
                timer.tick();
                match(elementBins[did])([&](const auto &eb) {
                    cuDev.compute_launch(
                        {(element_cnt[did] - 1) / 32 + 1, 32},
                        v2fem_FBar, (uint32_t)element_cnt[did], dt, nextDt,
                        device_vertices[did], device_element_IDs[did], eb);
                });
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} v2fem_FBar", did,
                                      curFrame, curStep));
                // fem2v_FBar - Halo and Non-Halo
                timer.tick();
                match(elementBins[did])([&](const auto &eb) {
                    cuDev.compute_launch(
                        {(element_cnt[did] - 1) / 32 + 1, 32},
                        fem2v_FBar, (uint32_t)element_cnt[did], dt, nextDt,
                        device_vertices[did], device_element_IDs[did], eb);
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
                      device_vertices[did]);
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
          {
            match(particleBins[rollid][did])([&](const auto &pb) {
              if (pb.use_ASFLIP == false  &&
                  pb.use_FEM    == false && 
                  pb.use_FBAR   == false) {
              cuDev.compute_launch(
                  {pbcnt[did], 128, (512 * 3 * 4) + (512 * 4 * 4)}, g2p2g, dt,
                  nextDt, (const ivec3 *)nullptr, pb,
                  get<typename std::decay_t<decltype(pb)>>(
                      particleBins[rollid ^ 1][did]),
                  partitions[rollid ^ 1][did], partitions[rollid][did],
                  gridBlocks[0][did], gridBlocks[1][did]);
                }
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
                    gridBlocks[0][did], gridBlocks[1][did]);
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
                    device_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_v2p2g", did,
                                      curFrame, curStep));
              }
            });
          } //< End Non-Halo FEM + ASFLIP

          // Grid-to-Vertices - Update FEM + F-Bar - Vertices-to-Grid
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
                    device_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_v2p2g_FBar", did,
                                      curFrame, curStep));
              }
            });
          } //< End Non-Halo FEM + ASFLIP + F-Bar

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
          if (checkedCnts[did][0] > 0) 
            gridBlocks[0][did].resize(device_allocator{}, curNumActiveBlocks[did]);
          
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
          gridBlocks[0][did].reset_FBar(ebcnt[did], cuDev);
          cuDev.syncStream<streamIdx::Compute>();

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

        // Output summed grid kinetic energy value to *.csv
        if (1)
        {
          if (fmod(curTime, (1.f/host_target_freq)) < dt || curTime + dt >= nextTime)
          {
            std::string fn = std::string{"grid_energy_time_series.csv"};
            energyFile.open(fn, std::ios::out | std::ios::app);
            energyFile << curTime << "," << sum_kinetic_energy_grid<< "," << sum_gravity_energy_grid << ",\n";
            energyFile.close();
          }
        }

        // Output summed grid kinetic energy value to *.csv
        if (1)
        {
          if (fmod(curTime, (1.f/host_target_freq)) < dt || curTime + dt >= nextTime)
          {
            issue([this](int did) {
              IO::flush();
              auto &cuDev = Cuda::ref_cuda_context(did);
              //cuDev.setContext();
              PREC_G *device_kinetic_energy_particles = tmps[did].device_kinetic_energy_particles;
              PREC_G *device_gravity_energy_particles = tmps[did].device_gravity_energy_particles;
              PREC_G *device_strain_energy_particles  = tmps[did].device_strain_energy_particles;

              CudaTimer timer{cuDev.stream_compute()};
              timer.tick();


              checkCudaErrors(cudaMemsetAsync(device_kinetic_energy_particles, 0, sizeof(PREC_G),
                                              cuDev.stream_compute()));
              checkCudaErrors(cudaMemsetAsync(device_gravity_energy_particles, 0, sizeof(PREC_G),
                                              cuDev.stream_compute()));
              checkCudaErrors(cudaMemsetAsync(device_strain_energy_particles, 0, sizeof(PREC_G),
                                              cuDev.stream_compute()));
              cuDev.syncStream<streamIdx::Compute>();

              match(particleBins[rollid ^ 1][did])([&](const auto &pb) {
                cuDev.compute_launch({pbcnt[did], 128}, query_energy_particles,
                                    partitions[rollid ^ 1][did], partitions[rollid][did],
                                    pb, device_kinetic_energy_particles, device_gravity_energy_particles, device_strain_energy_particles, grav);
              });
              cuDev.syncStream<streamIdx::Compute>();
              checkCudaErrors(cudaMemcpyAsync(&kinetic_energy_particle_vals[did], device_kinetic_energy_particles,
                                              sizeof(PREC_G), cudaMemcpyDefault,
                                              cuDev.stream_compute()));
              checkCudaErrors(cudaMemcpyAsync(&gravity_energy_particle_vals[did], device_gravity_energy_particles,
                                              sizeof(PREC_G), cudaMemcpyDefault,
                                              cuDev.stream_compute()));
              checkCudaErrors(cudaMemcpyAsync(&strain_energy_particle_vals[did], device_strain_energy_particles,
                                              sizeof(PREC_G), cudaMemcpyDefault,
                                              cuDev.stream_compute()));
              timer.tock(fmt::format("GPU[{}] frame {} step {} query_strain_energy_particles",
                                    did, curFrame, curStep));
              cuDev.syncStream<streamIdx::Compute>();
            });
            sync();

            PREC_G sum_kinetic_energy_particles = 0.0;
            PREC_G sum_gravity_energy_particles = 0.0;
            PREC_G sum_strain_energy_particles = 0.0;
            for (int did = 0; did < g_device_cnt; ++did)
            {
              sum_kinetic_energy_particles += kinetic_energy_particle_vals[did];
              sum_gravity_energy_particles += gravity_energy_particle_vals[did];
              sum_strain_energy_particles += strain_energy_particle_vals[did];
            }
            sum_gravity_energy_particles = - (init_gravity_energy_particles - sum_gravity_energy_particles); // Difference in gravity energy since start

            {
              std::string fn = std::string{"particle_energy_time_series.csv"};
              particleEnergyFile.open(fn, std::ios::out | std::ios::app);
              particleEnergyFile << curTime << "," << sum_kinetic_energy_particles << "," << sum_gravity_energy_particles << "," << sum_strain_energy_particles << ",\n";
              particleEnergyFile.close();
            }

          }
        }

        // Output gridTarget
        if (flag_gt) { 
          // Check if end of frame or output frequency
          if (fmod(curTime, (1.f/host_target_freq)) < dt || curTime + dt >= nextTime) {
            issue([this](int did) {
              IO::flush();    // Clear IO
              output_gridcell_target(did); // Output gridTarget as *.bgeo
            });
            sync();
          }
        } //< End-of Grid-Target output

        // Update for next time-step
        rollid ^= 1;
        dt = nextDt; 
      } //< End of time-step, re-loop or go next line end of frame
      // Output particle models
      issue([this](int did) {
        IO::flush();
        output_model(did);
        IO::flush();    // Clear IO

      });
      sync();
      // Output finite element models
      issue([this](int did) {
        IO::flush();
        output_finite_elements(did);
      });
      sync();
      

      nextTime = 1.f * (curFrame + 1) / fps;
      fmt::print(fmt::emphasis::bold | fg(fmt::color::red),
                 "-----------------------------------------------------------"
                 "-----\n");
    } //< End of frame, reloop or go next line if end of simulation
    //IO::flush();    // Clear IO
      issue([this](int did) {
        IO::flush();
        output_model(did);
      });
      sync();
    for (int GPU_ID = 0; GPU_ID < g_device_cnt; GPU_ID++)
    {
      tmps[GPU_ID].dealloc();
    }
    fmt::print(fmt::emphasis::bold | fg(fmt::color::red),
                "------------------------------ END -----------------------------\n");
  } //< End of main simulation loop

  void output_model(int did) {
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    CudaTimer timer{cuDev.stream_compute()};
    timer.tick();
    

    int parcnt, *device_parcnt = (int *)cuDev.borrow(sizeof(int));
    checkCudaErrors(
        cudaMemsetAsync(device_parcnt, 0, sizeof(int), cuDev.stream_compute()));

    // Setup forceSum to sum all forces in grid-target in a kernel
    PREC dispVal, *device_dispVal = (PREC *)cuDev.borrow(sizeof(PREC));
    checkCudaErrors(
        cudaMemsetAsync(device_dispVal, 0.0, sizeof(PREC), cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();

    match(particleBins[rollid][did])([&](const auto &pb) {
      cuDev.compute_launch({pbcnt[did], 128}, retrieve_particle_buffer_attributes,
                           partitions[rollid][did], partitions[rollid ^ 1][did],
                           pb, particles[did], pattribs[did], device_dispVal, device_parcnt, length);
    });
    checkCudaErrors(cudaMemcpyAsync(&parcnt, device_parcnt, sizeof(int),
                                    cudaMemcpyDefault, cuDev.stream_compute()));
    checkCudaErrors(cudaMemcpyAsync(&dispVal, device_dispVal, sizeof(PREC),
                                    cudaMemcpyDefault,
                                    cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();
    fmt::print(fg(fmt::color::red), "GPU[{}] Total number of particles: {}\n", did, parcnt);
    fmt::print(fg(fmt::color::red), "GPU[{}] Tracked value of particle ID {} in model: {} \n", did, g_track_ID, dispVal);
  

    if (1) {
      std::string fn_disp = std::string{"disp_time_series"} + "_target[0]_dev[" + std::to_string(did) + "].csv";
      dispFile[did].open(fn_disp, std::ios::out | std::ios::app);
      dispFile[did] << curTime << "," << dispVal << ",\n";
      dispFile[did].close();
      if (verb) fmt::print(fg(fmt::color::red), "GPU[{}] CSV write finished.\n", did);
    }

    models[did].resize(parcnt);
    checkCudaErrors(cudaMemcpyAsync(models[did].data(),
                                    (void *)&particles[did].val_1d(_0, 0),
                                    sizeof(std::array<PREC, 3>) * (parcnt),
                                    cudaMemcpyDefault, cuDev.stream_compute()));

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


  void output_finite_elements(int did) {
    if (flag_fem[did]) {
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    
    
    CudaTimer timer{cuDev.stream_compute()};
    timer.tick();
    

    int elcnt, *device_elcnt = (int *)cuDev.borrow(sizeof(int));
    checkCudaErrors(
        cudaMemsetAsync(device_elcnt, 0, sizeof(int), cuDev.stream_compute()));

    // Setup forceSum to sum all forces in grid-target in a kernel
    PREC dispVal, *device_dispVal = (PREC *)cuDev.borrow(sizeof(PREC));
    checkCudaErrors(
        cudaMemsetAsync(device_dispVal, 0, sizeof(PREC), cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();

    checkCudaErrors(
        cudaMemsetAsync(device_elcnt, 0, sizeof(int), cuDev.stream_compute()));

    checkCudaErrors(
        cudaMemsetAsync((void *)&device_element_attribs[did].val_1d(_0, 0), 0,
                        sizeof(std::array<PREC, 6>) * (element_cnt[did]),
                        cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();


    match(elementBins[did])([&](const auto &eb) {
      cuDev.compute_launch({element_cnt[did], 1}, 
                            retrieve_element_buffer_attributes,
                            (uint32_t)element_cnt[did], 
                            device_vertices[did], eb, device_element_IDs[did], device_element_attribs[did], 
                            device_dispVal, device_elcnt);
    });
    cuDev.syncStream<streamIdx::Compute>();
    checkCudaErrors(cudaMemcpyAsync(&elcnt, device_elcnt, sizeof(int),
                                    cudaMemcpyDefault, cuDev.stream_compute()));
  
    // Copy displacement to host
    checkCudaErrors(cudaMemcpyAsync(&dispVal, device_dispVal, sizeof(PREC),
                                    cudaMemcpyDefault,
                                    cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();
    fmt::print(fg(fmt::color::red), "GPU[{}] Total number of elements: {}\n", did, elcnt);
    fmt::print(fg(fmt::color::red), "GPU[{}] Tracked value of element ID {} in model: {} \n", did, g_track_ID, dispVal);
    fmt::print(fg(fmt::color::red), "GPU[{}] Total element count: {}\n", did, element_cnt[did]);


    // if (1) {
    //   std::string fn_disp = std::string{"element_time_series"} + "_target[0]_dev[" + std::to_string(did) + "].csv";
    //   dispFile[did].open (fn_disp, std::ios::out | std::ios::app);
    //   dispFile[did] << curTime << "," << dispVal << ",\n";
    //   dispFile[did].close();
    //   if (verb) fmt::print(fg(fmt::color::red), "GPU[{}] CSV write finished.\n", did);
    // }

    //host_element_IDs[did].resize(element_cnt[did]);
    host_element_attribs[did].resize(element_cnt[did]);
    cuDev.syncStream<streamIdx::Compute>();

    checkCudaErrors(cudaMemcpyAsync(host_element_attribs[did].data(),
                                    (void *)&device_element_attribs[did].val_1d(_0, 0),
                                    sizeof(std::array<PREC, 6>) * (element_cnt[did]),
                                    cudaMemcpyDefault, cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();

    std::string fn = std::string{"elements"} + "_dev[" + std::to_string(did) +
                     "]_frame[" + std::to_string(curFrame) + "].bgeo";
    IO::insert_job(
        [fn, e = host_element_attribs[did]]() { write_partio_finite_elements<PREC>(fn, e); });
    
    timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_elements", did,
                           curFrame, curStep));
    }
  }

  /// Output data from grid blocks (mass, momentum) to *.bgeo (JB)
  void output_gridcell_target(int did) {
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    CudaTimer timer{cuDev.stream_compute()};
    timer.tick();
    if (verb) fmt::print(fg(fmt::color::red), "GPU[{}] Entered output_gridcell_target\n", did);

    for (int i = 0; i < number_of_grid_targets; i++)
    {
      IO::flush();    // Clear IO
      int gridID = 0;
      if (curTime == 0){
        gridID += 1;
        rollid = rollid^1;
        host_gridTarget[did].resize(tarcnt[i][did]);
        cuDev.syncStream<streamIdx::Compute>();

        checkCudaErrors(
            cudaMemsetAsync((void *)&device_gridTarget[did].val_1d(_0, 0), 0,
                            sizeof(std::array<PREC_G, config::g_target_attribs>) * (config::g_target_cells),
                            cuDev.stream_compute()));
        cuDev.syncStream<streamIdx::Compute>();

        checkCudaErrors(
            cudaMemcpyAsync(host_gridTarget[did].data(), (void *)&device_gridTarget[did].val_1d(_0, 0),
                            sizeof(std::array<PREC_G, config::g_target_attribs>) * (tarcnt[i][did]),
                            cudaMemcpyDefault, cuDev.stream_compute()));
        cuDev.syncStream<streamIdx::Compute>();

      }

      int target_cnt, *device_target_cnt = (int *)cuDev.borrow(sizeof(int));
      checkCudaErrors(
          cudaMemsetAsync(device_target_cnt, 0, sizeof(int), 
          cuDev.stream_compute())); /// Reset memory
      //target_cnt = config::g_target_cells; // Max number of grid-nodes in target
      target_cnt = 0;
      cuDev.syncStream<streamIdx::Compute>();

      // Setup forceSum to sum all forces in grid-target in a kernel
      PREC_G forceSum, *device_forceSum = (PREC_G *)cuDev.borrow(sizeof(PREC_G));
      checkCudaErrors(
          cudaMemsetAsync(device_forceSum, 0, sizeof(PREC_G), cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();

      // Reset gridTarget (device) to zeroes asynchronously
      checkCudaErrors(
          cudaMemsetAsync((void *)&device_gridTarget[did].val_1d(_0, 0), 0,
                          sizeof(std::array<PREC_G, config::g_target_attribs>) * (config::g_target_cells),
                          cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();

      fmt::print(fg(fmt::color::red), "GPU[{}] Launch retrieve_selected_grid_cells\n", did);

      /// Copy down-sampled grid value from buffer (device) to target (device)
      cuDev.compute_launch(
                {curNumActiveBlocks[did], 32}, retrieve_selected_grid_cells, 
                (uint32_t)nbcnt[did], partitions[rollid ^ 1][did], 
                gridBlocks[gridID][did], device_gridTarget[did],
                nextDt, device_forceSum, device_target[i], device_target_cnt, length);
      cudaGetLastError();
      cuDev.syncStream<streamIdx::Compute>();

      // Copy force summation to host
      checkCudaErrors(cudaMemcpyAsync(&forceSum, device_forceSum, sizeof(PREC_G),
                                      cudaMemcpyDefault, cuDev.stream_compute()));
      checkCudaErrors(cudaMemcpyAsync(&target_cnt, device_target_cnt, sizeof(int),
                                      cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
      tarcnt[i][did] = target_cnt;

      fmt::print(fg(fmt::color::red), "GPU[{}] Total number of target grid-nodes: {}\n", did, tarcnt[i][did]);
      fmt::print(fg(fmt::color::red), "GPU[{}] Force summation in gridTarget: {} N\n", did, forceSum);

      // Output gridTarget binary geometry file with x,y,z, forces, etc.
      if (curTime + dt >= nextTime || curTime == 0) 
      {

        // Asynchronously copy data from target (device) to target (host)
        host_gridTarget[did].resize(tarcnt[i][did]);
        cuDev.syncStream<streamIdx::Compute>();

        checkCudaErrors(
            cudaMemcpyAsync(host_gridTarget[did].data(), (void *)&device_gridTarget[did].val_1d(_0, 0),
                            sizeof(std::array<PREC_G, config::g_target_attribs>) * (tarcnt[i][did]),
                            cudaMemcpyDefault, cuDev.stream_compute()));
        cuDev.syncStream<streamIdx::Compute>();
        // Output to Partio as 'gridTarget_target[ ]_dev[ ]_frame[ ].bgeo'
        std::string fn = std::string{"gridTarget["} + std::to_string(i) + "]" + "_dev[" + std::to_string(did) + "]_frame[" + std::to_string(curFrame) + "].bgeo";
        IO::insert_job([fn, m = host_gridTarget[did]]() { write_partio_gridTarget<float, config::g_target_attribs>(fn, m); });
        fmt::print(fg(fmt::color::red), "GPU[{}] gridTarget[{}] outputted.\n", did, i);
      }
    
      // Output summed grid-target value to *.csv
      if (fmod(curTime, (1.f/host_target_freq)) < dt) 
      {
        std::string fn = std::string{"force_time_series"} + "_gridTarget[" + std::to_string(i) + "]_dev[" + std::to_string(did) + "].csv";
        forceFile[did].open (fn, std::ios::out | std::ios::app);
        forceFile[did] << curTime << "," << forceSum << ",\n";
        forceFile[did].close();
      }
      if (curTime == 0){
        IO::flush();    // Clear IO

        rollid = rollid^1;
      }
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
      float waveMax, *device_waveMax = (float *)cuDev.borrow(sizeof(float));
      checkCudaErrors(
          cudaMemsetAsync(device_waveMax, 0.f, sizeof(float), cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();

      /// Copy down-sampled grid value from buffer (device) to target (device)
      cuDev.compute_launch(
                {curNumActiveBlocks[did], 32}, retrieve_wave_gauge, 
                (uint32_t)nbcnt[did], partitions[rollid][did], 
                gridBlocks[0][did],
                nextDt, device_waveMax, device_wg_point_a[i], device_wg_point_b[i], length);
      cuDev.syncStream<streamIdx::Compute>();

      // Copy force summation to host
      checkCudaErrors(cudaMemcpyAsync(&waveMax, device_waveMax, sizeof(float),
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
      // cuDev.compute_launch({(pcnt[did] + 255) / 256, 256}, activate_blocks,
      //                      pcnt[did], particles[did],
      //                      partitions[rollid ][did]); // JB
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
      // cuDev.compute_launch({(pcnt[did] + 255) / 256, 256},
      //                      build_particle_cell_buckets, pcnt[did],
      //                      particles[did], partitions[rollid ][did]); // JB
      // Partition buckets, binsts
      cuDev.syncStream<streamIdx::Compute>();
      partitions[rollid ^ 1][did].buildParticleBuckets(cuDev, pbcnt[did]);
      // partitions[rollid ][did].buildParticleBuckets(cuDev, pbcnt[did]); //JB

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
            cuDev.compute_launch({element_cnt[did], 1}, fem_precompute,
                                (uint32_t)element_cnt[did], device_vertices[did], device_element_IDs[did], eb);
          });
        cuDev.syncStream<streamIdx::Compute>();
        }
      }); //< End FEM Precompute
      cuDev.syncStream<streamIdx::Compute>();

      // Register neighbor Grid-Blocks
      cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
                           register_neighbor_blocks, (uint32_t)pbcnt[did],
                           partitions[rollid ^ 1][did]);
      // cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
      //                      register_neighbor_blocks, (uint32_t)pbcnt[did],
      //                      partitions[rollid ][did]); // JB
      checkCudaErrors(cudaMemcpyAsync(
          &nbcnt[did], partitions[rollid ^ 1][did]._cnt, sizeof(int),
          cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
      // Register exterior Grid-Blocks
      cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
                           register_exterior_blocks, (uint32_t)pbcnt[did],
                           partitions[rollid ^ 1][did]);
      // cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
      //                      register_exterior_blocks, (uint32_t)pbcnt[did],
      //                      partitions[rollid ][did]); // JB
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
      fmt::print(fg(fmt::color::green), "Start partitions.copy_to GPU[{}]\n", did);

      /// Need to copy halo tag info to next partition
      partitions[rollid ^ 1][did].copy_to(partitions[rollid][did], ebcnt[did],
                                          cuDev.stream_compute());
      fmt::print(fg(fmt::color::green), "Finish partitions.copy_to GPU[{}]\n", did);

      checkCudaErrors(cudaMemcpyAsync(
          partitions[rollid][did]._activeKeys,
          partitions[rollid ^ 1][did]._activeKeys, sizeof(ivec3) * ebcnt[did],
          cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
      fmt::print(fg(fmt::color::green), "Finish active_keys GPU[{}], extent: {}\n", did, partitions[rollid ^ 1][did].h_count) ;

      timer.tick();
      gridBlocks[0][did].reset(nbcnt[did], cuDev); // Zero out blocks on all Grids 0
      // Send initial information from Particle Arrays to Grids 0 (e.g. mass, velocity)
      match(particleBins[rollid][did])([&](const auto &pb) {
        cuDev.compute_launch({(pcnt[did] + 255) / 256, 256}, rasterize, pcnt[did],
                           particles[did], gridBlocks[0][did],
                           partitions[rollid][did], dt, pb.mass, pb.volume, vel0[did], pb.length, (PREC)grav);
      });
      cuDev.syncStream<streamIdx::Compute>();
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


    if (1)
    {
      {
        issue([this](int did) {
          IO::flush();
          auto &cuDev = Cuda::ref_cuda_context(did);
          //cuDev.setContext();
          PREC_G *device_kinetic_energy_particles = tmps[did].device_kinetic_energy_particles;
          PREC_G *device_gravity_energy_particles = tmps[did].device_gravity_energy_particles;
          PREC_G *device_strain_energy_particles  = tmps[did].device_strain_energy_particles;

          CudaTimer timer{cuDev.stream_compute()};
          timer.tick();


          checkCudaErrors(cudaMemsetAsync(device_kinetic_energy_particles, 0, sizeof(PREC_G),
                                          cuDev.stream_compute()));
          checkCudaErrors(cudaMemsetAsync(device_gravity_energy_particles, 0, sizeof(PREC_G),
                                          cuDev.stream_compute()));
          checkCudaErrors(cudaMemsetAsync(device_strain_energy_particles, 0, sizeof(PREC_G),
                                          cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();

          match(particleBins[rollid][did])([&](const auto &pb) {
            cuDev.compute_launch({pbcnt[did], 128}, query_energy_particles,
                                partitions[rollid][did], partitions[rollid ^ 1][did],
                                pb, device_kinetic_energy_particles, device_gravity_energy_particles, device_strain_energy_particles, grav);
          });
          cuDev.syncStream<streamIdx::Compute>();
          checkCudaErrors(cudaMemcpyAsync(&kinetic_energy_particle_vals[did], device_kinetic_energy_particles,
                                          sizeof(PREC_G), cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          checkCudaErrors(cudaMemcpyAsync(&gravity_energy_particle_vals[did], device_gravity_energy_particles,
                                          sizeof(PREC_G), cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          checkCudaErrors(cudaMemcpyAsync(&strain_energy_particle_vals[did], device_strain_energy_particles,
                                          sizeof(PREC_G), cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          timer.tock(fmt::format("GPU[{}] frame {} step {} query_strain_energy_particles",
                                did, curFrame, curStep));
          cuDev.syncStream<streamIdx::Compute>();
        });
        sync();

        PREC_G sum_kinetic_energy_particles = 0.0;
        PREC_G sum_gravity_energy_particles = 0.0;
        PREC_G sum_strain_energy_particles = 0.0;
        for (int did = 0; did < g_device_cnt; ++did)
        {
          sum_kinetic_energy_particles += kinetic_energy_particle_vals[did];
          sum_gravity_energy_particles += gravity_energy_particle_vals[did];
          sum_strain_energy_particles += strain_energy_particle_vals[did];
        }
        init_gravity_energy_particles = sum_gravity_energy_particles;
        sum_gravity_energy_particles -= init_gravity_energy_particles;

        {
          std::string fn = std::string{"particle_energy_time_series.csv"};
          particleEnergyFile.open(fn, std::ios::out | std::ios::app);
          particleEnergyFile << curTime << "," << sum_kinetic_energy_particles << "," << sum_gravity_energy_particles << "," << sum_strain_energy_particles << ",\n";
          particleEnergyFile.close();
        }

      }
    }

    // Output Grid-Target Frame 0
    if (flag_gt) {
      issue([this](int did) {
        IO::flush();    // Clear IO
        output_gridcell_target(did); // Output as *.bgeo and summed force *.csv
      });
      sync();
    }

    // Output particle models frame 0
    issue([this](int did) {
      IO::flush();
      output_model(did);
    });
    sync();

    // Output finite element models frame 0

    issue([this](int did) {
      IO::flush();
      output_finite_elements(did);
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
      fmt::print(fg(fmt::color::green), "Start collect halo grid blocks GPU[{}]\n", did);

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
  PREC length;
  float dt, nextDt, dtDefault, curTime, nextTime, maxVel, grav;
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
  std::vector<GridTarget> device_gridTarget; ///< Target node structure on device, 7+ f32 (x,y,z, mass, mx,my,mz, fx, fy, fz) (JB)
  std::vector<vec7> device_target; ///< Grid target boundaries and type (JB)

  std::vector<vec7> walls; ///< 
  std::vector<vec7> boxes; ///< 
  std::vector<vec3> device_wg_point_a; ///< Point A of target (JB)
  std::vector<vec3> device_wg_point_b; ///< Point B of target (JB)
  vec3 device_waveMaker; ///< OSU wm info (time, disp, vel) on device (JB)
  std::vector<std::string> output_attribs[config::g_device_cnt];
  //< Data-structures configured in particle_buffer.cuh
  vec<ParticleArray, config::g_device_cnt> particles; //< Basic GPU vector for Particle positions
  vec<ParticleAttrib, config::g_device_cnt> pattribs;  //< Basic GPU vector for Particle attributes
  //vec<ParticleAttrib, config::g_device_cnt> pattribs;  //< Basic GPU vector for Particle attributes

  //std::vector<particle_array_> particles; //< Basic GPU vector for Particle positions
  //std::vector<particle_array_> pattribs;  //< Basic GPU vector for Particle attributes
  
  vec<ElementAttrib, config::g_device_cnt> device_element_attribs;  //< Basic GPU vector for Finite Element Gauss Point attributes
  //std::vector<ElementAttrib> element_attribs;
  //< Data-structures configured in fem_buffer.cuh
  vec<VerticeArray, config::g_device_cnt> device_vertices; //< Basic GOU vector for FEM Vertice positions
  vec<ElementArray, config::g_device_cnt> device_element_IDs; //< Basic GPU vector for FEM Element node IDs

  struct Intermediates {
    void *base;
    PREC_G *device_strain_energy_particles;
    PREC_G *device_gravity_energy_particles;
    PREC_G *device_kinetic_energy_particles;
    PREC_G *device_gravity_energy_grid;
    PREC_G *device_kinetic_energy_grid;
    PREC_G *device_maxVel;
    int *device_tmp;
    int *activeBlockMarks;
    int *destinations;
    int *sources;
    int *binpbs;
    void alloc(int maxBlockCnt) {
      checkCudaErrors(cudaMalloc(&base, sizeof(int) * (maxBlockCnt * 5 + 6)));
      device_strain_energy_particles = (PREC_G *)((char *)base + sizeof(int) * (maxBlockCnt * 5 + 5));
      device_gravity_energy_particles = (PREC_G *)((char *)base + sizeof(int) * (maxBlockCnt * 5  + 4));
      device_kinetic_energy_particles = (PREC_G *)((char *)base + sizeof(int) * (maxBlockCnt * 5  + 3));
      device_gravity_energy_grid = (PREC_G *)((char *)base + sizeof(int) * (maxBlockCnt * 5 + 2));
      device_kinetic_energy_grid = (PREC_G *)((char *)base + sizeof(int) * (maxBlockCnt * 5 + 1));
      device_maxVel = (PREC_G *)((char *)base + sizeof(int) * maxBlockCnt * 5);
      device_tmp = (int *)((uintptr_t)base);
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
  }; 
  Intermediates tmps[config::g_device_cnt];

  // halo data
  vec<ivec3 *, config::g_device_cnt, config::g_device_cnt> haloBlockIds;

  /// data on host
  static_assert(std::is_same<GridBufferDomain::index_type, int>::value,
                "block index type is not int");
  char rollid;
  std::size_t curNumActiveBlocks[config::g_device_cnt],
      curNumActiveBins[config::g_device_cnt],
      checkedCnts[config::g_device_cnt][2];
  vec<PREC_G, config::g_device_cnt> maxVels;
  vec<PREC_G, config::g_device_cnt> kinetic_energy_grid_vals;
  vec<PREC_G, config::g_device_cnt> gravity_energy_grid_vals;
  vec<PREC_G, config::g_device_cnt> kinetic_energy_particle_vals;
  vec<PREC_G, config::g_device_cnt> gravity_energy_particle_vals;
  vec<PREC_G, config::g_device_cnt> strain_energy_particle_vals;
  PREC init_gravity_energy_particles;
  
  vec<int, config::g_device_cnt> pbcnt, nbcnt, ebcnt, bincnt; ///< num blocks
  vec<int, config::g_device_cnt> element_cnt;
  vec<int, config::g_device_cnt> vertice_cnt;
  vec<uint32_t, config::g_device_cnt> pcnt;                   ///< num particles
  std::vector<vec<uint32_t, config::g_device_cnt>> tarcnt; ///< Number of target grid nodes (JB)
  std::vector<float> durations[config::g_device_cnt + 1];
  std::vector<std::array<PREC, 3>> models[config::g_device_cnt];
  std::vector<std::array<PREC, 3>> attribs[config::g_device_cnt];
  int number_of_grid_targets=0;
  std::vector<std::array<PREC_G, config::g_target_attribs>> host_gridTarget[config::g_device_cnt];   ///< Grid target info (x,y,z,m,mx,my,mz,fx,fy,fz) on host (JB)
  std::vector<std::array<PREC, 13>> host_vertices[config::g_device_cnt];
  std::vector<std::array<int, 4>> host_element_IDs[config::g_device_cnt];
  std::vector<std::array<PREC, 6>> host_element_attribs[config::g_device_cnt];
  std::vector<std::array<float, 3>> host_waveMaker;   ///< wave-maker (time, disp, vel) on host (JB)
  std::array<int , config::g_device_cnt> flag_fem; // 0 if no grid targets, 1 if grid targets to output
  int flag_gt = 0; // 0 if no grid targets, 1 if grid targets to output
  PREC_G host_target_freq = 1.f; // Frequency of grid-target output
  int flag_wg = 0; // 0 if no wave-gauges, 1 if wave-gauges to output
  PREC_G host_wg_freq = 1.f; // Frequency of wave-gauge output

  std::ofstream dispFile[mn::config::g_device_cnt];
  std::ofstream forceFile[mn::config::g_device_cnt];
  std::ofstream energyFile;
  std::ofstream gridEnergyFile;
  std::ofstream particleEnergyFile;

  bool verb = false; //< If true, print more information to terminal

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