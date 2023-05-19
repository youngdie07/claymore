#ifndef __MGSP_BENCHMARK_CUH_
#define __MGSP_BENCHMARK_CUH_
#include "boundary_condition.cuh"
#include "halo_buffer.cuh"
#include "halo_kernels.cuh"
#include "hash_table.cuh"
#include "grid_buffer.cuh"
#include "fem_buffer.cuh"
#include "mgmpm_kernels.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include <MnBase/Concurrency/Concurrency.h>
#include <MnBase/Meta/ControlFlow.h>
#include <MnBase/Meta/TupleMeta.h>
#include <MnBase/Profile/CppTimers.hpp>
#include <MnBase/Profile/CudaTimers.cuh>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <MnSystem/IO/ParticleIO.hpp>
#include <fmt/color.h>
#include <fmt/core.h>
#include <array>
#include <vector>
#include <numeric>

#if CLUSTER_COMM_STYLE == 1
#include <mpi.h>
#endif

namespace mn {
using namespace config;

struct mgsp_benchmark {
  using streamIdx = Cuda::StreamIndex;
  using eventIdx = Cuda::EventIndex;
  using host_allocator = heap_allocator;
  struct device_allocator { // hide the global one
    int device_alloc_cnt = 0;
    void *allocate(std::size_t bytes) {
      void *ret;
      checkCudaErrors(cudaMalloc(&ret, bytes));
      device_alloc_cnt++;
      fmt::print("device_allocator: cudaMalloc [{}] megabytes to ptr[{}]. Depth [{}]\n", (float)bytes / 1000 / 1000, ret, device_alloc_cnt);
      return ret;
    }
    void deallocate(void *p, std::size_t) { 
      if (p) { 
        fmt::print("cudaFree device_allocator ptr[{}]\n", p);
        checkCudaErrors(cudaFree(p));
        //if (p) { p = nullptr;  fmt::print("Set pointer to nullptr.\n"); }
      }
      device_alloc_cnt--;
      fmt::print("Deallocated device_allocator. Depth [{}]\n", device_alloc_cnt);
    }
  };
  void printDiv() { 
    fmt::print("----------------------------------------------------------------\n"); 
  }
  struct temp_allocator {
    int did;
    int temp_alloc_cnt = 0;
    explicit temp_allocator(int did) : did{did} {}
    void *allocate(std::size_t bytes) {
      temp_alloc_cnt++;
      if (mn::config::g_log_level >= 3) fmt::print("temp_allocator: CUDA::ref_cuda_context(GPU[{}]).borrow() [{}] megabytes. Depth [{}]\n", did, (float)bytes /1000.f/1000.f, temp_alloc_cnt);
      return Cuda::ref_cuda_context(did).borrow(bytes);
    }
    void deallocate(void *p, std::size_t) {      
      temp_alloc_cnt--;
      if (mn::config::g_log_level >= 3) fmt::print("temp_allocator: Deallocate ptr[{}] in CUDA::ref_cuda_context(GPU[{}]). Depth: [{}]\n", p, did, temp_alloc_cnt);
    }
  };

  // Initialize GPU data-structures on all GPUs, assign a host thread to each GPU and set the CUDA context of the host thread to that GPU.
  template <std::size_t GPU_ID> void initParticles() {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID); // Reference CUDA context of GPU_ID
    cuDev.setContext(); // Set CUDA context of host thread. Host thread-to-GPU is 1:1
    // NOTE: The following code is executed on the host thread of GPU_ID, but the CUDA context is of GPU_ID. It is what organizes device memory allocation, etc. Device memory deallocation MUST use the same thread and CUDA context or the program will not close/undefined behavior.
    tmps[GPU_ID].alloc(g_max_active_block); // Temporary memory array on GPU
    for (int copyid = 0; copyid < 2; copyid++) {
      fmt::print("Allocating gridBlocks[{}][{}].\n", copyid, GPU_ID);
      gridBlocks[copyid].emplace_back(device_allocator{});
      printDiv();

      fmt::print("Allocating partitions[{}][{}].\n", copyid, GPU_ID);
      partitions[copyid].emplace_back(device_allocator{}, g_max_active_block, domainCellCnt);
      printDiv();
    }
    cuDev.syncStream<streamIdx::Compute>();

    fmt::print("Allocating inputHaloGridBlocks[{}].\n", GPU_ID);
    inputHaloGridBlocks.emplace_back(g_device_cnt);
    printDiv();

    fmt::print("Allocating outputHaloGridBlocks[{}].\n", GPU_ID);
    outputHaloGridBlocks.emplace_back(g_device_cnt);
    printDiv();


    fmt::print("Allocating d_element_attribs[{}].\n", GPU_ID);
    d_element_attribs[GPU_ID] = spawn<element_attrib_, orphan_signature>(device_allocator{}); //< Element attributes on device
    printDiv();

    // Add/initialize a gridTarget data-structure per GPU within d_gridTarget vector. 
    fmt::print("Allocating d_gridTarget[{}].\n", GPU_ID);
    d_gridTarget.emplace_back(std::move(GridTarget{spawn<grid_target_, orphan_signature>(device_allocator{}, sizeof(std::array<PREC_G, g_grid_target_attribs>) * g_grid_target_cells)}));
    checkCudaErrors(  cudaMemsetAsync((void *)&d_gridTarget[GPU_ID].val_1d(_0, 0), 0,
                        sizeof(std::array<PREC_G, g_grid_target_attribs>) * g_grid_target_cells,
                        cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();
    printDiv();
    d_particleTarget.emplace_back(std::move(ParticleTarget{spawn<particle_target_, orphan_signature>(device_allocator{}, sizeof(std::array<PREC, g_particle_target_attribs>) * g_particle_target_cells)}));
    checkCudaErrors(  cudaMemsetAsync((void *)&d_particleTarget[GPU_ID].val_1d(_0, 0), 0,
                        sizeof(std::array<PREC, g_particle_target_attribs>) * g_particle_target_cells,
                        cuDev.stream_compute()));
    fmt::print("Allocating d_particleTarget for GPU[{}].\n", GPU_ID);
    cuDev.syncStream<streamIdx::Compute>();

    for (int MODEL_ID=0; MODEL_ID<getModelCnt(GPU_ID); MODEL_ID++){
      for (int d = 0; d<3; d++) vel0[GPU_ID][MODEL_ID][d] = 0.0; //< Init. vel.if no initial attributes
    }
    printDiv();
    flag_pe = true; //< Flag for particle energy output
    flag_ge = true; //< Flag for grid energy output
    //flag_pi[GPU_ID][MODEL_ID] = false; //< Flag for use particle initial attributes
    flag_fem[GPU_ID] = false; //< Flag for Finite Element Method
    element_cnt[GPU_ID] = g_max_fem_element_num; //< Number of elements
    vertice_cnt[GPU_ID] = g_max_fem_vertice_num; //< Number of vertices
    checkedCnts[GPU_ID][0] = 0; //< Init. flag for checked grid block count
    checkedCnts[GPU_ID][1] = 0; //< Init. flag for checked particle bin count
    curNumActiveBlocks[GPU_ID] = g_max_active_block; //< Number of active grid blocks
    for (int MODEL_ID=0; MODEL_ID<getModelCnt(GPU_ID); MODEL_ID++){
      checkedBinCnts[GPU_ID][MODEL_ID] = 0;
      curNumActiveBins[GPU_ID][MODEL_ID] = g_max_particle_bin; //< Number of active particle bins
    }
    // "if constexpr" is a C++ 17 feature. Change if not compiling.
    // Loop and initialize each device/CUDA context.
    if constexpr (GPU_ID + 1 < g_device_cnt) initParticles<GPU_ID + 1>();
  }

  // Constructor
  mgsp_benchmark(PREC l = g_length, uint64_t dCC = (g_grid_size_x*g_grid_size_y*g_grid_size_z), float dt = 1e-4, uint64_t fp = 24, uint64_t frames = 60, mn::pvec3 g = mn::pvec3{0., -9.81, 0.}, std::string suffix = ".bgeo")
      : length(l), domainCellCnt(dCC), dtDefault(dt), curTime(0.f), rollid(0), curFrame(0), curStep{0}, fps(fp), nframes(frames), grav(g), save_suffix(suffix), bRunning(true) {

    fmt::print(fg(fmt::color::white),"Entered simulator. Start GPU set-up...\n");
    printDiv();

    // Set-up Multi-GPU Multi-Node communication if enabled
#if CLUSTER_COMM_STYLE == 1
    // Obtain our rank (Node ID) and the total number of ranks (Num Nodes)
    // Assume we launch MPI with one rank per GPU Node
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    fmt::print(fg(fmt::color::cyan),"MPI Rank {} of {}.\n", rank, num_ranks);
#endif


    initTime = curTime;
    collisionObjs.resize(g_device_cnt); // Deprecated boundary method
    initParticles<0>(); // Set-up basic data-strucutres for each GPU
    printDiv();
    fmt::print("GPU[{}] gridBlocks[0] and gridBlocks[1]  size [{}] + [{}] megabytes.\n", 0,
               (float)std::decay<decltype(*gridBlocks[0].begin())>::type::size / 1000.f / 1000.f,
               (float)std::decay<decltype(*gridBlocks[1].begin())>::type::size / 1000.f / 1000.f);   
    // TODO: Report preallocated partition and halo partition size
    // fmt::print("GPU[{}] Partitions, Double-Buffered with Padding: Partitions[0] size {} bytes -vs- Paritions[1] size {} bytes.\n", 0,
    //            sizeof(partitions[0]),
    //            std::decay<decltype(*partitions[1].begin())>::type::base_t::size); 

    // Grid-energy output
    if (flag_ge) {
      gridEnergyFile.open(std::string{"grid_energy_time_series.csv"}, std::ios::out | std::ios::trunc); 
      gridEnergyFile << "Time" << "," << "Kinetic_FLIP" <<  "," << "Kinetic_PIC" << "\n";
      gridEnergyFile.close();
    }
    // Particle-energy output
    if (flag_pe) {
      particleEnergyFile.open(std::string{"particle_energy_time_series.csv"}, std::ios::out | std::ios::trunc); 
      particleEnergyFile << "Time" << "," << "Kinetic" << "," << "Gravity" << "," << "Strain" << "\n";
      particleEnergyFile.close();
    }
    printDiv();

    // Tasks for host threads
    fmt::print("Setting host threads as GPU workers.\n");
    for (int did = 0; did < g_device_cnt; ++did) {
      ths[did] = std::thread([this](int did) { this->gpu_worker(did); }, did);
      fmt::print("GPU[{}] Set thread.\n", did);
    }
    printDiv();
  }

  // Destructor
  ~mgsp_benchmark() {
    fmt::print("Simulator destructor. Wait on threads to finish work...\n");
    auto is_empty = [this]() {
      for (int did = 0; did < g_device_cnt; ++did)
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
    {
      IO::flush(); // JB
    }
    fmt::print("Threads finished.\n");
    printDiv();
  }

  // Set time step for sim. min() sets to smallest step needed for a material in sim.
  template <typename T = float>
  void set_time_step(T dt) { dtDefault = std::min(dtDefault, (float) dt); }

  // Initialize particle models. Allow for varied materials on GPUs.
  template <material_e m>
  void initModel(int GPU_ID, int MODEL_ID, const std::vector<std::array<PREC, 3>> &model,
                  const vec<PREC, 3> &v0, const std::vector<int> &trackIDs) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();

    // Check for valid model size
    if (model.size() > config::g_max_particle_num)
      throw std::runtime_error("ERROR: Particle count exceeds maximum particle size.");
    if (model.size() == 0)
      throw std::runtime_error("ERROR: Model has zero particles.");
    pcnt[GPU_ID][MODEL_ID] = model.size(); // Initial particle count

    h_model_cnt[GPU_ID] += 1; // Increment model count on GPU
    for (int copyid = 0; copyid < 2; copyid++) {
      fmt::print("NODE[{}] GPU[{}] MODEL[{}] Allocating ParticleBins[{}][{}].\n", rank, GPU_ID, MODEL_ID, copyid, MODEL_ID);
      particleBins[copyid][GPU_ID].emplace_back(ParticleBuffer<m>(device_allocator{},
        config::g_max_particle_bin));
      cuDev.syncStream<streamIdx::Compute>();
      if (g_buckets_on_particle_buffer)
        match(particleBins[copyid][GPU_ID][MODEL_ID])([&](auto &pb) {
          pb.reserveBuckets(device_allocator{}, config::g_max_active_block);
        });
      printDiv();
    }
    for (int i = 0; i < 3; ++i) vel0[GPU_ID][MODEL_ID][i] = v0[i]; // Initial velocity
    cuDev.syncStream<streamIdx::Compute>();

    fmt::print("NODE[{}] GPU[{}] MODEL[{}] Allocating ParticleArray.\n", rank, GPU_ID, MODEL_ID);

    particles[GPU_ID][MODEL_ID] = spawn<particle_array_, orphan_signature>(device_allocator{});
    bincnt[GPU_ID][MODEL_ID] = 0;
    checkedBinCnts[GPU_ID][MODEL_ID] = 0;
    curNumActiveBins[GPU_ID][MODEL_ID] = config::g_max_particle_bin;
    cudaMemcpyAsync((void *)&particles[GPU_ID][MODEL_ID].val_1d(_0, 0), model.data(),
                    sizeof(std::array<PREC, 3>) * pcnt[GPU_ID][MODEL_ID],
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    fmt::print(fg(fmt::color::green), "NODE[{}] GPU[{}] MODEL[{}] Initialized device array with [{}] particles.\n", rank, GPU_ID, MODEL_ID, pcnt[GPU_ID][MODEL_ID]);
    printDiv();

    fmt::print("NODE[{}] GPU[{}] MODEL[{}] ParticleBins[0] and [1]: size [{}] and [{}] megabytes.\n", rank, GPU_ID, MODEL_ID,
               (float)match(particleBins[0][GPU_ID][MODEL_ID])([&](auto &pb) {return pb.size;})/1000/1000,
               (float)match(particleBins[1][GPU_ID][MODEL_ID])([&](auto &pb) {return pb.size;})/1000/1000);    
    printDiv();


    // Set-up particle ID tracker file
    num_particle_trackers[GPU_ID][MODEL_ID] = trackIDs.size();
    std::string fn_track = std::string{"particleTrack"} + "_model[" + std::to_string(MODEL_ID) + "]" + "_dev[" + std::to_string(GPU_ID + rank * mn::config::g_device_cnt) + "].csv";
    particleTrackFile[GPU_ID][MODEL_ID].open (fn_track, std::ios::out | std::ios::trunc); 
    particleTrackFile[GPU_ID][MODEL_ID] << "Time";
    for (int i = 0; i < trackIDs.size(); ++i) 
     particleTrackFile[GPU_ID][MODEL_ID] << "," << "ParticleID" << trackIDs[i]; 
    particleTrackFile[GPU_ID][MODEL_ID] << "\n";
    particleTrackFile[GPU_ID][MODEL_ID].close();
    printDiv();
    // Output initial particle model
    std::string fn = std::string{"model["} + std::to_string(MODEL_ID) + "]"  "_dev[" + std::to_string(GPU_ID + rank * mn::config::g_device_cnt) +
                     "]_frame[-1]" + save_suffix;
    IO::insert_job([fn, model]() { write_partio<PREC, 3>(fn, model); });
    IO::flush();
  }
  
  /// @brief Initialize particle attributes on host and device. Allow for varied material and outputs. Number of attributes is restricted to numbers defined in enumerator num_attribs_e
  /// @param GPU_ID Unique ID for GPU device, particle attributes will be initialized per GPU.
  /// @param  model_attribs Initial attributes (e.g. Velocity) for each particle.
  /// @param has_init_attribs True if initial attributes given, false if not (defaults will be used).
  template <num_attribs_e N>
  void initInitialAttribs(int GPU_ID, int MODEL_ID, const std::vector<std::vector<PREC>>& model_attribs, const bool has_init_attribs) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    if (MODEL_ID >= getModelCnt(GPU_ID)) throw std::runtime_error("ERROR: Exceeds particle models for all GPUs. Increase g_models_per_gpu.\n");

    cuDev.setContext();
    constexpr int n = static_cast<int>(N);
    flag_pi[GPU_ID][MODEL_ID] = has_init_attribs;
    fmt::print("GPU[{}] MODEL[{}] Allocating ParticleAttribs.\n", GPU_ID, MODEL_ID);
    pattribs_init[GPU_ID].emplace_back(ParticleAttrib<N>(device_allocator{}));
    cuDev.syncStream<streamIdx::Compute>();
    printDiv();

    std::vector<PREC> flattened;
    flattened.reserve(n * model_attribs.size());
    size_t reserve_size = 0;
    for (int i=0; i<model_attribs.size(); ++i)
      reserve_size += model_attribs[i].size();
    flattened.reserve(reserve_size);
    for (int i=0; i<model_attribs.size(); ++i) {
      const std::vector<PREC> & v = model_attribs[i];
      flattened.insert( flattened.end() , v.begin() , v.end() );
    }
    match(pattribs_init[GPU_ID][MODEL_ID])([&](auto &pa) {
      checkCudaErrors(cudaMemcpyAsync((void *)&get<typename std::decay_t<decltype(pa)>>(
                          pattribs_init[GPU_ID][MODEL_ID]).val_1d(_0, 0), flattened.data(),
                      sizeof(PREC) * n * model_attribs.size(),
                      cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
    });

    fmt::print(fg(fmt::color::green), "GPU[{}] MODEL[{}] Initialized input device attribute vector of vectors with [{}] particles and [{}] attributes.\n", GPU_ID, MODEL_ID, flattened.size()/n, n);
    printDiv();
  }

  /// @brief Initialize particle attributes on host and device. Allow for varied material and outputs. Number of attributes is restricted to numbers defined in enumerator num_attribs_e
  /// @param GPU_ID Unique ID for GPU device, particle attributes will be initialized per GPU.
  /// @param  model_attribs Initial attributes (e.g. Velocity) for each particle.
  /// @param has_init_attribs True if initial attributes given, false if not (defaults will be used).
  template <num_attribs_e N>
  void initOutputAttribs(int GPU_ID, int MODEL_ID, const std::vector<std::vector<PREC>>& model_attribs) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    constexpr int n = static_cast<int>(N);
    fmt::print("GPU[{}] MODEL[{}] Allocating ParticleAttribs.\n", GPU_ID, MODEL_ID);
    pattribs[GPU_ID].emplace_back(ParticleAttrib<N>(device_allocator{}));
    cuDev.syncStream<streamIdx::Compute>();
    printDiv();

    std::vector<PREC> flattened;
    flattened.reserve(n * model_attribs.size());
    size_t reserve_size = 0;
    for (int i=0; i<model_attribs.size(); ++i)
      reserve_size += model_attribs[i].size();
    flattened.reserve(reserve_size);
    for (int i=0; i<model_attribs.size(); ++i){
      const std::vector<PREC> & v = model_attribs[i]; 
      flattened.insert( flattened.end() , v.begin() , v.end() );
    }

    match(pattribs[GPU_ID][MODEL_ID])([&](auto &pa) {
      checkCudaErrors(cudaMemcpyAsync((void *)&pa.val_1d(_0, 0), flattened.data(),
                      sizeof(PREC) * n * model_attribs.size(),
                      cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
    });
    fmt::print(fg(fmt::color::green), "GPU[{}] MODEL[{}] Initialized output device attribute vector of vectors with [{}] particles and [{}] attributes.\n", GPU_ID, MODEL_ID, model_attribs.size(), model_attribs[0].size());
    printDiv();
  }


  // Initialize FEM vertices and elements
  template<fem_e f>
  void initFEM(int GPU_ID, const std::vector<std::array<PREC, 13>> &input_vertices,
                const std::vector<std::array<int, 4>> &input_elements,
                const std::vector<std::array<PREC, 6>> &input_element_attribs) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    
    flag_fem[GPU_ID] = true; // Set true flag for Finite Element Method
    fmt::print("GPU[{}] Allocating FEM d_vertices.\n", GPU_ID);
    d_vertices[GPU_ID] = spawn<vertice_array_13_, orphan_signature>(device_allocator{}); //< FEM vertices
    printDiv();
    fmt::print("GPU[{}] Allocating FEM d_element_IDs.\n", GPU_ID);
    d_element_IDs[GPU_ID] = spawn<element_array_, orphan_signature>(device_allocator{}); //< FEM elements
    cuDev.syncStream<streamIdx::Compute>();
    printDiv();

    fmt::print("GPU[{}] Allocating FEM elementBins.\n", GPU_ID);
    cuDev.syncStream<streamIdx::Compute>();
    elementBins.emplace_back(ElementBuffer<f>(device_allocator{}));
    printDiv();

    vertice_cnt[GPU_ID] = input_vertices.size(); // Vertice count
    element_cnt[GPU_ID] = input_elements.size(); // Element count

    // Set FEM vertices in GPU array
    fmt::print("GPU[{}] Initialize device array with {} vertices.\n", GPU_ID, vertice_cnt[GPU_ID]);
    cudaMemcpyAsync((void *)&d_vertices[GPU_ID].val_1d(_0, 0), input_vertices.data(),
                    sizeof(std::array<PREC, 13>) * vertice_cnt[GPU_ID],
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    // Set FEM elements in GPU array
    fmt::print("GPU[{}] Initialize FEM elements with {} ID arrays\n", GPU_ID, element_cnt[GPU_ID]);
    cudaMemcpyAsync((void *)&d_element_IDs[GPU_ID].val_1d(_0, 0), input_elements.data(),
                    sizeof(std::array<int, 4>) * element_cnt[GPU_ID],
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    // Set FEM elements in GPU array
    fmt::print("GPU[{}] Initialize FEM element attribs with {} arrays\n", GPU_ID, element_cnt[GPU_ID]);
    cudaMemcpyAsync((void *)&d_element_attribs[GPU_ID].val_1d(_0, 0), input_element_attribs.data(),
                    sizeof(std::array<PREC, 6>) * element_cnt[GPU_ID],
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    host_element_IDs[GPU_ID].resize(element_cnt[GPU_ID]);
    host_element_attribs[GPU_ID].resize(element_cnt[GPU_ID]);
    printDiv();
  }

// ! Mostly deprecated, was used to have SDF set boundary conditions
// TODO : Reimplement for engineering + improve memory usage
  void initBoundary(std::string fn) {
    initFromSignedDistanceFile(fn,
                               vec<std::size_t, 3>{(std::size_t)1024,
                                                   (std::size_t)1024,
                                                   (std::size_t)512},
                               _hostData);
    for (int did = 0; did < g_device_cnt; ++did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      cuDev.setContext();
      fmt::print("GPU[{}] Allocating SDF collisionObjs.\n", did);
      collisionObjs[did] = SignedDistanceGrid{device_allocator{}};
      collisionObjs[did]->init(_hostData, cuDev.stream_compute());
      cuDev.syncStream<streamIdx::Compute>();
    }
  }

  /// Initialize target from host setting (&input_gridTarget), output as *.bgeo (JB)
  void initGridTarget(int GPU_ID,
                      const std::vector<std::array<PREC_G, g_grid_target_attribs>> &input_gridTarget, 
                      const vec<PREC_G, 7> &host_target,  
                      float freq) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    fmt::print("Entered initGridTarget in mgsp_benchmark.cuh.\n");
    flag_gt = true;
    // Set points a/b (device) for grid-target volume using (host, from JSON)
    if (GPU_ID == 0)  {
      number_of_grid_targets += 1; 
      d_grid_target.emplace_back();
      grid_tarcnt.emplace_back();
      for (int d = 0; d < 7; d++)
        d_grid_target.back()[d] = host_target[d];
    }
    int target_ID = number_of_grid_targets-1;
    host_gt_freq = freq; // Set output frequency [Hz] for target (host)
    std::string fn_force = std::string{"gridTarget"} + "[" + std::to_string(target_ID)+"]_dev[" + std::to_string(GPU_ID) + "].csv";
    gridTargetFile[GPU_ID].open (fn_force, std::ios::out | std::ios::trunc); // Initialize *.csv
    gridTargetFile[GPU_ID] << "Time [s]" << "," << "Force [n]" << "\n";
    gridTargetFile[GPU_ID].close();

    grid_tarcnt.back()[GPU_ID] = input_gridTarget.size(); // Set size
    printf("GPU[%d] Target[%d] node count: %d \n", GPU_ID, target_ID, grid_tarcnt[target_ID][GPU_ID]);

    /// Populate target (device) with data from target (host) (JB)
    cudaMemcpyAsync((void *)&d_gridTarget[GPU_ID].val_1d(_0, 0), input_gridTarget.data(),
                    sizeof(std::array<PREC_G, g_grid_target_attribs>) *  input_gridTarget.size(),
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    /// Write target data to a *.bgeo output file using Partio  
    std::string fn = std::string{"gridTarget"}  +"[" + std::to_string(target_ID) + "]" + "_dev[" + std::to_string(GPU_ID) + "]_frame[-1]" + save_suffix;
    IO::insert_job([fn, input_gridTarget]() { write_partio_gridTarget<float, g_grid_target_attribs>(fn, input_gridTarget); });
    IO::flush();
  }

  /// Initialize particleTarget from host setting
  void initParticleTarget(int GPU_ID, int MODEL_ID,
                      const std::vector<std::array<PREC, g_particle_target_attribs>> &input_particleTarget, 
                      const vec<PREC, 7> &host_target,  
                      float freq) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    fmt::print("Entered initParticleTarget in mgsp_benchmark.cuh.\n");
    // Set points a/b (device) for particle-target volume using (host, from JSON)
    if (MODEL_ID >= getModelCnt(GPU_ID)) return;
    if (MODEL_ID == 0 && GPU_ID == 0) 
    {
      number_of_particle_targets += 1; 
      d_particle_target.emplace_back();
      particle_tarcnt.emplace_back();
      for (int d = 0; d < 7; d++)
        d_particle_target.back()[d] = host_target[d];
    }
    flag_pt = true; // Set flag for particle-target
    host_pt_freq = freq; // Set output frequency [Hz] for particle-target aggregate value
    int particle_target_ID = number_of_particle_targets-1;
    std::string fn_particle_target = std::string{"particleTarget"} + "[" + std::to_string(particle_target_ID) + "]_model[" + std::to_string(MODEL_ID) + "]_dev[" + std::to_string(GPU_ID) + "]" + ".csv";
    particleTargetFile[GPU_ID][MODEL_ID].open (fn_particle_target, std::ios::out | std::ios::trunc); 
    particleTargetFile[GPU_ID][MODEL_ID] << "Time" << "," << "Aggregate" << "\n";
    particleTargetFile[GPU_ID][MODEL_ID].close();

    particle_tarcnt.back()[GPU_ID] = input_particleTarget.size(); // Set size
    fmt::print("GPU[{}] MODEL[{}] particleTarget[{}] particle count: {} \n", GPU_ID, MODEL_ID, particle_target_ID, particle_tarcnt[particle_target_ID][GPU_ID]);

    /// Populate target (device) with data from target (host) (JB)
    cudaMemcpyAsync((void *)&d_particleTarget[GPU_ID].val_1d(_0, 0), input_particleTarget.data(),
                    sizeof(std::array<PREC, g_particle_target_attribs>) *  input_particleTarget.size(),
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();
    fmt::print("Finished initializing particleTarget[{}] in initParticleTarget mgsp_benchmark.cuh.\n", particle_target_ID);

    /// Write target data to a *.bgeo output file using Partio  
    std::string fn = std::string{"particleTarget"}  +"[" + std::to_string(particle_target_ID) + "]" + "_model["+ std::to_string(MODEL_ID) + "]" + "_dev[" + std::to_string(GPU_ID) + "]_frame[-1]" + save_suffix;
    IO::insert_job([fn, input_particleTarget]() { write_partio_particleTarget<PREC, g_particle_target_attribs>(fn, input_particleTarget); });
    IO::flush();
  }

  /// Initialize basic boundaries on the grid
  void initGridBoundaries(int GPU_ID,
                      const vec<PREC_G, g_grid_boundary_attribs> &host_gridBoundary,
                      const GridBoundaryConfigs &gridBoundaryConfigs,
                      int boundary_ID) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    fmt::print("Entered initGridBoundaries in simulator.\n");
    // Initialize placeholder boundaries

    h_gridBoundaryConfigs[boundary_ID]._ID = gridBoundaryConfigs._ID;
    h_gridBoundaryConfigs[boundary_ID]._object = gridBoundaryConfigs._object;
    h_gridBoundaryConfigs[boundary_ID]._contact = gridBoundaryConfigs._contact;
    h_gridBoundaryConfigs[boundary_ID]._domain_start = gridBoundaryConfigs._domain_start;
    h_gridBoundaryConfigs[boundary_ID]._domain_end = gridBoundaryConfigs._domain_end;
    h_gridBoundaryConfigs[boundary_ID]._time = gridBoundaryConfigs._time;
    h_gridBoundaryConfigs[boundary_ID]._velocity = gridBoundaryConfigs._velocity;
    h_gridBoundaryConfigs[boundary_ID]._array = gridBoundaryConfigs._array;
    h_gridBoundaryConfigs[boundary_ID]._spacing = gridBoundaryConfigs._spacing;
    
    h_gridBoundaryConfigs[boundary_ID]._friction_static = gridBoundaryConfigs._friction_static;
    h_gridBoundaryConfigs[boundary_ID]._friction_dynamic = gridBoundaryConfigs._friction_dynamic;

    fmt::print("gridBoundary[{}]: object[{}], contact[{}]\n", boundary_ID, h_gridBoundaryConfigs[boundary_ID]._object, h_gridBoundaryConfigs[boundary_ID]._contact);
    if (boundary_ID == 0)
    {
      for (int b=0; b<g_max_grid_boundaries; b++)
        for (int d=0; d<g_grid_boundary_attribs; d++) 
        gridBoundary[b][d] = -1;
    
    }
    for (int d=0; d<g_grid_boundary_attribs; d++) 
      gridBoundary[boundary_ID][d] = host_gridBoundary[d];
    cuDev.syncStream<streamIdx::Compute>();

  }

  /// Init OSU Motion-Path on device (device_motionPath) from host (&host_motionPath) (JB)
  void initMotionPath(int GPU_ID,
                     const std::vector<std::array<PREC_G, 3>> &motionPath, 
                     float frequency) {
    auto &cuDev = Cuda::ref_cuda_context(GPU_ID);
    cuDev.setContext();
    fmt::print("Entered initMotionPath. \n");
    flag_mp = 1;
    host_mp_freq = frequency; // Frequency of motion file rows, 1 / time-step
    host_motionPath = motionPath;
    /// Populate Motion-Path (device) with data from Motion-Path (host) (JB)
    for (int d = 0; d < 3; d++) d_motionPath[d] = (PREC_G)host_motionPath[0][d]; //< Set vals
    fmt::print("Init motionPath with time {}s, disp {}m, vel {}m/s \n", d_motionPath[0], d_motionPath[1], d_motionPath[2]);
  }  
  
  /// Set a Motion-Path for time-step (JB)
  void setMotionPath(std::vector<std::array<PREC_G, 3>> &host_motionPath, float curTime) {
    if (flag_mp) {
      float check_point_time = 0.f; // TODO : Time of check-point 
      int step = (int)((curTime + check_point_time)  * host_mp_freq); //< Index for time
      if (step >= host_motionPath.size()) step = (int)(host_motionPath.size() - 1); //< Index-limit
      else if (step < 0) step = 0;
      for (int d = 0; d < 3; d++) d_motionPath[d] = (PREC_G)host_motionPath[step][d]; //< Set vals
      fmt::print("Set motionPath with step[{}], dt[{} s], time[{} s], disp[{} m], vel[{} m/s]\n", step, (1.0/host_mp_freq), d_motionPath[0], d_motionPath[1], d_motionPath[2]);
    }
    else 
      for (int d = 0; d < 3; d++) d_motionPath[d] = 0.f;
  }

  template<material_e mt>
  void updateParameters(int did, int mid, MaterialConfigs materialConfigs,
                              AlgoConfigs algoConfigs, 
                              std::vector<std::string> names,
                              std::vector<int> trackIDs, std::vector<std::string> trackNames,
                              std::vector<std::string> targetNames) {
    //int mid = mid + did * g_models_per_gpu;
    if (mid >= getModelCnt(did)) {
      fmt::print("Error: mid[{}] > getModelCnt(did)[{}] in updateParameters mgsp_benchmark.cuh.\n", mid, getModelCnt(did));
      return;
    }
    for (int copyid = 0; copyid<2; copyid++) {
      match(particleBins[copyid][did][mid])([&](auto &pb) {},
          [&](ParticleBuffer<mt> &pb) {
            pb.updateParameters(length, materialConfigs, algoConfigs);
            pb.updateOutputs(names);
            pb.updateTrack(trackNames, trackIDs);
            pb.updateTargets(targetNames);
          });
    }
    return;
  }
  

  void updateMeshedParameters(int did, MaterialConfigs materialConfigs,
                              AlgoConfigs algoConfigs, 
                              std::vector<std::string> names) 
  {
    int mid = 0;
    match(particleBins[0][did][mid])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::Meshed> &pb) {
          pb.updateParameters(length, materialConfigs, algoConfigs);
          pb.updateOutputs(names);
        });
    match(particleBins[1][did][mid])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::Meshed> &pb) {
          pb.updateParameters(length, materialConfigs, algoConfigs);
          pb.updateOutputs(names);
        });
    match(elementBins[did])(
        [&](auto &eb) {},
        [&](ElementBuffer<fem_e::Tetrahedron> &eb) {
          eb.updateParameters(length, materialConfigs, algoConfigs);
        });   
    std::cout << "Update Meshed parameters." << '\n';
 
  }
  void updateMeshedFBARParameters(int did, MaterialConfigs materialConfigs,
                              AlgoConfigs algoConfigs, 
                              std::vector<std::string> names) 
  {    
    int mid = 0;
    match(particleBins[0][did][mid])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::Meshed> &pb) {
          pb.updateParameters(length, materialConfigs, algoConfigs);
          pb.updateOutputs(names);
        });
    match(particleBins[1][did][mid])(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::Meshed> &pb) {
          pb.updateParameters(length, materialConfigs, algoConfigs);
          pb.updateOutputs(names);
        });
    match(elementBins[did])(
        [&](auto &eb) {},
        [&](ElementBuffer<fem_e::Tetrahedron_FBar> &eb) {
          eb.updateParameters(length, materialConfigs, algoConfigs);
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
    void *d_tmp = tmps[cuDev.getDevId()].d_tmp;
    checkCudaErrors(cub::DeviceScan::ExclusiveScan(d_tmp, temp_storage_bytes,
                                                   in, out, plus_op, 0, cnt,
                                                   cuDev.stream_compute()));
#endif
  }
  // Used to set initial mass, momentum on grid
  PREC getMass(int did, int mid) {
    return match(particleBins[rollid][did][mid])(
        [&](const auto &particleBuffer) { return (PREC)particleBuffer.mass; });
  }
  // Used to set initial volume on grid (Simple FBar Method)
  PREC getVolume(int did, int mid) {
    return match(particleBins[rollid][did][mid])(
        [&](const auto &particleBuffer) { return (PREC)particleBuffer.volume; });
  }
  int getModelCnt(int did) const noexcept { return (int)h_model_cnt[did]; }
  void checkCapacity(int did) {
    // Check active block capacity
    if (ebcnt[did] > curNumActiveBlocks[did] * 4/5 && checkedCnts[did][0] == 0) {
      curNumActiveBlocks[did] = std::max(curNumActiveBlocks[did] * 5/4, (std::size_t)ebcnt[did]);
      checkedCnts[did][0] = 2; // 2 means resize of GPU blocks needed
      fmt::print(fg(fmt::color::orange), "GPU[{}] Will resize active blocks soon: [{}] -> [{}]\n", did, ebcnt[did], curNumActiveBlocks[did]);
    }
    // Check active particle bin capacity
    if (g_buckets_on_particle_buffer) {
      for (int mid = 0; mid < getModelCnt(did); mid++) {
        if (bincnt[did][mid] >= curNumActiveBins[did][mid] * 9/10 && checkedBinCnts[did][mid] == 0) {
          curNumActiveBins[did][mid] = std::max(curNumActiveBins[did][mid] * 9/8, (std::size_t)bincnt[did][mid]);
          checkedBinCnts[did][mid] = 2; // 2 means resize of GPU model's bins needed
          fmt::print(fg(fmt::color::orange),"GPU[{}] MODEL[{}] Will resize active particle bins soon: [{}] -> [{}]\n", did, mid, bincnt[did][mid], curNumActiveBins[did][mid]);
        } 
      }
    } else {
      if (bincnt[did][0] >= curNumActiveBins[did][0] * 9/10 && checkedCnts[did][1] == 0) {
        curNumActiveBins[did][0] = std::max(curNumActiveBins[did][0] * 9/8, (std::size_t)bincnt[did][0]);
        checkedBinCnts[did][0] = 2; // 2 means resize of GPU's bins needed
        fmt::print(fg(fmt::color::orange), "GPU[{}] Will resize active particle bins soon: [{}] -> [{}]\n", did, bincnt[did][0], curNumActiveBins[did][0]);
      } 
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
               "{}-th GPU worker operates on GPU[{}].\n", did, cuDev.getDevId());
    while (this->bRunning) {
      wait();
      auto job = this->jobs[did].try_pop();
      if (job)
        (*job)(did);
      signal();
    }
    fmt::print(fg(fmt::color::light_blue), "{}-th GPU worker exits GPU[{}].\n", did, cuDev.getDevId());
  }
  void sync() {
    std::unique_lock<std::mutex> lk{mut_ctrl};
    cv_ctrl.wait(lk, [this]() { return this->idleCnt == g_device_cnt; });
    printDiv();
  }
  void issue(std::function<void(int)> job) {
    std::unique_lock<std::mutex> lk{mut_slave};
    for (int did = 0; did < g_device_cnt; ++did)
      jobs[did].push(job);
    idleCnt = 0;
    lk.unlock();
    cv_slave.notify_all();
  }
  void main_loop() {
    curFrame = 0;
    nextTime = curTime + 1.0 / fps; // Initial next time
    dt = dtDefault;
    //dt = compute_dt(0.f, curTime, nextTime, dtDefault);
    fmt::print(fmt::emphasis::bold, "curTime[{}] --dt[{}]--> nextTime[{}], defaultDt[{}].\n", curTime, dt, nextTime, dtDefault);
    initial_setup();
    fmt::print("Begin main loop.\n");
    curTime += dt;
    for (curFrame = 1; curFrame <= nframes; ++curFrame) {
      for (; curTime < nextTime; curTime += dt, curStep++) {
        setMotionPath(host_motionPath, curTime); //< Update motion-path for time
        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          cuDev.setContext(); // JB
          checkCapacity(did);
          PREC_G *d_maxVel = tmps[did].d_maxVel;
          PREC_G *d_kinetic_energy_grid = tmps[did].d_kinetic_energy_grid;
          PREC_G *d_gravity_energy_grid = tmps[did].d_gravity_energy_grid;

          CudaTimer timer{cuDev.stream_compute()};
          timer.tick();
          checkCudaErrors(cudaMemsetAsync(d_maxVel, 0, sizeof(PREC_G),
                                          cuDev.stream_compute()));
          checkCudaErrors(cudaMemsetAsync(d_kinetic_energy_grid, 0, sizeof(PREC_G),
                                          cuDev.stream_compute()));
          checkCudaErrors(cudaMemsetAsync(d_gravity_energy_grid, 0, sizeof(PREC_G),
                                          cuDev.stream_compute()));
          //if (curStep == 0) dt = dt/2; //< Init. grid vel. update shifted 1/2 dt. Leap-frog time-integration instead of symplectic Euler for extra stability
          // Grid Update
          if (collisionObjs[did]) // If using SDF boundaries
            cuDev.compute_launch(
                {(nbcnt[did] + g_num_grid_blocks_per_cuda_block - 1) /
                     g_num_grid_blocks_per_cuda_block,
                 g_num_warps_per_cuda_block * 32, g_num_warps_per_cuda_block * sizeof(PREC_G)},
                update_grid_velocity_query_max, (uint32_t)nbcnt[did],
                gridBlocks[0][did], partitions[rollid][did], dt,
                (const SignedDistanceGrid)(*collisionObjs[did]), d_maxVel, curTime, grav[1]);
          else { // If using basic geometry boundaries
            cuDev.compute_launch(
                {(nbcnt[did] + g_num_grid_blocks_per_cuda_block - 1) /
                     g_num_grid_blocks_per_cuda_block,
                 g_num_warps_per_cuda_block * 32, g_num_warps_per_cuda_block * sizeof(PREC_G)},
                update_grid_velocity_query_max, (uint32_t)nbcnt[did],
                gridBlocks[0][did], partitions[rollid][did], dt, d_maxVel, curTime, grav, gridBoundary, d_gridBoundaryConfigs, d_motionPath, length);
            

            // TODO: Query grid-energy at user-specified frequency
            if (flag_ge && (fmod(curTime, 1/host_ge_freq) < dt || curTime + dt > nextTime)) {
              cuDev.syncStream<streamIdx::Compute>(); // May not be needed
              cuDev.compute_launch(
                  {(nbcnt[did] + g_num_grid_blocks_per_cuda_block - 1) / g_num_grid_blocks_per_cuda_block, g_num_warps_per_cuda_block * 32, g_num_warps_per_cuda_block * sizeof(PREC_G)},
                  query_energy_grid, (uint32_t)nbcnt[did],
                  gridBlocks[0][did], partitions[rollid][did], dt, d_kinetic_energy_grid, d_gravity_energy_grid, curTime, grav[1], gridBoundary, d_motionPath, length);
            }
          }
          cuDev.syncStream<streamIdx::Compute>();
          if (flag_ge && (fmod(curTime, 1/host_ge_freq) < dt || curTime + dt > nextTime)) {
            checkCudaErrors(cudaMemcpyAsync(&maxVels[did], d_maxVel,
                                            sizeof(PREC_G), cudaMemcpyDefault,
                                            cuDev.stream_compute()));
            checkCudaErrors(cudaMemcpyAsync(&kinetic_energy_grid_vals[did], d_kinetic_energy_grid,
                                            sizeof(PREC_G), cudaMemcpyDefault,
                                            cuDev.stream_compute()));
            checkCudaErrors(cudaMemcpyAsync(&gravity_energy_grid_vals[did], d_gravity_energy_grid,
                                            sizeof(PREC_G), cudaMemcpyDefault,
                                            cuDev.stream_compute()));
          }
          timer.tock(fmt::format("GPU[{}] frame {} step {} grid_update_query.\n",
                                 did, curFrame, curStep));
        });
        sync();
        //if (curStep == 0) dt = dt*2; //< Use regular time-step after leap-frog init.

        /// * Host: Aggregate values from grid update
        PREC_G maxVel = 0.0; //< Velocity max across all GPUs
        for (int did = 0; did < g_device_cnt; ++did)
          if (maxVels[did] > maxVel) maxVel = maxVels[did];
        maxVel = std::sqrt(maxVel);

        PREC_G sum_kinetic_energy_grid = 0.0; //< Kinetic energy summed across all GPUs
        PREC_G sum_gravity_energy_grid = 0.0; //< Gravity energy summed across all GPUs
        if (flag_ge && (fmod(curTime, 1/host_ge_freq) < dt || curTime + dt > nextTime)){
          for (int did = 0; did < g_device_cnt; ++did) {
            if (maxVels[did] > maxVel) maxVel = maxVels[did];
            sum_kinetic_energy_grid += kinetic_energy_grid_vals[did];
            sum_gravity_energy_grid += gravity_energy_grid_vals[did];
          }
        }

        if (false) nextDt = compute_dt(maxVel, curTime, nextTime, dtDefault);
        else nextDt = dtDefault;
        fmt::print(fmt::emphasis::bold, "curTime[{}] [s] --nextDt[{}]--> nextTime[{}] [s], defaultDt[{}] [s], maxVel[{}] [m/s], sum_kinetic_energy_grid[{}] [J]\n", curTime,  nextDt, nextTime, dtDefault, (maxVel*length), sum_kinetic_energy_grid);

        // * Run g2p2g, g2p-p2g, or g2p2v-v2fem2v-v2p2g pipeline on each GPU
        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          CudaTimer timer{cuDev.stream_compute()};

          // Check capacity of particle bins, resize if needed
          if (g_buckets_on_particle_buffer) {
            for (int mid=0; mid<getModelCnt(did); mid++) {
              if (checkedBinCnts[did][mid] > 0) {
                match(particleBins[rollid ^ 1][did][mid])([&](auto &pb) {
                  pb.resize(device_allocator{}, curNumActiveBins[did][mid]);
                });
                checkedBinCnts[did][mid]--;
                cuDev.syncStream<streamIdx::Compute>(); // JB
              }
            }
          } else {
            if (checkedCnts[did][1] > 0) {
              int mid = 0;
              match(particleBins[rollid ^ 1][did][mid])([&](auto &pb) {
                pb.resize(device_allocator{}, curNumActiveBins[did][mid]);
              });
              checkedCnts[did][1]--;
            }
          }

          //timer.tick();
          gridBlocks[1][did].reset(nbcnt[did], cuDev);

          // Advection map
          if (g_buckets_on_particle_buffer) { 
            for (int mid=0; mid<getModelCnt(did); mid++) {
              match(particleBins[rollid ^ 1][did][mid])([&](auto &pb) {
                checkCudaErrors(cudaMemsetAsync(pb._ppcs, 0, sizeof(int) * ebcnt[did] * g_blockvolume, cuDev.stream_compute())); 
              }); 
            }
          } else {
            checkCudaErrors(cudaMemsetAsync(partitions[rollid][did]._ppcs, 0,
                                sizeof(int) * ebcnt[did] * g_blockvolume,
                                cuDev.stream_compute()));
          }


          // Loop over all particleBins models on this GPU
          for (int mid=0; mid<getModelCnt(did); mid++) {
            // int mid = mid + did*g_models_per_gpu;
            match(particleBins[rollid][did][mid])([&](const auto &pb) {

              // Set shared memory carveout initially, once, for functions
              // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
              //if (curFrame == 1) {
              // Grid-to-Particle-to-Grid - g2p2g
              checkCudaErrors(cudaFuncSetAttribute(g2p2g<std::decay_t<decltype(particleBins[rollid][did][mid])>, std::decay_t<decltype(partitions[rollid ^ 1][did])>, std::decay_t<decltype(gridBlocks[0][did])>>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
              
              // Grid-to-Particle - F-Bar Update
              checkCudaErrors(cudaFuncSetAttribute(g2p_FBar<std::decay_t<decltype(particleBins[rollid][did][mid])>, std::decay_t<decltype(partitions[rollid ^ 1][did])>, std::decay_t<decltype(gridBlocks[0][did])>>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
              // Particle-to-Grid - F-Bar Update
              checkCudaErrors(cudaFuncSetAttribute(p2g_FBar<std::decay_t<decltype(particleBins[rollid][did][mid])>, std::decay_t<decltype(partitions[rollid ^ 1][did])>, std::decay_t<decltype(gridBlocks[0][did])>>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
              //}
              
              // Grid-to-Particle-to-Grid - g2p2g 
              if (!pb.use_ASFLIP && !pb.use_FEM && !pb.use_FBAR) {
                fmt::print("GPU[{}] ERROR: Still need to reimplement g2p2g without ASFLIP, FBAR, or FEM. Try turning on ASFLIP or FBAR.\n", did);
              }

              // Grid-to-Particle-to-Grid - g2p2g 
              else if (pb.use_ASFLIP && !pb.use_FEM && !pb.use_FBAR) {
                if (partitions[rollid][did].h_count) {
                  cuDev.compute_launch(
                      {partitions[rollid][did].h_count, g_particle_batch, (6 * (g_blockvolume << 3) * sizeof(PREC_G) + 7 * (g_blockvolume << 3) * sizeof(PREC_G))},
                      g2p2g, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did]);
                }
              }
          
              // Grid-to-Particle - F-Bar Update - Particle-to-Grid
              else if (!pb.use_ASFLIP && !pb.use_FEM && pb.use_FBAR) {
                timer.tick();
                // g2g_FBar Halo
                if (partitions[rollid][did].h_count) {
                  cuDev.compute_launch({partitions[rollid][did].h_count, g_particle_batch,
                      (512 * 3 * sizeof(PREC_G)) + (512 * 2 * sizeof(PREC_G))},
                      g2p_FBar, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did]);
                }
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] MODEL[{}] frame {} step {} halo_g2p_FBar", did, mid, curFrame, curStep));
              
                // p2g_FBar Halo
                timer.tick();
                if (partitions[rollid][did].h_count) {
                  cuDev.compute_launch({partitions[rollid][did].h_count, g_particle_batch,
                      (512 * 5 * sizeof(PREC_G)) + (512 * 4 * sizeof(PREC_G))},
                      p2g_FBar, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did]);
                }
                timer.tock(fmt::format("GPU[{}] MODEL[{}] frame {} step {} halo_p2g_FBar", did, mid, curFrame, curStep));
              } //< End F-Bar

              // Grid-to-Particle - F-Bar Update - Particle-to-Grid
              else if (pb.use_ASFLIP && !pb.use_FEM && pb.use_FBAR) {
                timer.tick();
                // g2g_FBar Halo
                int shmem = (3 + 2) * (512 * sizeof(PREC_G));
                if (partitions[rollid][did].h_count) {
                  cuDev.compute_launch({partitions[rollid][did].h_count, g_particle_batch,
                      shmem},
                      g2p_FBar, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did]);
                }
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] MODEL[{}] frame {} step {} halo_g2p_ASFLIP_FBar", did, mid, curFrame, curStep));
              
                // p2g_FBar Halo
                timer.tick();
#if DEBUG_COUPLED_UP
                  shmem = (10 + 9) * (g_arenavolume * sizeof(PREC_G));
#else
                  shmem = (8 + 7) * (g_arenavolume * sizeof(PREC_G));
#endif
                if (partitions[rollid][did].h_count) {

                  cuDev.compute_launch({partitions[rollid][did].h_count, g_particle_batch,
                      shmem},
                      p2g_FBar, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did]);
                }
                timer.tock(fmt::format("GPU[{}] MODEL[{}] frame {} step {} halo_p2g_ASFLIP_FBar", did, mid, curFrame, curStep));
              } //< End ASFLIP + F-Bar

              // Grid-to-Vertices - Update FEM - Vertices-to-Grid
              else if (pb.use_ASFLIP && pb.use_FEM && !pb.use_FBAR) {
                // g2p2v - Halo
                timer.tick();
                if (partitions[rollid][did].h_count) 
                  cuDev.compute_launch({partitions[rollid][did].h_count, g_particle_batch,
                      (512 * 6 * 4)},
                      g2p2v, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did],
                      d_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} halo_g2p2v", did,
                                      curFrame, curStep));
                // g2p2v - Non-Halo
                timer.tick();
                cuDev.compute_launch({pbcnt[did], g_particle_batch, (512 * 6 * 4)}, 
                    g2p2v, dt, nextDt, 
                    (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did],
                    d_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non-halo_g2p2g", did,
                                      curFrame, curStep));
                // v2fem2v - Halo and Non-Halo
                timer.tick();
                match(elementBins[did])([&](const auto &eb) {
                  cuDev.compute_launch({(element_cnt[did] - 1) / 32 + 1, 32},
                      v2fem2v, (uint32_t)element_cnt[did], dt, nextDt,
                      d_vertices[did], d_element_IDs[did], eb);
                });
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} v2fem2v", did,
                                      curFrame, curStep));
                // v2p2g - Halo
                timer.tick();
                if (partitions[rollid][did].h_count)
                  cuDev.compute_launch({partitions[rollid][did].h_count, g_particle_batch,
                      (512 * 6 * 4) + (512 * 7 * 4)},
                      v2p2g, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did],
                      d_vertices[did]);
                timer.tock(fmt::format("GPU[{}] frame {} step {} halo_v2p2g", did,
                                      curFrame, curStep));
              } //< End FEM

              // Grid-to-Vertices - Update FEM + Simple F-Bar - Vertices-to-Grid
              else if (pb.use_ASFLIP && pb.use_FEM && pb.use_FBAR) {
                // g2p2v - F-Bar - Halo
                timer.tick();
                if (partitions[rollid][did].h_count)
                  cuDev.compute_launch({partitions[rollid][did].h_count, g_particle_batch, (512 * 6 * 4)},
                      g2p2v_FBar, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did],
                      d_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} halo_g2p2v_FBar", did,
                                      curFrame, curStep));
            
                // g2p2v - F-Bar - Non-Halo
                timer.tick();
                cuDev.compute_launch({pbcnt[did], g_particle_batch, (512 * 6 * 4)}, 
                    g2p2v_FBar, dt, nextDt, 
                    (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did],
                    d_vertices[did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non-halo_g2p2v_FBar", did,
                                      curFrame, curStep));

                // v2fem_FBar - Halo and Non-Halo
                timer.tick();
                match(elementBins[did])([&](const auto &eb) {
                    cuDev.compute_launch({(element_cnt[did] - 1) / 32 + 1, 32},
                        v2fem_FBar, (uint32_t)element_cnt[did], dt, nextDt,
                        d_vertices[did], d_element_IDs[did], eb);
                });
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} v2fem_FBar", did,
                                      curFrame, curStep));
                // fem2v_FBar - Halo and Non-Halo
                timer.tick();
                match(elementBins[did])([&](const auto &eb) {
                    cuDev.compute_launch({(element_cnt[did] - 1) / 32 + 1, 32},
                        fem2v_FBar, (uint32_t)element_cnt[did], dt, nextDt,
                        d_vertices[did], d_element_IDs[did], eb);
                });
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} fem2v_FBar", did,
                                      curFrame, curStep));
                // v2p2g_FBar - Halo
                timer.tick();
                if (partitions[rollid][did].h_count) {
                  cuDev.compute_launch(
                      {partitions[rollid][did].h_count, g_particle_batch, (512 * 6 * 4) + (512 * 7 * 4)},
                      v2p2g_FBar, dt, nextDt,
                      (const ivec3 *)partitions[rollid][did]._haloBlocks, pb,
                      get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did],
                      d_vertices[did]);
                }
                timer.tock(fmt::format("GPU[{}] frame {} step {} halo_v2p2g_FBar", did,
                                      curFrame, curStep));
              }  //< End FEM + F-Bar
            }); //< End Match for ParticleBins
            cuDev.syncStream<streamIdx::Compute>(); // JB
          } //< End Model Loop
        }); //< End Algorithm
        sync(); //< Sync all worker GPUs

        collect_halo_grid_blocks();

        /// * Non-Halo
        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          CudaTimer timer{cuDev.stream_compute()};

          //timer.tick();
          for (int mid=0; mid<getModelCnt(did); mid++) {
            match(particleBins[rollid][did][mid])([&](const auto &pb) {
              // Grid-to-Particle-to-Grid - No ASFLIP
              if (!pb.use_ASFLIP  && !pb.use_FEM && !pb.use_FBAR) {
                int shmem = (3 + 4) * (512 * sizeof(PREC_G));
                cuDev.compute_launch(
                    {pbcnt[did], g_particle_batch, shmem}, g2p2g, dt,
                    nextDt, (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did]);
              }
                
              // Grid-to-Particle-to-Grid - ASFLIP
              else if (pb.use_ASFLIP && !pb.use_FEM && !pb.use_FBAR) {
                // g2p2g - Non-Halo
                timer.tick();
                int shmem = (6 + 7) * (512 * sizeof(PREC_G));
                cuDev.compute_launch(
                    {pbcnt[did], g_particle_batch, shmem}, g2p2g, dt,
                    nextDt, (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(
                        particleBins[rollid ^ 1][did][mid]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did]);
                timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_g2p2g", did,
                                      curFrame, curStep));
              } //< End G2P2G (MPM + ASFLIP)

              // Grid-to-Particle - Update F-Bar - Particle-to-Grid 
              else if (!pb.use_ASFLIP && !pb.use_FEM && pb.use_FBAR) {
                // g2p F-Bar - Non-Halo
                timer.tick();
                // Recheck this
                int shmem_g2p = (3*216 + 2*512) * (sizeof(PREC_G));
                cuDev.compute_launch(
                    {pbcnt[did], g_particle_batch, shmem_g2p}, 
                    g2p_FBar, dt, nextDt, 
                    (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(
                        particleBins[rollid ^ 1][did][mid]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_g2p_FBar", did,
                                      curFrame, curStep));          
                // p2g F-Bar - Non-Halo
                timer.tick();
                int shmem_p2g = (5 + 4) * (512 * sizeof(PREC_G));
                cuDev.compute_launch(
                    {pbcnt[did], g_particle_batch, shmem_p2g}, 
                    p2g_FBar, dt, nextDt, 
                    (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did]);
                timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_p2g_FBar", did,
                                      curFrame, curStep));
              } //< End Non-Halo F-Bar 

              // Grid-to-Particle - Update F-Bar - Particle-to-Grid ASFLIP
              else if (pb.use_ASFLIP && !pb.use_FEM && pb.use_FBAR) {
                // g2p F-Bar ASFLIP - Non-Halo
                timer.tick();
                int shmem = (3 + 2) * (g_arenavolume * sizeof(PREC_G));
                cuDev.compute_launch(
                    {pbcnt[did], g_particle_batch, shmem}, 
                    g2p_FBar, dt, nextDt, 
                    (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(
                        particleBins[rollid ^ 1][did][mid]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did]);
                cuDev.syncStream<streamIdx::Compute>();
                timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_g2p_FBar_ASFLIP", did,
                                      curFrame, curStep));          

                // p2g F-Bar - Non-Halo
                timer.tick();
#if DEBUG_COUPLED_UP
                shmem = (10 + 9) * (g_arenavolume * sizeof(PREC_G));
#else 
                shmem = (8 + 7) * (g_arenavolume * sizeof(PREC_G));
#endif
                cuDev.compute_launch(
                    {pbcnt[did], g_particle_batch, shmem}, 
                    p2g_FBar, dt, nextDt, 
                    (const ivec3 *)nullptr, pb,
                    get<typename std::decay_t<decltype(pb)>>(
                        particleBins[rollid ^ 1][did][mid]),
                    partitions[rollid ^ 1][did], partitions[rollid][did],
                    gridBlocks[0][did], gridBlocks[1][did]);
                timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_p2g_FBar_ASFLIP", did,
                                      curFrame, curStep));
              } //< End Non-Halo F-Bar + ASFLIP

              // Grid-to-Vertices - Update FEM - Vertices-to-Grid
              else if (pb.use_ASFLIP && pb.use_FEM && !pb.use_FBAR) {
                  timer.tick();
                  cuDev.compute_launch(
                      {pbcnt[did], g_particle_batch, (512 * 6 * 4) + (512 * 7 * 4)}, v2p2g, dt,
                      nextDt, (const ivec3 *)nullptr, pb,
                      get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did],
                      d_vertices[did]);
                  timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_v2p2g", did,
                                        curFrame, curStep));
              } //< End Non-Halo FEM + ASFLIP

              // Grid-to-Vertices - Update FEM + F-Bar - Vertices-to-Grid
              else if (pb.use_ASFLIP && pb.use_FEM && pb.use_FBAR) {
                  timer.tick();
                  cuDev.compute_launch(
                      {pbcnt[did], g_particle_batch, (512 * 6 * 4) + (512 * 7 * 4)}, 
                      v2p2g_FBar, dt, nextDt, (const ivec3 *)nullptr, pb,
                      get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]),
                      partitions[rollid ^ 1][did], partitions[rollid][did],
                      gridBlocks[0][did], gridBlocks[1][did],
                      d_vertices[did]);
                  timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_v2p2g_FBar", 
                                          did, curFrame, curStep));
              } //< End Non-Halo FEM + ASFLIP + F-Bar
            });
            cuDev.syncStream<streamIdx::Compute>();
          }
          // Resize partition, particles have moved into/out of cells/blocks in the domain.
          timer.tick();
          if (checkedCnts[did][0] > 0) {
            if (g_buckets_on_particle_buffer) {
              partitions[rollid^1][did].resizePartition(device_allocator{}, curNumActiveBlocks[did]);
              for (int mid=0; mid<getModelCnt(did); mid++) {
                // int mid = mid + did*g_models_per_gpu;
                match(particleBins[rollid][did][mid])([&](auto &pb) {
                  pb.reserveBuckets(device_allocator{}, curNumActiveBlocks[did]); // JB
                });
              }
            } else {
              partitions[rollid^1][did].resizePartition(device_allocator{}, curNumActiveBlocks[did]);
            }
            checkedCnts[did][0]--;
          }
          cuDev.syncStream<streamIdx::Compute>();
          timer.tock(fmt::format("GPU[{}] frame {} step {} resize_buckets", 
                                  did, curFrame, curStep));
        });
        sync();

        reduce_halo_grid_blocks(); //< Communicate halo grid data between GPUs

        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          CudaTimer timer{cuDev.stream_compute()};
          timer.tick();
          /// * Mark particle blocks
          if (g_buckets_on_particle_buffer) {
            for (int mid=0; mid<getModelCnt(did); mid++) {
              // int mid = mid + did*g_models_per_gpu;
              match(particleBins[rollid ^ 1][did][mid])([&](auto &pb) {
                checkCudaErrors(cudaMemsetAsync(pb._ppbs, 0,
                                                sizeof(int) * (ebcnt[did] + 1),
                                                cuDev.stream_compute()));
                cuDev.compute_launch({ebcnt[did], g_blockvolume},
                                    cell_bucket_to_block, pb._ppcs,
                                    pb._cellbuckets, pb._ppbs, pb._blockbuckets);
                // partitions[rollid].buildParticleBuckets(cuDev, ebcnt);
              });
            }
          } else {
            partitions[rollid][did].buildParticleBuckets(cuDev, ebcnt[did]);
          }

          int *activeBlockMarks = tmps[did].activeBlockMarks,
              *destinations = tmps[did].destinations,
              *sources = tmps[did].sources;
          checkCudaErrors(cudaMemsetAsync(activeBlockMarks, 0,
                                          sizeof(int) * nbcnt[did],
                                          cuDev.stream_compute()));
          /// * Mark grid blocks active if they contain any mass
          cuDev.compute_launch({(nbcnt[did] * g_blockvolume + 127) / 128, 128},
                               mark_active_grid_blocks, (uint32_t)nbcnt[did],
                               gridBlocks[1][did], activeBlockMarks);
          if (g_buckets_on_particle_buffer) {
            checkCudaErrors(cudaMemsetAsync(sources, 0, sizeof(int) * (ebcnt[did] + 1),
                                            cuDev.stream_compute())); // Maybe MGSP reset sources? From GMPM (JB)
            for (int mid=0; mid<getModelCnt(did); mid++) {
              // int mid = mid + did*g_models_per_gpu;
              match(particleBins[rollid ^ 1][did][mid])([&](auto &pb) {
                cuDev.compute_launch({(ebcnt[did] + 1 + 127) / 128, 128},
                                    mark_active_particle_blocks, ebcnt[did] + 1,
                                    pb._ppbs, sources);
              });
            }
          } else {
            cuDev.compute_launch({(ebcnt[did] + 1 + 127) / 128, 128},
                               mark_active_particle_blocks, ebcnt[did] + 1,
                               partitions[rollid][did]._ppbs, sources);
          }
          exclScan(ebcnt[did] + 1, sources, destinations, cuDev); //< Exclusive scan

          /// * Build new partition
          // Particle block count
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
          if (g_buckets_on_particle_buffer) {
            cuDev.compute_launch({(pbcnt[did]+ 127) / 128, 128}, update_partition,
                               (uint32_t)pbcnt[did], (const int *)sources,
                               partitions[rollid][did],
                               partitions[rollid ^ 1][did]);
          } else {
            cuDev.compute_launch({pbcnt[did], 128}, update_partition,
                               (uint32_t)pbcnt[did], (const int *)sources,
                               partitions[rollid][did],
                               partitions[rollid ^ 1][did]);
          }

          if (g_buckets_on_particle_buffer) {
            for (int mid=0; mid<getModelCnt(did); mid++) {
              // int mid = mid + did*g_models_per_gpu;
              match(particleBins[rollid ^ 1][did][mid])([&](auto &pb) {
                auto &next_pb = get<typename std::decay_t<decltype(pb)>>(
                    particleBins[rollid][did][mid]);
                cuDev.compute_launch({pbcnt[did], 128}, update_buckets,
                                    (uint32_t)pbcnt[did], (const int *) sources, pb,
                                    next_pb);
              });
            }
          } else { 
            // Don'tupdate buckets if on partition here, leave blank
          }

          // Determine offset of each particle bin by calculating _binsts
          // In sparse memory, knowing bin offset lets us find each particle within
          {
            int *binpbs = tmps[did].binpbs; //< Particle bins per block
            if (g_buckets_on_particle_buffer) {
              for (int mid=0; mid<getModelCnt(did); mid++) {
                // int mid = mid + did*g_models_per_gpu;
                match(particleBins[rollid][did][mid])([&](auto &pb) {
                  cuDev.compute_launch({(pbcnt[did] + 1 + 127) / 128, 128},
                                      compute_bin_capacity, pbcnt[did] + 1,
                                      (const int *)pb._ppbs, binpbs);
                  exclScan(pbcnt[did] + 1, binpbs, pb._binsts, cuDev);
                  checkCudaErrors(cudaMemcpyAsync(&bincnt[did][mid], pb._binsts + pbcnt[did],
                                                  sizeof(int), cudaMemcpyDefault,
                                                  cuDev.stream_compute()));
                  cuDev.syncStream<streamIdx::Compute>();
                });
              }
            } else { 
              cuDev.compute_launch({(pbcnt[did] + 1 + 127) / 128, 128},
                                 compute_bin_capacity, pbcnt[did] + 1,
                                 (const int *)partitions[rollid ^ 1][did]._ppbs,
                                 binpbs);
              exclScan(pbcnt[did] + 1, binpbs, partitions[rollid ^ 1][did]._binsts, cuDev);
              checkCudaErrors(cudaMemcpyAsync(
                  &bincnt[did], partitions[rollid ^ 1][did]._binsts + pbcnt[did],
                  sizeof(int), cudaMemcpyDefault, cuDev.stream_compute()));
              cuDev.syncStream<streamIdx::Compute>();
            }
          }
          timer.tock(fmt::format("GPU[{}] frame {} step {} update_partition",
                                 did, curFrame, curStep));

          /// * Register neighboring blocks 
          timer.tick();
          cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
                               register_neighbor_blocks, (uint32_t)pbcnt[did],
                               partitions[rollid ^ 1][did]);
          auto prev_nbcnt = nbcnt[did]; //< Previous neighbor block count 
          checkCudaErrors(cudaMemcpyAsync(&nbcnt[did], partitions[rollid ^ 1][did]._cnt, 
                                sizeof(int), cudaMemcpyDefault, cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();
          timer.tock(
              fmt::format("GPU[{}] frame {} step {} build_partition_for_grid",
                          did, curFrame, curStep));

          /// * Check grid-block capacity, resize if needed
          if (checkedCnts[did][0] > 0) {
            gridBlocks[0][did].resize(device_allocator{}, curNumActiveBlocks[did]);
          }
          /// * Rearrange grid blocks
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

          /// * Check gridBlocks and temp. array capacity, resize if needed
          /// * Flag set by checkCapacity() start of step, before MPM grid_update
          if (checkedCnts[did][0] > 0) {
            gridBlocks[1][did].resize(device_allocator{}, curNumActiveBlocks[did]);
            tmps[did].resize(curNumActiveBlocks[did]);
          }
        });
        sync();

        halo_tagging(); //< Tag grid-blocks as Halo if appropiate

        issue([this](int did) {
          auto &cuDev = Cuda::ref_cuda_context(did);
          CudaTimer timer{cuDev.stream_compute()};

          /// * Register exterior blocks
          timer.tick();
          cuDev.compute_launch({(pbcnt[did] + 127) / 128, 128},
                               register_exterior_blocks, (uint32_t)pbcnt[did],
                               partitions[rollid ^ 1][did]);
          checkCudaErrors(cudaMemcpyAsync(
              &ebcnt[did], partitions[rollid ^ 1][did]._cnt, sizeof(int),
              cudaMemcpyDefault, cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();
          fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
                     "GPU[{}] Block Count: Particle {}, Neighbor {}, Exterior {}, Allocated [{}]\n",
                     did, pbcnt[did], nbcnt[did], ebcnt[did], curNumActiveBlocks[did]);
          for (int mid=0; mid < getModelCnt(did); mid++) {
            fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
                      "GPU[{}] MODEL[{}] Bin Count: Particle Bins {}, Allocated Bins [{}]\n",
                      did, mid, bincnt[did][mid], curNumActiveBins[did][mid]);
          }
          timer.tock(fmt::format(
              "GPU[{}] frame {} step {} build_partition_for_particles", did,
              curFrame, curStep));
        });
        sync();

        // * Write grid energy values
        host_ge_freq = host_gt_freq;
        if (flag_ge && (fmod(curTime, (1.f/host_ge_freq)) < dt || curTime + dt >= nextTime))
        {
          std::string fn = std::string{"grid_energy_time_series.csv"};
          gridEnergyFile.open(fn, std::ios::out | std::ios::app);
          gridEnergyFile << curTime << "," << sum_kinetic_energy_grid<< "," << sum_gravity_energy_grid << "\n";
          gridEnergyFile.close();
        }
        
        // * Write particle energy values
        host_pe_freq = host_gt_freq;
        if (flag_pe && (fmod(curTime, (1.f/host_pe_freq)) < dt || curTime + dt >= nextTime))
        {
          issue([this](int did) {
            get_particle_energy(did);
          });
          sync();
          output_particle_energy();
        } //< End of particle energy output
        sync();

        // Check if end of frame or output frequency
        if (flag_gt && (fmod(curTime, (1.f/host_gt_freq)) < dt || curTime + dt >= nextTime)) {
          issue([this](int did) {
            IO::flush();
            output_gridcell_target(did); // Output gridTarget
          });
          sync();
        } //< End-of Grid-Target output

        // Check if end of frame or output frequency
        if (flag_pt && (fmod(curTime, (1.f/host_pt_freq)) < dt || curTime + dt >= nextTime)) {
          issue([this](int did) {
            IO::flush();
            output_particle_target(did); // Output particleTarget
          });
          sync();
        } //< End-of Particle-Target output

        dt = nextDt; // Update time-step
        rollid ^= 1; // Update rolling index
      } //< End of time-step

      // Output particle models
      issue([this](int did) {
        IO::flush();
        output_model(did);
        IO::flush(); 
      });
      sync();

      // Output finite element models
      issue([this](int did) {
        IO::flush();
        output_finite_elements(did);
      });
      sync();

      nextTime = 1.f * (curFrame + 1) / fps; // Next frame end time
      fmt::print(fmt::emphasis::bold | fg(fmt::color::red),
                 "----------------------------------------------------------------\n");
    } //< End of frame
    
    // Clean-up main simulation loop before exit
    issue([this](int did) {
      IO::flush();
      output_model(did);
    });
    sync();

    cudaDeviceSynchronize();

    issue([this](int did) {
      manual_deallocate(did);
    });
    sync();

    cudaDeviceSynchronize();
    fmt::print(fmt::emphasis::bold | fg(fmt::color::red),
                "------------------------------ END -----------------------------\n");
  } //< End of main simulation loop

  void manual_deallocate(int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      cuDev.setContext();
      CudaTimer timer{cuDev.stream_compute()};
      fmt::print("Deallocating tmps[{}]\n", did);
      for (int COPY_ID = 0; COPY_ID < 2; COPY_ID++) {
        fmt::print("Deallocating partitions[{}][{}]\n", COPY_ID, did);
        partitions[COPY_ID][did].deallocate_partition(device_allocator{}); // Deallocate partitions
        //partitions[COPY_ID][did].deallocate(device_allocator{}); // Deallocate partitions
        fmt::print("Deallocating gridBlocks[{}][{}]\n", COPY_ID, did);
        gridBlocks[COPY_ID][did].deallocate(device_allocator{}); // Deallocate grid blocks
        for (int MODEL_ID=0; MODEL_ID < getModelCnt(did); MODEL_ID++) {
          fmt::print("Deallocating particleBins[{}][{}][{}]\n", COPY_ID, did, MODEL_ID);
          match(particleBins[COPY_ID][did][MODEL_ID])([&](auto &pb) {
            pb.deallocateBuckets(device_allocator{}); // Deallocate particle buckets
            pb.deallocate(device_allocator{}); // Deallocate particle bins
          });
        }
      }
      for (int MODEL_ID=0; MODEL_ID < getModelCnt(did); MODEL_ID++) {
        fmt::print("Deallocating particles[{}][{}]\n", did, MODEL_ID);
        particles[did][MODEL_ID].deallocate(device_allocator{}); // Deallocate particles
        fmt::print("Deallocating pattribs[{}][{}] and pattribs_init[{}][{}]\n", did, MODEL_ID, did, MODEL_ID);
        match(pattribs[did][MODEL_ID], pattribs_init[did][MODEL_ID])([&](auto &pa, auto &pi) {
          pa.deallocate(device_allocator{}); // Deallocate particle attributes
          pi.deallocate(device_allocator{}); // Deallocate particle attributes initial
        });
      }
      fmt::print("Deallocating inputHaloGridBlocks[{}]\n", did);
      inputHaloGridBlocks[did].deallocate(device_allocator{}); // Deallocate input halo blocks
      fmt::print("Deallocating outputHaloGridBlocks[{}]\n", did);
      outputHaloGridBlocks[did].deallocate(device_allocator{}); // Deallocate output halo blocks
      
      fmt::print("Deallocating d_gridTarget[{}]\n", did);
      d_gridTarget[did].deallocate(device_allocator{}); // Deallocate grid-targets
      fmt::print("Deallocating d_particleTarget[{}]\n", did);
      d_particleTarget[did].deallocate(device_allocator{}); // Deallocate particle-targets

      if (flag_fem[did]) {
        fmt::print("Deallocating d_element_IDs[{}]\n", did);
        d_element_IDs[did].deallocate(device_allocator{}); // Deallocate element attributes
        fmt::print("Deallocating d_vertices[{}]\n", did);
        d_vertices[did].deallocate(device_allocator{}); // Deallocate element attributes
        fmt::print("Deallocating d_elementBins[{}]\n", did);
        match(elementBins[did])([&](auto &eb) {
          eb.deallocate(device_allocator{});
        });
      }
      fmt::print("Deallocating d_element_attribs[{}]\n", did);
      d_element_attribs[did].deallocate(device_allocator{}); // Deallocate element attributes
    
      cuDev.resetMem(); // Reset memory of temp allocator per CUDA context?
      tmps[did].dealloc(); // Deallocate temporary arrays
  }

  void get_particle_energy(int did) {
    //IO::flush();
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();

    if (curTime == 0) rollid = rollid^1;
    PREC_G *d_kinetic_energy_particles = tmps[did].d_kinetic_energy_particles;
    PREC_G *d_gravity_energy_particles = tmps[did].d_gravity_energy_particles;
    PREC_G *d_strain_energy_particles  = tmps[did].d_strain_energy_particles;

    CudaTimer timer{cuDev.stream_compute()};
    timer.tick();

    checkCudaErrors(cudaMemsetAsync(d_kinetic_energy_particles, 0, sizeof(PREC_G),
                                    cuDev.stream_compute()));
    checkCudaErrors(cudaMemsetAsync(d_gravity_energy_particles, 0, sizeof(PREC_G),
                                    cuDev.stream_compute()));
    checkCudaErrors(cudaMemsetAsync(d_strain_energy_particles, 0, sizeof(PREC_G),
                                    cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();
    for (int mid=0; mid<getModelCnt(did); mid++) {
      // int mid = mid + did*g_models_per_gpu;
      match(particleBins[rollid ^ 1][did][mid])([&](const auto &pb) {
        cuDev.compute_launch({pbcnt[did], 128}, query_energy_particles,
                            partitions[rollid ^ 1][did], partitions[rollid][did],
                            pb, d_kinetic_energy_particles, d_gravity_energy_particles, d_strain_energy_particles, grav[1]);
      });
    }
    //cuDev.syncStream<streamIdx::Compute>();
    checkCudaErrors(cudaMemcpyAsync(&kinetic_energy_particle_vals[did], d_kinetic_energy_particles,
                                    sizeof(PREC_G), cudaMemcpyDefault,
                                    cuDev.stream_compute()));
    checkCudaErrors(cudaMemcpyAsync(&gravity_energy_particle_vals[did], d_gravity_energy_particles,
                                    sizeof(PREC_G), cudaMemcpyDefault,
                                    cuDev.stream_compute()));
    checkCudaErrors(cudaMemcpyAsync(&strain_energy_particle_vals[did], d_strain_energy_particles,
                                    sizeof(PREC_G), cudaMemcpyDefault,
                                    cuDev.stream_compute()));
    if (curTime == 0 ) rollid = rollid^1;
    timer.tock(fmt::format("GPU[{}] frame {} step {} query_energy_particles",
                          did, curFrame, curStep));
    cuDev.syncStream<streamIdx::Compute>();
  }

  void output_particle_energy() {
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
            particleEnergyFile << curTime << "," << sum_kinetic_energy_particles << "," << sum_gravity_energy_particles << "," << sum_strain_energy_particles << "\n";
            particleEnergyFile.close();
          }
  }

  /// @brief Output full particle model to disk. Called at end of frame.
  /// @param did GPU ID of the particle model.
  void output_model(int did) {
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    CudaTimer timer{cuDev.stream_compute()};

    int parcnt, *d_parcnt = (int *)cuDev.borrow(sizeof(int));
    PREC trackVal[g_max_particle_trackers], *d_trackVal = (PREC *)cuDev.borrow(g_max_particle_trackers * sizeof(PREC));
    int particle_target_cnt, *d_particle_target_cnt = (int *)cuDev.borrow(sizeof(int));
    PREC valAgg, *d_valAgg = (PREC *)cuDev.borrow(sizeof(PREC));
    cuDev.syncStream<streamIdx::Compute>();

    for (int mid=0; mid<getModelCnt(did); mid++) {
      timer.tick();
      int i = 0;
      parcnt = 0;
      for (int j=0;j<g_max_particle_trackers; j++) 
        trackVal[j] = 0.0;
      particle_target_cnt = 0;
      valAgg = 0;
      checkCudaErrors(
          cudaMemsetAsync(d_parcnt, 0, sizeof(int), cuDev.stream_compute()));
      checkCudaErrors(
          cudaMemsetAsync(d_trackVal, 0.0, sizeof(PREC) * g_max_particle_trackers, cuDev.stream_compute()));
      checkCudaErrors(
          cudaMemsetAsync(d_particle_target_cnt, 0, sizeof(int), cuDev.stream_compute()));
      checkCudaErrors(
          cudaMemsetAsync(d_valAgg, 0, sizeof(PREC), cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();

      fmt::print(fg(fmt::color::red), "GPU[{}] Launch retrieve_particle_buffer_attributes\n", did);
      match(particleBins[rollid][did][mid], pattribs[did][mid])([&](const auto &pb, auto &pa) {
        cuDev.compute_launch({pbcnt[did], 128}, retrieve_particle_buffer_attributes_general,
                            partitions[rollid][did], partitions[rollid ^ 1][did],
                            pb, get<typename std::decay_t<decltype(pb)>>(particleBins[rollid ^ 1][did][mid]), particles[did][mid],  get<typename std::decay_t<decltype(pa)>>(
                          pattribs[did][mid]), d_trackVal, d_parcnt,
                            d_particleTarget[did], d_valAgg, d_particle_target[i],d_particle_target_cnt, false);
      });
      // Copy device to host
      checkCudaErrors(cudaMemcpyAsync(&parcnt, d_parcnt, sizeof(int),
                                      cudaMemcpyDefault, cuDev.stream_compute()));
      checkCudaErrors(cudaMemcpyAsync(&trackVal, d_trackVal, sizeof(PREC) * g_max_particle_trackers,
                                      cudaMemcpyDefault,
                                      cuDev.stream_compute()));
      checkCudaErrors(cudaMemcpyAsync(&particle_target_cnt, d_particle_target_cnt, sizeof(int),
                                      cudaMemcpyDefault, cuDev.stream_compute()));
      checkCudaErrors(cudaMemcpyAsync(&valAgg, d_valAgg, sizeof(PREC),
                                      cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();

      fmt::print(fg(fmt::color::red), "GPU[{}] Total number of particles: {}\n", did, parcnt);
      fmt::print(fg(fmt::color::red), "GPU[{}] Tracked value of particle ID {} in model: {} \n", did, g_track_ID, trackVal[0]); // TODO: Fix to track multiple particles / not be required
      fmt::print(fg(fmt::color::red), "GPU[{}] Total number of target particles: {}\n", did, particle_tarcnt[i][did]);
      fmt::print(fg(fmt::color::red), "GPU[{}] Aggregate value in particleTarget: {} \n", did, valAgg);

      {
        std::string fn_track = std::string{"particleTrack"} + "_model[" + std::to_string(mid) +  "]" + "_dev[" + std::to_string(did + rank * g_device_cnt) + "].csv";
        particleTrackFile[did][mid].open(fn_track, std::ios::out | std::ios::app);
        particleTrackFile[did][mid] << curTime;
        for (int j=0; j<g_max_particle_trackers; j++) {
          if (j >= num_particle_trackers[did][mid]) continue;
          particleTrackFile[did][mid] << "," << trackVal[j];
        }
        particleTrackFile[did][mid] << "\n";
        particleTrackFile[did][mid].close();
      }      
      
      //host_particleTarget[did][mid].resize(particle_tarcnt[i][did][mid]);
      models[did][mid].resize(parcnt);
      checkCudaErrors(cudaMemcpyAsync(models[did][mid].data(),
                                      (void *)&particles[did][mid].val_1d(_0, 0),
                                      sizeof(std::array<PREC, 3>) * (parcnt),
                                      cudaMemcpyDefault, cuDev.stream_compute()));

      // * Write full particle files
      {
        match(particleBins[rollid][did][mid], pattribs[did][mid])([&](const auto &pb, const auto &pa) {
          attribs[did][mid].resize(pa.numAttributes*parcnt);
          if (pa.numAttributes){
            checkCudaErrors(cudaMemcpyAsync(attribs[did][mid].data(), (void *)&pa.val_1d(_0, 0),
                                            sizeof(PREC) * (pa.numAttributes) * (parcnt),
                                            cudaMemcpyDefault, cuDev.stream_compute()));
            cuDev.syncStream<streamIdx::Compute>();
            fmt::print("Updated attribs, [{}] particles with [{}] elements for [{}] output attributes.\n", attribs[did][mid].size() / pa.numAttributes, attribs[did][mid].size() / parcnt, pa.numAttributes);
            std::string fn = std::string{"model["} + std::to_string(mid) + "]" + "_dev[" + std::to_string(did + rank * g_device_cnt) +
                            "]_frame[" + std::to_string(curFrame) + "]" + save_suffix;
            IO::insert_job([fn, m = models[did][mid], a = attribs[did][mid], labels = pb.output_labels, dim_out = pa.numAttributes]() { write_partio_particles<PREC>(fn, m, a, labels); });
          }
        });
      }
      cuDev.syncStream<streamIdx::Compute>();
      timer.tock(fmt::format("NODE[{}] GPU[{}] MODEL[{}] frame {} step {} retrieve_particles", rank, did, mid, curFrame, curStep));
    }
  }


  void output_finite_elements(int did) {
    if (flag_fem[did]) {
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    
    
    CudaTimer timer{cuDev.stream_compute()};
    timer.tick();
    

    int elcnt, *d_elcnt = (int *)cuDev.borrow(sizeof(int));
    checkCudaErrors(
        cudaMemsetAsync(d_elcnt, 0, sizeof(int), cuDev.stream_compute()));

    // Setup forceSum to sum all forces in grid-target in a kernel
    PREC trackVal, *d_trackVal = (PREC *)cuDev.borrow(sizeof(PREC));
    checkCudaErrors(
        cudaMemsetAsync(d_trackVal, 0, sizeof(PREC), cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();

    checkCudaErrors(
        cudaMemsetAsync(d_elcnt, 0, sizeof(int), cuDev.stream_compute()));

    checkCudaErrors(
        cudaMemsetAsync((void *)&d_element_attribs[did].val_1d(_0, 0), 0,
                        sizeof(std::array<PREC, 6>) * (element_cnt[did]),
                        cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();


    match(elementBins[did])([&](const auto &eb) {
      cuDev.compute_launch({element_cnt[did], 1}, 
                            retrieve_element_buffer_attributes,
                            (uint32_t)element_cnt[did], 
                            d_vertices[did], eb, d_element_IDs[did], d_element_attribs[did], 
                            d_trackVal, d_elcnt);
    });
    cuDev.syncStream<streamIdx::Compute>();
    checkCudaErrors(cudaMemcpyAsync(&elcnt, d_elcnt, sizeof(int),
                                    cudaMemcpyDefault, cuDev.stream_compute()));
  
    // Copy tracklacement to host
    checkCudaErrors(cudaMemcpyAsync(&trackVal, d_trackVal, sizeof(PREC),
                                    cudaMemcpyDefault,
                                    cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();
    fmt::print(fg(fmt::color::red), "GPU[{}] Total number of elements: {}\n", did, elcnt);
    fmt::print(fg(fmt::color::red), "GPU[{}] Tracked value of element ID {} in model: {} \n", did, g_track_ID, trackVal);
    fmt::print(fg(fmt::color::red), "GPU[{}] Total element count: {}\n", did, element_cnt[did]);


    //host_element_IDs[did].resize(element_cnt[did]);
    host_element_attribs[did].resize(element_cnt[did]);
    cuDev.syncStream<streamIdx::Compute>();

    checkCudaErrors(cudaMemcpyAsync(host_element_attribs[did].data(),
                                    (void *)&d_element_attribs[did].val_1d(_0, 0),
                                    sizeof(std::array<PREC, 6>) * (element_cnt[did]),
                                    cudaMemcpyDefault, cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();

    std::string fn = std::string{"elements"} + "_dev[" + std::to_string(did) +
                     "]_frame[" + std::to_string(curFrame) + "]" + save_suffix;
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
      // * IO::flush();    // Clear IO
      int gridID = 0;
      if (curTime == 0){
        gridID += 1;
        rollid = rollid^1;
        host_gridTarget[did].resize(grid_tarcnt[i][did]);
        cuDev.syncStream<streamIdx::Compute>();

        checkCudaErrors(
            cudaMemsetAsync((void *)&d_gridTarget[did].val_1d(_0, 0), 0,
                            sizeof(std::array<PREC_G, g_grid_target_attribs>) * (g_grid_target_cells),
                            cuDev.stream_compute()));
        cuDev.syncStream<streamIdx::Compute>();

        checkCudaErrors(
            cudaMemcpyAsync(host_gridTarget[did].data(), (void *)&d_gridTarget[did].val_1d(_0, 0),
                            sizeof(std::array<PREC_G, g_grid_target_attribs>) * (grid_tarcnt[i][did]),
                            cudaMemcpyDefault, cuDev.stream_compute()));
        cuDev.syncStream<streamIdx::Compute>();

      }

      int target_cnt, *d_target_cnt = (int *)cuDev.borrow(sizeof(int));
      target_cnt = 0;
      checkCudaErrors(
          cudaMemsetAsync(d_target_cnt, 0, sizeof(int), 
          cuDev.stream_compute())); /// Reset memory
      cuDev.syncStream<streamIdx::Compute>();

      // Setup valAgg to sum all forces in grid-target in a kernel
      PREC_G valAgg, *d_valAgg = (PREC_G *)cuDev.borrow(sizeof(PREC_G));
      checkCudaErrors(
          cudaMemsetAsync(d_valAgg, 0, sizeof(PREC_G), cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();

      // Reset gridTarget (device) to zeroes asynchronously
      checkCudaErrors(
          cudaMemsetAsync((void *)&d_gridTarget[did].val_1d(_0, 0), 0,
                          sizeof(std::array<PREC_G, g_grid_target_attribs>) * (g_grid_target_cells), cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();

      fmt::print(fg(fmt::color::red), "GPU[{}] Launch retrieve_selected_grid_cells\n", did);
      cuDev.compute_launch(
                {curNumActiveBlocks[did], 32}, retrieve_selected_grid_cells, 
                (uint32_t)nbcnt[did], partitions[rollid ^ 1][did], 
                gridBlocks[gridID][did], d_gridTarget[did],
                nextDt, d_valAgg, d_grid_target[i], d_target_cnt, length);
      cudaGetLastError();
      cuDev.syncStream<streamIdx::Compute>();

      // Copy grid-target aggregate value to host
      checkCudaErrors(cudaMemcpyAsync(&valAgg, d_valAgg, sizeof(PREC_G),
                                      cudaMemcpyDefault, cuDev.stream_compute()));
      checkCudaErrors(cudaMemcpyAsync(&target_cnt, d_target_cnt, sizeof(int),
                                      cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
      grid_tarcnt[i][did] = target_cnt;

      fmt::print(fg(fmt::color::red), "GPU[{}] Total number of gridTarget[{}] grid-nodes: {}\n", did, i, grid_tarcnt[i][did]);
      fmt::print(fg(fmt::color::red), "GPU[{}] Value aggregate in gridTarget[{}]: {} \n", did, i, valAgg);

      if (curTime + dt >= nextTime || curTime == 0) 
      {
        // Asynchronously copy data from target (device) to target (host)
        host_gridTarget[did].resize(grid_tarcnt[i][did]);
        cuDev.syncStream<streamIdx::Compute>();

        checkCudaErrors(
            cudaMemcpyAsync(host_gridTarget[did].data(), (void *)&d_gridTarget[did].val_1d(_0, 0),
                            sizeof(std::array<PREC_G, g_grid_target_attribs>) * (grid_tarcnt[i][did]),
                            cudaMemcpyDefault, cuDev.stream_compute()));
        cuDev.syncStream<streamIdx::Compute>();
        // Output to Partio as 'gridTarget_target[ ]_dev[ ]_frame[ ].bgeo'
        std::string fn = std::string{"gridTarget["} + std::to_string(i) + "]" + "_dev[" + std::to_string(did) + "]_frame[" + std::to_string(curFrame) + "]" + save_suffix;
        IO::insert_job([fn, m = host_gridTarget[did]]() { write_partio_gridTarget<float, g_grid_target_attribs>(fn, m); });
        fmt::print(fg(fmt::color::red), "GPU[{}] gridTarget[{}] outputted.\n", did, i);
      }
    
      // Output summed grid-target value to *.csv
      if (fmod(curTime, (1.f/host_gt_freq)) < dt) 
      {
        std::string fn = std::string{"gridTarget"} + "[" + std::to_string(i) + "]_dev[" + std::to_string(did) + "].csv";
        gridTargetFile[did].open (fn, std::ios::out | std::ios::app);
        gridTargetFile[did] << curTime << "," << valAgg << "\n";
        gridTargetFile[did].close();
      }
      if (curTime == 0){
        IO::flush();    // Clear IO
        rollid = rollid^1;
      }
    }
    timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_cells", did,
                           curFrame, curStep));
  }


  /// @brief Output particle-targets to disk. Write aggregate values.
  /// @param did GPU ID of the particle model.
  void output_particle_target(int did) {
    auto &cuDev = Cuda::ref_cuda_context(did);
    cuDev.setContext();
    CudaTimer timer{cuDev.stream_compute()};

    int parcnt, *d_parcnt = (int *)cuDev.borrow(sizeof(int));
    PREC trackVal, *d_trackVal = (PREC *)cuDev.borrow(sizeof(PREC));
    int particle_target_cnt, *d_particle_target_cnt = (int *)cuDev.borrow(sizeof(int));
    PREC valAgg, *d_valAgg = (PREC *)cuDev.borrow(sizeof(PREC));
    cuDev.syncStream<streamIdx::Compute>();

    for (int mid=0; mid<getModelCnt(did); mid++) {
      timer.tick();
      for (int i = 0; i < number_of_particle_targets; i++) {
        int particleID = rollid^1;
        parcnt = 0;
        trackVal = 0.0;
        particle_target_cnt = 0;
        valAgg = 0.0;
        checkCudaErrors(
            cudaMemsetAsync(d_parcnt, 0, sizeof(int), cuDev.stream_compute()));
        checkCudaErrors(
            cudaMemsetAsync(d_trackVal, 0.0, sizeof(PREC), cuDev.stream_compute()));
        checkCudaErrors(
          cudaMemsetAsync(d_particle_target_cnt, 0, sizeof(int), cuDev.stream_compute()));
        checkCudaErrors(
            cudaMemsetAsync(d_valAgg, 0, sizeof(PREC), cuDev.stream_compute()));
        cuDev.syncStream<streamIdx::Compute>();

        if (curTime == 0) {
          particleID = rollid;
          host_particleTarget[did][mid].resize(particle_tarcnt[i][did]);
          checkCudaErrors(
              cudaMemsetAsync((void *)&d_particleTarget[did].val_1d(_0, 0), 0,
                              sizeof(std::array<PREC, g_particle_target_attribs>) * (g_particle_target_cells),
                              cuDev.stream_compute()));
          checkCudaErrors(
              cudaMemcpyAsync(host_particleTarget[did][mid].data(), 
                              (void *)&d_particleTarget[did].val_1d(_0, 0),
                              sizeof(std::array<PREC, g_particle_target_attribs>) * (particle_tarcnt[i][did]),
                              cudaMemcpyDefault, cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();
        }

        // Zero-out particleTarget
        if (curTime + dt >= nextTime || curTime == 0) {
          checkCudaErrors(
              cudaMemsetAsync((void *)&d_particleTarget[did].val_1d(_0, 0), 0,
                              sizeof(std::array<PREC, g_particle_target_attribs>) * (g_particle_target_cells),
                              cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();
        }

        match(particleBins[particleID][did][mid])([&](const auto &pb) {
          cuDev.compute_launch({pbcnt[did], 128}, retrieve_particle_target_attributes_general,
                              partitions[rollid ^ 1][did], partitions[rollid][did],
                              pb, get<typename std::decay_t<decltype(pb)>>(particleBins[particleID ^ 1][did][mid]), d_trackVal, d_parcnt,
                              d_particleTarget[did], d_valAgg, d_particle_target[i],d_particle_target_cnt);
        });
        cuDev.syncStream<streamIdx::Compute>();

        // checkCudaErrors(cudaMemcpyAsync(&trackVal, d_trackVal, sizeof(PREC),
        //                                 cudaMemcpyDefault,
        //                                 cuDev.stream_compute()));
        checkCudaErrors(cudaMemcpyAsync(&particle_target_cnt, d_particle_target_cnt, sizeof(int),
                                        cudaMemcpyDefault, cuDev.stream_compute()));
        checkCudaErrors(cudaMemcpyAsync(&valAgg, d_valAgg, sizeof(PREC),
                                        cudaMemcpyDefault, cuDev.stream_compute()));
        cuDev.syncStream<streamIdx::Compute>();
        particle_tarcnt[i][did] = particle_target_cnt;
        fmt::print(fg(fmt::color::red), "GPU[{}] MODEL[{}] Total number of target[{}] particles: {}, with aggregate value: {} \n", did, mid, i, particle_tarcnt[i][did], valAgg);

        // {
        //   std::string fn_track = std::string{"track_time_series"} + "_ID[" + std::to_string(0) + "]_dev[" + std::to_string(did) + "].csv";
        //   trackFile[did].open(fn_track, std::ios::out | std::ios::app);
        //   trackFile[did] << curTime << "," << trackVal << "\n";
        //   trackFile[did].close();
        // }      

        // * particleTarget per-frame full output      
        if (curTime + dt >= nextTime || curTime == 0)
        {
          host_particleTarget[did][mid].resize(particle_tarcnt[i][did]);
          checkCudaErrors(
              cudaMemcpyAsync(host_particleTarget[did][mid].data(), (void *)&d_particleTarget[did].val_1d(_0, 0), sizeof(std::array<PREC, g_particle_target_attribs>) * (particle_tarcnt[i][did]), cudaMemcpyDefault, cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();
        
          std::string fn = std::string{"particleTarget"}  +"[" + std::to_string(i) + "]" + "_model["+ std::to_string(mid) + "]" + "_dev[" + std::to_string(did + rank * g_device_cnt) + "]_frame[" + std::to_string(curFrame) + "]" + save_suffix;
          IO::insert_job([fn, m = host_particleTarget[did][mid]]() { write_partio_particleTarget<PREC, g_particle_target_attribs>(fn, m); });
          fmt::print(fg(fmt::color::red), "NODE[{}] GPU[{}] particleTarget[{}] outputted.\n", rank, did, i);
        }

        // * particleTarget frequency-set aggregate ouput
        {
          std::string fn = std::string{"particleTarget"} + "[" + std::to_string(i) + "]" + "_model["+ std::to_string(mid) + "]" + "_dev[" + std::to_string(did + rank * g_device_cnt) + "]" + ".csv";
          particleTargetFile[did][mid].open (fn, std::ios::out | std::ios::app);
          particleTargetFile[did][mid] << curTime << "," << valAgg << "\n";
          particleTargetFile[did][mid].close();
        }
        // * IO::flush(); 
        cuDev.syncStream<streamIdx::Compute>();
        timer.tock(fmt::format("NODE[{}] GPU[{}] frame {} step {} retrieve_particle_targets", rank, did, curFrame, curStep));
      }
    }
  }


  void initial_setup() {
    issue([this](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      cuDev.setContext();
      CudaTimer timer{cuDev.stream_compute()};
      timer.tick();

      // Initialize grid-boundaries
      {
        checkCudaErrors( cudaMalloc((void**)&d_gridBoundaryConfigs, sizeof(GridBoundaryConfigs)*g_max_grid_boundaries) );
        checkCudaErrors( cudaMemcpy(d_gridBoundaryConfigs, &h_gridBoundaryConfigs, sizeof(GridBoundaryConfigs)*g_max_grid_boundaries, cudaMemcpyHostToDevice) );
      }

      // Partition activate particle blocks
      for (int mid=0; mid < getModelCnt(did); mid++) {
        cuDev.compute_launch({(pcnt[did][mid] + 255) / 256, 256}, activate_blocks,
                            pcnt[did][mid], particles[did][mid],
                            partitions[rollid ^ 1][did]);
      }
      // Set particle block cnt.(grid-blocks containing particles) on GPU
      checkCudaErrors(cudaMemcpyAsync(
          &pbcnt[did], partitions[rollid ^ 1][did]._cnt, sizeof(int),
          cudaMemcpyDefault, cuDev.stream_compute()));
      timer.tock(fmt::format("GPU[{}] step {} init_table", did, curStep));

      timer.tick();
      cuDev.resetMem(); // Reset device memory pool

      // Build particle cell bucket arrays to hold particle indices per grid-cell
      if (g_buckets_on_particle_buffer) {
        for (int mid=0; mid < getModelCnt(did); mid++) {
          cuDev.syncStream<streamIdx::Compute>();
          match(particleBins[rollid][did][mid])([&](auto &pb) { // check rollid
          cuDev.compute_launch({(pcnt[did][mid] + 255) / 256, 256},
                        build_particle_cell_buckets, pcnt[did][mid],
                        particles[did][mid], pb, partitions[rollid ^ 1][did]);
          });
        }
      } 
      else {
        int mid = 0; // single model per GPU
        cuDev.compute_launch({(pcnt[did][mid] + 255) / 256, 256},
                      build_particle_cell_buckets, pcnt[did][mid],
                      particles[did][mid], partitions[rollid ^ 1][did]);  
      }      
      cuDev.syncStream<streamIdx::Compute>();

      // Build block buckets using cell buckets
      if (g_buckets_on_particle_buffer) {
        for (int mid=0; mid < getModelCnt(did); mid++) {
          match(particleBins[rollid][did][mid])([&](auto &pb) { // recheck rollid
            checkCudaErrors(cudaMemsetAsync(
              pb._ppbs, 0, sizeof(int) * (pbcnt[did] + 1), cuDev.stream_compute()));
            cuDev.compute_launch({pbcnt[did], g_blockvolume},
                                cell_bucket_to_block, pb._ppcs,
                                pb._cellbuckets, pb._ppbs, pb._blockbuckets);
          });
        }
      }
      else {
        partitions[rollid ^ 1][did].buildParticleBuckets(cuDev, pbcnt[did]); 
      }

      // Compute bins in each particle block
      // Exclusive scan on particle block buckets to compute bin offsets
      {
        int *binpbs = tmps[did].binpbs;
        if (g_buckets_on_particle_buffer) {
          for (int mid=0; mid < getModelCnt(did); mid++) {
            match(particleBins[rollid][did][mid], pattribs_init[did][mid])([&](auto &pb, const auto &pi) {
              cuDev.compute_launch({(pbcnt[did] + 1 + 127) / 128, 128},
                              compute_bin_capacity, pbcnt[did] + 1,
                              (const int *)pb._ppbs,
                              binpbs);
              exclScan(pbcnt[did] + 1, binpbs, pb._binsts, cuDev);
              checkCudaErrors(cudaMemcpyAsync(
                  &bincnt[did][mid], pb._binsts + pbcnt[did],
                  sizeof(int), cudaMemcpyDefault, cuDev.stream_compute()));
              cuDev.syncStream<streamIdx::Compute>();
              if (flag_pi[did][mid]) {
                  fmt::print("GPU[{}] MODEL[{}] array_to_buffer with initial attributes.\n", did, mid);
                  cuDev.compute_launch({pbcnt[did], 128}, array_to_buffer, particles[did][mid], 
                  pi, pb, partitions[rollid ^ 1][did], vel0[did][mid]);
              } else {
                fmt::print("GPU[{}] MODEL[{}] array_to_buffer without initial attributes.\n", did, mid);
                cuDev.compute_launch({pbcnt[did], 128}, array_to_buffer, particles[did][mid],
                                    pb, partitions[rollid ^ 1][did], vel0[did][mid]);
              }
            });
          }
        } else {
          cuDev.compute_launch({(pbcnt[did] + 1 + 127) / 128, 128},
                             compute_bin_capacity, pbcnt[did] + 1,
                             (const int *)partitions[rollid ^ 1][did]._ppbs,
                             binpbs);
          exclScan(pbcnt[did] + 1, binpbs, partitions[rollid ^ 1][did]._binsts,
                  cuDev);
          checkCudaErrors(cudaMemcpyAsync(
              &bincnt[did], partitions[rollid ^ 1][did]._binsts + pbcnt[did],
              sizeof(int), cudaMemcpyDefault, cuDev.stream_compute()));
        }
        cuDev.syncStream<streamIdx::Compute>();
      }
      if (g_buckets_on_particle_buffer == false) {
        int mid = 0;  
        match(particleBins[rollid][did][mid], pattribs_init[did][mid])([&](const auto &pb, const auto &pi) {
          if (flag_pi[did][mid]) {
              fmt::print("array_to_buffer with initial attributes.\n");
              cuDev.compute_launch({pbcnt[did], 128}, array_to_buffer, particles[did][mid], 
                                pi, pb, partitions[rollid ^ 1][did], vel0[did][mid]);
          } else {
            fmt::print("array_to_buffer without initial attributes.\n");
            cuDev.compute_launch({pbcnt[did], 128}, array_to_buffer, particles[did][mid],
                                pb, partitions[rollid ^ 1][did], vel0[did][mid]);
          }
        });
      }
      // match(particleBins[rollid][did])([&](const auto &pb) {
      //   cuDev.compute_launch({pbcnt[did], 128}, array_to_buffer, particles[did],
      //                        pb, partitions[rollid ^ 1][did], vel0[did]);
      // }
      // // Copy GPU particle data from basic device arrays to Particle Buffer
      // for (int mid=0; mid<getModelCnt(did); mid++) {
      //   int mid = mid + did*g_models_per_gpu;
      //   match(particleBins[rollid][did][mid], pattribs_init[did][mid])([&](const auto &pb, const auto &pi) {
      //   });
      //   cuDev.syncStream<streamIdx::Compute>();
      // }
      // FEM Precompute - Resize elementBins and then precompute values
      for (int mid=0; mid<getModelCnt(did); mid++) {
        match(particleBins[rollid][did][mid])([&](const auto &pb) {
          if (pb.use_FEM) {
            match(elementBins[did])([&](auto &eb) {
              eb.resize(device_allocator{}, element_cnt[did]);
            });
            cuDev.syncStream<streamIdx::Compute>();
            match(elementBins[did])([&](auto &eb) {
              cuDev.compute_launch({element_cnt[did], 1}, fem_precompute,
                                  (uint32_t)element_cnt[did], d_vertices[did], d_element_IDs[did], eb);
            });
            cuDev.syncStream<streamIdx::Compute>();
          }
        }); //< End FEM Precompute
        cuDev.syncStream<streamIdx::Compute>();
      }
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
      timer.tock(fmt::format("GPU[{}] step {} init_partition\n", did, curStep));

      for (int mid=0; mid<getModelCnt(did); mid++) {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
                  "GPU[{}] MODEL[{}] Block count: pbcnt[{}], nbcnt[{}], ebcnt[{}], curNumActiveBlocks[{}]; Bin count: bincnt[{}] curNumActiveBins[{}]\n", did, mid, pbcnt[did], nbcnt[did], ebcnt[did], curNumActiveBlocks[did], bincnt[did][mid], curNumActiveBins[did][mid]);
      }
    });
    sync();

    halo_tagging(); //< Tag Halo Grid-Blocks (Multi-GPU shared blocks)

    issue([this](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      CudaTimer timer{cuDev.stream_compute()};

      // // curNumActiveBlocks[did] must be >= exterior block count ebcnt[did]
      // partitions[rollid][did].halo_base_t::resizePartition(device_allocator{}, (std::size_t)g_max_active_block, (std::size_t)curNumActiveBlocks[did]); //< Reset overlap marks // JB

      // JB
      // partitions[rollid][did].resetOverlapMarks(curNumActiveBlocks[did], cuDev.stream_compute());
      // partitions[rollid][did].resetHaloBlocks(curNumActiveBlocks[did], cuDev.stream_compute()); 
      cuDev.syncStream<streamIdx::Compute>();

      /// Copy halo tag info to next partition for following time-steps
      partitions[rollid ^ 1][did].copy_to(partitions[rollid][did], ebcnt[did],
                                          cuDev.stream_compute());
      checkCudaErrors(cudaMemcpyAsync(
          partitions[rollid][did]._activeKeys,
          partitions[rollid ^ 1][did]._activeKeys, sizeof(ivec3) * ebcnt[did],
          cudaMemcpyDefault, cuDev.stream_compute()));

      // fmt::print(fg(fmt::color::green), "Finish active_keys GPU[{}], extent: {}\n", did, partitions[rollid ^ 1][did].h_count) ;
      
      if (g_buckets_on_particle_buffer) {
        for (int mid=0; mid<getModelCnt(did); mid++) {
          match(particleBins[rollid][did][mid])([&](const auto &pb) {
            // binsts, ppbs
            pb.copy_to(get<typename std::decay_t<decltype(pb)>>(
                          particleBins[rollid ^ 1][did][mid]),
                      pbcnt[did], cuDev.stream_compute());
          });
        }
      }
      fmt::print(fg(fmt::color::green), "GPU[{}] partitions and particleBins copy_to\n", did);
      cuDev.syncStream<streamIdx::Compute>();
      // Reset gridBlocks
      // Rasterize (shape-functions) init. data from Particles to Grids (mass vel. fint)
      // Init. advection buckets on partition or particle buffer
      timer.tick();
      gridBlocks[0][did].reset(nbcnt[did], cuDev); // Zero out blocks on all Grids 0
      if (g_buckets_on_particle_buffer) {
        for (int mid=0; mid<getModelCnt(did); mid++) {
          // TODO : Need to retrofit for init attribs and pbuffer usage for vars
          match(particleBins[rollid][did][mid], pattribs_init[did][mid])([&](auto &pb, auto &pi) {
            if (flag_pi[did][mid]) {
              cuDev.compute_launch({(pcnt[did][mid] + 255) / 256, 256}, rasterize, pcnt[did][mid],
                                  pb, particles[did][mid],  pi, gridBlocks[0][did],
                                  partitions[rollid][did], dt, vel0[did][mid], (PREC)grav[1]);
              fmt::print("GPU[{}] MODEl[{}] Rasterized init. attributes to grid.\n", did, mid);
            } else {
              cuDev.compute_launch({(pcnt[did][mid] + 255) / 256, 256}, rasterize, pcnt[did][mid],
                              pb, particles[did][mid], gridBlocks[0][did],
                              partitions[rollid][did], dt, vel0[did][mid], (PREC)grav[1]); 
            }
          });
          match(particleBins[rollid ^ 1][did][mid])([&](auto &pb) {
            cuDev.compute_launch({pbcnt[did], 128}, init_adv_bucket,
                                (const int *)pb._ppbs, pb._blockbuckets);
          });
        }
      } else { 
        int mid = 0;
        match(particleBins[rollid][did][mid], pattribs_init[did][mid])([&](auto &pb, auto &pi) {
          if (flag_pi[did][mid]) {
              cuDev.compute_launch({(pcnt[did][mid] + 255) / 256, 256}, rasterize, pcnt[did][mid],
                                  pb, particles[did][mid],  pi, gridBlocks[0][did],
                                  partitions[rollid][did], dt, vel0[did][mid], (PREC)grav[1]);
            fmt::print("GPU[{}] Rasterized init. attributes to grid.\n", did);
          } else {
            cuDev.compute_launch({(pcnt[did][mid] + 255) / 256, 256}, rasterize, pcnt[did][mid],
                            pb, particles[did][mid], gridBlocks[0][did],
                            partitions[rollid][did], dt, vel0[did][mid], (PREC)grav[1]); 
          }
        });
        cuDev.compute_launch({pbcnt[did], 128}, init_adv_bucket,
                          (const int *)partitions[rollid][did]._ppbs,
                          partitions[rollid][did]._blockbuckets);
      }
      cuDev.syncStream<streamIdx::Compute>();
      timer.tock(fmt::format("GPU[{}] step {} init_grid", did, curStep));
    });
    sync();

    collect_halo_grid_blocks(0); //< Send/collect halo blocks between GPUs
    reduce_halo_grid_blocks(0); //< Add recieved halo blocks into GPU grids

    if (flag_pe) {
      issue([this](int did) {
        get_particle_energy(did);
      });
      sync();

      PREC_G sum_kinetic_energy_particles = 0.0;
      PREC_G sum_gravity_energy_particles = 0.0;
      PREC_G sum_strain_energy_particles = 0.0;
      for (int did = 0; did < g_device_cnt; ++did) {
        sum_kinetic_energy_particles += kinetic_energy_particle_vals[did];
        sum_gravity_energy_particles += gravity_energy_particle_vals[did];
        sum_strain_energy_particles += strain_energy_particle_vals[did];
      }
      init_gravity_energy_particles = sum_gravity_energy_particles;
      sum_gravity_energy_particles -= init_gravity_energy_particles;
      {
        std::string fn = std::string{"particle_energy_time_series.csv"};
        particleEnergyFile.open(fn, std::ios::out | std::ios::app);
        particleEnergyFile << curTime << "," << sum_kinetic_energy_particles << "," << sum_gravity_energy_particles << "," << sum_strain_energy_particles << "\n";
        particleEnergyFile.close();
      }
    }

    // Output Grid-Targets Frame 0
    if (flag_gt) {
      issue([this](int did) {
        IO::flush();
        output_gridcell_target(did); 
      });
      sync();
    }

    // Output particleTargets Frame 0
    if (flag_pt) {
      issue([this](int did) {
        IO::flush(); 
        output_particle_target(did); 
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

    // issue([this](int did) {
    //   auto &cuDev = Cuda::ref_cuda_context(did);
    //   CudaTimer timer{cuDev.stream_compute()};
    //   cuDev.resetMem();
    // });
    // sync();

    fmt::print(fg(fmt::color::green),"Finished initial setup. Return to main loop...\n");
  } //< Return to main simulation loop.

  void halo_tagging() {
    issue([this](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      cuDev.resetMem();
      for (int otherdid = 0; otherdid < g_device_cnt; otherdid++)
        if (otherdid != did)
          haloBlockIds[did][otherdid] =
              (ivec3 *)cuDev.borrow(sizeof(ivec3) * nbcnt[otherdid]);
      /// Init. Halo grid block _blockIds
      outputHaloGridBlocks[did].initBlocks(temp_allocator{did}, nbcnt[did]);
      inputHaloGridBlocks[did].initBlocks(temp_allocator{did}, nbcnt[did]);
      // cuDev.syncStream<streamIdx::Compute>(); // JB
      // outputHaloGridBlocks[did].resetBlocks(nbcnt[did], cuDev.stream_compute()); // JB
      // inputHaloGridBlocks[did].resetBlocks(nbcnt[did], cuDev.stream_compute()); // JB
    });
    sync();
    issue([this](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      /// Reset halo block counts
      outputHaloGridBlocks[did].resetCounts(cuDev.stream_compute());
      cuDev.syncStream<streamIdx::Compute>();
      /// sharing local active blocks
      for (int otherdid = 0; otherdid < g_device_cnt; otherdid++)
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
      /// Initialize GPU grid overlap marks
      partitions[rollid ^ 1][did].resetOverlapMarks(nbcnt[did], cuDev.stream_compute());
      cuDev.syncStream<streamIdx::Compute>();
      /// Receiving active blocks from other GPU devices
      for (int otherdid = 0; otherdid < g_device_cnt; otherdid++)
        if (otherdid != did) {
          cuDev.spareStreamWaitForEvent(
              otherdid, Cuda::ref_cuda_context(otherdid).event_spare(did));
          cuDev.spare_launch(otherdid, {(nbcnt[otherdid] + 127) / 128, 128},
                             mark_overlapping_blocks, (uint32_t)nbcnt[otherdid],
                             otherdid, (const ivec3 *)haloBlockIds[did][otherdid],
                             partitions[rollid ^ 1][did],
                             outputHaloGridBlocks[did]._counts + otherdid,
                             outputHaloGridBlocks[did]._buffers[otherdid]);
          cuDev.spare_event_record(otherdid);
          cuDev.computeStreamWaitForEvent(cuDev.event_spare(otherdid));
        }
      /// Reset halo blocks
      partitions[rollid ^ 1][did].resetHaloCount(cuDev.stream_compute());
      cuDev.compute_launch(
          {(pbcnt[did] + 127) / 128, 128}, collect_blockids_for_halo_reduction,
          (uint32_t)pbcnt[did], did, partitions[rollid ^ 1][did]);
      /// Retrieve halo-tag counts
      partitions[rollid ^ 1][did].retrieveHaloCount(cuDev.stream_compute());
      outputHaloGridBlocks[did].retrieveCounts(cuDev.stream_compute());
      cuDev.syncStream<streamIdx::Compute>();
      timer.tock(fmt::format("GPU[{}] step {} halo_tagging.", did, curStep));

      fmt::print(fg(fmt::color::green), "Halo particle blocks GPU[{}]: [{}]\n", did,
                 partitions[rollid ^ 1][did].h_count);
      for (int otherdid = 0; otherdid < g_device_cnt; otherdid++)
        fmt::print(fg(fmt::color::green), "Halo grid blocks GPU[{}] with GPU[{}]: [{}]\n", did,
                   otherdid, outputHaloGridBlocks[did].h_counts[otherdid]);
      //cuDev.syncStream<streamIdx::Compute>(); // JB?
      //inputHaloGridBlocks[did].temp_deallocate(temp_allocator{did}); // JB
      //outputHaloGridBlocks[did].temp_deallocate(temp_allocator{did}); // JB
    });
    sync();
  }
  void collect_halo_grid_blocks(int gid = 1) {
    /// init halo grid blocks
    issue([this](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did); //< JB, may need to remove
      fmt::print(fg(fmt::color::green), " GPU[{}] Start collect halo grid blocks\n", did);
      std::vector<uint32_t> counts(g_device_cnt);
      outputHaloGridBlocks[did].initBuffer(temp_allocator{did}, outputHaloGridBlocks[did].h_counts);
      for (int otherdid = 0; otherdid < g_device_cnt; otherdid++)
        counts[otherdid] = (otherdid != did)
                               ? outputHaloGridBlocks[otherdid].h_counts[did]
                               : 0; //< If other GPUs, set halo grid-blocks count
      inputHaloGridBlocks[did].initBuffer(temp_allocator{did}, counts);
    });
    sync();
    issue([this, gid](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      CppTimer timer{};
      timer.tick();
      /// Share local GPU active Halo Blocks with neighboring GPUs
      for (int otherdid = 0; otherdid < g_device_cnt; otherdid++)
        if (otherdid != did) {
          if (outputHaloGridBlocks[did].h_counts[otherdid] > 0) {
            auto &cnt = outputHaloGridBlocks[did].h_counts[otherdid];
            cuDev.spare_launch(otherdid, {cnt, g_blockvolume},
                               collect_grid_blocks, gridBlocks[gid][did],
                               partitions[rollid][did],
                               outputHaloGridBlocks[did]._buffers[otherdid]);
            outputHaloGridBlocks[did].send(inputHaloGridBlocks[otherdid], did,
                                           otherdid, cuDev.stream_spare(otherdid));
            cuDev.spare_event_record(otherdid);
          } else
            inputHaloGridBlocks[otherdid].h_counts[did] = 0; //< Zero count if no other GPUs
        }
      timer.tock(
          fmt::format("GPU[{}] step {} collect_send_halo_grid.\n", did, curStep));
      //inputHaloGridBlocks[did].temp_deallocate(temp_allocator{did});
      //outputHaloGridBlocks[did].temp_deallocate(temp_allocator{did});
    });
    sync();
  }
  
  void reduce_halo_grid_blocks(int gid = 1) {
    issue([this, gid](int did) {
      auto &cuDev = Cuda::ref_cuda_context(did);
      CppTimer timer{};
      timer.tick();
      /// Receiving active Halo Blocks from other GPU devices, merge with local Blocks
      for (int otherdid = 0; otherdid < g_device_cnt; otherdid++)
        if (otherdid != did) {
          if (inputHaloGridBlocks[did].h_counts[otherdid] > 0) {
            cuDev.spareStreamWaitForEvent(
                otherdid, Cuda::ref_cuda_context(otherdid).event_spare(did));
            cuDev.spare_launch(otherdid,
                               {inputHaloGridBlocks[did].h_counts[otherdid],
                                g_blockvolume},
                               reduce_grid_blocks, gridBlocks[gid][did],
                               partitions[rollid][did],
                               inputHaloGridBlocks[did]._buffers[otherdid]);
            cuDev.spare_event_record(otherdid);
            cuDev.computeStreamWaitForEvent(cuDev.event_spare(otherdid));
          }
        }
      cuDev.syncStream<streamIdx::Compute>();
      timer.tock(fmt::format("GPU[{}] step {} receive_reduce_halo_grid.\n", did,
                             curStep));
      //cuDev.resetMem(); // JB
      //inputHaloGridBlocks[did].temp_deallocate(temp_allocator{did}) ; // JB
      //outputHaloGridBlocks[did].temp_deallocate(temp_allocator{did}); // JB
    });
    sync();
  } //< End


  // * Declare Simulation basic run-time settings
  PREC length; ///< Max length of domain, [m]
  uint64_t domainCellCnt; ///< Max number of cells in domain
  float dt, nextDt, dtDefault, curTime, nextTime, maxVel;
  pvec3 grav; ///< Gravity 3D vector, [m/s^2]
  uint64_t curFrame, curStep, fps, nframes;
  pvec3 vel0[g_device_cnt][g_models_per_gpu]; ///< Sets initial velocities per gpu model, not individual particles
  float initTime = 0.0f; ///< Start time of sim, [sec]

  // * Data-structures on GPUs or cast by kernels
  std::vector<Partition<1>> partitions[2]; ///< Organizes partition + halo info, halo_buffer.cuh
  std::vector<GridBuffer> gridBlocks[2]; //< Organizes grid data in blocks
  std::vector<particle_buffer_t> particleBins[2][g_device_cnt]; //< Organizes particle data in bins
  std::vector<element_buffer_t> elementBins; //< Organizes FEM element data in bins
  std::vector<HaloGridBlocks> inputHaloGridBlocks, outputHaloGridBlocks;
  // std::vector<HaloParticleBlocks> inputHaloParticleBlocks, outputHaloParticleBlocks;
  std::vector<optional<SignedDistanceGrid>> collisionObjs; // Needs to be reimplemented

  std::array<uint32_t, g_device_cnt> h_model_cnt = {0}; ///< Particle models per GPU device
  std::vector<GridTarget> d_gridTarget; ///< Grid-target device arrays 
  std::vector<ParticleTarget> d_particleTarget; ///< Particle-target device arrays
  std::vector<vec7> d_grid_target; ///< Grid target boundaries and type 
  std::vector<vec7> d_particle_target; ///< Grid target boundaries and type 

  GridBoundaryConfigs h_gridBoundaryConfigs[g_max_grid_boundaries]; ///< Grid boundaries
  GridBoundaryConfigs * d_gridBoundaryConfigs; ///< Grid boundaries

  vec<vec7, g_max_grid_boundaries> gridBoundary; ///< Grid boundaries
  vec3 d_motionPath; ///< Motion path info (time, disp, vel) to send to device kernels

  // Labels for attribute I/O
  std::vector<std::string> output_attribs[g_device_cnt][g_models_per_gpu];
  std::vector<std::string> track_attribs[g_device_cnt][g_models_per_gpu];
  std::vector<std::string> particle_target_attribs[g_device_cnt][g_models_per_gpu];
  std::vector<std::string> grid_target_attribs[g_device_cnt];

  vec<ParticleArray, g_models_per_gpu> particles[g_device_cnt]; //< Basic GPU arrays for Particle positions
  std::vector<particle_attrib_t> pattribs[g_device_cnt];  //< Basic GPU arrays for Particle attributes
  std::vector<particle_attrib_t> pattribs_init[g_device_cnt];  //< Basic GPU arrays for Particle init. attributes

  vec<VerticeArray, g_device_cnt> d_vertices; //< Device arrays for FEM Vertice positions
  vec<ElementArray, g_device_cnt> d_element_IDs; //< Device arrays FEM Element node IDs
  vec<ElementAttrib, g_device_cnt> d_element_attribs; //< Device arrays for FEM Gauss Point attributes
  // struct Intermediates
  struct Intermediates {
    void *base;
    PREC_G *d_strain_energy_particles;
    PREC_G *d_gravity_energy_particles;
    PREC_G *d_kinetic_energy_particles;
    PREC_G *d_gravity_energy_grid;
    PREC_G *d_kinetic_energy_grid;
    PREC_G *d_maxVel;
    int *d_tmp;
    int *activeBlockMarks;
    int *destinations;
    int *sources;
    int *binpbs;
    void alloc(int maxBlockCnt) {
      checkCudaErrors(cudaMalloc(&base, sizeof(int) * (maxBlockCnt * 5 + 6)));
      d_strain_energy_particles = (PREC_G *)((char *)base + sizeof(int) * (maxBlockCnt * 5 + 5));
      d_gravity_energy_particles = (PREC_G *)((char *)base + sizeof(int) * (maxBlockCnt * 5  + 4));
      d_kinetic_energy_particles = (PREC_G *)((char *)base + sizeof(int) * (maxBlockCnt * 5  + 3));
      d_gravity_energy_grid = (PREC_G *)((char *)base + sizeof(int) * (maxBlockCnt * 5 + 2));
      d_kinetic_energy_grid = (PREC_G *)((char *)base + sizeof(int) * (maxBlockCnt * 5 + 1));
      d_maxVel = (PREC_G *)((char *)base + sizeof(int) * maxBlockCnt * 5);
      d_tmp = (int *)((uintptr_t)base);
      activeBlockMarks = (int *)((char *)base + sizeof(int) * maxBlockCnt);
      destinations = (int *)((char *)base + sizeof(int) * maxBlockCnt * 2);
      sources = (int *)((char *)base + sizeof(int) * maxBlockCnt * 3);
      binpbs = (int *)((char *)base + sizeof(int) * maxBlockCnt * 4);
      fmt::print("Allocated tmp device array: bytes[{}].\n", sizeof(int) * (maxBlockCnt * 5 + 6));
    }
    void dealloc() {
      cudaDeviceSynchronize();
      if (base) { 
        checkCudaErrors(cudaFree(base)); 
        fmt::print("Deallocated tmp device array.\n");
      }
    }
    void resize(int maxBlockCnt) {
      dealloc();
      alloc(maxBlockCnt);
    }
  }; 
  Intermediates tmps[g_device_cnt]; //< Pointers to temp. GPU array 

  // Halo grid-block data
  vec<ivec3 *, g_device_cnt, g_device_cnt> haloBlockIds;
  static_assert(std::is_same<GridBufferDomain::index_type, int>::value,
                "Grid Block index type is not int.\n");

  /// * Declare variables for host
  char rollid;
  std::size_t curNumActiveBlocks[g_device_cnt], // Reserved blocks
              checkedCnts[g_device_cnt][2], // Checks need to resize block reservation
              curNumActiveBins[g_device_cnt][g_models_per_gpu], // Reserved particle bins
              checkedBinCnts[g_device_cnt][g_models_per_gpu]; // Checks need to resize particle bin reservation
  vec<PREC_G, g_device_cnt> maxVels; //< Max. velocity of particles in each GPU
  vec<PREC_G, g_device_cnt> kinetic_energy_grid_vals; //< Total Kinetic energy of grid on GPU
  vec<PREC_G, g_device_cnt> gravity_energy_grid_vals; //< Total Gravity energy of grid on GPU
  vec<PREC_G, g_device_cnt> kinetic_energy_particle_vals; //< Total GPU Kinetic energy of particles
  vec<PREC_G, g_device_cnt> gravity_energy_particle_vals; //< Total GPU Gravity energy of particles
  vec<PREC_G, g_device_cnt> strain_energy_particle_vals; //< Total GPU Strain energy of particles 
  PREC init_gravity_energy_particles;
  
  vec<int, g_device_cnt> pbcnt, nbcnt, ebcnt; ///< Num blocks
  vec<vec<int, g_models_per_gpu>, g_device_cnt> bincnt; ///< Num par. bins
  vec<vec<uint32_t, g_models_per_gpu>, g_device_cnt> pcnt; ///< Num. particles
  vec<int, g_device_cnt> element_cnt;
  vec<int, g_device_cnt> vertice_cnt;
  std::vector<vec<uint32_t, g_device_cnt>> grid_tarcnt; //< Num. gridTarget nodes 
  std::vector<vec<uint32_t, g_device_cnt>> particle_tarcnt; //< Num. particleTarget particles

  std::vector<float> durations[g_device_cnt + 1]; // should this be floats...?
  std::vector<std::array<PREC, 3>> models[g_device_cnt][g_models_per_gpu];
  std::vector<PREC> attribs[g_device_cnt][g_models_per_gpu];

  int number_of_grid_targets = 0;
  int number_of_particle_targets = 0;
  std::vector<std::array<PREC_G, g_grid_target_attribs>> host_gridTarget[g_device_cnt]; ///< Grid target data (x,y,z,m,mx,my,mz,fx,fy,fz) on host (JB)
  std::vector<std::array<PREC, g_particle_target_attribs>> host_particleTarget[g_device_cnt][g_models_per_gpu]; ///< Particle target data on host (JB)

  int num_particle_trackers[g_device_cnt][g_models_per_gpu] = {0}; //< Number of particle trackers
  std::vector<std::array<PREC, 13>> host_vertices[g_device_cnt];
  std::vector<std::array<int, 4>> host_element_IDs[g_device_cnt];
  std::vector<std::array<PREC, 6>> host_element_attribs[g_device_cnt];
  std::vector<std::array<PREC_G, 3>> host_motionPath; ///< Motion-Path (time, disp, vel) on host (JB)
  std::array<bool, g_device_cnt> flag_fem = {false}; // Toggle finite elements
  std::array<std::array<bool, g_models_per_gpu>, g_device_cnt> flag_pi = {false};
  bool flag_pe = false; // Toggle particle energy output
  bool flag_pt = false; // Toggle particle target
  bool flag_ti = false; // Toggle particle tracked ID 
  bool flag_ge = false; // Toggle grid energy output 
  bool flag_gt = false; // Toggle grid target
  bool flag_mp = false; // Toggle motion path

  PREC_G host_pe_freq = 60.f; // Frequency of particle-energy output
  PREC_G host_pt_freq = 60.f; // Frequency of particle-target output
  PREC_G host_ti_freq = 60.f; // Frequency of particle tracked ID output
  PREC_G host_ge_freq = 60.f; // Frequency of grid-energy output
  PREC_G host_gt_freq = 60.f; // Frequency of grid-target output
  PREC_G host_gb_freq = 60.f; // Frequency of grid-boundary output
  PREC_G host_mp_freq = 60.f; // Frequency of motion path sampling

  std::ofstream particleEnergyFile;
  std::ofstream particleTargetFile[g_device_cnt][g_models_per_gpu];
  std::ofstream particleTrackFile[g_device_cnt][g_models_per_gpu];
  std::ofstream gridEnergyFile;
  std::ofstream gridTargetFile[g_device_cnt];

  Instance<signed_distance_field_> _hostData;

  /// Set-up host CPU threads, tasks, locks, etc.
  bool bRunning;
  threadsafe_queue<std::function<void(int)>> jobs[g_device_cnt];
  std::thread ths[g_device_cnt]; ///< thread is not trivial
  std::mutex mut_slave, mut_ctrl;
  std::condition_variable cv_slave, cv_ctrl;
  std::atomic_uint idleCnt{0};

  /// Computations per substep
  std::vector<std::function<void(int)>> init_tasks;
  std::vector<std::function<void(int)>> loop_tasks;

  // Node Communication 
  int rank = 0; //< MPI rank, i.e. GPU Node ID, default root = 0
  int num_ranks = 1; //< Num. of MPI ranks, i.e. total GPU nodes

  bool verb = false; //< If true, print more information to terminal
  std::string save_suffix; //< Suffix for output files, e.g. bgeo, using PartIO
};

} // namespace mn

#endif