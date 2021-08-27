#include "mgsp_benchmark.cuh"
#include "partition_domain.h"
#include <MnBase/Geometry/GeometrySampler.h>
#include <MnBase/Math/Vec.h>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <MnSystem/IO/ParticleIO.hpp>


#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
typedef std::vector<std::array<float, 3>> WaveHolder;
typedef std::vector<std::array<float, 10>> VerticeHolder;
typedef std::vector<std::array<int, 4>> ElementHolder;
float verbose = 0;

// dragon_particles.bin, 775196
// cube256_2.bin, 1048576
// two_dragons.bin, 388950

decltype(auto) load_model(std::size_t pcnt, std::string filename) {
  std::vector<std::array<float, 3>> rawpos(pcnt);
  auto addr_str = std::string(AssetDirPath) + "MpmParticles/";
  auto f = fopen((addr_str + filename).c_str(), "rb");
  auto res = std::fread((float *)rawpos.data(), sizeof(float), rawpos.size() * 3, f);
  std::fclose(f);
  return rawpos;
}

void load_waveMaker(const std::string& filename, char sep, WaveHolder& fields){
  auto addr_str = std::string(AssetDirPath) + "WaveMaker/";
  std::ifstream in((addr_str + filename).c_str());
  if (in) {
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          int col = 0;
          std::array<float, 3> arr;
          while (getline(sep, field, ',')) {
              if (col >= 3) break;
              arr[col] = stof(field);
              col++;
          }
          fields.push_back(arr);
      }
  }
  if (verbose) {
    for (auto row : fields) {
        for (auto field : row) std::cout << field << ' ';
        std::cout << '\n';
    }
  }
}


void load_FEM_Particles(const std::string& filename, char sep, 
                        std::vector<std::array<float, 3>>& fields, 
                        mn::vec<float, 3> offset, 
                        mn::vec<float, 3> lengths){
  auto addr_str = std::string(AssetDirPath) + "TetMesh/";
  std::ifstream in((addr_str + filename).c_str());
  if (in) {
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          const int el = 3; // x, y, z - Default
          std::array<float, 3> arr;
          int col = 0;
          while (getline(sep, field, ',')) {
              if (col >= el) break;
              arr[col] = (stof(field) + offset[col]) / mn::config::g_length + (8.f * mn::config::g_dx);
              col++;
          }
          fields.push_back(arr);
      }
  }
  if (verbose) {
    for (auto row : fields) {
        for (auto field : row) std::cout << field << ' ';
        std::cout << '\n';
    }
  }
}

void load_FEM_Vertices(const std::string& filename, char sep, 
                       VerticeHolder& fields, 
                       mn::vec<float, 3> offset, 
                       mn::vec<float, 3> lengths){
  auto addr_str = std::string(AssetDirPath) + "TetMesh/";
  std::ifstream in((addr_str + filename).c_str());
  if (in) {
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          const int el = 3; // x, y, z - Default
          std::array<float, 10> arr;
          int col = 0;
          while (getline(sep, field, ',')) {
              if (col >= el) break;
              arr[col] = (stof(field) + offset[col]) / mn::config::g_length + (8.f * mn::config::g_dx);
              arr[col+el] = arr[col]; 
              col++;
          }
          arr[6] = 0.f;
          arr[7] = 0.f;
          arr[8] = 0.f;
          arr[9] = 0.f;
          fields.push_back(arr);
      }
  }
  if (verbose) {
    for (auto row : fields) {
        for (auto field : row) std::cout << field << ' ';
        std::cout << '\n';
    }
  }
}

void load_FEM_Elements(const std::string& filename, char sep, 
                       ElementHolder& fields){
  auto addr_str = std::string(AssetDirPath) + "TetMesh/";
  std::ifstream in((addr_str + filename).c_str());
  if (in) {
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          int col = 0;
          // Elements hold integer IDs of vertices
          const int el = 4; // 1st-Order Tetrahedron
          std::array<int, el> arr;
          while (getline(sep, field, ',')) {
              if (col >= el) break;
              arr[col] = stoi(field); // string to integer
              col++;
          }
          fields.push_back(arr);
      }
  }
  if (verbose) {
    for (auto row : fields) {
        for (auto field : row) std::cout << field << ' ';
        std::cout << '\n';
    }
  }
}

// load from analytic levelset
// init models
void init_models(
    std::vector<std::array<float, 3>> models[mn::config::g_device_cnt],
    int opt = 0) {
  using namespace mn;
  using namespace config;
  switch (opt) {
  case 0:
    models[0] = load_model(775196, "dragon_particles.bin");
    models[1] = read_sdf(std::string{"two_dragons.sdf"}, 8.f, g_dx,
                         vec<float, 3>{0.5f, 0.5f, 0.5f},
                         vec<float, 3>{0.2f, 0.2f, 0.2f});
    break;
  case 1:
    models[0] = load_model(775196, "dragon_particles.bin");
    models[1] = load_model(775196, "dragon_particles.bin");
    for (auto &pt : models[1])
      pt[1] -= 0.3f;
    break;
  case 2: {
    constexpr auto LEN = 54;
    constexpr auto STRIDE = 56;
    constexpr auto MODEL_CNT = 1;
    for (int did = 0; did < g_device_cnt; ++did) {
      models[did].clear();
      std::vector<std::array<float, 3>> model;
      for (int i = 0; i < MODEL_CNT; ++i) {
        auto idx = (did * MODEL_CNT + i);
        model = sample_uniform_box(
            g_dx, ivec3{18 + (idx & 1 ? STRIDE : 0), 18, 18},
            ivec3{18 + (idx & 1 ? STRIDE : 0) + LEN, 18 + LEN, 18 + LEN});
        models[did].insert(models[did].end(), model.begin(), model.end());
      }
    }
  } break;
  case 3: {
    constexpr auto LEN = 32; // 54;
    constexpr auto STRIDE = (g_domain_size / 4);
    constexpr auto MODEL_CNT = 1;
    for (int did = 0; did < g_device_cnt; ++did) {
      models[did].clear();
      std::vector<std::array<float, 3>> model;
      for (int i = 0; i < MODEL_CNT; ++i) {
        auto idx = (did * MODEL_CNT + i);
        model = sample_uniform_box(
            g_dx,
            ivec3{8 + (idx & 1 ? STRIDE : 0), 8, 8},
            ivec3{8 + (idx & 1 ? STRIDE : 0) + LEN, 8 + LEN / 3,
                  8 + LEN});
                  
        models[did].insert(models[did].end(), model.begin(), model.end());
      }
    }
  } break;
  case 4: {
      float off = 8.f * g_dx;
      float f = 1.f;

      // float water_ppc = MODEL_PPC;
      // vec<float, 3> water_lengths;
      // water_lengths[0] = 281.5735f * 0.3048f / g_length * f;
      // water_lengths[1] = 2.f / g_length * f;
      //water_lengths[2] = 12.f * 0.3048f / g_length * f;

      if (g_device_cnt == 1) {
        // vec<float, 3> debris_offset{0.25f, 0.5f, 0.5f};
        // debris_offset /= g_length;
        // debris_offset = debris_offset + off;
        // vec<float, 3> debris_lengths{0.5f, 0.051f, 0.102f};
        // debris_lengths /= g_length;
        // float debris_ppc = MODEL_PPC_FC;
        // models[0] = read_sdf(std::string{"Debris/OSU_Debris_0.5x_0.051y_0.102z_dx0.001_pad1.sdf"}, 
        //                   debris_ppc, mn::config::g_dx, mn::config::g_domain_size,
        //                   debris_offset, debris_lengths);
        load_FEM_Particles(std::string{"Debris/OSU_AT162_spacing_2.5cm_res8_Vertices.csv"}, ',', models[0], 
                            vec<float, 3>{1.525f, 1.95f, 1.0375f}, 
                            vec<float, 3>{0.475f, 0.1f, 2.075f});

      } else if (g_device_cnt == 2) {
        load_FEM_Particles(std::string{"Debris/OSU_AT162_spacing_2.5cm_res8_Vertices.csv"}, ',', models[0], 
                            vec<float, 3>{1.525f, 0.125f, 1.0375f}, 
                            vec<float, 3>{0.475f, 0.1f, 2.075f});
        // models[0] = read_sdf(std::string{"Water/OSU_Water_Bath_ft_281.5735x_6.5617y_12z_dx0.2_pad1.sdf"}, 
        //                   water_ppc, mn::config::g_dx, mn::config::g_domain_size,
        //                   vec<float, 3>{off, off, off},
        //                   water_lengths);     
        //vec<float, 3> debris_offset{42.f, 2.f, 0.7538f};
        // vec<float, 3> debris_offset{2.525f, 2.5f, 1.0375f};
        // debris_offset /= g_length;
        // debris_offset = debris_offset + off;
        // vec<float, 3> debris_lengths{0.558f, 0.051f, 2.15f};
        // debris_lengths /= g_length;
        // float debris_ppc = MODEL_PPC_FC;
        // models[1] = read_sdf(std::string{"Debris/OSU_AT162_spacing_5cm_dx0.005_pad1.sdf"}, 
        //                   debris_ppc, mn::config::g_dx, mn::config::g_domain_size,
        //                   debris_offset, debris_lengths);
        load_FEM_Particles(std::string{"Debris/OSU_AT162_spacing_2.5cm_res8_Vertices.csv"}, ',', models[1], 
                            vec<float, 3>{1.525f, 0.0f, 1.0375f}, 
                            vec<float, 3>{0.475f, 0.1f, 2.075f});
      }
  }
  default:
    break;
  }
}

int main() {
  using namespace mn;
  using namespace config;
  Cuda::startup();

  ElementHolder h_FEM_Elements;
  VerticeHolder h_FEM_Vertices;

  std::vector<std::array<float, 3>> models[g_device_cnt];
  std::vector<std::array<float, 10>> h_gridTarget(mn::config::g_target_cells, 
                                                  std::array<float, 10>{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f});
  float off = 8.f * g_dx;
  
  vec<float, 3> h_point_a;
  vec<float, 3> h_point_b;
  h_point_a[0] = 2.f / g_length + (2.f * g_dx);
  h_point_a[1] = 0.05f / g_length;
  h_point_a[2] = 1.6f / g_length;
  h_point_a = h_point_a + off;
  h_point_b[0] = h_point_a[0] + (1.f * g_dx);
  h_point_b[1] = h_point_a[1] + 0.4f / g_length;
  h_point_b[2] = h_point_a[2] + 0.8f / g_length;

  /// Initialize
  auto benchmark = std::make_unique<mgsp_benchmark>();

  std::cout << "Load MPM Particles" << '\n';
  //getchar();
  init_models(models, 4);

  std::cout << "Load FEM Elements" << '\n';
  //getchar();
  load_FEM_Elements(std::string{"Debris/OSU_AT162_spacing_2.5cm_res8_Elements.csv"}, ',', h_FEM_Elements);


  std::cout << "Load FEM Vertices" << '\n';
  //getchar();
  load_FEM_Vertices(std::string{"Debris/OSU_AT162_spacing_2.5cm_res8_Vertices.csv"}, ',', h_FEM_Vertices,
                    vec<float, 3>{1.525f, 0.125f, 1.0375f}, 
                    vec<float, 3>{0.475f, 0.1f, 2.075f});

  std::cout << "Initialize Simulation" << '\n';
  //getchar();

  /// Loop through GPU devices
  for (int did = 0; did < g_device_cnt; ++did) {
    benchmark->initModel(did, models[did]);
    benchmark->initGridTarget(did, h_gridTarget, h_point_a, h_point_b, 1.f);
    benchmark->initFEM(did, h_FEM_Vertices, h_FEM_Elements);
    // benchmark->initFEMElements();
  }
  benchmark->main_loop();
  ///
  IO::flush();
  benchmark.reset();
  ///
  Cuda::shutdown();
  return 0;
}