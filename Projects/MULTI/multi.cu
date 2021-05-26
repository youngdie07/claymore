#include "mgsp_benchmark.cuh"
#include "partition_domain.h"
#include <MnBase/Geometry/GeometrySampler.h>
#include <MnBase/Math/Vec.h>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <MnSystem/IO/ParticleIO.hpp>

// dragon_particles.bin, 775196
// cube256_2.bin, 1048576
// two_dragons.bin, 388950

decltype(auto) load_model(std::size_t pcnt, std::string filename) {
  std::vector<std::array<float, 3>> rawpos(pcnt);
  auto addr_str = std::string(AssetDirPath) + "MpmParticles/";
  auto f = fopen((addr_str + filename).c_str(), "rb");
  std::fread((float *)rawpos.data(), sizeof(float), rawpos.size() * 3, f);
  std::fclose(f);
  return rawpos;
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
        // model = sample_uniform_box(
        //     g_dx,
        //     ivec3{18 + (idx & 1 ? STRIDE : 0), 18, 18 + (idx & 2 ? STRIDE : 0)},
        //     ivec3{18 + (idx & 1 ? STRIDE : 0) + LEN, 18 + LEN / 3,
        //           18 + (idx & 2 ? STRIDE : 0) + LEN});
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
      float length_x = 82.85f / g_length * 0.3048f;
      float length_y = 1.7526f / 4 / g_length * 0.3048f;
      float length_z = 3.6576f / g_length * 0.3048f;
      float spacing = 1.7526f / 4 / g_length * 0.3048f;
      models[0] = read_sdf(std::string{"Water/OSU_Water_Bath_ft_x271.826_y5.75_z3_dx0.2_pad1.sdf"}, 1.f, g_dx,
                        vec<float, 3>{off, off, off},
                        vec<float, 3>{length_x, length_y, length_z});
      models[1] = read_sdf(std::string{"Water/OSU_Water_Bath_ft_x271.826_y5.75_z3_dx0.2_pad1.sdf"}, 1.f, g_dx,
                        vec<float, 3>{off, off, off},
                        vec<float, 3>{length_x, length_y, (length_z + spacing)});
      models[2] = read_sdf(std::string{"Water/OSU_Water_Bath_ft_x271.826_y5.75_z3_dx0.2_pad1.sdf"}, 1.f, g_dx,
                        vec<float, 3>{off, off, off},
                        vec<float, 3>{length_x, length_y, (length_z + 2.f*spacing)});
      models[3] = read_sdf(std::string{"Water/OSU_Water_Bath_ft_x271.826_y5.75_z3_dx0.2_pad1.sdf"}, 1.f, g_dx,
                        vec<float, 3>{off, off, off},
                        vec<float, 3>{length_x, length_y, (length_z + 3.f*spacing)});
  }
  default:
    break;
  }
}

int main() {
  using namespace mn;
  using namespace config;
  Cuda::startup();

  std::vector<std::array<float, 3>> models[g_device_cnt];

  auto benchmark = std::make_unique<mgsp_benchmark>();
  /// init
  init_models(models, 4);

  for (int did = 0; did < g_device_cnt; ++did)
    benchmark->initModel(did, models[did]);
  // benchmark->initBoundary("candy_base");

  benchmark->main_loop();
  ///
  IO::flush();
  benchmark.reset();
  ///
  Cuda::shutdown();
  return 0;
}