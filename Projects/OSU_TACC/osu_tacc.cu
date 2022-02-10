#include "mgsp_benchmark.cuh"
#include "partition_domain.h"
#include <MnBase/Geometry/GeometrySampler.h>
#include <MnBase/Math/Vec.h>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <MnSystem/IO/ParticleIO.hpp>

#include <cxxopts.hpp>
#include <fmt/color.h>
#include <fmt/core.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if 0
#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
namespace rj = rapidjson;
static const char *kTypeNames[] = {"Null",  "False",  "True",  "Object",
                                   "Array", "String", "Number"};


typedef std::vector<std::array<float, 3>> WaveHolder;
typedef std::vector<std::array<float, 10>> VerticeHolder;
typedef std::vector<std::array<int, 4>> ElementHolder;
float verbose = 0;


struct SimulatorConfigs {
  int _dim;
  float _dx, _dxInv;
  int _resolution;
  float _gravity;
  std::vector<float> _offset;
} simConfigs;

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
                        mn::vec<float, 3> offset){
  auto addr_str = std::string(AssetDirPath) + "TetMesh/";
  std::ifstream in((addr_str + filename).c_str());
  if (in) {
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          float f = 1.f; // Scale factor
          const int el = 3; // x, y, z - Default
          std::array<float, 3> arr;
          int col = 0;
          while (getline(sep, field, ',')) {
              if (col >= el) break;
              arr[col] = (stof(field) * f + offset[col]) / mn::config::g_length + (8.f * mn::config::g_dx);
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
                       mn::vec<float, 3> offset){
  auto addr_str = std::string(AssetDirPath) + "TetMesh/";
  std::ifstream in((addr_str + filename).c_str());
  if (in) {
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          float f = 1.f; // Scale factor
          const int el = 3; // x, y, z - Default
          std::array<float, 10> arr;
          int col = 0;
          while (getline(sep, field, ',')) {
              if (col >= el) break;
              arr[col] = (stof(field) * f + offset[col]) / mn::config::g_length + (8.f * mn::config::g_dx);
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



void load_FEM(const std::string& filename, char sep, 
                       VerticeHolder& vertice_fields, 
                       ElementHolder& element_fields,
                       mn::vec<float, 3> offset){
  auto addr_str = std::string(AssetDirPath) + "TetMesh/";
  std::ifstream in((addr_str + filename).c_str());
  if (in) {
      std::string line;
      int vertice_flag = 0;
      int element_flag = 0;
      int rel_row;
      float vertice_max; 
      float element_max;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          
          if (vertice_flag == 1) {
            vertice_max = stof(field); // Set num of vertice from file
            std::cout << "Vertice Num " << vertice_max << ', ' << "g_max_vert";
            std::cout << '\n';
            break;
          }

          int col = 0;
          while (getline(sep, field, ',')) {
            if (col >= 1) break;
            if (field == "Vertices") {
              std::cout << field << ' ';
              std::cout << '\n';
              vertice_flag = 1;  // Trip flag for vertice start in file
            }          
            col++;
          }
      }

      rel_row = 0;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          
          
          if (rel_row >= vertice_max) break;

          float f = 1.f; // Scale factor
          const int el = 3; // x, y, z - Default
          std::array<float, 10> arr;
          int col = 0;
          while (getline(sep, field, ',')) {
              if (col >= el) break;
              arr[col] = (stof(field) * f + offset[col]) / mn::config::g_length + (8.f * mn::config::g_dx);
              arr[col+el] = arr[col]; 
              col++;
          }
          arr[6] = 0.f;
          arr[7] = 0.f;
          arr[8] = 0.f;
          arr[9] = 0.f;
          vertice_fields.push_back(arr);
          rel_row++;
      }

      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          
          if (element_flag == 1) {
            element_max = stoi(field); // Set num of elements from file
            std::cout << "Element Num " << element_max << ', ' << "g_max_element" ;
            std::cout << '\n';
            break;
          }

          int col = 0;
          while (getline(sep, field, ',')) {
            if (col >= 1) break;
            if (field == "Tetrahedra") {
              std::cout << field << ' ';
              std::cout << '\n';
              element_flag = 1;  // Trip flag for element start in file
            }          
            col++;
          }
      }

      rel_row = 0;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          
          if (rel_row >= element_max) break;

          int col = 0;
          // Elements hold integer IDs of vertices
          const int el = 4; // 1st-Order Tetrahedron
          std::array<int, el> arr;
          while (getline(sep, field, ',')) {
              if (col >= el) break;
              arr[col] = stoi(field); // string to integer
              col++;
          }
          element_fields.push_back(arr);
          rel_row++;
      }
  }
  if (verbose) {
    for (auto row : vertice_fields) {
        for (auto field : row) std::cout << field << ' ';
        std::cout << '\n';
    }
    for (auto row : element_fields) {
        for (auto field : row) std::cout << field << ' ';
        std::cout << '\n';
    }
  }
}



void parse_scene(std::string fn,
                 std::unique_ptr<mn::mgsp_benchmark> &benchmark,
                 std::vector<std::array<float, 3>> models[mn::config::g_device_cnt]) {

  fs::path p{fn};
  if (p.empty())
    fmt::print("file not exist {}\n", fn);
  else {
    std::size_t size = fs::file_size(p);
    std::string configs;
    configs.resize(size);

    std::ifstream istrm(fn);
    if (!istrm.is_open())
      fmt::print("cannot open file {}\n", fn);
    else
      istrm.read(const_cast<char *>(configs.data()), configs.size());
    istrm.close();
    fmt::print("load the scene file of size {}\n", size);

    rj::Document doc;
    doc.Parse(configs.data());
    for (rj::Value::ConstMemberIterator itr = doc.MemberBegin();
         itr != doc.MemberEnd(); ++itr) {
      fmt::print("Scene member {} is {}\n", itr->name.GetString(),
                 kTypeNames[itr->value.GetType()]);
    }
    {
      auto it = doc.FindMember("simulation");
      if (it != doc.MemberEnd()) {
        auto &sim = it->value;
        if (sim.IsObject()) {
          fmt::print(
              fg(fmt::color::cyan),
              "simulation: gpuid[{}], defaultDt[{}], fps[{}], frames[{}]\n",
              sim["gpu"].GetInt(), sim["default_dt"].GetFloat(),
              sim["fps"].GetInt(), sim["frames"].GetInt());
          benchmark = std::make_unique<mn::mgsp_benchmark>(
              sim["default_dt"].GetFloat(),
              sim["fps"].GetInt(), sim["frames"].GetInt());
        }
      }
    } ///< end simulation parsing
    {
      auto it = doc.FindMember("meshes");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print("has {} meshes\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
            if (model["gpu"].GetInt() >= mn::config::g_device_cnt) {
              fmt::print(fg(fmt::color::green),
                       "Mesh gpu[{}] exceeds global device count!\n", 
                       model["gpu"].GetInt());
              break;
            }
            std::string constitutive{model["constitutive"].GetString()};
            fmt::print(fg(fmt::color::green),
                       "model constitutive[{}], file[{}]\n", constitutive,
                       model["file"].GetString());
            fs::path p{model["file"].GetString()};
            if (constitutive == "Meshed") {

              benchmark->updateMeshedParameters(
                model["gpu"].GetInt(),
                model["rho"].GetFloat(), model["ppc"].GetFloat(),
                model["youngs_modulus"].GetFloat(),
                model["poisson_ratio"].GetFloat(),
                model["alpha"].GetFloat(),
                model["beta_min"].GetFloat(),
                model["beta_max"].GetFloat());
              fmt::print("Did Meshed Update!! [{}]\n", model["youngs_modulus"].GetFloat());
            }
            getchar();

            ElementHolder h_FEM_Elements;
            VerticeHolder h_FEM_Vertices;

            float g_length = mn::config::g_length;
            mn::vec<float, 3> offset, velocity;
            for (int d = 0; d < 3; ++d) {
              offset[d] = model["offset"].GetArray()[d].GetFloat(); //< Adjusted in load_FEM_Vertices/Particles
              velocity[d] = model["velocity"].GetArray()[d].GetFloat() / g_length; 
            }
            std::cout << "Load FEM Elements by JSON" << '\n';
            load_FEM_Elements(model["file_elements"].GetString(), ',', h_FEM_Elements);
            getchar();

            std::cout << "Load FEM Vertices by JSON" << '\n';
            load_FEM_Vertices(model["file_vertices"].GetString(), ',', h_FEM_Vertices,
                              offset);
            getchar();

            benchmark->initFEM(model["gpu"].GetInt(), h_FEM_Vertices, h_FEM_Elements);


            load_FEM_Particles(model["file_vertices"].GetString(), ',', 
                                models[model["gpu"].GetInt()], 
                                offset);
            
            benchmark->initModel(model["gpu"].GetInt(), 
                                models[model["gpu"].GetInt()], 
                                velocity);
          }
        }
      }
    } ///< end mesh parsing
    {
      auto it = doc.FindMember("models");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print("has {} models\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
            if (model["gpu"].GetInt() >= mn::config::g_device_cnt) {
              fmt::print(fg(fmt::color::green),
                       "Model gpu[{}] exceeds global device count!\n", 
                       model["gpu"].GetInt());
              break;
            }
            std::string constitutive{model["constitutive"].GetString()};
            fmt::print(fg(fmt::color::green),
                       "model constitutive[{}], file[{}]\n", constitutive,
                       model["file"].GetString());
            fs::path p{model["file"].GetString()};

            auto initModel = [&](auto &positions, auto &velocity) {

              if (constitutive == "jfluid") {
                benchmark->updateJFluidParameters( model["gpu"].GetInt(),
                    model["rho"].GetFloat(), model["ppc"].GetFloat(),
                    model["bulk_modulus"].GetFloat(), model["gamma"].GetFloat(),
                    model["viscosity"].GetFloat());
              } else if (constitutive == "JFluid_ASFLIP") {
                benchmark->updateJFluidASFLIPParameters( model["gpu"].GetInt(),
                    model["rho"].GetFloat(), model["ppc"].GetFloat(),
                    model["bulk_modulus"].GetFloat(), model["gamma"].GetFloat(),
                    model["viscosity"].GetFloat(), 
                    model["alpha"].GetFloat(), 
                    model["beta_min"].GetFloat(), model["beta_max"].GetFloat());
              } else if (constitutive == "fixed_corotated") {
                benchmark->updateFRParameters( model["gpu"].GetInt(),
                    model["rho"].GetFloat(), model["ppc"].GetFloat(),
                    model["youngs_modulus"].GetFloat(),
                    model["poisson_ratio"].GetFloat());
              } else if (constitutive == "FixedCorotated_ASFLIP") {
                benchmark->updateFRASFLIPParameters( model["gpu"].GetInt(),
                    model["rho"].GetFloat(), model["ppc"].GetFloat(),
                    model["youngs_modulus"].GetFloat(),
                    model["poisson_ratio"].GetFloat(), 
                    model["alpha"].GetFloat(), 
                    model["beta_min"].GetFloat(), model["beta_max"].GetFloat());
              } else if (constitutive == "nacc") {
                benchmark->updateNACCParameters( model["gpu"].GetInt(),
                    model["rho"].GetFloat(), model["ppc"].GetFloat(),
                    model["youngs_modulus"].GetFloat(),
                    model["poisson_ratio"].GetFloat(), model["beta"].GetFloat(),
                    model["xi"].GetFloat());
              } else if (constitutive == "sand") { 
                benchmark->updateSandParameters( model["gpu"].GetInt(),
                    model["rho"].GetFloat(), model["ppc"].GetFloat(),
                    model["youngs_modulus"].GetFloat(),
                    model["poisson_ratio"].GetFloat());              
              }
              
              benchmark->initModel(model["gpu"].GetInt(), positions, velocity);
            };
            float g_length = mn::config::g_length;
            float off = mn::config::g_dx * 8.f;
            mn::vec<float, 3> offset, span, velocity;
            for (int d = 0; d < 3; ++d) {
              offset[d] = model["offset"].GetArray()[d].GetFloat() / g_length + off;
              span[d] = model["span"].GetArray()[d].GetFloat() / g_length;
              velocity[d] = model["velocity"].GetArray()[d].GetFloat() / g_length;
            }
            if (p.extension() == ".sdf") {
              if (model["partition"].GetFloat()){
                mn::vec<float, 3> point_a, point_b;
                for (int d = 0; d < 3; ++d) {
                  point_a[d] = model["point_a"].GetArray()[d].GetFloat() / g_length;
                  point_b[d] = model["point_b"].GetArray()[d].GetFloat() / g_length;
                }
                auto positions = mn::read_sdf(
                    model["file"].GetString(), model["ppc"].GetFloat(),
                    mn::config::g_dx, mn::config::g_domain_size, offset, span,
                    point_a, point_b);  
                mn::IO::insert_job([&]() {
                  mn::write_partio<float, 3>(std::string{p.stem()} + ".bgeo",
                                            positions);
                });              
                mn::IO::flush();
                initModel(positions, velocity);
              }
              else {
                auto positions = mn::read_sdf(
                    model["file"].GetString(), model["ppc"].GetFloat(),
                    mn::config::g_dx, mn::config::g_domain_size, offset, span);
                mn::IO::insert_job([&]() {
                mn::write_partio<float, 3>(std::string{p.stem()} + ".bgeo",
                                           positions);
                });
                mn::IO::flush();
                initModel(positions, velocity);
              }

            }
          }
        }
      }
    } ///< end models parsing
    {
      auto it = doc.FindMember("targets");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print("has {} targets\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
          
            std::vector<std::array<float, 10>> h_gridTarget(mn::config::g_target_cells, 
                                                            std::array<float, 10>{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f});
            float g_length = mn::config::g_length;
            float g_dx = mn::config::g_dx;
            float off = 8.f * g_dx;
            mn::vec<float, 3> h_point_a, h_point_b;
            for (int d = 0; d < 3; ++d) {
              h_point_a[d] = model["point_a"].GetArray()[d].GetFloat() / g_length + off;
              h_point_b[d] = model["point_b"].GetArray()[d].GetFloat() / g_length + off;
            }

            if (1) {
              h_point_b[0] = h_point_b[0] + (1.f * g_dx);              
              h_point_a[1] = h_point_a[1] + (1.f * g_dx);
              h_point_b[1] = h_point_b[1] + (1.f * g_dx);
            }

            // ----------------
            /// Loop through GPU devices
            //if (model["gpu"].GetString() == "all") {
            for (int did = 0; did < mn::config::g_device_cnt; ++did) {
              fmt::print("device {} target\n", did);
              benchmark->initGridTarget(did, h_gridTarget, h_point_a, h_point_b, 
                model["output_frequency"].GetFloat());
            }
          }
        }
      }
    } ///< end grid-target parsing
    {
      auto it = doc.FindMember("wave-gauges");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print("has {} wave-gauges\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {

            float g_length = mn::config::g_length;
            float g_dx = mn::config::g_dx;
            float off = 8.f * g_dx;
            mn::vec<float, 3> h_point_a, h_point_b;
            for (int d = 0; d < 3; ++d) {
              h_point_a[d] = model["point_a"].GetArray()[d].GetFloat() / g_length + off;
              h_point_b[d] = model["point_b"].GetArray()[d].GetFloat() / g_length + off;
            }

            if (1) {
              h_point_b[0] = h_point_b[0] + (1.f * g_dx);              
            }

            // ----------------
            /// Loop through GPU devices
            for (int did = 0; did < mn::config::g_device_cnt; ++did) {
              fmt::print("device {} wave-gauge\n", did);
              benchmark->initWaveGauge(did, h_point_a, h_point_b, 
                model["output_frequency"].GetFloat());
            }
          }
        }
      }
    } ///< end wave-gauge parsing
    {
      auto it = doc.FindMember("wave");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print("has {} wave\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
          
            WaveHolder waveMaker;
            load_waveMaker(model["file"].GetString(), ',', waveMaker);

            for (int did = 0; did < mn::config::g_device_cnt; ++did) {
              fmt::print("device {} wave.\n", did);
              benchmark->initWaveMaker(did, waveMaker);
            }

          }
        }
      }
    } ///< end wave parsing
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
  case 0: {
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
  case 1: {
      float off = 8.f * g_dx;
      float water_ppc = MODEL_PPC;

      if (g_device_cnt == 1) {
        // load_FEM_Particles(std::string{"Debris/WASIRF_Debris_B_L10_res0.635cm_Vertices.csv"}, ',', models[0], 
        //                     vec<float, 3>{11.75f, 0.11f, 0.2555f});
        
        // vec<float, 3> debris_offset{11.5f, 0.11f, 0.2976f};
        // debris_offset /= g_length;
        // debris_offset = debris_offset + off;
        // vec<float, 3> debris_lengths{0.3048f, 0.0681f, 0.3048f};
        // debris_lengths /= g_length;
        // models[0] = read_sdf(std::string{"Debris/WASIRF_Debris_G_Cross1_spacing_1.5cm_dx0.0025_pad1.sdf"}, 
        //                   MODEL_PPC_FC, mn::config::g_dx, mn::config::g_domain_size,
        //                   debris_offset, debris_lengths);

        // vec<float, 3> water_offset{4.f, 0.f, 0.f};
        // water_offset /= g_length;
        // water_offset = water_offset + off;
        // vec<float, 3> water_lengths{8.f, 0.1f, 0.9f};
        // water_lengths /= g_length;
        // models[0] = read_sdf(std::string{"Water/Water_x8_y0.1_z0.9_dx0.02_pad1.sdf"}, 
        //                   water_ppc, mn::config::g_dx, mn::config::g_domain_size,
        //                   water_offset, water_lengths);


      } else if (g_device_cnt == 2) {
        // load_FEM_Particles(std::string{"Debris/WASIRF_Debris_G_Cross1_spacing_1.5cm_res_1cm_Vertices.csv"}, ',', models[0], 
        //                     vec<float, 3>{11.2f, 0.11f, 0.4246f});

        load_FEM_Particles(std::string{"Debris/WASIRF_Debris_B_L10_res0.635cm_Vertices.csv"}, ',', models[1], 
                            vec<float, 3>{11.75f, 0.11f, 0.2555f});

        vec<float, 3> water_offset{4.f, 0.f, 0.f};
        water_offset /= g_length;
        water_offset = water_offset + off;
        vec<float, 3> water_lengths{8.f, 0.1f, 0.9f};
        water_lengths /= g_length;
        models[1] = read_sdf(std::string{"Water/Water_x8_y0.1_z0.9_dx0.02_pad1.sdf"}, 
                          water_ppc, mn::config::g_dx, mn::config::g_domain_size,
                          water_offset, water_lengths);

        // vec<float, 3> debris_offset{42.f, 2.f, 0.7538f};
        // debris_offset /= g_length;
        // debris_offset = debris_offset + off;
        // vec<float, 3> debris_lengths{0.558f, 0.051f, 2.15f};
        // debris_lengths /= g_length;
        // float debris_ppc = MODEL_PPC_FC;
        // models[1] = read_sdf(std::string{"Debris/OSU_AT162_spacing_5cm_dx0.005_pad1.sdf"}, 
        //                   debris_ppc, mn::config::g_dx, mn::config::g_domain_size,
        //                   debris_offset, debris_lengths);
      }
  }
  default:
    break;
  }
}

/// Start simulation here.
int main(int argc, char *argv[]) {
  using namespace mn;
  using namespace config;
  // ----------------
  // Start GPU
  Cuda::startup();
  // ----------------
  // Set scene with input file
  cxxopts::Options options("Scene_Loader", "Read simulation scene");
  options.add_options()(
      "f,file", "Scene Configuration File",
      cxxopts::value<std::string>()->default_value("scene.json")); //< scene.json !
  auto results = options.parse(argc, argv);
  auto fn = results["file"].as<std::string>();
  fmt::print("loading scene [{}]\n", fn);
  // ----------------
  // Data-structures for loading onto host
  std::vector<std::array<float, 3>> models[g_device_cnt];
  std::vector<mn::vec<float,3>> v0(g_device_cnt, mn::vec<float,3>{0.f,0.f,0.f});
  float off = 8.f * g_dx; //< Grid-cell buffer size (see off-by-2, Xinlei Wang)
  // ----------------
  /// Initialize the scene
  std::unique_ptr<mn::mgsp_benchmark> benchmark;
  parse_scene(fn, benchmark, models); //< Load from input file  
  if (false) {
    benchmark = std::make_unique<mgsp_benchmark>(9e-6, 10);
  }
  std::cout << "Finished scene initialization." << '\n';
  getchar();
  // ----------------
  /// Grid-Target
  std::cout << "Set grid-target..." << '\n';
  getchar();
  // std::vector<std::array<float, 10>> h_gridTarget(mn::config::g_target_cells, 
  //                                                 std::array<float, 10>{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f});
  // vec<float, 3> h_point_a;
  // vec<float, 3> h_point_b;
  // h_point_a[0] = 12.f / g_length + (2.f * g_dx);
  // h_point_a[1] = 0.075f / g_length;
  // h_point_a[2] = (0.9f - 0.254f) / 2.f / g_length;
  // h_point_a = h_point_a + off;
  // h_point_b[0] = h_point_a[0] + (2.01f * g_dx);
  // h_point_b[1] = h_point_a[1] + 0.254f / g_length;
  // h_point_b[2] = h_point_a[2] + 0.254f / g_length;
  // // ----------------
  // /// Loop through GPU devices
  // for (int did = 0; did < g_device_cnt; ++did) {
  //   //benchmark->initModel(did, models[did], v0[did]);
  //   benchmark->initGridTarget(did, h_gridTarget, h_point_a, h_point_b, 96.f);
  // }
  // ----------------
  /// Run simulation
  std::cout << "Run simulation..." << '\n';
  getchar();
  benchmark->main_loop();
  // ----------------
  /// Clear
  IO::flush();
  benchmark.reset();
  // ----------------
  /// Shutdown GPU
  Cuda::shutdown();
  // ----------------
  /// End program
  return 0;
}