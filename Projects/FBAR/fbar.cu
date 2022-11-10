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
typedef std::vector<std::array<PREC, 13>> VerticeHolder;
typedef std::vector<std::array<int, 4>> ElementHolder;

float verbose = 0;
PREC o = mn::config::g_offset; //< Grid-cell buffer size (see off-by-2, Xinlei Wang)
PREC l = mn::config::g_length; //< Grid-length
PREC dx = mn::config::g_dx; //< Grid-cell length [1x1x1]
//__device__ __constant__ PREC length;

struct SimulatorConfigs {
  int _dim;
  float _dx, _dxInv;
  int _resolution;
  float _gravity;
  std::vector<float> _offset;
} simConfigs;

decltype(auto) load_model(std::size_t pcnt, std::string filename) {
  std::vector<std::array<float, 3>> rawpos(pcnt);
  auto addr_str = std::string(AssetDirPath) + "MpmParticles/";
  auto f = fopen((addr_str + filename).c_str(), "rb");
  auto res = std::fread((float *)rawpos.data(), sizeof(float), rawpos.size() * 3, f);
  std::fclose(f);
  return rawpos;
}

void load_waveMaker(const std::string& filename, char sep, WaveHolder& fields, int rate=10){
  auto addr_str = std::string(AssetDirPath) + "WaveMaker/";
  std::ifstream in((addr_str + filename).c_str());
  if (in) {
      int iter = 0;
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          int col = 0;
          std::array<float, 3> arr;
          while (getline(sep, field, ',')) {
              if (col >= 3) break;
              if ((iter % rate) == 0) arr[col] = stof(field);
              col++;
          }
          if ((iter % rate) == 0) fields.push_back(arr);
          iter++;
      }
  }
  if (verbose) {
    for (auto row : fields)for (auto field : row) std::cout << field << ' ' << '\n'; 
  }
}


void load_csv_particles(const std::string& filename, char sep, 
                        std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> offset, const std::string& addr=std::string{"TetMesh/"}){
  auto addr_str = std::string(AssetDirPath) + addr;
  std::ifstream in((addr_str + filename).c_str());
  if (in) {
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          const int el = 3; // x, y, z - Default
          int col = 0;
          std::array<PREC, 3> arr;
          while (getline(sep, field, ',')) {
              if (col >= el) break;
              arr[col] = stof(field) / l + offset[col];
              col++;
          }
          fields.push_back(arr);
      }
  }
  if (verbose) {
    for (auto row : fields) for (auto field : row) std::cout << field << ' ' << '\n';
  }
}

void make_box(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset,
                        PREC ppc) {
  // Make a rectangular prism of particles, write to fields
  // Span sets dimensions, offset is starting corner, ppc is particles-per-cell
  // Assumes span and offset are pre-adjusted to 1x1x1 box with 8*g_dx offset
  PREC ppl_dx = dx / cbrt(ppc); // Linear spacing of particles [1x1x1]
  int i_lim, j_lim, k_lim; // Number of par. per span direction
  i_lim = (int)((span[0]) / ppl_dx + 1.0); 
  j_lim = (int)((span[1]) / ppl_dx + 1.0); 
  k_lim = (int)((span[2]) / ppl_dx + 1.0); 

  for (int i = 0; i < i_lim ; i++){
    for (int j = 0; j < j_lim ; j++){
      for (int k = 0; k < k_lim ; k++){
          std::array<PREC, 3> arr;
          arr[0] = (i + 0.5) * ppl_dx + offset[0];
          arr[1] = (j + 0.5) * ppl_dx + offset[1];
          arr[2] = (k + 0.5) * ppl_dx + offset[2];
          if (arr[0] < (span[0] + offset[0]) && arr[1] < (span[1] + offset[1]) && arr[2] < (span[2] + offset[2])) {
            fields.push_back(arr);
        }
      }
    }
  } 
}

void load_FEM_Vertices(const std::string& filename, char sep, 
                       VerticeHolder& fields, 
                       mn::vec<PREC, 3> offset){
  auto addr_str = std::string(AssetDirPath) + "TetMesh/";
  std::ifstream in((addr_str + filename).c_str());
  if (in) {
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          const int el = 3; // x, y, z - Default
          std::array<PREC, 13> arr;
          int col = 0;
          while (getline(sep, field, ',')) {
              if (col >= el) break;
              arr[col] = stof(field) / l + offset[col];
              arr[col+el] = arr[col]; 
              col++;
          }
          arr[3] = (PREC)0.;
          arr[4] = (PREC)0.;
          arr[5] = (PREC)0.;
          arr[6] = (PREC)0.;
          arr[7] = (PREC)0.;
          arr[8] = (PREC)0.;
          arr[9] = (PREC)0.;
          arr[10] = (PREC)0.;
          arr[11] = (PREC)0.;
          arr[12] = (PREC)0.;
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
          const int el = 4; // 4-node Tetrahedron
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
    for (auto row : fields) for (auto field : row) std::cout << field << ' ' << '\n';
  }
}


void parse_scene(std::string fn,
                 std::unique_ptr<mn::mgsp_benchmark> &benchmark,
                 std::vector<std::array<PREC, 3>> models[mn::config::g_device_cnt]) {

  fs::path p{fn};
  if (p.empty())
    fmt::print("File does not exist: {}\n", fn);
  else {
    std::size_t size = fs::file_size(p);
    std::string configs;
    configs.resize(size);

    std::ifstream istrm(fn);
    if (!istrm.is_open())
      fmt::print("Cannot open file: {}\n", fn);
    else
      istrm.read(const_cast<char *>(configs.data()), configs.size());
    istrm.close();
    fmt::print("Load the scene file of size {}\n", size);

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

          l = sim["default_dx"].GetDouble() * mn::config::g_dx_inv_d; 
          //printf("length %f \n", length);
          //cudaMemcpyToSymbol("length", &l, sizeof(PREC), cudaMemcpyHostToDevice);

          fmt::print(
              fg(fmt::color::cyan),
              "Simulation: gpuid[{}], Domain Length [{}], default_dx[{}], default_dt[{}], fps[{}], frames[{}]\n",
              sim["gpu"].GetInt(), l, sim["default_dx"].GetFloat(), sim["default_dt"].GetFloat(),
              sim["fps"].GetInt(), sim["frames"].GetInt());
          benchmark = std::make_unique<mn::mgsp_benchmark>(
              l, sim["default_dt"].GetFloat(),
              sim["fps"].GetInt(), sim["frames"].GetInt(), sim["gravity"].GetFloat());
        }
      }
    } ///< end simulation parsing
    {
      auto it = doc.FindMember("meshes");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print("Scene file has [{}] Finite Element meshes.\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
            if (model["gpu"].GetInt() >= mn::config::g_device_cnt) {
              fmt::print(fg(fmt::color::green),
                       "ERROR! Mesh model GPU[{}] exceeds global device count (settings.h)! Skipping mesh.\n", 
                       model["gpu"].GetInt());
              break;
            }
            std::string constitutive{model["constitutive"].GetString()};
            fmt::print(fg(fmt::color::green),
                       "Mesh model using constitutive[{}], file_elements[{}], file_vertices[{}].\n", constitutive,
                       model["file_elements"].GetString(), model["file_vertices"].GetString());
            fs::path p{model["file"].GetString()};

            auto initModel = [&](auto &positions, auto &velocity) {
              benchmark->initModel<mn::material_e::Meshed>(model["gpu"].GetInt(), 
                                models[model["gpu"].GetInt()], 
                                velocity); //< Initalize particle model
              std::vector<std::string> output_attribs;
              for (int d = 0; d < 3; ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
              std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
              benchmark->updateMeshedParameters(
                model["gpu"].GetInt(),
                model["rho"].GetDouble(), model["ppc"].GetDouble(),
                model["youngs_modulus"].GetDouble(), model["poisson_ratio"].GetDouble(),
                model["alpha"].GetDouble(), model["beta_min"].GetDouble(), model["beta_max"].GetDouble(),
                model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                output_attribs); //< Update particle material with run-time inputs
              fmt::print("FEM mesh material set.\n");
            };
            
            ElementHolder h_FEM_Elements; //< Declare Host elements
            VerticeHolder h_FEM_Vertices; //< Declare Host vertices

            mn::vec<PREC, 3> offset, velocity;
            for (int d = 0; d < 3; ++d) {
              offset[d]   = model["offset"].GetArray()[d].GetDouble() / l + o;
              velocity[d] = model["velocity"].GetArray()[d].GetDouble() / l; 
            }
            std::cout << "Load FEM Element..." << '\n';
            load_FEM_Elements(model["file_elements"].GetString(), ',', h_FEM_Elements);
            std::cout << "Load FEM Vertices..." << '\n';
            load_FEM_Vertices(model["file_vertices"].GetString(), ',', h_FEM_Vertices,
                              offset);
            // Initialize FEM model on GPU arrays
            benchmark->initFEM(model["gpu"].GetInt(), h_FEM_Vertices, h_FEM_Elements);
            std::cout << "Initialized FEM model." << '\n';
            // Load particles into model using vertice positions
            load_csv_particles(model["file_vertices"].GetString(), ',', 
                                models[model["gpu"].GetInt()], 
                                offset);
            // Initialize particle model
            initModel(models[model["gpu"].GetInt()], velocity);
          }
        }
      }
    } ///< end mesh parsing
    {
      auto it = doc.FindMember("models");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print("Scene file has {} particle models. \n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
            if (model["gpu"].GetInt() >= mn::config::g_device_cnt) {
              fmt::print(fg(fmt::color::green),
                       "ERROR! Particle model GPU[{}] exceeds g_device_cnt (settings.h)! Skipping model.\n", 
                       model["gpu"].GetInt());
              break;
            }
            std::string constitutive{model["constitutive"].GetString()};
            fmt::print(fg(fmt::color::green),
                       "Model constitutive[{}], file[{}]\n", constitutive,
                       model["file"].GetString());
            fs::path p{model["file"].GetString()};

            auto initModel = [&](auto &positions, auto &velocity) {

              if (constitutive == "jfluid") {
                benchmark->initModel<mn::material_e::JFluid>(model["gpu"].GetInt(), positions, velocity);
                std::vector<std::string> output_attribs;
                for (int d = 0; d < 3; ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
                std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
                benchmark->updateJFluidParameters( model["gpu"].GetInt(),
                    model["rho"].GetDouble(), model["ppc"].GetDouble(),
                    model["bulk_modulus"].GetDouble(), model["gamma"].GetDouble(),
                    model["viscosity"].GetDouble(),
                    model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                    output_attribs);
              } else if (constitutive == "JFluid_ASFLIP") {
                benchmark->initModel<mn::material_e::JFluid_ASFLIP>(model["gpu"].GetInt(), positions, velocity);
                std::vector<std::string> output_attribs;
                for (int d = 0; d < 3; ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
                std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
                benchmark->updateJFluidASFLIPParameters( model["gpu"].GetInt(),
                    model["rho"].GetDouble(), model["ppc"].GetDouble(),
                    model["bulk_modulus"].GetDouble(), model["gamma"].GetDouble(),
                    model["viscosity"].GetDouble(), 
                    model["alpha"].GetDouble(), 
                    model["beta_min"].GetDouble(), model["beta_max"].GetDouble(),
                    model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                    output_attribs);
              } else if (constitutive == "JBarFluid") {
                benchmark->initModel<mn::material_e::JBarFluid>(model["gpu"].GetInt(), positions, velocity);
                std::vector<std::string> output_attribs;
                for (int d = 0; d < 3; ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
                std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
                benchmark->updateJBarFluidParameters( model["gpu"].GetInt(),
                    model["rho"].GetDouble(), model["ppc"].GetDouble(),
                    model["bulk_modulus"].GetDouble(), model["gamma"].GetDouble(),
                    model["viscosity"].GetDouble(), 
                    model["alpha"].GetDouble(), 
                    model["beta_min"].GetDouble(), model["beta_max"].GetDouble(),
                    model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                    output_attribs);
              } else if (constitutive == "fixed_corotated") {
                benchmark->initModel<mn::material_e::FixedCorotated>(model["gpu"].GetInt(), positions, velocity);
                std::vector<std::string> output_attribs;
                for (int d = 0; d < 3; ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
                std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
                benchmark->updateFRParameters( model["gpu"].GetInt(),
                    model["rho"].GetDouble(), model["ppc"].GetDouble(),
                    model["youngs_modulus"].GetDouble(),
                    model["poisson_ratio"].GetDouble(),
                    model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                    output_attribs);
              } else if (constitutive == "FixedCorotated_ASFLIP") {
                benchmark->initModel<mn::material_e::FixedCorotated_ASFLIP>(model["gpu"].GetInt(), positions, velocity);
                std::vector<std::string> output_attribs;
                for (int d = 0; d < 3; ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
                std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
                benchmark->updateFRASFLIPParameters( model["gpu"].GetInt(),
                    model["rho"].GetDouble(), model["ppc"].GetDouble(),
                    model["youngs_modulus"].GetDouble(),
                    model["poisson_ratio"].GetDouble(), 
                    model["alpha"].GetDouble(), 
                    model["beta_min"].GetDouble(), model["beta_max"].GetDouble(),
                    model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                    output_attribs);
              } else if (constitutive == "nacc") {
                benchmark->initModel<mn::material_e::NACC>(model["gpu"].GetInt(), positions, velocity);
                std::vector<std::string> output_attribs;
                for (int d = 0; d < 3; ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
                std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
                benchmark->updateNACCParameters( model["gpu"].GetInt(),
                    model["rho"].GetDouble(), model["ppc"].GetDouble(),
                    model["youngs_modulus"].GetDouble(),
                    model["poisson_ratio"].GetDouble(), model["beta"].GetDouble(),
                    model["xi"].GetDouble(),
                    model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                    output_attribs);
              } else if (constitutive == "sand") { 
                benchmark->initModel<mn::material_e::Sand>(model["gpu"].GetInt(), positions, velocity); 
                std::vector<std::string> output_attribs;
                for (int d = 0; d < 3; ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
                std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
                benchmark->updateSandParameters( model["gpu"].GetInt(),
                    model["rho"].GetDouble(), model["ppc"].GetDouble(),
                    model["youngs_modulus"].GetDouble(),
                    model["poisson_ratio"].GetDouble(),
                    model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                    output_attribs);              
              }
              
              //benchmark->initModel<mn::material_e::FixedCorotated>(model["gpu"].GetInt(), positions, velocity);
              fmt::print("Particle material model updated.\n");

            };
            mn::vec<PREC, 3> offset, span, velocity;
            mn::vec<PREC, 3> point_a, point_b, inter_a, inter_b;
            for (int d = 0; d < 3; ++d) {
              offset[d]   = model["offset"].GetArray()[d].GetDouble() / l + o;
              span[d]     = model["span"].GetArray()[d].GetDouble() / l;
              velocity[d] = model["velocity"].GetArray()[d].GetDouble() / l;
              point_a[d]  = model["point_a"].GetArray()[d].GetDouble() / l;
              point_b[d]  = model["point_b"].GetArray()[d].GetDouble() / l;
              inter_a[d]  = model["inter_a"].GetArray()[d].GetDouble() / l;
              inter_b[d]  = model["inter_b"].GetArray()[d].GetDouble() / l;
            }
            // Signed-Distance-Field MPM particle input (make with SDFGen or SideFX Houdini)
            if (p.extension() == ".sdf") {
              if (model["partition"].GetInt()){
                auto positions = mn::read_sdf(
                    model["file"].GetString(), model["ppc"].GetFloat(),
                    mn::config::g_dx, mn::config::g_domain_size, offset, span,
                    point_a, point_b, inter_a, inter_b);
                mn::IO::insert_job([&]() {
                  mn::write_partio<PREC, 3>(std::string{p.stem()} + ".bgeo",
                                            positions);
                });              
                mn::IO::flush();
                initModel(positions, velocity);
              } else {
                auto positions = mn::read_sdf(
                    model["file"].GetString(), model["ppc"].GetFloat(),
                    mn::config::g_dx, mn::config::g_domain_size, offset, span);
                mn::IO::insert_job([&]() {
                  mn::write_partio<PREC, 3>(std::string{p.stem()} + ".bgeo",
                                            positions);
                });              
                mn::IO::flush();
                initModel(positions, velocity);
              }
            }
            // CSV MPM particle input (Make with Excel, Notepad, etc.)
            if (p.extension() == ".csv") {
              load_csv_particles(model["file"].GetString(), ',', 
                                  models[model["gpu"].GetInt()], 
                                  offset);
              auto positions = models[model["gpu"].GetInt()];
              mn::IO::insert_job([&]() {
                mn::write_partio<PREC, 3>(std::string{p.stem()} + ".bgeo",
                                          positions);
              });              
              mn::IO::flush();
              initModel(positions, velocity);
            }
            // Auto-generate MPM particles in specified box dimensions. Must use *.box suffix in "file"
            if (p.extension() == ".box") {
              make_box(models[model["gpu"].GetInt()], 
                        span, offset, model["ppc"].GetFloat());
              auto positions = models[model["gpu"].GetInt()];
              mn::IO::insert_job([&]() {
                mn::write_partio<PREC, 3>(std::string{p.stem()} + ".bgeo",
                                          positions);
              });              
              mn::IO::flush();
              initModel(positions, velocity);
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
          
            std::vector<std::array<float, mn::config::g_target_attribs>> h_gridTarget(mn::config::g_target_cells, 
                                                            std::array<float, mn::config::g_target_attribs>{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f});
            // Load and scale target domain
            mn::vec<float, 3> h_point_a, h_point_b;
            for (int d = 0; d < 3; ++d) {
              h_point_a[d] = model["point_a"].GetArray()[d].GetFloat() / l + o;
              h_point_b[d] = model["point_b"].GetArray()[d].GetFloat() / l + o;
            }

            // Check for thin target domain, grow 1 grid-cell if so
            for (int d=0; d < 3; ++d){
              if (h_point_a[d] == h_point_b[d]) h_point_b[d] = h_point_b[d] + dx;         
            }

            // ----------------
            /// Loop through GPU devices
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
      auto it = doc.FindMember("grid-boundaries");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print("has {} grid-boundaries\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {

            mn::vec<float, 7> h_walls, h_boxes;
            for (int d = 0; d < 3; ++d) {
              h_walls[d] = model["wall_start"].GetArray()[d].GetFloat() / l + o;
              h_boxes[d] = model["box_start"].GetArray()[d].GetFloat() / l + o;
            }
            for (int d = 0; d < 3; ++d) {
              h_walls[d+3] = model["wall_end"].GetArray()[d].GetFloat() / l + o;
              h_boxes[d+3] = model["box_end"].GetArray()[d].GetFloat() / l + o;
            }
            h_walls[6] = model["wall_type"].GetFloat(); // Contact
            h_boxes[6] = model["box_type"].GetFloat(); // Contact

            // ----------------
            /// Loop through GPU devices
            //for (int did = 0; did < mn::config::g_device_cnt; ++did) {
            benchmark->initGridBoundaries(0, h_walls, h_boxes);
            //}
          }
        }
      }
    } ///< end wave-gauge parsing
    {
      auto it = doc.FindMember("wave-gauges");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print("has {} wave-gauges\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {

            mn::vec<float, 3> h_point_a, h_point_b;
            for (int d = 0; d < 3; ++d) {
              h_point_a[d] = model["point_a"].GetArray()[d].GetFloat() / l + o;
              h_point_b[d] = model["point_b"].GetArray()[d].GetFloat() / l + o;
            }
            if (h_point_a[0] == h_point_b[0]) h_point_b[0] = h_point_b[0] + dx;              

            // ----------------
            /// Loop through GPU devices
            //for (int did = 0; did < mn::config::g_device_cnt; ++did) {
            benchmark->initWaveGauge(0, h_point_a, h_point_b, 
              model["output_frequency"].GetFloat());
            //}
          }
        }
      }
    } ///< End wave-gauge parsing
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
    } ///< End wave parsing
  }
} ///< End scene file parsing



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
  std::vector<std::array<PREC, 3>> models[g_device_cnt];
  std::vector<mn::vec<PREC, 3>> v0(g_device_cnt, mn::vec<PREC, 3>{0.,0.,0.});
  // ----------------
  /// Initialize the scene
  std::unique_ptr<mn::mgsp_benchmark> benchmark;
  parse_scene(fn, benchmark, models); //< Load from input file  
  if (false) {
    benchmark = std::make_unique<mgsp_benchmark>(9e-6, 10);
  }
  std::cout << "Finished scene initialization." << '\n';
  // ----------------
  /// Grid-Target
  std::cout << "Set grid-target..." << '\n';
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