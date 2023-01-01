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
typedef std::vector<std::array<PREC, 6>> ElementAttribsHolder;

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
          PREC x, y, z;
          x = ((arr[0] - offset[0]) * l);
          y = ((arr[1] - offset[1]) * l);
          z = ((arr[2] - offset[2]) * l);
          if (arr[0] < (span[0] + offset[0]) && arr[1] < (span[1] + offset[1]) && arr[2] < (span[2] + offset[2])) {
            if (0)
            {
              PREC m, b, surf;
              
              m = -0.2/3.2;
              b = (3.2 + 0.2/2.0);
              surf =  m * x + b;
              
              if (y <= surf)
              fields.push_back(arr);
            }  
            else if (0)
            {
              PREC m, b, surf_one, surf_two;
              
              m = -(0.1-0.00001)/0.5;
              b = 0.1;
              surf_one =  m * (x) + b;
              m = (0.1-0.00001)/0.5;
              b = 0.1;
              surf_two =  m * (x) + b;

              if (z >= surf_one && z <= surf_two)
                fields.push_back(arr);
            }
            else if (0)
            {
              PREC R = 0.126156626101; // Volume = pi * R^2 * 0.05 
              PREC R_out = 0.13570370042; // Volume = pi * R^2 * 0.05 
              PREC R_in = 0.05; // Volume = pi * R^2 * 0.05 
			//"offset": [0.728592599149, 0.1, 0.114296299574],
			//"span": [0.5, 0.05, 0.5],

              PREC xo = R_out ; 
              PREC zo = R_out ;
              PREC r = std::sqrt((x-xo)*(x-xo) + (z-zo)*(z-zo));
              // "offset": [0.747686747798, 0.1, 0.123843373899],
              //			"offset": [0.728592599149, 0.1, 0.114296299574],
			        // "span": [0.5, 0.05, 0.5],
              //fmt::print("r: {}", r);
              if (r <= R_out && r >= R_in)
                fields.push_back(arr);
            }
            else 
            {
              fields.push_back(arr);
            }
        }
      }
    }
  } 
}


void make_cylinder(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset,
                        PREC ppc, PREC radius, std::string axis) {
  // Make a cylinder of particles, write to fields
  // Span sets dimensions, offset is starting corner, ppc is particles-per-cell
  // Assumes span and offset are pre-adjusted to 1x1x1 box with 8*g_dx offset
  PREC ppl_dx = dx / cbrt(ppc); // Linear spacing of particles [1x1x1]
  int i_lim, j_lim, k_lim; // Number of par. per span direction
  i_lim = (int)((span[0]) / ppl_dx + 1.0); 
  j_lim = (int)((span[1]) / ppl_dx + 1.0); 
  k_lim = (int)((span[2]) / ppl_dx + 1.0); 

  for (int i = 0; i < i_lim ; i++) {
    for (int j = 0; j < j_lim ; j++) {
      for (int k = 0; k < k_lim ; k++) {
          std::array<PREC, 3> arr;
          arr[0] = (i + 0.5) * ppl_dx + offset[0];
          arr[1] = (j + 0.5) * ppl_dx + offset[1];
          arr[2] = (k + 0.5) * ppl_dx + offset[2];
          PREC x, y, z;
          x = ((arr[0] - offset[0]) * l);
          y = ((arr[1] - offset[1]) * l);
          z = ((arr[2] - offset[2]) * l);
          if (arr[0] < (span[0] + offset[0]) && arr[1] < (span[1] + offset[1]) && arr[2] < (span[2] + offset[2])) {
            PREC xo = radius; 
            PREC yo = radius;
            PREC zo = radius;
            PREC r;
            if (axis == "x" || axis == "X") 
              r = std::sqrt((y-yo)*(y-yo) + (z-zo)*(z-zo));
            else if (axis == "y" || axis == "Y") 
              r = std::sqrt((x-xo)*(x-xo) + (z-zo)*(z-zo));
            else if (axis == "z" || axis == "Z") 
              r = std::sqrt((x-xo)*(x-xo) + (y-yo)*(y-yo));
            else 
            {
              r = 0;
              fmt::print("ERRROR: Value of axis[{}] is not applicable for a Cylinder. Use X, Y, or Z.", axis);
            }

            if (r <= radius) fields.push_back(arr);
        }
      }
    }
  } 
}



void make_sphere(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset,
                        PREC ppc, PREC radius) {
  // Make a sphere of particles, write to fields
  // Span sets dimensions, offset is starting corner, ppc is particles-per-cell
  // Assumes span and offset are pre-adjusted to 1x1x1 box with 8*g_dx offset
  PREC ppl_dx = dx / cbrt(ppc); // Linear spacing of particles [1x1x1]
  int i_lim, j_lim, k_lim; // Number of par. per span direction
  i_lim = (int)((span[0]) / ppl_dx + 1.0); 
  j_lim = (int)((span[1]) / ppl_dx + 1.0); 
  k_lim = (int)((span[2]) / ppl_dx + 1.0); 

  for (int i = 0; i < i_lim ; i++) {
    for (int j = 0; j < j_lim ; j++) {
      for (int k = 0; k < k_lim ; k++) {
          std::array<PREC, 3> arr;
          arr[0] = (i + 0.5) * ppl_dx + offset[0];
          arr[1] = (j + 0.5) * ppl_dx + offset[1];
          arr[2] = (k + 0.5) * ppl_dx + offset[2];
          PREC x, y, z;
          x = ((arr[0] - offset[0]) * l);
          y = ((arr[1] - offset[1]) * l);
          z = ((arr[2] - offset[2]) * l);
          if (arr[0] < (span[0] + offset[0]) && arr[1] < (span[1] + offset[1]) && arr[2] < (span[2] + offset[2])) {
            PREC xo = radius; 
            PREC yo = radius;
            PREC zo = radius;
            PREC r;
            r = std::sqrt((x-xo)*(x-xo) + (y-yo)*(y-yo) + (z-zo)*(z-zo));
            if (r <= radius) fields.push_back(arr);
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

          //dx = sim["default_dx"].GetDouble();
          l = sim["default_dx"].GetDouble() * mn::config::g_dx_inv_d; 
          //o =  mn::config::g_offset /  mn::config::g_dx * dx;
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
              continue;
            }
            std::string constitutive{model["constitutive"].GetString()};
            fmt::print(fg(fmt::color::green),
                       "Mesh model using constitutive[{}], file_elements[{}], file_vertices[{}].\n", constitutive,
                       model["file_elements"].GetString(), model["file_vertices"].GetString());
            fs::path p{model["file"].GetString()};
            
            std::vector<std::string> output_attribs;
            for (int d = 0; d < 3; ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
            std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
                      
            auto initModel = [&](auto &positions, auto &velocity, auto &vertices, auto &elements, auto &attribs)
            {
              if (constitutive == "Meshed") 
              {
                // Initialize FEM model on GPU arrays
                if (model["use_FBAR"].GetBool() == true)
                {
                  std::cout << "Initialize FEM FBAR Model." << '\n';
                  benchmark->initFEM<mn::fem_e::Tetrahedron_FBar>(model["gpu"].GetInt(), vertices, elements, attribs);
                  benchmark->initModel<mn::material_e::Meshed>(model["gpu"].GetInt(), positions, velocity); //< Initalize particle model

                  std::cout << "Initialize Mesh Parameters." << '\n';
                  benchmark->updateMeshedFBARParameters(
                    model["gpu"].GetInt(),
                    model["rho"].GetDouble(), model["ppc"].GetDouble(),
                    model["youngs_modulus"].GetDouble(), model["poisson_ratio"].GetDouble(),
                    model["alpha"].GetDouble(), model["beta_min"].GetDouble(), model["beta_max"].GetDouble(),
                    model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                    output_attribs); //< Update particle material with run-time inputs
                }
                else if (model["use_FBAR"].GetBool() == false)
                {
                  std::cout << "Initialize FEM Model." << '\n';
                  benchmark->initFEM<mn::fem_e::Tetrahedron>(model["gpu"].GetInt(), vertices, elements, attribs);
                  benchmark->initModel<mn::material_e::Meshed>(model["gpu"].GetInt(), positions, velocity); //< Initalize particle model

                  std::cout << "Initialize Mesh Parameters." << '\n';
                  benchmark->updateMeshedParameters(
                    model["gpu"].GetInt(),
                    model["rho"].GetDouble(), model["ppc"].GetDouble(),
                    model["youngs_modulus"].GetDouble(), model["poisson_ratio"].GetDouble(),
                    model["alpha"].GetDouble(), model["beta_min"].GetDouble(), model["beta_max"].GetDouble(),
                    model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                    output_attribs); //< Update particle material with run-time inputs
                }
                std::cout << "Initialized FEM model." << '\n';
              }
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
            
            std::vector<std::array<PREC, 6>> h_FEM_Element_Attribs(mn::config::g_max_fem_element_num, 
                                                            std::array<PREC, 6>{0., 0., 0., 0., 0., 0.});
            
            std::cout << "Load FEM Vertices..." << '\n';
            load_FEM_Vertices(model["file_vertices"].GetString(), ',', h_FEM_Vertices,
                              offset);
            // Load particles into model using vertice positions
            load_csv_particles(model["file_vertices"].GetString(), ',', 
                                models[model["gpu"].GetInt()], 
                                offset);

            // Initialize particle and finite element model
            initModel(models[model["gpu"].GetInt()], velocity, h_FEM_Vertices, h_FEM_Elements, h_FEM_Element_Attribs);
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
            std::vector<std::string> output_attribs;
            int track_particle_id;
            std::vector<std::string> track_attribs;
            std::vector<std::string> target_attribs;
            auto initModel = [&](auto &positions, auto &velocity) {

              if (constitutive == "JFluid" || constitutive == "J-Fluid" || constitutive == "J_Fluid" || constitutive == "J Fluid" ||  constitutive == "jfluid" || constitutive == "j-fluid" || constitutive == "j_fluid" || constitutive == "j fluid" || constitutive == "Fluid" || constitutive == "fluid" || constitutive == "Water" || constitutive == "Liquid") {
                if(!model["use_ASFLIP"].GetBool() && !model["use_FBAR"].GetBool() && !model["use_FEM"].GetBool())
                {
                  benchmark->initModel<mn::material_e::JFluid>(model["gpu"].GetInt(), positions, velocity);
                  benchmark->updateJFluidParameters( model["gpu"].GetInt(),
                      model["rho"].GetDouble(), model["ppc"].GetDouble(),
                      model["bulk_modulus"].GetDouble(), model["gamma"].GetDouble(),
                      model["viscosity"].GetDouble(),
                      model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                      output_attribs);
                  fmt::print("GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else if (model["use_ASFLIP"].GetBool() && !model["use_FBAR"].GetBool() && !model["use_FEM"].GetBool())
                {
                  benchmark->initModel<mn::material_e::JFluid_ASFLIP>(model["gpu"].GetInt(), positions, velocity);
                  benchmark->updateJFluidASFLIPParameters( model["gpu"].GetInt(),
                      model["rho"].GetDouble(), model["ppc"].GetDouble(),
                      model["bulk_modulus"].GetDouble(), model["gamma"].GetDouble(), model["viscosity"].GetDouble(), 
                      model["alpha"].GetDouble(), model["beta_min"].GetDouble(), model["beta_max"].GetDouble(),
                      model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                      output_attribs);
                  fmt::print("GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else if (model["use_ASFLIP"].GetBool() && model["use_FBAR"].GetBool() && !model["use_FEM"].GetBool())
                {
                  benchmark->initModel<mn::material_e::JBarFluid>(model["gpu"].GetInt(), positions, velocity);
                  benchmark->updateJBarFluidParameters( model["gpu"].GetInt(),
                      model["rho"].GetDouble(), model["ppc"].GetDouble(),
                      model["bulk_modulus"].GetDouble(), model["gamma"].GetDouble(), model["viscosity"].GetDouble(), 
                      model["alpha"].GetDouble(), model["beta_min"].GetDouble(), model["beta_max"].GetDouble(),
                      model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                      output_attribs,
                      track_particle_id, track_attribs, 
                      target_attribs);
                  fmt::print("GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else 
                {
                  fmt::print(fg(fmt::color::red),
                       "ERROR: GPU[{}] Improper/undefined settings for material [{}] with: use_ASFLIP[{}], use_FEM[{}], and use_FBAR[{}]! \n", 
                       model["gpu"].GetInt(), constitutive,
                       model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool());
                  getchar();
                }
              } else if (constitutive == "FixedCorotated" || constitutive == "Fixed_Corotated" || constitutive == "Fixed-Corotated" || constitutive == "Fixed Corotated" || constitutive == "fixedcorotated" || constitutive == "fixed_corotated" || constitutive == "fixed-corotated"|| constitutive == "fixed corotated") {

                if(!model["use_ASFLIP"].GetBool() && !model["use_FBAR"].GetBool() && !model["use_FEM"].GetBool())
                {
                  benchmark->initModel<mn::material_e::FixedCorotated>(model["gpu"].GetInt(), positions, velocity);
                  benchmark->updateFRParameters( model["gpu"].GetInt(),
                      model["rho"].GetDouble(), model["ppc"].GetDouble(),
                      model["youngs_modulus"].GetDouble(), model["poisson_ratio"].GetDouble(),
                      model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                      output_attribs);
                  fmt::print("GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else if (model["use_ASFLIP"].GetBool() && !model["use_FBAR"].GetBool() && !model["use_FEM"].GetBool())
                {
                  benchmark->initModel<mn::material_e::FixedCorotated_ASFLIP>(model["gpu"].GetInt(), positions, velocity);
                  benchmark->update_FR_ASFLIP_Parameters( model["gpu"].GetInt(),
                      model["rho"].GetDouble(), model["ppc"].GetDouble(),
                      model["youngs_modulus"].GetDouble(), model["poisson_ratio"].GetDouble(), 
                      model["alpha"].GetDouble(), model["beta_min"].GetDouble(), model["beta_max"].GetDouble(),
                      model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                      output_attribs);
                  fmt::print("GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else if (model["use_ASFLIP"].GetBool() && model["use_FBAR"].GetBool() && !model["use_FEM"].GetBool())
                {
                  benchmark->initModel<mn::material_e::FixedCorotated_ASFLIP_FBAR>(model["gpu"].GetInt(), positions, velocity);
                  benchmark->update_FR_ASFLIP_FBAR_Parameters( model["gpu"].GetInt(),
                      model["rho"].GetDouble(), model["ppc"].GetDouble(),
                      model["youngs_modulus"].GetDouble(), model["poisson_ratio"].GetDouble(), 
                      model["alpha"].GetDouble(), model["beta_min"].GetDouble(), model["beta_max"].GetDouble(),
                      model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                      output_attribs, 
                      track_particle_id, track_attribs, 
                      target_attribs);
                  fmt::print("GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else 
                {
                  fmt::print(fg(fmt::color::red),
                       "ERROR: GPU[{}] Improper/undefined settings for material [{}] with: use_ASFLIP[{}], use_FEM[{}], and use_FBAR[{}]! \n", 
                       model["gpu"].GetInt(), constitutive,
                       model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool());
                  getchar();
                }
              } else if (constitutive == "Sand" || constitutive == "sand" || constitutive == "DruckerPrager" || constitutive == "Drucker_Prager" || constitutive == "Drucker-Prager" || constitutive == "Drucker Prager") { 
                benchmark->initModel<mn::material_e::Sand>(model["gpu"].GetInt(), positions, velocity); 
                benchmark->updateSandParameters( model["gpu"].GetInt(),
                    model["rho"].GetDouble(), model["ppc"].GetDouble(),
                    model["youngs_modulus"].GetDouble(), model["poisson_ratio"].GetDouble(),
                    model["logJp0"].GetDouble(), model["friction_angle"].GetDouble(),model["cohesion"].GetDouble(),model["beta"].GetDouble(),model["Sand_volCorrection"].GetBool(),
                    model["alpha"].GetDouble(), model["beta_min"].GetDouble(), model["beta_max"].GetDouble(),
                    model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                    output_attribs);        
                fmt::print("GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
              } else if (constitutive == "NACC" || constitutive == "nacc" || constitutive == "CamClay" || constitutive == "Cam_Clay" || constitutive == "Cam-Clay" || constitutive == "Cam Clay") {
                benchmark->initModel<mn::material_e::NACC>(model["gpu"].GetInt(), positions, velocity);
                benchmark->updateNACCParameters( model["gpu"].GetInt(),
                    model["rho"].GetDouble(), model["ppc"].GetDouble(),
                    model["youngs_modulus"].GetDouble(), model["poisson_ratio"].GetDouble(), 
                    model["beta"].GetDouble(), model["xi"].GetDouble(),
                    model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                    output_attribs);
                fmt::print("GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
              } 

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
            for (int d = 0; d < 3; ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
            std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
            track_particle_id = model["track_particle_id"].GetInt();
            for (int d = 0; d < 1; ++d) track_attribs.emplace_back(model["track_attribs"].GetArray()[d].GetString());
            std::cout <<"Track particle ID: " << track_particle_id << " for Attributes: [ " << track_attribs[0] <<" ]"<<'\n';
            for (int d = 0; d < 1; ++d) target_attribs.emplace_back(model["target_attribs"].GetArray()[d].GetString());
            std::cout <<"Target Attributes: [ " << target_attribs[0] <<" ]"<<'\n';
            

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
            if (p.extension() == ".cylinder") {
              make_cylinder(models[model["gpu"].GetInt()], 
                        span, offset, model["ppc"].GetFloat(), model["radius"].GetDouble(), model["axis"].GetString());
              auto positions = models[model["gpu"].GetInt()];
              mn::IO::insert_job([&]() {
                mn::write_partio<PREC, 3>(std::string{p.stem()} + ".bgeo",
                                          positions);
              });              
              mn::IO::flush();
              initModel(positions, velocity);
            }
            if (p.extension() == ".sphere") {
              make_sphere(models[model["gpu"].GetInt()], 
                        span, offset, model["ppc"].GetFloat(), model["radius"].GetDouble());
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
      auto it = doc.FindMember("grid-targets");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print("has {} grid-targets\n", it->value.Size());
          int target_ID = 0;
          for (auto &model : it->value.GetArray()) {
          
            std::vector<std::array<PREC_G, mn::config::g_target_attribs>> h_gridTarget(mn::config::g_target_cells, 
                                                            std::array<PREC_G, mn::config::g_target_attribs>{0.f,0.f,0.f,
                                                            0.f,0.f,0.f,
                                                            0.f,0.f,0.f,0.f});
            // Load and scale target domain
            mn::vec<float, 7> target;
            target[0] = model["type"].GetFloat();
            for (int d = 0; d < 3; ++d) 
            {
              target[d+1] = model["point_a"].GetArray()[d].GetFloat() / l + o;
              target[d+4] = model["point_b"].GetArray()[d].GetFloat() / l + o;
            }

            // Check for thin target domain, grow 1 grid-cell if so
            for (int d=0; d < 3; ++d)
              if (target[d+1] == target[d+4]) target[d+4] = target[d+4] + dx;         

            // ----------------
            /// Loop through GPU devices
            for (int did = 0; did < mn::config::g_device_cnt; ++did) {
              benchmark->initGridTarget(did, h_gridTarget, target, 
                model["output_frequency"].GetFloat());
            fmt::print("GPU[{}] gridTarget[{}] Initialized.\n", did, target_ID);
            }
            target_ID += 1;
          }
        }
      }
    } ///< end grid-target parsing
    {
      auto it = doc.FindMember("particle-targets");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print("has {} particle-targets\n", it->value.Size());
          int target_ID = 0;
          for (auto &model : it->value.GetArray()) {
          
            std::vector<std::array<PREC, mn::config::g_particle_target_attribs>> h_particleTarget(mn::config::g_particle_target_cells, 
                                                            std::array<PREC, mn::config::g_particle_target_attribs>{0.f,0.f,0.f,
                                                            0.f,0.f,0.f,
                                                            0.f,0.f,0.f,0.f});
            // Load and scale target domain
            mn::vec<PREC, 7> target;
            target[0] = model["type"].GetFloat();
            for (int d = 0; d < 3; ++d) 
            {
              target[d+1] = model["point_a"].GetArray()[d].GetFloat() / l + o;
              target[d+4] = model["point_b"].GetArray()[d].GetFloat() / l + o;
            }

            // ----------------
            /// Loop through GPU devices
            for (int did = 0; did < mn::config::g_device_cnt; ++did) {
              benchmark->initParticleTarget(did, h_particleTarget, target, 
                model["output_frequency"].GetFloat());
            fmt::print("GPU[{}] particleTarget[{}] Initialized.\n", did, target_ID);
            }
            target_ID += 1;
          }
        }
      }
    } ///< end particle-target parsing
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
            std::string wall_type{model["wall_type"].GetString()};
            std::string box_type{model["box_type"].GetString()};


            if (wall_type == "Rigid" || wall_type == "Sticky" || wall_type == "Stick") h_walls[6] = 0;
            else if (wall_type == "Slip") h_walls[6] = 1;
            else if (wall_type == "Separable") h_walls[6] = 2;
            else h_walls[6] = -1;

            if (box_type == "Rigid" || box_type == "Sticky" || box_type == "Stick") h_boxes[6] = 0;
            else if (box_type == "Slip") h_boxes[6] = 1;
            else if (box_type == "Separable") h_boxes[6] = 2;
            else h_boxes[6] = -1;

            // h_walls[6] = model["wall_type"].GetFloat(); // Contact
            // h_boxes[6] = model["box_type"].GetFloat(); // Contact
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
  if (g_log_level > 1){
    std::cout << "Press ENTER to start simulation. Disable with g_log_level 0 or 1 in 'settings.h'." << '\n';
    getchar();
  }
  benchmark->main_loop();
  std::cout << "Finished main loop of simulation." << '\n';
  // ----------------
  /// Clear
  IO::flush();
  std::cout << "Cleared I/O." << '\n';
  benchmark.reset();
  std::cout << "Reset simulation structure." << '\n';
  // ----------------
  /// Shutdown GPU
  Cuda::shutdown();
  std::cout << "Shut-down CUDA GPUs." << '\n';
  // ----------------
  /// End program
  std::cout << "Simulation finished." << '\n';
  return 0;
}