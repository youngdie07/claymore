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

typedef std::vector<std::array<PREC, 3>> PositionHolder;
typedef std::vector<std::array<PREC, 13>> VerticeHolder;
typedef std::vector<std::array<int, 4>> ElementHolder;
typedef std::vector<std::array<PREC, 6>> ElementAttribsHolder;
typedef std::vector<std::array<PREC_G, 3>> MotionHolder;

float verbose = 0;
PREC o = mn::config::g_offset; //< Grid-cell buffer size (see off-by-2, Xinlei Wang)
PREC l = mn::config::g_length; //< Grid-length
PREC dx = mn::config::g_dx; //< Grid-cell length [1x1x1]
std::string save_suffix; //< File-format to save particles with
//__device__ __constant__ PREC length;



decltype(auto) load_model(std::size_t pcnt, std::string filename) {
  std::vector<std::array<float, 3>> rawpos(pcnt);
  auto f = fopen(filename.c_str(), "rb");
  auto res = std::fread((float *)rawpos.data(), sizeof(float), rawpos.size() * 3, f);
  std::fclose(f);
  return rawpos;
}

void load_csv_particles(const std::string& filename, char sep, 
                        std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> offset, const std::string& addr=std::string{"TetMesh/"}){
  std::ifstream in(filename.c_str());
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
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset, PREC ppc) {
  // Make a rectangular prism of particles, write to fields
  // Span sets dimensions, offset is starting corner, ppc is particles-per-cell
  // Assumes span and offset are pre-adjusted to 1x1x1 domain with 8*g_dx offset
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
  // Assumes span and offset are pre-adjusted to 1x1x1 domain with 8*g_dx offset
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
              fmt::print(fg(fmt::color::red), "ERROR: Value of axis[{}] is not applicable for a Cylinder. Use X, Y, or Z.", axis);
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

void make_OSU_LWF(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset,
                        PREC ppc) {
  // Make OSU LWF flume fluid as particles, write to fields
  // Span sets dimensions, offset is starting corner, ppc is particles-per-cell
  // Assumes span and offset are pre-adjusted to 1x1x1 box with 8*g_dx offset
  PREC ppl_dx = dx / cbrt(ppc); // Linear spacing of particles [1x1x1]
  int i_lim, j_lim, k_lim; // Number of par. per span direction
  i_lim = (int)((span[0]) / ppl_dx + 1.0); 
  j_lim = (int)((span[1]) / ppl_dx + 1.0); 
  k_lim = (int)((span[2]) / ppl_dx + 1.0); 

  //PREC wave_maker_neutral = -2.0; 
  // Assume JSON input offsets model 2 meters forward in X
  PREC bathx[7];
  PREC bathy[7];
  PREC bath_slope[7];

  bathx[0] = 0.0; //- wave_maker_neutral; // Start of bathymetry X direction
  bathx[1] = 14.275 + bathx[0];
  bathx[2] = 3.658 + bathx[1];
  bathx[3] = 10.973 + bathx[2];
  bathx[4] = 14.63 + bathx[3];
  bathx[5] = 36.57 + bathx[4];
  bathx[6] = 7.354 + bathx[5];

  bathy[0] = 0.0;
  bathy[1] = (0.15 + 0.076) + bathy[0]; // Bathymetry slap raised ~0.15m, 0.076m thick
  bathy[2] = 0.0 + bathy[1];
  bathy[3] = (10.973 / 12.0) + bathy[2];
  bathy[4] = 1.75; //(14.63f / 24.f) + bathy[3];
  bathy[5] = 0.0 + bathy[4];
  bathy[6] = (7.354 / 12.0) + bathy[5]; 

  bath_slope[0] = 0;
  bath_slope[1] = 0;
  bath_slope[2] = 0;
  bath_slope[3] = 1.0 / 12.0;
  bath_slope[4] = 1.0 / 24.0;
  bath_slope[5] = 0;
  bath_slope[6] = 1.0 / 12.0;

  for (int i = 0; i < i_lim ; i++) {
    for (int j = 0; j < j_lim ; j++) {
      for (int k = 0; k < k_lim ; k++) {
        std::array<PREC, 3> arr;
        arr[0] = (i + 0.5) * ppl_dx + offset[0];
        arr[1] = (j + 0.5) * ppl_dx + offset[1];
        arr[2] = (k + 0.5) * ppl_dx + offset[2];
        PREC x, y;
        x = ((arr[0] - offset[0]) * l);
        y = ((arr[1] - offset[1]) * l);
        if (arr[0] < (span[0] + offset[0]) && arr[1] < (span[1] + offset[1]) && arr[2] < (span[2] + offset[2])) {
          // Start ramp segment definition for OSU flume
          // Based on bathymetry diagram, February
          for (int d = 1; d < 7; d++)
          {
            if (x < bathx[d])
            {
              if (y >= ( bath_slope[d] * (x - bathx[d-1]) + bathy[d-1]) )
              {
                fields.push_back(arr);
                break;
              }
              else break;
            }
          }
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


void load_motionPath(const std::string& filename, char sep, MotionHolder& fields, int rate=1){
  std::ifstream in((filename).c_str());
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
    for (auto row : fields) {
      for (auto field : row) std::cout << field << ' '; 
      std::cout << '\n';
    }
    std::cout << '\n';
  }
}

/// @brief Parses an input JSON script to set-up a Multi-GPU simulation.
/// @param fn Filename of the simulation input JSON script. Default: ./scene.json 
/// @param benchmark Simulation object to initalize.
/// @param models Contains initial particle positions for simulation.
void parse_scene(std::string fn,
                 std::unique_ptr<mn::mgsp_benchmark> &benchmark,
                 std::vector<std::array<PREC, 3>> models[mn::config::g_device_cnt]) {
  fs::path p{fn};
  if (p.empty()) fmt::print(fg(fmt::color::red), "ERROR: Input file[{}] does not exist.\n", fn);
  else {
    std::size_t size = fs::file_size(p);
    std::string configs;
    configs.resize(size);

    std::ifstream istrm(fn);
    if (!istrm.is_open())  fmt::print(fg(fmt::color::red), "ERROR: Cannot open file[{}]\n", fn);
    else istrm.read(const_cast<char *>(configs.data()), configs.size());
    istrm.close();
    fmt::print(fg(fmt::color::green), "Opened scene file[{}] of size[{}] bytes.\n", fn, size);

    rj::Document doc;
    doc.Parse(configs.data());
    for (rj::Value::ConstMemberIterator itr = doc.MemberBegin();
         itr != doc.MemberEnd(); ++itr) {
      fmt::print("Scene member {} is type {}. \n", itr->name.GetString(),
                 kTypeNames[itr->value.GetType()]);
    }
    {
      auto it = doc.FindMember("simulation");
      if (it != doc.MemberEnd()) {
        auto &sim = it->value;
        if (sim.IsObject()) {


          l = sim["default_dx"].GetDouble() * mn::config::g_dx_inv_d; 
          mn::vec<PREC, 3> domain;
          for (int d = 0; d < 3; ++d)
            domain[d] = sim["domain"].GetArray()[d].GetDouble();
            
          if (domain[0] > l || domain[1] > l || domain[2] > l) {
            fmt::print(fg(fmt::color::red), "ERROR: Simulation domain[{},{},{}] exceeds max domain length[{}]\n", domain[0], domain[1], domain[2], l);
            fmt::print(fg(fmt::color::red), "TIP: Shrink domain, grow default_dx, or increase DOMAIN_BITS (settings.h) and recompile. \n" );
            fmt::print(fg(fmt::color::red), "Press Enter to continue...\n");
            getchar();
          }
          
          auto save_suffix_check = sim.FindMember("save_suffix");
          if (save_suffix_check != sim.MemberEnd()) save_suffix = sim["save_suffix"].GetString();
          else save_suffix = std::string{".bgeo"};
          fmt::print("Saving particle outputs with save_suffix[{}] file format.\n", save_suffix);

          fmt::print(
              fg(fmt::color::cyan),
              "Simulation Scene: gpuid[{}], Domain Length [{}], default_dx[{}], default_dt[{}], fps[{}], frames[{}]\n",
              sim["gpu"].GetInt(), l, sim["default_dx"].GetFloat(), sim["default_dt"].GetFloat(),
              sim["fps"].GetInt(), sim["frames"].GetInt());
          benchmark = std::make_unique<mn::mgsp_benchmark>(
              l, sim["default_dt"].GetFloat(),
              sim["fps"].GetInt(), sim["frames"].GetInt(), sim["gravity"].GetFloat(), save_suffix);
        }
      }
    } ///< End basic simulation scene parsing
    {
      auto it = doc.FindMember("meshes");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(fmt::color::cyan),"Scene file has [{}] Finite Element meshes.\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
            if (model["gpu"].GetInt() >= mn::config::g_device_cnt) {
              fmt::print(fg(fmt::color::red),
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
                    model["FBAR_ratio"].GetDouble(),
                    output_attribs); //< Update particle material with run-time inputs
                  fmt::print(fg(fmt::color::green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
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
                    model["FBAR_ratio"].GetDouble(),
                    output_attribs); //< Update particle material with run-time inputs
                  fmt::print(fg(fmt::color::green),"GPU[{}] Mesh material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else 
                {
                  fmt::print(fg(fmt::color::red),
                       "ERROR: GPU[{}] Improper/undefined settings for material [{}] with: use_ASFLIP[{}], use_FEM[{}], and use_FBAR[{}]! \n", 
                       model["gpu"].GetInt(), constitutive,
                       model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool());
                  fmt::print(fg(fmt::color::red), "Press Enter to continue...");
                  getchar();
                }
              }
              else 
              {
                fmt::print(fg(fmt::color::red),
                      "ERROR: GPU[{}] No material [{}] implemented for finite element meshes! \n", 
                      model["gpu"].GetInt(), constitutive);
                fmt::print(fg(fmt::color::red), "Press Enter to continue...");
                getchar();
              }
            };
            
            ElementHolder h_FEM_Elements; //< Declare Host elements
            VerticeHolder h_FEM_Vertices; //< Declare Host vertices

            mn::vec<PREC, 3> offset, velocity;
            for (int d = 0; d < 3; ++d) {
              offset[d]   = model["offset"].GetArray()[d].GetDouble() / l + o;
              velocity[d] = model["velocity"].GetArray()[d].GetDouble() / l; 
            }
            fmt::print(fg(fmt::color::blue),"GPU[{}] Load FEM elements file[{}]...", model["gpu"].GetInt(),  model["file_elements"].GetString());
            load_FEM_Elements(model["file_elements"].GetString(), ',', h_FEM_Elements);
            
            std::vector<std::array<PREC, 6>> h_FEM_Element_Attribs(mn::config::g_max_fem_element_num, 
                                                            std::array<PREC, 6>{0., 0., 0., 0., 0., 0.});
            
            fmt::print(fg(fmt::color::blue),"GPU[{}] Load FEM vertices file[{}]...", model["gpu"].GetInt(),  model["file_vertices"].GetString());
            load_FEM_Vertices(model["file_vertices"].GetString(), ',', h_FEM_Vertices,
                              offset);
            fmt::print(fg(fmt::color::blue),"GPU[{}] Load FEM-MPM particle file[{}]...", model["gpu"].GetInt(),  model["file_vertices"].GetString());
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
          fmt::print(fg(fmt::color::cyan), "Scene file has [{}] particle models. \n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
            if (model["gpu"].GetInt() >= mn::config::g_device_cnt) {
              fmt::print(fg(fmt::color::red),
                       "ERROR! Particle model gpu[{}] exceeds GPUs reserved by g_device_cnt (settings.h)! Skipping model. Increase g_device_cnt and recompile. \n", 
                       model["gpu"].GetInt());
              break;
            } 
            else if (model["gpu"].GetInt() < 0) {
              fmt::print(fg(fmt::color::red),
                       "ERROR! GPU ID[{}] cannot be negative. \n", model["gpu"].GetInt());
              break;
            }
            //fs::path p{model["file"].GetString()};
            std::string constitutive{model["constitutive"].GetString()};
            fmt::print(fg(fmt::color::green),
                       "GPU[{}] Read model constitutive[{}].\n", model["gpu"].GetInt(), constitutive);
            std::vector<std::string> output_attribs;
            std::vector<std::string> target_attribs;
            std::vector<std::string> track_attribs;
            int track_particle_id;

            mn::config::AlgoConfigs algoConfigs( model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),  model["alpha"].GetDouble(), model["beta_min"].GetDouble(), model["beta_max"].GetDouble(), model["FBAR_ratio"].GetDouble());

            auto initModel = [&](auto &positions, auto &velocity) {

              if (constitutive == "JFluid" || constitutive == "J-Fluid" || constitutive == "J_Fluid" || constitutive == "J Fluid" ||  constitutive == "jfluid" || constitutive == "j-fluid" || constitutive == "j_fluid" || constitutive == "j fluid" || constitutive == "Fluid" || constitutive == "fluid" || constitutive == "Water" || constitutive == "Liquid") {
                if(!model["use_ASFLIP"].GetBool() && !model["use_FBAR"].GetBool() && !model["use_FEM"].GetBool())
                {
                  benchmark->initModel<mn::material_e::JFluid>(model["gpu"].GetInt(), positions, velocity);
                  benchmark->updateJFluidParameters( model["gpu"].GetInt(),
                      model["rho"].GetDouble(), model["ppc"].GetDouble(),
                      model["bulk_modulus"].GetDouble(), model["gamma"].GetDouble(),
                      model["viscosity"].GetDouble(),
                      algoConfigs,
                      output_attribs);
                  fmt::print(fg(fmt::color::green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
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
                  fmt::print(fg(fmt::color::green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else if (model["use_ASFLIP"].GetBool() && model["use_FBAR"].GetBool() && !model["use_FEM"].GetBool())
                {
                  benchmark->initModel<mn::material_e::JBarFluid>(model["gpu"].GetInt(), positions, velocity);
                  benchmark->updateJBarFluidParameters( model["gpu"].GetInt(),
                      model["rho"].GetDouble(), model["ppc"].GetDouble(),
                      model["bulk_modulus"].GetDouble(), model["gamma"].GetDouble(), model["viscosity"].GetDouble(), 
                      algoConfigs,
                      output_attribs,
                      track_particle_id, track_attribs, 
                      target_attribs);
                  fmt::print(fg(fmt::color::green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                } 
                else if (!model["use_ASFLIP"].GetBool() && model["use_FBAR"].GetBool() && !model["use_FEM"].GetBool())
                {
                  benchmark->initModel<mn::material_e::JFluid_FBAR>(model["gpu"].GetInt(), positions, velocity);
                  benchmark->updateJFluidFBARParameters( model["gpu"].GetInt(),
                      model["rho"].GetDouble(), model["ppc"].GetDouble(),
                      model["bulk_modulus"].GetDouble(), model["gamma"].GetDouble(), model["viscosity"].GetDouble(), 
                      algoConfigs,
                      output_attribs,
                      track_particle_id, track_attribs, 
                      target_attribs);
                  fmt::print(fg(fmt::color::green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
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
                  fmt::print(fg(fmt::color::green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
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
                  fmt::print(fg(fmt::color::green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else if (model["use_ASFLIP"].GetBool() && model["use_FBAR"].GetBool() && !model["use_FEM"].GetBool())
                {
                  benchmark->initModel<mn::material_e::FixedCorotated_ASFLIP_FBAR>(model["gpu"].GetInt(), positions, velocity);
                  benchmark->update_FR_ASFLIP_FBAR_Parameters( model["gpu"].GetInt(),
                      model["rho"].GetDouble(), model["ppc"].GetDouble(),
                      model["youngs_modulus"].GetDouble(), model["poisson_ratio"].GetDouble(), 
                      model["alpha"].GetDouble(), model["beta_min"].GetDouble(), model["beta_max"].GetDouble(),
                      model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                      model["FBAR_ratio"].GetDouble(),
                      output_attribs, 
                      track_particle_id, track_attribs, 
                      target_attribs);
                  fmt::print(fg(fmt::color::green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else 
                {
                  fmt::print(fg(fmt::color::red),
                       "ERROR: GPU[{}] Improper/undefined settings for material [{}] with: use_ASFLIP[{}], use_FEM[{}], and use_FBAR[{}]! \n", 
                       model["gpu"].GetInt(), constitutive,
                       model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool());
                  getchar();
                }
              } else if (constitutive == "NeoHookean" || constitutive == "neohookena" || constitutive == "Neo-Hookean" || constitutive == "neo-hookean") {

                mn::config::MaterialConfigs materialConfigs(model["ppc"].GetDouble(), model["rho"].GetDouble(), 0, 0, 0, model["youngs_modulus"].GetDouble(), model["poisson_ratio"].GetDouble(), 0, 0, 0, 0, 0, 0);

                if (model["use_ASFLIP"].GetBool() && model["use_FBAR"].GetBool() && !model["use_FEM"].GetBool())
                {
                  benchmark->initModel<mn::material_e::NeoHookean_ASFLIP_FBAR>(model["gpu"].GetInt(), positions, velocity);
                  benchmark->update_NH_ASFLIP_FBAR_Parameters( model["gpu"].GetInt(),
                      materialConfigs, algoConfigs,
                      output_attribs, 
                      track_particle_id, track_attribs, 
                      target_attribs);
                  fmt::print(fg(fmt::color::green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
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
                    model["FBAR_ratio"].GetDouble(),
                    output_attribs);        
                fmt::print(fg(fmt::color::green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
              } else if (constitutive == "NACC" || constitutive == "nacc" || constitutive == "CamClay" || constitutive == "Cam_Clay" || constitutive == "Cam-Clay" || constitutive == "Cam Clay") {
                benchmark->initModel<mn::material_e::NACC>(model["gpu"].GetInt(), positions, velocity);
                benchmark->updateNACCParameters( model["gpu"].GetInt(),
                    model["rho"].GetDouble(), model["ppc"].GetDouble(),
                    model["youngs_modulus"].GetDouble(), model["poisson_ratio"].GetDouble(), 
                    model["beta"].GetDouble(), model["xi"].GetDouble(),
                    model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool(),
                    model["FBAR_ratio"].GetDouble(),
                    output_attribs);
                fmt::print(fg(fmt::color::green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
              } 
              else 
              {
                fmt::print(fg(fmt::color::red),"ERROR: GPU[{}] particle model constititive[{}] does not exist! \n", 
                      model["gpu"].GetInt(), constitutive);
                fmt::print(fg(fmt::color::red), "Press Enter to continue...");                
                getchar();
              }
            };
            mn::vec<PREC, 3> velocity, partition_start, partition_end, inter_a, inter_b;
            for (int d = 0; d < 3; ++d) {
              //offset[d]   = model["offset"].GetArray()[d].GetDouble() / l + o;
              //span[d]     = model["span"].GetArray()[d].GetDouble() / l;
              velocity[d] = model["velocity"].GetArray()[d].GetDouble() / l;
              partition_start[d]  = model["partition_start"].GetArray()[d].GetDouble() / l;
              partition_end[d]  = model["partition_end"].GetArray()[d].GetDouble() / l;
              //inter_a[d]  = model["inter_a"].GetArray()[d].GetDouble() / l;
              //inter_b[d]  = model["inter_b"].GetArray()[d].GetDouble() / l;
              inter_a[d] = -1; // TODO : Deprecate inter_a/b for JSON "subtract" in "geometry"
              inter_b[d] = -1;
            }
            for (int d = 0; d < 3; ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
            std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
            track_particle_id = model["track_particle_id"].GetArray()[0].GetInt();
            for (int d = 0; d < 1; ++d) track_attribs.emplace_back(model["track_attribs"].GetArray()[d].GetString());
            std::cout <<"Track particle ID: " << track_particle_id << " for Attributes: [ " << track_attribs[0] <<" ]"<<'\n';
            for (int d = 0; d < 1; ++d) target_attribs.emplace_back(model["target_attribs"].GetArray()[d].GetString());
            std::cout <<"Target Attributes: [ " << target_attribs[0] <<" ]"<<'\n';
            
            auto geo = model.FindMember("geometry");
            if (geo != model.MemberEnd()) {
              if (geo->value.IsArray()) {
                fmt::print(fg(fmt::color::blue),"Model has [{}] particle geometry operations to perform. \n", geo->value.Size());
                for (auto &geometry : geo->value.GetArray()) {
                  std::string operation{geometry["operation"].GetString()};
                  std::string type{geometry["object"].GetString()};
                  fmt::print("GPU[{}] Operation[{}] with object[{}]... \n", model["gpu"].GetInt(), operation, type);

                  mn::vec<PREC, 3> geometry_offset, geometry_span, geometry_spacing;
                  mn::vec<int, 3> geometry_array;
                  for (int d = 0; d < 3; ++d) {
                    geometry_offset[d]   = geometry["offset"].GetArray()[d].GetDouble() / l + o;
                    geometry_span[d]     = geometry["span"].GetArray()[d].GetDouble() / l;
                    geometry_spacing[d]  = geometry["spacing"].GetArray()[d].GetDouble() / l;
                    geometry_array[d]    = geometry["array"].GetArray()[d].GetInt();
                  }

                  mn::vec<PREC, 3> geometry_offset_updated;
                  geometry_offset_updated[0] = geometry_offset[0];
                  for (int i = 0; i < geometry_array[0]; i++)
                  {
                  geometry_offset_updated[1] = geometry_offset[1];
                  for (int j = 0; j < geometry_array[1]; j++)
                  {
                  geometry_offset_updated[2] = geometry_offset[2];
                  for (int k = 0; k < geometry_array[2]; k++)
                  {
                  if (type == "Box" || type == "box")
                  {
                    if (operation == "Add" || operation == "add") {
                      make_box(models[model["gpu"].GetInt()], 
                          geometry_span, geometry_offset_updated, model["ppc"].GetFloat());
                    }
                    else if (operation == "Subtract" || operation == "subtract") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Union" || operation == "union") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Intersect" || operation == "intersect") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Difference" || operation == "difference") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else 
                    {
                      fmt::print(fg(fmt::color::red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", model["gpu"].GetInt(), operation); 
                      getchar(); 
                    }
                  }
                  else if (type == "Cylinder" || type == "cylinder")
                  {
                    if (operation == "Add" || operation == "add") {
                      make_cylinder(models[model["gpu"].GetInt()], 
                              geometry_span, geometry_offset_updated, model["ppc"].GetFloat(), geometry["radius"].GetDouble(), geometry["axis"].GetString());
                    }
                    else if (operation == "Subtract" || operation == "subtract") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Union" || operation == "union") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Intersect" || operation == "intersect") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Difference" || operation == "difference") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else 
                    {
                      fmt::print(fg(fmt::color::red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", model["gpu"].GetInt(), operation); 
                      getchar(); 
                    }
                  }
                  else if (type == "Sphere" || type == "sphere")
                  {
                    if (operation == "Add" || operation == "add") {
                      make_sphere(models[model["gpu"].GetInt()], 
                              geometry_span, geometry_offset_updated, model["ppc"].GetFloat(), geometry["radius"].GetDouble());
                    }
                    else if (operation == "Subtract" || operation == "subtract") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Union" || operation == "union") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Intersect" || operation == "intersect") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Difference" || operation == "difference") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else 
                    {
                      fmt::print(fg(fmt::color::red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", model["gpu"].GetInt(), operation); 
                      getchar(); 
                    }
                  }
                  else if (type == "OSU LWF" || type == "OSU Water")
                  {
                    if (operation == "Add" || operation == "add") {
                      make_OSU_LWF(models[model["gpu"].GetInt()], 
                            geometry_span, geometry_offset_updated, model["ppc"].GetFloat());
                    }
                    else if (operation == "Subtract" || operation == "subtract") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Union" || operation == "union") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Intersect" || operation == "intersect") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Difference" || operation == "difference") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else 
                    {
                      fmt::print(fg(fmt::color::red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", model["gpu"].GetInt(), operation); 
                      getchar(); 
                    }
                  }
                  else if (type == "File" || type == "file") 
                  {
                    // * NOTE : Assumes geometry "file" specified by scene.json is in AssetDirPath/MpmParticles, i.e. ~/claymore/Data/MpmParticles/file
                    std::string geometry_fn = std::string(AssetDirPath) + "MpmParticles/" + geometry["file"].GetString();
                    fs::path geometry_file_path{geometry_fn};
                    if (geometry_file_path.empty()) fmt::print(fg(fmt::color::red), "ERROR: Input file[{}] does not exist.\n", geometry_fn);
                    else {
                      std::ifstream istrm(geometry_fn);
                      if (!istrm.is_open())  fmt::print(fg(fmt::color::red), "ERROR: Cannot open file[{}]\n", geometry_fn);
                      istrm.close();
                    }
                    if (operation == "Add" || operation == "add") {
                      // TODO : Reimplement signed-distance-field (*.sdf) particle input files to match current scene set-up (e.g. appropiate offset and scale).
                      if (geometry_file_path.extension() == ".sdf") 
                      {
                        if (model["partition"].GetInt()){
                          auto sdf_particles = mn::read_sdf(
                              geometry_fn, model["ppc"].GetFloat(),
                              mn::config::g_dx, mn::config::g_domain_size, geometry_offset_updated, geometry_span,
                              partition_start, partition_end, inter_a, inter_b);
                        } else {
                          auto sdf_particles = mn::read_sdf(
                              geometry_fn, model["ppc"].GetFloat(),
                              mn::config::g_dx, mn::config::g_domain_size, geometry_offset_updated, geometry_span);
                        }
                      }
                      if (geometry_file_path.extension() == ".csv") 
                      {
                        load_csv_particles(geometry_fn, ',', 
                                            models[model["gpu"].GetInt()], geometry_offset_updated);
                      }
                      // TODO : Include particle file input for ".bgeo", ".bin", ".pdb", ".ptc", ".vtk" using PartIO readers
                    }
                    else if (operation == "Subtract" || operation == "subtract") { fmt::print(fg(fmt::color::red),"Operation not implemented...\n"); }
                    else if (operation == "Union" || operation == "union") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Intersect" || operation == "intersect") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else if (operation == "Difference" || operation == "difference") {fmt::print(fg(fmt::color::red),"Operation not implemented...\n");}
                    else 
                    {
                      fmt::print(fg(fmt::color::red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", model["gpu"].GetInt(), operation); 
                      getchar(); 
                    }

                  }
                  else 
                  {
                    fmt::print(fg(fmt::color::red), "GPU[{}] ERROR: Geometry object[{}] does not exist!\n", model["gpu"].GetInt(), type);
                    fmt::print(fg(fmt::color::red), "Press Enter...\n");
                    getchar();
                  } 
                  geometry_offset_updated[2] += geometry_spacing[2];
                  } 
                  geometry_offset_updated[1] += geometry_spacing[1];
                  } 
                  geometry_offset_updated[0] += geometry_spacing[0];
                  }

                }
              }
            } //< End geometry
            else {
              fmt::print(fg(fmt::color::red), "ERROR: GPU[{}] has no geometry object!\n", model["gpu"].GetInt());
              fmt::print(fg(fmt::color::red), "Press enter to continue...\n");
              getchar();
            }
              
            auto positions = models[model["gpu"].GetInt()];
            mn::IO::insert_job([&]() {
              mn::write_partio<PREC, 3>(std::string{p.stem()} + save_suffix,
                                        positions);});              
            mn::IO::flush();
            fmt::print(fg(fmt::color::green), "GPU[{}] Saved particle model to [{}].\n", model["gou"].GetInt(), std::string{p.stem()} + save_suffix);
            
            if (positions.size() > mn::config::g_max_particle_num) {
              fmt::print(fg(fmt::color::red), "ERROR: GPU[{}] Particle count [{}] exceeds g_max_particle_num in settings.h! Increase and recompile to avoid problems. \n", model["gpu"].GetInt(), positions.size());
              fmt::print(fg(fmt::color::red), "Press Enter to continue anyways... \n");
              getchar();
            }

            initModel(positions, velocity);
          }
        }
      }
    } ///< end models parsing
    {
      auto it = doc.FindMember("grid-targets");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(fmt::color::cyan),"Scene has [{}] grid-targets.\n", it->value.Size());
          int target_ID = 0;
          for (auto &model : it->value.GetArray()) {
          
            std::vector<std::array<PREC_G, mn::config::g_target_attribs>> h_gridTarget(mn::config::g_target_cells, 
                                                            std::array<PREC_G, mn::config::g_target_attribs>{0.f,0.f,0.f,
                                                            0.f,0.f,0.f,
                                                            0.f,0.f,0.f,0.f});
            mn::vec<PREC_G, 7> target; // TODO : Make structure for grid-target data

            if (0) // TODO : Implement attribute selection for grid-target
            {
              std::string attribute{model["attribute"].GetString()};
              if      (attribute == "Force"  || attribute == "force")  target[0] = 0;
              else if (attribute == "Velocity" || attribute == "velocity") target[0] = 1;
              else if (attribute == "Momentum" || attribute == "momentum") target[0] = 2;
              else if (attribute == "Mass"  || attribute == "mass")  target[0] = 3;
              else if (attribute == "JBar" || attribute == "J Bar") target[0] = 4;
              else if (attribute == "Volume" || attribute == "volume") target[0] = 5;
              else if (attribute == "X"  || attribute == "x")  target[0] = 6;
              else if (attribute == "Z-" || attribute == "z-") target[0] = 7;
              else if (attribute == "Z+" || attribute == "z+") target[0] = 8;
              else {
                target[0] = -1;
                fmt::print(fg(fmt::color::red), "ERROR: gridTarget[{}] has invalid attribute[{}].\n", target_ID, attribute);
              }
            }

            std::string direction{model["direction"].GetString()};
            if      (direction == "X"  || direction == "x")  target[0] = 0;
            else if (direction == "X-" || direction == "x-") target[0] = 1;
            else if (direction == "X+" || direction == "x+") target[0] = 2;
            else if (direction == "Y"  || direction == "y")  target[0] = 3;
            else if (direction == "Y-" || direction == "y-") target[0] = 4;
            else if (direction == "Y+" || direction == "y+") target[0] = 5;
            else if (direction == "X"  || direction == "x")  target[0] = 6;
            else if (direction == "Z-" || direction == "z-") target[0] = 7;
            else if (direction == "Z+" || direction == "z+") target[0] = 8;
            else {
              target[0] = -1;
              fmt::print(fg(fmt::color::red), "ERROR: gridTarget[{}] has invalid direction[{}].\n", target_ID, direction);
              getchar();
            }
            // * Load and scale target domain
            for (int d = 0; d < 3; ++d) 
            {
              target[d+1] = model["domain_start"].GetArray()[d].GetFloat() / l + o;
              target[d+4] = model["domain_end"].GetArray()[d].GetFloat() / l + o;
            }

            // * NOTE: Checks for zero length target dimensions, grows by 1 grid-cell if so
            for (int d=0; d < 3; ++d)
              if (target[d+1] == target[d+4]) target[d+4] = target[d+4] + dx;         
            PREC_G freq = model["output_frequency"].GetFloat();

            mn::config::GridTargetConfigs gridTargetConfigs(target[6], target[6], target[6], make_float3(target[1], target[2], target[3]), make_float3(target[4], target[5], target[6]), freq);

            // * Loop through GPU devices to initialzie
            for (int did = 0; did < mn::config::g_device_cnt; ++did) {
              benchmark->initGridTarget(did, h_gridTarget, target, 
                freq); // TODO : Allow more than one frequency for grid-targets
            fmt::print(fg(fmt::color::green), "GPU[{}] gridTarget[{}] Initialized.\n", did, target_ID);
            }
            target_ID += 1;
          }
        }
      }
    } ///< End grid-target parsing
    {
      auto it = doc.FindMember("particle-targets");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(fmt::color::cyan),"Scene has [{}] particle-targets.\n", it->value.Size());
          int target_ID = 0;
          for (auto &model : it->value.GetArray()) {
          
            std::vector<std::array<PREC, mn::config::g_particle_target_attribs>> h_particleTarget(mn::config::g_particle_target_cells, 
                                                            std::array<PREC, mn::config::g_particle_target_attribs>{0.f,0.f,0.f,
                                                            0.f,0.f,0.f,
                                                            0.f,0.f,0.f,0.f});
            mn::vec<PREC, 7> target; // TODO : Make structure for particle-target data
            // TODO : Implement attribute selection for particle-targets (only elevation currently)
            std::string operation{model["operation"].GetString()};
            if      (operation == "Maximum" || operation == "maximum" || operation == "Max" || operation == "max") target[0] = 0;
            else if (operation == "Minimum" || operation == "minimum" || operation == "Min" || operation == "min") target[0] = 1;
            else if (operation == "Add" || operation == "add" || operation == "Sum" || operation == "sum") target[0] = 2;
            else if (operation == "Subtract" || operation == "subtract") target[0] = 3;
            else if (operation == "Average" || operation == "average" ||  operation == "Mean" || operation == "mean") target[0] = 4;
            else if (operation == "Variance" || operation == "variance") target[0] = 5;
            else if (operation == "Standard Deviation" || operation == "stdev") target[0] = 6;
            else {
              target[0] = -1;
              fmt::print(fg(fmt::color::red), "ERROR: particleTarget[{}] has invalid operation[{}].\n", target_ID, operation);
              getchar();
            }
            // Load and scale target domain to 1 x 1 x 1 domain + off-by-2 offset
            for (int d = 0; d < 3; ++d) 
            {
              target[d+1] = model["domain_start"].GetArray()[d].GetFloat() / l + o;
              target[d+4] = model["domain_end"].GetArray()[d].GetFloat() / l + o;
            }

            PREC freq = model["output_frequency"].GetDouble();

            mn::config::ParticleTargetConfigs particleTargetConfigs(target[6], target[6], target[6], {target[1], target[2], target[3]}, {target[4], target[5], target[6]}, freq);


            // Initialize on GPUs
            for (int did = 0; did < mn::config::g_device_cnt; ++did) {
              benchmark->initParticleTarget(did, h_particleTarget, target, 
                freq);
              fmt::print(fg(fmt::color::green), "GPU[{}] particleTarget[{}] Initialized.\n", did, target_ID);
            }
            target_ID += 1; // TODO : Count targets using static variable in a structure
          }
        }
      }
    } ///< End particle-target parsing
    {
      auto it = doc.FindMember("grid-boundaries");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(fmt::color::cyan), "Scene has [{}] grid-boundaries.\n", it->value.Size());
          int boundary_ID = 0;
          for (auto &model : it->value.GetArray()) {

            mn::vec<float, 7> h_boundary;
            for (int d = 0; d < 3; ++d) {
              h_boundary[d] = model["domain_start"].GetArray()[d].GetFloat() / l + o;
            }
            for (int d = 0; d < 3; ++d) {
              h_boundary[d+3] = model["domain_end"].GetArray()[d].GetFloat() / l + o;
            }
            std::string object{model["object"].GetString()};
            std::string contact{model["contact"].GetString()};

            if (object == "Wall" || object == "wall")
            {
              if (contact == "Rigid" || contact == "Sticky" || contact == "Stick") h_boundary[6] = 0;
              else if (contact == "Slip") h_boundary[6] = 1;
              else if (contact == "Separable") h_boundary[6] = 2;
              else h_boundary[6] = -1;
            }
            else if (object == "Box" || object == "box")
            {
              if (contact == "Rigid" || contact == "Sticky" || contact == "Stick") h_boundary[6] = 3;
              else if (contact == "Slip") h_boundary[6] = 4;
              else if (contact == "Separable") h_boundary[6] = 5;
              else h_boundary[6] = -1;
            }
            else if (object == "Sphere" || object == "sphere")
            {
              if (contact == "Rigid" || contact == "Sticky" || contact == "Stick") h_boundary[6] = 6;
              else if (contact == "Slip") h_boundary[6] = 7;
              else if (contact == "Separable") h_boundary[6] = 8;
              else h_boundary[6] = -1;
            }
            else if (object == "OSU LWF" || object == "OSU Flume" || object == "OSU")
            {
              if (contact == "Rigid" || contact == "Sticky" || contact == "Stick") h_boundary[6] = 9;
              else if (contact == "Slip") h_boundary[6] = 9;
              else if (contact == "Separable") h_boundary[6] = 9;
              else h_boundary[6] = -1;
            }
            else if (object == "OSU Paddle" || object == "OSU Wave Maker")
            {
              if (contact == "Rigid" || contact == "Sticky" || contact == "Stick") h_boundary[6] = 12;
              else if (contact == "Slip") h_boundary[6] = 12;
              else if (contact == "Separable") h_boundary[6] = 12;
              else h_boundary[6] = -1;
            }            
            else if (object == "Cylinder" || object == "cylinder")
            {
              if (contact == "Rigid" || contact == "Sticky" || contact == "Stick") h_boundary[6] = 15;
              else if (contact == "Slip") h_boundary[6] = 16;
              else if (contact == "Separable") h_boundary[6] = 17;
              else h_boundary[6] = -1;
            }
            else 
            {
              fmt::print(fg(fmt::color::red), "ERROR: gridBoundary[{}] object[{}] or contact[{}] is not valid! \n", boundary_ID, object, contact);
              h_boundary[6] = -1;
            }

            // Set up moving grid-boundary if applicable
            auto motion_file = model.FindMember("file"); // Check for motion file
            auto motion_velocity = model.FindMember("velocity"); // Check for velocity
            if (motion_file != model.MemberEnd()) 
            {
              fmt::print(fg(fmt::color::blue),"Found motion file for grid-boundary[{}]. Loading... \n", boundary_ID);
              MotionHolder motionPath;
              std::string motion_fn = std::string(AssetDirPath) + model["file"].GetString();
              fs::path motion_file_path{motion_fn};
              if (motion_file_path.empty()) fmt::print(fg(fmt::color::red), "ERROR: Input file[{}] does not exist.\n", motion_fn);
              else {
                std::ifstream istrm(motion_fn);
                if (!istrm.is_open())  fmt::print(fg(fmt::color::red), "ERROR: Cannot open file[{}]\n", motion_fn);
                istrm.close();
              }

              load_motionPath(motion_fn, ',', motionPath);
              
              PREC_G gb_freq = 1;
              auto motion_freq = model.FindMember("output_frequency");
              if (motion_freq != model.MemberEnd()) gb_freq = model["output_frequency"].GetFloat();

              for (int did = 0; did < mn::config::g_device_cnt; ++did) {
                benchmark->initMotionPath(did, motionPath, gb_freq);
                fmt::print(fg(fmt::color::green),"GPU[{}] gridBoundary[{}] motion file[{}] initialized with frequency[{}].\n", did, boundary_ID, model["file"].GetString(), gb_freq);
              }
            }
            else if (motion_velocity != model.MemberEnd() && motion_file == model.MemberEnd())
            {
              fmt::print(fg(fmt::color::blue),"Found velocity for grid-boundary[{}]. Loading...", boundary_ID);
              mn::vec<PREC_G, 3> velocity;
              for (int d=0; d<3; d++) velocity[d] = model["velocity"].GetArray()[d].GetDouble() / l;
            }
            else 
              fmt::print(fg(fmt::color::orange),"No motion file or velocity specified for grid-boundary. Assuming static. \n");
            
            // ----------------  Initialize grid-boundaries ---------------- 
            benchmark->initGridBoundaries(0, h_boundary, boundary_ID);
            fmt::print(fg(fmt::color::green), "Initialized gridBoundary[{}]: object[{}], contact[{}].\n", boundary_ID, object, contact);
            boundary_ID += 1;
          }
        }
      }
    }
  }
} ///< End scene file parsing




/// @brief Multi-GPU MPM for Engineers. Builds on the original, open-source Claymore MPM, all rights reserved. Claymore MPM for Engineers: https://github.com/penn-graphics-research/claymore . Original Claymore MPM : https://github.com/penn-graphics-research/claymore . For an executable [test] with scene file [scene_test.json]
/// @param argv  For an executable [test] with desired scene file [scene_test.json], use Command-line: ./test --file = scene_test.json  (NOTE : defaults to scene.json if not specified)
int main(int argc, char *argv[]) {
  using namespace mn;
  using namespace config;
  Cuda::startup(); //< Start CUDA GPUs if available.

  // ---------------- Read JSON input file for simulation ---------------- 
  cxxopts::Options options("Scene_Loader", "Read simulation scene");
  options.add_options()("f,file", "Scene Configuration File",
      cxxopts::value<std::string>()->default_value("scene.json")); //< scene.json is default
  auto results = options.parse(argc, argv);
  auto fn = results["file"].as<std::string>();
  fmt::print(fg(fmt::color::green),"Loading scene file [{}].\n", fn);

  // ---------------- Initialize the simulation ---------------- 
  {
    std::unique_ptr<mn::mgsp_benchmark> benchmark; //< Simulation object pointer
    std::vector<std::array<PREC, 3>> models[g_device_cnt]; //< Initial particle positions
    std::vector<mn::vec<PREC, 3>> v0(g_device_cnt, mn::vec<PREC, 3>{0.,0.,0.}); //< Initial velocities

    parse_scene(fn, benchmark, models); //< Initialize from input scene file  
    fmt::print(fg(fmt::color::green),"Finished scene initialization.\n");

    // ---------------- Run Simulation
    if (g_log_level > 1) {
      fmt::print(fg(fmt::color::blue),"Press ENTER to start simulation... (Disable via g_log_level < 2). \n");
      getchar();
    }

    fmt::print(fg(fmt::color::cyan),"Starting simulation...\n");
    // CppTimer sim_timer{};
    // sim_timer.tick();
    benchmark->main_loop();
    // sim_timer.tock(
    //       fmt::format("Finished simulation in [{}] minutes. Average of [{}] seconds per frame.", did, curStep));
    fmt::print(fg(fmt::color::green), "Finished simulation.\n");
    // ---------------- Clear
    IO::flush();
    fmt::print(fg(fmt::color::green),"Cleared I/O.\n");

    benchmark.reset();
    fmt::print(fg(fmt::color::green),"Reset simulation structure.\n");
  }
  // ---------------- Shutdown GPU / CUDA
  Cuda::shutdown();
  fmt::print(fg(fmt::color::green),"Simulation finished. Shut-down CUDA GPUs.\n");
  // ---------------- Finish
  return 0;
}