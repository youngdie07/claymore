#ifndef __READ_SCENE_INPUT_H_
#define __READ_SCENE_INPUT_H_
#include "mgsp_benchmark.cuh"
#include "partition_domain.h"
#include <MnBase/Geometry/GeometrySampler.h>
#include <MnBase/Math/Vec.h>
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
#include <array>
#include <cassert>

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
static const auto red = fmt::color::red;
static const auto blue = fmt::color::blue;
static const auto green = fmt::color::green;
static const auto yellow = fmt::color::yellow;
static const auto orange = fmt::color::orange;
static const auto cyan = fmt::color::cyan;
static const auto white = fmt::color::white;

typedef std::vector<std::array<PREC, 3>> PositionHolder;
typedef std::vector<std::array<PREC, 13>> VerticeHolder;
typedef std::vector<std::array<int, 4>> ElementHolder;
typedef std::vector<std::array<PREC, 6>> ElementAttribsHolder;
typedef std::vector<std::array<PREC_G, 3>> MotionHolder;

PREC o = mn::config::g_offset; //< Grid-cell buffer size (see off-by-2, Xinlei Wang)
PREC l = mn::config::g_length; //< Domain max length default
PREC dx = mn::config::g_dx; //< Grid-cell length [1x1x1] default
std::string save_suffix; //< File-format to save particles with
float verbose = 0;


/// @brief Check if a particle is inside a box partition.
/// @tparam Data-type, e.g. float or double. 
/// @param arr Array of [x,y,z] position to check for being inside partition.
/// @param partition_start Starting corner of box partition.
/// @param partition_end Far corner of box partition.
/// @return Returns true if an inputted position is inside the specified partition. False otherwise.
template <typename T>
bool inside_partition(std::array<T,3> arr, mn::vec<T,3> partition_start, mn::vec<T,3> partition_end) {
  if (arr[0] >= partition_start[0] && arr[0] < partition_end[0])
    if (arr[1] >= partition_start[1] && arr[1] < partition_end[1])
      if (arr[2] >= partition_start[2] && arr[2] < partition_end[2])
        return true;
  return false;
}

/// @brief  Load in binary file (*.bin) as particles. Assumes data is sequential particles [x,y,z] positions.
/// @tparam T Data-type in file, e.g. 'int', 'float' or 'double'.
/// @param pcnt Particle count in file. If too low, not all particles will be read.
/// @param filename Path to file (e.g. MpmParticles/my_particles.bin), starting from AssetDirectory (e.g. claymore/Data/).
/// @return Positions of particles returned as vector of arrays, containing [x,y,z] per particle
template <typename T>
decltype(auto) load_binary_particles(std::size_t pcnt, std::string filename) {
  std::vector<std::array<T, 3>> fields(pcnt);
  auto f = fopen(filename.c_str(), "rb");
  auto res = std::fread((T *)fields.data(), sizeof(T), fields.size() * 3, f);
  std::fclose(f);
  int i = 0;
  fmt::print(fg(fmt::color::white), "Particle count[{}] read from file[{}].\n", filename, fields.size()); 
  fmt::print(fg(fmt::color::white), "Printing first/last 4 particle positions (NOTE: Not scaled): \n");
  for (auto row : fields) {
    if ((i >= 4) && (i < fields.size() - 4))  {i++; continue;}
    std::cout << "Particle["<< i << "] (x, y, z): ";
    for (auto field : row) std::cout << " "<< field << ", ";
    std::cout << '\n';
    i++;
  }
  return fields;
}

/// @brief Load a comma-delimited *.csv file as particles. Reads in [x, y, z] positions per row. Outputs particles into fields.
/// @brief Assume offset, partition_start/end are already scaled to 1x1x1 domain. Does not assumed input file is scaled to 1x1x1, this function will scale it for you.
/// @param filename Path to file (e.g. MpmParticles/my_particles.csv), starting from AssetDirectory (e.g. claymore/Data/).
/// @param sep Delimiter of data. Typically a comma (',') for CSV files.
/// @param fields Vector of array to output data into
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
/// @param partition_start Start corner of particle GPU partition, cut out everything before.
/// @param partition_end End corner of particle GPU partition, cut out everything beyond.
void load_csv_particles(const std::string& filename, char sep, 
                        std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> offset, mn::vec<PREC,3> partition_start, mn::vec<PREC,3> partition_end) {
  std::ifstream in(filename.c_str());
  if (in) {
    std::string line;
    while (getline(in, line)) {
      std::stringstream sep(line);
      std::string field;
      const int el = 3; // 3 = x, y, z - Default,
      int col = 0;
      std::array<PREC, 3> arr;
      while (getline(sep, field, ',')) 
      {
        if (col >= el) break;
        arr[col] = stof(field) / l + offset[col];
        col++;
      }
      if (inside_partition(arr, partition_start, partition_end))
        fields.push_back(arr);
    }
  }
  fmt::print(fg(fmt::color::white), "Particle count[{}] read from file[{}].\n", fields.size(), filename); 
  fmt::print(fg(fmt::color::white), "Printing first/last 4 particle positions (NOTE: scaled to 1x1x1 domain): \n");
  int i = 0;
  for (auto row : fields) {
    if ((i >= 4) && (i < fields.size() - 4))  {i++; continue;}
    std::cout << "Particle["<< i << "] (x, y, z): ";
    for (auto field : row) std::cout << " "<< field << ", ";
    std::cout << '\n';
    i++;
  }
}

/// @brief Load a comma-delimited *.csv file as particles. Reads in [x, y, z] positions per row. Outputs particles into fields.
/// @brief Assume offset already scaled to 1x1x1 domain. Does not assumed input file is scaled to 1x1x1, this function will scale it for you.
/// @param filename Path to file (e.g. MpmParticles/my_particles.csv), starting from AssetDirectory (e.g. claymore/Data/)
/// @param sep Delimiter of data. Typically a comma (',') for CSV files.
/// @param fields Vector of array to output data into
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
void load_csv_particles(const std::string& filename, char sep, 
                        std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> offset) {
  std::ifstream in(filename.c_str());
  if (in) {
    std::string line;
    while (getline(in, line)) {
      std::stringstream sep(line);
      std::string field;
      const int el = 3; // 3 = x, y, z - Default,
      int col = 0;
      std::array<PREC, 3> arr;
      while (getline(sep, field, ',')) 
      {
        if (col >= el) break;
        arr[col] = stof(field) / l + offset[col];
        col++;
      }
      fields.push_back(arr);
    }
  }
  fmt::print(fg(fmt::color::white), "Particle count[{}] read from file[{}].\n", fields.size(), filename); 
  fmt::print(fg(fmt::color::white), "Printing first/last 4 particle positions (NOTE: scaled to 1x1x1 domain): \n");
  int i = 0;
  for (auto row : fields) {
    if ((i >= 4) && (i < fields.size() - 4))  {i++; continue;}
    std::cout << "Particle["<< i << "] (x, y, z): ";
    for (auto field : row) std::cout << " " << field << ", ";
    std::cout << '\n';
    i++;
  }
}

/// @brief Make box as particles, write to fields as [x,y,z] data.
/// @brief Assume span, offset, radius, partition_start/end are already scaled to 1x1x1 domain. 
/// @param fields Vector of arrays to write particle position [x,y,z] data into
/// @param span Sets max span to look in (for efficiency). Erases particles if too low.
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
/// @param ppc Particle-per-cell, number of particles to sample per 3D grid-cell (e.g. 8).
/// @param partition_start Start corner of particle GPU partition, cut out everything before.
/// @param partition_end End corner of particle GPU partition, cut out everything beyond.
void make_box(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset, PREC ppc, mn::vec<PREC,3> partition_start, mn::vec<PREC,3> partition_end) {
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
                if (inside_partition(arr, partition_start, partition_end))
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
                if (inside_partition(arr, partition_start, partition_end))
                  fields.push_back(arr);
            }
            else 
            {
            if (inside_partition(arr, partition_start, partition_end))
              fields.push_back(arr);
          }
        }
      }
    }
  } 
}
/// @brief Make cylinder as particles, write to fields as [x,y,z] data.
/// @brief Assume span, offset, radius, partition_start/end are already scaled to 1x1x1 domain. 
/// @param fields Vector of arrays to write particle position [x,y,z] data into
/// @param span Sets max span to look in (for efficiency). Erases particles if too low.
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
/// @param ppc Particle-per-cell, number of particles to sample per 3D grid-cell (e.g. 8).
/// @param radius Radius of cylinder.
/// @param axis Longitudinal axis of cylinder, e.g. std::string{"X"} for X oriented cylinder.
/// @param partition_start Start corner of particle GPU partition, cut out everything before.
/// @param partition_end End corner of particle GPU partition, cut out everything beyond.
void make_cylinder(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset,
                        PREC ppc, PREC radius, std::string axis, mn::vec<PREC,3> partition_start, mn::vec<PREC,3> partition_end) {
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
            PREC xo, yo, zo;
            xo = yo = zo = radius; 
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
              fmt::print(fg(red), "ERROR: Value of axis[{}] is not applicable for a Cylinder. Use X, Y, or Z.", axis);
            }
            if (r <= radius)
              if (inside_partition(arr, partition_start, partition_end))
                fields.push_back(arr);
        }
      }
    }
  } 
}
/// @brief Make sphere as particles, write to fields as [x,y,z] data.
/// @brief Assume span, offset, radius, partition_start/end are already scaled to 1x1x1 domain. 
/// @param fields Vector of arrays to write particle position [x,y,z] data into
/// @param span Sets max span to look in (for efficiency). Erases particles if too low.
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
/// @param ppc Particle-per-cell, number of particles to sample per 3D grid-cell (e.g. 8).
/// @param radius Radius of cylinder.
/// @param partition_start Start corner of particle GPU partition, cut out everything before.
/// @param partition_end End corner of particle GPU partition, cut out everything beyond.
void make_sphere(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset,
                        PREC ppc, PREC radius, mn::vec<PREC,3> partition_start, mn::vec<PREC,3> partition_end) {
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
            PREC xo, yo, zo;
            xo = yo = zo = radius; 
            PREC r;
            r = std::sqrt((x-xo)*(x-xo) + (y-yo)*(y-yo) + (z-zo)*(z-zo));
            if (r <= radius) 
              if (inside_partition(arr, partition_start, partition_end))
                fields.push_back(arr);
        }
      }
    }
  } 
}


/// @brief Make OSU LWF flume fluid as particles, write to fields as [x,y,z] data.
/// @brief Assume span, offset, radius, partition_start/end are already scaled to 1x1x1 domain. 
/// @param fields Vector of arrays to write particle position [x,y,z] data into
/// @param span Sets max span to look in (for efficiency). Erases particles if too low.
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
/// @param ppc Particle-per-cell, number of particles to sample per 3D grid-cell (e.g. 8).
/// @param partition_start Start corner of particle GPU partition, cut out everything before.
/// @param partition_end End corner of particle GPU partition, cut out everything beyond.
void make_OSU_LWF(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset,
                        PREC ppc, mn::vec<PREC,3> partition_start, mn::vec<PREC,3> partition_end) {
  PREC ppl_dx = dx / cbrt(ppc); // Linear spacing of particles [1x1x1]
  int i_lim, j_lim, k_lim; // Number of par. per span direction
  i_lim = (int)((span[0]) / ppl_dx + 1.0); 
  j_lim = (int)((span[1]) / ppl_dx + 1.0); 
  k_lim = (int)((span[2]) / ppl_dx + 1.0); 

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
  bathy[1] = (0.15 + 0.076) + bathy[0]; // Bathymetry slab raised ~0.15m, 0.076m thick
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
                if (inside_partition(arr, partition_start, partition_end))
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

template <typename T>
bool inside_box(std::array<T,3>& arr, mn::vec<T,3> span, mn::vec<T,3> offset) 
{
  if (arr[0] >= offset[0] && arr[0] < span[0] + offset[0])
    if (arr[1] >= offset[1] && arr[1] < span[1] + offset[1])
      if (arr[2] >= offset[2] && arr[2] < span[2] + offset[2])
        return true;
  return false;
}

template <typename T>
bool inside_cylinder(std::array<T,3>& arr, T radius, std::string axis, mn::vec<T,3> span, mn::vec<T,3> offset) 
{
  std::array<T,3> center;
  for (int d=0; d<3; d++) center[d] = offset[d] + radius;

  if (axis == "X" || axis == "x")
  { 
    PREC r = std::sqrt((arr[1]-center[1])*(arr[1]-center[1]) + (arr[2]-center[2])*(arr[2]-center[2]));
    if (r <= radius)
       if (arr[0] >= offset[0] && arr[0] < offset[0] + span[0])
          return true;
  }
  else if (axis == "Y" || axis == "Y")
  { 
    PREC r = std::sqrt((arr[0]-center[0])*(arr[0]-center[0]) + (arr[2]-center[2])*(arr[2]-center[2]));
    if (r <= radius)
       if (arr[1] >= offset[1] && arr[1] < offset[1] + span[1])
          return true;
  }
  else if (axis == "Z" || axis == "z")
  { 
    PREC r = std::sqrt((arr[1]-center[1])*(arr[1]-center[1]) + (arr[0]-center[0])*(arr[0]-center[0]));
    if (r <= radius)
       if (arr[2] >= offset[2] && arr[2] < offset[2] + span[2])
          return true;
  }
  return false;
}

template <typename T>
bool inside_sphere(std::array<T,3>& arr, T radius, mn::vec<T,3> offset) 
{
  std::array<T,3> center;
  for (int d=0; d<3; d++) center[d] = offset[d] + radius;
  PREC r = std::sqrt((arr[0]-center[0])*(arr[0]-center[0]) + (arr[1]-center[1])*(arr[1]-center[1]) + (arr[2]-center[2])*(arr[2]-center[2]));
  if (r <= radius)
      return true;
  return false;
}

template <typename T>
void subtract_box(std::vector<std::array<T,3>>& particles, mn::vec<T,3> span, mn::vec<T,3> offset) {
  fmt::print("Previous particle count: {}\n", particles.size());
  particles.erase(std::remove_if(particles.begin(), particles.end(),
                              [&](std::array<T,3> x){ return inside_box(x, span, offset); }), particles.end());
  fmt::print("Updated particle count: {}\n", particles.size());
}

template <typename T>
void subtract_cylinder(std::vector<std::array<T,3>>& particles, T radius, std::string axis, mn::vec<T,3> span, mn::vec<T,3> offset) {
  fmt::print("Previous particle count: {}\n", particles.size());
  particles.erase(std::remove_if(particles.begin(), particles.end(),
                              [&](std::array<T,3> x){ return inside_cylinder(x, radius / l, axis, span, offset); }), particles.end());
  fmt::print("Updated particle count: {}\n", particles.size());
}


template <typename T>
void subtract_sphere(std::vector<std::array<T,3>>& particles, T radius, mn::vec<T,3> offset) {
  fmt::print("Previous particle count: {}\n", particles.size());
  particles.erase(std::remove_if(particles.begin(), particles.end(),
                              [&](std::array<T,3> x){ return inside_sphere(x, radius / l, offset); }), particles.end());
  fmt::print("Updated particle count: {}\n", particles.size());
}


void load_FEM_Vertices(const std::string& filename, char sep, 
                       VerticeHolder& fields, 
                       mn::vec<PREC, 3> offset){
  std::ifstream in(filename.c_str());
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
  std::ifstream in(filename.c_str());
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

/// @brief Make 3D rotation matrix that can rotate a point around origin
/// @brief Order: [Z Rot.]->[Y Rot.]->[X Rot.]=[ZYX Rot.]. (X Rot. 1st).
/// @brief Rot. on fixed axis (X,Y,Z) (i.e. NOT Euler or Quat. rotation)
/// @param angles Rotation angles [a,b,c] (degrees) for [X,Y,Z] axis.
/// @param R Original rotation matrix, use 3x3 Identity mat. if none.
/// @returns Multiply new rotation matrix into R.
template <typename T>
constexpr void rotation_matrix(const mn::vec<T,3> &angles, mn::vec<T,3,3> &R = {{1,0,0},{0,1,0},{0,0,0,1}}) {
    // TODO: Can be way more efficient, made this quickly
    for (int d=0;d<3;d++) angles[d] *= PI_AS_A_DOUBLE;
    mn::vec3x3 prev_R = R;
    mn::vec3x3 tmp_R; tmp_R.set(0.0);
    // X-Axis Rotation
    tmp_R(0, 0) = 1;
    tmp_R(1, 1) = tmp_R(2, 2) = std::cos(angles[0]);
    tmp_R(2, 1) = tmp_R(1, 2) = std::sin(angles[0]);
    tmp_R(1, 2) = -tmp_R(1, 2);
    matrixMatrixMultiplication3d(tmp_R.data(), prev_R.data(), R.data());
    prev_R = tmp_R; tmp_R.set(0.0);
    // Z-Axis * Y-Axis * X-Axis Rotation
    tmp_R(1, 1) = 1;
    tmp_R(0, 0) = tmp_R(2, 2) = std::cos(angles[1]);
    tmp_R(2, 0) = tmp_R(0, 2) = std::sin(angles[1]);
    tmp_R(2, 0) = -tmp_R(2, 0);
    matrixMatrixMultiplication3d(tmp_R.data(), prev_R.data(), R.data());
    prev_R = R; tmp_R.set(0.0);
    // Z-Axis * Y-Axis * X-Axis Rotation
    tmp_R(2, 2) = 1;
    tmp_R(0, 0) = tmp_R(1, 1) = std::cos(angles[2]);
    tmp_R(1, 0) = tmp_R(0, 1) = std::sin(angles[2]);
    tmp_R(0, 1) = -tmp_R(0, 1);
    matrixMatrixMultiplication3d(tmp_R.data(), prev_R.data(), R.data());
  }

template <typename T>
constexpr void translate_rotate_translate_point(const mn::vec<T,3> &fulcrum, const mn::vec<T,3,3> &rotate, std::array<T,3> point) {
  std::array<T,3> tmp_point;
  for (int d=0;d<3;d++) tmp_point[d] = point[d] - fulcrum[d]; // Translate to rotation fulcrum
  matrixVectorMultiplication3d(rotate.data(), tmp_point.data(), point.data()); // Rotate
  for (int d=0;d<3;d++) point[d] += fulcrum[d]; // Translate back.
}


/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a string.
/// Return retrieved string from JSON or return backup value if not found/is not a string.
std::string CheckString(rapidjson::Value &object, std::string key, std::string backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsString()){
      fmt::print(fg(red), "ERROR: Input [{}] not a string! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetString());
      return object[key.c_str()].GetString();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a number array.
/// Return retrieved double-precision floating point array from JSON or return backup value if not found/is not a number array.
std::vector<std::string> CheckStringArray(rapidjson::Value &object, std::string key, std::vector<std::string> backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (object[key.c_str()].IsArray())
    {
      int dim = object[key.c_str()].GetArray().Size();
      if (dim > 0) {
        if (!object[key.c_str()].GetArray()[0].IsString()){
          fmt::print(fg(red), "ERROR: Input [{}] not an array of strings! Fix and retry. Current type: {}.\n", key, kTypeNames[object[key.c_str()].GetArray()[0].GetType()]);
        }
        else {
          std::vector<std::string> arr;
          for (int d=0; d<dim; d++) assert(object[key.c_str()].GetArray()[d].GetString());
          for (int d=0; d<dim; d++) arr.push_back(object[key.c_str()].GetArray()[d].GetString());
          fmt::print(fg(green), "Input [{}] found: ", key);
          fmt::print(fg(green), " [ ");
          for (int d=0; d<dim; d++) fmt::print(fg(green), " {}, ", arr[d]);
          fmt::print(fg(green), "]\n");
          return arr;
        }
      } else {
        fmt::print(fg(red), "ERROR: Input [{}] is an Array! Populate and retry.\n", key);
      }
    }
    else 
    {
      fmt::print(fg(red), "ERROR: Input [{}] not an Array! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] not in scene file! \n ", key);
    fmt::print(fg(orange), "WARNING: Using default value: [ ");
    for (int d=0; d<backup.size(); d++) fmt::print(fg(orange), " {}, ", backup[d]);
    fmt::print(fg(orange), "]\n");
    return backup;
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: ", key);
  fmt::print(fg(orange), " [ ");
  for (int d=0; d<backup.size(); d++) fmt::print(fg(orange), " {}, ", backup[d]);
  fmt::print(fg(orange), "]\n");
  getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a number array.
/// Return retrieved double-precision floating point array from JSON or return backup value if not found/is not a number array.
template<int dim=3>
mn::vec<PREC, dim> CheckDoubleArray(rapidjson::Value &object, std::string key, mn::vec<PREC,dim> backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (object[key.c_str()].IsArray())
    {
      //assert(object[key.c_str()].IsArray());
      mn::vec<PREC,dim> arr; 
      if (object[key.c_str()].GetArray().Size() != dim) 
      {
        fmt::print(fg(red), "ERROR: Input [{}] must be an array of size [{}]! Fix and retry. Current size: {}.\n", key, dim, object[key.c_str()].GetArray().Size());
      }
      else {
        if (!object[key.c_str()].GetArray()[0].IsNumber()){
          fmt::print(fg(red), "ERROR: Input [{}] not an array of numbers! Fix and retry. Current type: {}.\n", key, kTypeNames[object[key.c_str()].GetArray()[0].GetType()]);
        }
        else {
          for (int d=0; d<dim; d++) assert(object[key.c_str()].GetArray()[d].GetDouble());
          for (int d=0; d<dim; d++) arr[d] = object[key.c_str()].GetArray()[d].GetDouble();
          fmt::print(fg(green), "Input [{}] found: [{}, {}, {}].\n", key, arr[0],arr[1],arr[2]);
          return arr;
        }
      }
    }
    else 
    {
      fmt::print(fg(red), "ERROR: Input [{}] not an Array! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] not in scene file! \n ", key);
    fmt::print(fg(orange), "WARNING: Using default value: [ ");
    for (int d=0; d<dim; d++) fmt::print(fg(orange), " {}, ", backup[d]);
    fmt::print(fg(orange), "]\n");
    return backup;
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: ", key);
  fmt::print(fg(orange), " [ ");
  for (int d=0; d<dim; d++) fmt::print(fg(orange), " {}, ", backup[d]);
  fmt::print(fg(orange), "]\n");
  getchar();
  return backup;
}
/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is an integer array.
/// Return retrieved integer array from JSON or return backup value if not found/is not a double.
template<int dim>
mn::vec<int, dim> CheckIntArray(rapidjson::Value &object, std::string key, mn::vec<int,dim> backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd()) { 
    if (object[key.c_str()].IsArray()) {
      mn::vec<int,dim> arr; 
      if (object[key.c_str()].GetArray().Size() != dim) {
        fmt::print(fg(red), "ERROR: Input [{}] must be an array of size [{}]! Fix and retry. Current size: {}.\n", key, dim, object[key.c_str()].GetArray().Size());
      }
      else {
        if (!object[key.c_str()].GetArray()[0].IsInt()) {
          fmt::print(fg(red), "ERROR: Input [{}] not an array of integers! Fix and retry. Current type: {}.\n", key, kTypeNames[object[key.c_str()].GetArray()[0].GetType()]);
        }
        else {
          for (int d=0; d<dim; d++) assert(object[key.c_str()].GetArray()[d].GetInt());
          for (int d=0; d<dim; d++) arr[d] = object[key.c_str()].GetArray()[d].GetInt();
          fmt::print(fg(green), "Input [{}] found: [{}, {}, {}].\n", key, arr[0],arr[1],arr[2]);
          return arr;
        }
      }
    }
    else {
      fmt::print(fg(red), "ERROR: Input [{}] not an Array! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] not in scene file! \n ", key);
    fmt::print(fg(orange), "WARNING: Using default value: [ ");
    for (int d=0; d<dim; d++) fmt::print(fg(orange), " {}, ", backup[d]);
    fmt::print(fg(orange), "]\n");
    return backup;
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: ", key);
  fmt::print(fg(orange), " [ ");
  for (int d=0; d<dim; d++) fmt::print(fg(orange), " {}, ", backup[d]);
  fmt::print(fg(orange), "]\n");
  getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a double.
/// Return retrieved double from JSON or return backup value if not found/is not a double.
double CheckDouble(rapidjson::Value &object, std::string key, double backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsDouble() && !object[key.c_str()].IsNumber()){
      fmt::print(fg(red), "ERROR: Input [{}] not a number! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetDouble());
      return object[key.c_str()].GetDouble();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a booelean.
/// Return retrieved boolean from JSON or return backup value if not found/is not a boolean.
bool CheckBool(rapidjson::Value &object, std::string key, bool backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsBool()){
      fmt::print(fg(red), "ERROR: Input [{}] not a boolean (true/false! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetBool());
      return object[key.c_str()].GetBool();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a float.
/// Return retrieved float from JSON or return backup value if not found/is not a float.
float CheckFloat(rapidjson::Value &object, std::string key, float backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    assert(object[key.c_str()].IsFloat());
    return object[key.c_str()].GetFloat();
  }
  else return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is an int.
/// Return retrieved int from JSON or return backup value if not found/is not an int.
int CheckInt(rapidjson::Value &object, std::string key, int backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsNumber()){
      fmt::print(fg(red), "ERROR: Input [{}] not a number! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetInt());
      return object[key.c_str()].GetInt();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is an uint32.
/// Return retrieved int from JSON or return backup value if not found/is not an uint32.
uint32_t CheckUint(rapidjson::Value &object, std::string key, uint32_t backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsNumber() && !object[key.c_str()].IsUint() ){
      fmt::print(fg(red), "ERROR: Input [{}] not an 32-bit Unsigned Integer! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetUint());
      return object[key.c_str()].GetUint();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is an uint64.
/// Return retrieved int from JSON or return backup value if not found/is not an uint64.
uint64_t CheckUint64(rapidjson::Value &object, std::string key, uint64_t backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsNumber() && !object[key.c_str()].IsUint64() ){
      fmt::print(fg(red), "ERROR: Input [{}] not a 64-bit Unsigned Integer! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetUint64());
      return object[key.c_str()].GetUint64();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  getchar();
  return backup;
}

/// @brief Parses an input JSON script to set-up a Multi-GPU simulation.
/// @param fn Filename of input JSON script. Default: scene.json in current working directory
/// @param benchmark Simulation object to initalize. Calls GPU/Host functions and manages memory.
/// @param models Contains initial particle positions for simulation. One per GPU.
void parse_scene(std::string fn,
                 std::unique_ptr<mn::mgsp_benchmark> &benchmark,
                 std::vector<std::array<PREC, 3>> models[mn::config::g_device_cnt]) {
  fs::path p{fn};
  if (p.empty()) fmt::print(fg(red), "ERROR: Input file[{}] does not exist.\n", fn);
  else {
    std::size_t size=0;
    try{
      size = fs::file_size(p);
    } catch(fs::filesystem_error& e) { std::cout << e.what() << '\n'; }
    std::string configs;
    configs.resize(size);
    std::ifstream istrm(fn);
    if (!istrm.is_open())  fmt::print(fg(red), "ERROR: Cannot open file[{}]\n", fn);
    else istrm.read(const_cast<char *>(configs.data()), configs.size());
    istrm.close();
    fmt::print(fg(green), "Opened scene file[{}].\n", fn);
    
    mn::vec<PREC, 3> domain;

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

          PREC sim_default_dx = CheckDouble(sim, std::string{"default_dx"}, 0.1);
          float sim_default_dt = (float)CheckDouble(sim, std::string{"default_dt"}, sim_default_dx/100.);
          uint64_t sim_fps = CheckUint64(sim, std::string{"fps"}, 60);
          uint64_t sim_frames = CheckUint64(sim, std::string{"frames"}, 60);
          float sim_gravity = (float)CheckDouble(sim, std::string{"gravity"}, -9.81);
          
          auto save_suffix_check = sim.FindMember("save_suffix");
          if (save_suffix_check != sim.MemberEnd()) save_suffix = sim["save_suffix"].GetString();
          else save_suffix = std::string{".bgeo"};

          l = sim_default_dx * mn::config::g_dx_inv_d; 
          domain = CheckDoubleArray(sim, std::string{"domain"}, mn::pvec3{1.,1.,1.});
          if (domain[0] > (l-16*sim_default_dx) || domain[1] > (l-16*sim_default_dx) || domain[2] > (l-16*sim_default_dx)) {
            fmt::print(fg(red), "ERROR: Simulation domain[{},{},{}] exceeds max domain length[{}]\n", domain[0], domain[1], domain[2], (l-16*sim_default_dx));
            fmt::print(fg(yellow), "TIP: Shrink domain, grow default_dx, and/or increase DOMAIN_BITS (settings.h) and recompile. Press Enter to continue...\n" ); getchar();
          } 
          fmt::print(fg(cyan),
              "Scene simulation: Domain Length [{}], default_dx[{}], default_dt[{}], fps[{}], frames[{}], gravity[{}]\n",
              l, sim_default_dx, sim_default_dt,
              sim_fps, sim_frames, sim_gravity);
          benchmark = std::make_unique<mn::mgsp_benchmark>(
              l, sim_default_dt,
              sim_fps, sim_frames, sim_gravity, save_suffix);
          fmt::print(fmt::emphasis::bold,
              "-----------------------------------------------------------"
              "-----\n");
        }
      }
    } ///< End basic simulation scene parsing
    {
      auto it = doc.FindMember("meshes");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(cyan),"Scene file has [{}] Finite Element meshes.\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
            if (model["gpu"].GetInt() >= mn::config::g_device_cnt) {
              fmt::print(fg(red),
                       "ERROR! Mesh model GPU[{}] exceeds global device count (settings.h)! Skipping mesh.\n", 
                       model["gpu"].GetInt());
              continue;
            }
            std::string constitutive{model["constitutive"].GetString()};
            fmt::print(fg(green),
                       "Mesh model using constitutive[{}], file_elements[{}], file_vertices[{}].\n", constitutive,
                       model["file_elements"].GetString(), model["file_vertices"].GetString());
            fs::path p{model["file"].GetString()};
            
            std::vector<std::string> output_attribs;
            for (int d = 0; d < model["output_attribs"].GetArray().Size(); ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
            std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
                      
            mn::config::AlgoConfigs algoConfigs;
            algoConfigs.use_FEM = CheckBool(model,std::string{"use_FEM"}, false);
            algoConfigs.use_ASFLIP = CheckBool(model, std::string{"use_ASFLIP"}, true);
            algoConfigs.ASFLIP_alpha = CheckDouble(model, std::string{"alpha"}, 0.);
            algoConfigs.ASFLIP_beta_min = CheckDouble(model, std::string{"beta_min"}, 0.);
            algoConfigs.ASFLIP_beta_max = CheckDouble(model, std::string{"beta_max"}, 0.);
            algoConfigs.use_FBAR = CheckBool(model, std::string{"use_FBAR"}, true);
            algoConfigs.FBAR_ratio = CheckDouble(model, std::string{"FBAR_ratio"}, 0.25);


            mn::config::MaterialConfigs materialConfigs;
            materialConfigs.ppc = model["ppc"].GetDouble();
            materialConfigs.rho = model["rho"].GetDouble();

            auto initModel = [&](auto &positions, auto &velocity, auto &vertices, auto &elements, auto &attribs)
            {
              if (constitutive == "Meshed") 
              {
                materialConfigs.E = model["youngs_modulus"].GetDouble(); 
                materialConfigs.nu = model["poisson_ratio"].GetDouble();

                // Initialize FEM model on GPU arrays
                if (model["use_FBAR"].GetBool() == true)
                {
                  std::cout << "Initialize FEM FBAR Model." << '\n';
                  benchmark->initFEM<mn::fem_e::Tetrahedron_FBar>(model["gpu"].GetInt(), vertices, elements, attribs);
                  benchmark->initModel<mn::material_e::Meshed>(model["gpu"].GetInt(), positions, velocity); //< Initalize particle model

                  std::cout << "Initialize Mesh Parameters." << '\n';
                  benchmark->updateMeshedFBARParameters(
                    model["gpu"].GetInt(),
                    materialConfigs, algoConfigs,
                    output_attribs); //< Update particle material with run-time inputs
                  fmt::print(fg(green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else if (model["use_FBAR"].GetBool() == false)
                {
                  std::cout << "Initialize FEM Model." << '\n';
                  benchmark->initFEM<mn::fem_e::Tetrahedron>(model["gpu"].GetInt(), vertices, elements, attribs);
                  benchmark->initModel<mn::material_e::Meshed>(model["gpu"].GetInt(), positions, velocity); //< Initalize particle model

                  std::cout << "Initialize Mesh Parameters." << '\n';
                  benchmark->updateMeshedParameters(
                    model["gpu"].GetInt(),
                    materialConfigs, algoConfigs,
                    output_attribs); //< Update particle material with run-time inputs
                  fmt::print(fg(green),"GPU[{}] Mesh material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else 
                {
                  fmt::print(fg(red),
                       "ERROR: GPU[{}] Improper/undefined settings for material [{}] with: use_ASFLIP[{}], use_FEM[{}], and use_FBAR[{}]! \n", 
                       model["gpu"].GetInt(), constitutive,
                       model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool());
                  fmt::print(fg(red), "Press Enter to continue...");
                  getchar();
                }
              }
              else 
              {
                fmt::print(fg(red),
                      "ERROR: GPU[{}] No material [{}] implemented for finite element meshes! \n", 
                      model["gpu"].GetInt(), constitutive);
                fmt::print(fg(red), "Press Enter to continue...");
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


            // * NOTE : Assumes geometry "file" specified by scene.json is in AssetDirPath/, e.g. ~/claymore/Data/file
            std::string elements_fn = std::string(AssetDirPath) + model["file_elements"].GetString();
            std::string vertices_fn = std::string(AssetDirPath) + model["file_vertices"].GetString();

            fmt::print(fg(blue),"GPU[{}] Load FEM elements file[{}]...", model["gpu"].GetInt(),  elements_fn);
            load_FEM_Elements(elements_fn, ',', h_FEM_Elements);
            
            std::vector<std::array<PREC, 6>> h_FEM_Element_Attribs(mn::config::g_max_fem_element_num, 
                                                            std::array<PREC, 6>{0., 0., 0., 0., 0., 0.});
            
            fmt::print(fg(blue),"GPU[{}] Load FEM vertices file[{}]...", model["gpu"].GetInt(),  vertices_fn);
            load_FEM_Vertices(vertices_fn, ',', h_FEM_Vertices,
                              offset);
            fmt::print(fg(blue),"GPU[{}] Load FEM-MPM particle file[{}]...", model["gpu"].GetInt(),  vertices_fn);
            load_csv_particles(model["file_vertices"].GetString(), ',', 
                                models[model["gpu"].GetInt()], 
                                offset);

            // Initialize particle and finite element model
            initModel(models[model["gpu"].GetInt()], velocity, h_FEM_Vertices, h_FEM_Elements, h_FEM_Element_Attribs);
            fmt::print(fmt::emphasis::bold,
                      "-----------------------------------------------------------"
                      "-----\n");
          }
        }
      }
    } ///< end mesh parsing
    {
      auto it = doc.FindMember("models");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(cyan), "Scene file has [{}] particle models. \n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
            int gpu_id = model["gpu"].GetInt();
            if (gpu_id >= mn::config::g_device_cnt) {
              fmt::print(fg(red), "ERROR! Particle model gpu[{}] exceeds GPUs reserved by g_device_cnt (settings.h)! Skipping model. Increase g_device_cnt and recompile. \n", gpu_id);
              continue;
            } 
            else if (gpu_id < 0) {
              fmt::print(fg(red), "ERROR! GPU ID[{}] cannot be negative. \n", gpu_id);
              continue;
            }
            //std::string constitutive{model["constitutive"].GetString()};
            std::string constitutive = CheckString(model, std::string{"constitutive"}, std::string{"JFluid"});
            fmt::print(fg(green),  "GPU[{}] Read model constitutive[{}].\n", gpu_id, constitutive);
            std::vector<std::string> output_attribs;
            std::vector<std::string> input_attribs;
            std::vector<std::string> target_attribs;
            std::vector<std::string> track_attribs;
            mn::vec<int, 1> track_particle_id;
            bool has_attributes = false;
            std::vector<std::vector<PREC>> attributes; //< Initial attributes (not incl. position)

            mn::config::AlgoConfigs algoConfigs;
            algoConfigs.use_FEM = CheckBool(model,std::string{"use_FEM"}, false);
            algoConfigs.use_ASFLIP = CheckBool(model, std::string{"use_ASFLIP"}, true);
            algoConfigs.ASFLIP_alpha = CheckDouble(model, std::string{"alpha"}, 0.);
            algoConfigs.ASFLIP_beta_min = CheckDouble(model, std::string{"beta_min"}, 0.);
            algoConfigs.ASFLIP_beta_max = CheckDouble(model, std::string{"beta_max"}, 0.);
            algoConfigs.use_FBAR = CheckBool(model, std::string{"use_FBAR"}, true);
            algoConfigs.FBAR_ratio = CheckDouble(model, std::string{"FBAR_ratio"}, 0.25);

            mn::config::MaterialConfigs materialConfigs;
            materialConfigs.ppc = CheckDouble(model, std::string{"ppc"}, 8.0); 
            materialConfigs.rho = CheckDouble(model, std::string{"rho"}, 1e3); 

            auto initModel = [&](auto &positions, auto &velocity) {
              bool algo_error = false, mat_error  = false;
              if (constitutive == "JFluid" || constitutive == "J-Fluid" || constitutive == "J_Fluid" || constitutive == "J Fluid" ||  constitutive == "jfluid" || constitutive == "j-fluid" || constitutive == "j_fluid" || constitutive == "j fluid" || constitutive == "Fluid" || constitutive == "fluid" || constitutive == "Water" || constitutive == "Liquid") {
                materialConfigs.bulk = CheckDouble(model, std::string{"bulk_modulus"}, 2e7); 
                materialConfigs.gamma = CheckDouble(model, std::string{"gamma"}, 7.1); 
                materialConfigs.visco = CheckDouble(model, std::string{"viscosity"}, 0.001);
                if(!algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::JFluid>(gpu_id, positions, velocity);                    
                  benchmark->updateParameters<mn::material_e::JFluid>( 
                      gpu_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_id[0], track_attribs, target_attribs);
                }
                else if (algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::JFluid_ASFLIP>(gpu_id, positions, velocity);                    
                  benchmark->updateParameters<mn::material_e::JFluid_ASFLIP>( 
                      gpu_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_id[0], track_attribs, target_attribs);
                }
                else if (algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::JBarFluid>(gpu_id, positions, velocity);
                  benchmark->updateParameters<mn::material_e::JBarFluid>( 
                      gpu_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_id[0], track_attribs, target_attribs);
                } 
                else if (!algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::JFluid_FBAR>(gpu_id, positions, velocity);
                  benchmark->updateParameters<mn::material_e::JFluid_FBAR>( 
                      gpu_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_id[0], track_attribs, target_attribs);
                }
                else { algo_error = true; }
              } 
              else if (constitutive == "FixedCorotated" || constitutive == "Fixed_Corotated" || constitutive == "Fixed-Corotated" || constitutive == "Fixed Corotated" || constitutive == "fixedcorotated" || constitutive == "fixed_corotated" || constitutive == "fixed-corotated"|| constitutive == "fixed corotated") {
                materialConfigs.E = CheckDouble(model, std::string{"youngs_modulus"}, 1e7);
                materialConfigs.nu = CheckDouble(model, std::string{"poisson_ratio"}, 0.2);
                if(!algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::FixedCorotated>(gpu_id, positions, velocity);
                  benchmark->updateParameters<mn::material_e::FixedCorotated>(
                      gpu_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_id[0], track_attribs, target_attribs);
                }
                else if (algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::FixedCorotated_ASFLIP>(gpu_id, positions, velocity);
                  benchmark->updateParameters<mn::material_e::FixedCorotated_ASFLIP>(
                      gpu_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_id[0], track_attribs, target_attribs);
                }
                else if (algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::FixedCorotated_ASFLIP_FBAR>(gpu_id, positions, velocity);
                  benchmark->updateParameters<mn::material_e::FixedCorotated_ASFLIP_FBAR>(
                      gpu_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_id[0], track_attribs, target_attribs);
                }
                else { algo_error = true; }
              } 
              else if (constitutive == "NeoHookean" || constitutive == "neohookean" || 
                      constitutive == "Neo-Hookean" || constitutive == "neo-hookean") {
                materialConfigs.E = CheckDouble(model, std::string{"youngs_modulus"}, 1e7); 
                materialConfigs.nu = CheckDouble(model, std::string{"poisson_ratio"}, 0.2);
                if (algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::NeoHookean_ASFLIP_FBAR>(gpu_id, positions, velocity);
                  benchmark->updateParameters<mn::material_e::NeoHookean_ASFLIP_FBAR>( 
                      gpu_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_id[0], track_attribs, target_attribs);
                }
                else { algo_error = true; }
              } 
              else if (constitutive == "Sand" || constitutive == "sand" || constitutive == "DruckerPrager" || constitutive == "Drucker_Prager" || constitutive == "Drucker-Prager" || constitutive == "Drucker Prager") { 
                materialConfigs.E = CheckDouble(model, std::string{"youngs_modulus"}, 1e7); 
                materialConfigs.nu = CheckDouble(model, std::string{"poisson_ratio"}, 0.2);
                materialConfigs.logJp0 = CheckDouble(model, std::string{"logJp0"}, 0.0);
                materialConfigs.frictionAngle = CheckDouble(model, std::string{"friction_angle"}, 30.0);
                materialConfigs.cohesion = CheckDouble(model, std::string{"cohesion"}, 0.0);
                materialConfigs.beta = CheckDouble(model, std::string{"beta"}, 0.5);
                materialConfigs.volumeCorrection = CheckBool(model, std::string{"SandVolCorrection"}, true); 
                if (algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::Sand>(gpu_id, positions, velocity); 
                  benchmark->updateParameters<mn::material_e::Sand>( 
                        gpu_id, materialConfigs, algoConfigs,
                        output_attribs, track_particle_id[0], track_attribs, target_attribs);
                }
                else { algo_error = true; }
              } 
              else if (constitutive == "NACC" || constitutive == "nacc" || constitutive == "CamClay" || constitutive == "Cam_Clay" || constitutive == "Cam-Clay" || constitutive == "Cam Clay") {
                materialConfigs.E = CheckDouble(model, std::string{"youngs_modulus"}, 1e7); 
                materialConfigs.nu = CheckDouble(model, std::string{"poisson_ratio"}, 0.2);
                materialConfigs.logJp0 = CheckDouble(model, std::string{"logJp0"}, 0.0);
                materialConfigs.xi = CheckDouble(model, std::string{"xi"}, 0.8);
                materialConfigs.frictionAngle = CheckDouble(model, std::string{"friction_angle"}, 30.0);
                materialConfigs.beta = CheckDouble(model, std::string{"beta"}, 0.5);
                materialConfigs.hardeningOn = CheckBool(model, std::string{"hardeningOn"}, true); 
                if (algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::NACC>(gpu_id, positions, velocity);
                  benchmark->updateParameters<mn::material_e::NACC>( 
                        gpu_id, materialConfigs, algoConfigs,
                        output_attribs, track_particle_id[0], track_attribs, target_attribs);
                }
                else { algo_error = true; }
              } 
              else { mat_error = true; } //< Requested material doesn't exist in code
              if (mat_error) {
                fmt::print(fg(red),"ERROR: GPU[{}] constititive[{}] does not exist! Press ENTER to continue...\n",  gpu_id, constitutive); getchar(); return; 
              }
              if (algo_error) {
                fmt::print(fg(red), "ERROR: GPU[{}] Undefined algorithm use for material [{}]: use_ASFLIP[{}], use_FEM[{}], use_FBAR[{}]! Press ENTER to continue...\n", gpu_id, constitutive, algoConfigs.use_ASFLIP, algoConfigs.use_FEM, algoConfigs.use_FBAR); getchar(); return; 
              }
              fmt::print(fg(green),"GPU[{}] Material[{}] model set and updated.\n", gpu_id, constitutive);
            };

            mn::vec<PREC, 3> velocity, partition_start, partition_end;
            velocity = CheckDoubleArray(model, std::string{"velocity"}, mn::pvec3{0.,0.,0.});
            partition_start = CheckDoubleArray(model, std::string{"partition_start"}, mn::pvec3{0.,0.,0.});
            partition_end = CheckDoubleArray(model, std::string{"partition_end"}, domain);
            for (int d = 0; d < 3; ++d) {
              velocity[d] = velocity[d] / l;
              partition_start[d]  = partition_start[d] / l + o;
              partition_end[d]  = partition_end[d] / l + o;
              if (partition_start[d] > partition_end[d]) {
                fmt::print(fg(red), "GPU[{}] ERROR: Inverted partition (Element of partition_start > partition_end). Fix and Retry.", gpu_id); getchar();
              } else if (partition_end[d] == partition_start[d]) {
                fmt::print(fg(red), "GPU[{}] ERROR: Zero volume partition (Element of partition_end == partition_start). Fix and Retry.", gpu_id); getchar();
              }
            }
            output_attribs = CheckStringArray(model, std::string{"output_attribs"}, std::vector<std::string> {{"ID"}});
            track_attribs = CheckStringArray(model, std::string{"track_attribs"}, std::vector<std::string> {{"Position_Y"}});
            track_particle_id = CheckIntArray(model, std::string{"track_particle_id"}, mn::vec<int, 1> {0});
            target_attribs = CheckStringArray(model, std::string{"target_attribs"}, std::vector<std::string> {{"Position_Y"}});
            if (output_attribs.size() > mn::config::g_max_particle_attribs) { fmt::print(fg(red), "ERROR: GPU[{}] Only [{}] output_attribs value supported.\n", gpu_id, mn::config::g_max_particle_attribs); }
            if (track_attribs.size() > 1) { fmt::print(fg(red), "ERROR: GPU[{}] Only [1] track_attribs value supported currently.\n", gpu_id); }
            if (sizeof(track_particle_id) / sizeof(int) > 1) { fmt::print(fg(red), "ERROR: Only [1] track__particle_id value supported currently.\n"); }
            fmt::print("GPU[{}] Track attributes [{}] on particle IDs [{}].\n", gpu_id, track_attribs[0], track_particle_id[0]);
            if (target_attribs.size() > 1) { fmt::print(fg(red), "ERROR: GPU[{}] Only [1] target_attribs value supported currently.\n", gpu_id); }

            // * Begin particle geometry construction 
            auto geo = model.FindMember("geometry");
            if (geo != model.MemberEnd()) {
              if (geo->value.IsArray()) {
                fmt::print(fg(blue),"Model has [{}] particle geometry operations to perform. \n", geo->value.Size());
                for (auto &geometry : geo->value.GetArray()) {
                  // std::string operation{geometry["operation"].GetString()};
                  // std::string type{geometry["object"].GetString()};
                  std::string operation = CheckString(geometry, std::string{"operation"}, std::string{"add"});
                  std::string type = CheckString(geometry, std::string{"object"}, std::string{"box"});
                  fmt::print(fg(white), "GPU[{}] Begin operation[{}] with object[{}]... \n", gpu_id, operation, type);

                  mn::vec<PREC, 3> geometry_offset, geometry_span, geometry_spacing;
                  mn::vec<int, 3> geometry_array;
                  mn::vec<PREC, 3> geometry_rotate, geometry_fulcrum, geometry_shear;
                  geometry_span = CheckDoubleArray(geometry, std::string{"span"}, domain);
                  geometry_offset = CheckDoubleArray(geometry, std::string{"offset"}, mn::pvec3{0.,0.,0.});
                  geometry_array = CheckIntArray(geometry, std::string{"array"}, mn::ivec3{1,1,1});
                  geometry_spacing = CheckDoubleArray(geometry, std::string{"spacing"}, mn::pvec3{0.,0.,0.});
                  geometry_rotate = CheckDoubleArray(geometry, std::string{"rotate"}, mn::pvec3{0.,0.,0.});
                  geometry_fulcrum = CheckDoubleArray(geometry, std::string{"fulcrum"}, mn::pvec3{0.,0.,0.});           
                  for (int d = 0; d < 3; ++d) {
                    geometry_span[d]    = geometry_span[d]    / l;
                    geometry_offset[d]  = geometry_offset[d]  / l + o;
                    geometry_spacing[d] = geometry_spacing[d] / l;                    
                    geometry_fulcrum[d] = geometry_fulcrum[d] / l + o;
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
                      make_box(models[gpu_id], geometry_span, geometry_offset_updated, materialConfigs.ppc, partition_start, partition_end); }
                    else if (operation == "Subtract" || operation == "subtract") {
                      subtract_box(models[gpu_id], geometry_span, geometry_offset_updated); }
                    else if (operation == "Union" || operation == "union") { fmt::print(fg(red),"Operation not implemented...\n");}
                    else if (operation == "Intersect" || operation == "intersect") { fmt::print(fg(red),"Operation not implemented...\n");}
                    else if (operation == "Difference" || operation == "difference") { fmt::print(fg(red),"Operation not implemented...\n");}
                    else { fmt::print(fg(red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", gpu_id, operation); getchar(); }
                  }
                  else if (type == "Cylinder" || type == "cylinder")
                  {
                    PREC geometry_radius = CheckDouble(geometry, std::string{"radius"}, 0.);
                    std::string geometry_axis = CheckString(geometry, std::string{"axis"}, std::string{"X"});

                    if (operation == "Add" || operation == "add") {
                      make_cylinder(models[gpu_id], geometry_span, geometry_offset_updated, materialConfigs.ppc, geometry_radius, geometry_axis, partition_start, partition_end); }
                    else if (operation == "Subtract" || operation == "subtract") {             subtract_cylinder(models[gpu_id], geometry_radius, geometry_axis, geometry_span, geometry_offset_updated); }
                    else { fmt::print(fg(red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", gpu_id, operation); getchar(); }
                  }
                  else if (type == "Sphere" || type == "sphere")
                  {
                    PREC geometry_radius = CheckDouble(geometry, std::string{"radius"}, 0.);
                    if (operation == "Add" || operation == "add") {
                      make_sphere(models[gpu_id], geometry_span, geometry_offset_updated, materialConfigs.ppc, geometry_radius, partition_start, partition_end); }
                    else if (operation == "Subtract" || operation == "subtract") {
                      subtract_sphere(models[gpu_id], geometry_radius, geometry_offset_updated); }
                    else {  fmt::print(fg(red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", gpu_id, operation); getchar(); }
                  }
                  else if (type == "OSU LWF" || type == "OSU Water")
                  {
                    if (operation == "Add" || operation == "add") {
                      make_OSU_LWF(models[gpu_id], geometry_span, geometry_offset_updated, materialConfigs.ppc, partition_start, partition_end); }
                    else if (operation == "Subtract" || operation == "subtract") { fmt::print(fg(red),"Operation not implemented yet...\n"); }
                    else { fmt::print(fg(red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", gpu_id, operation); getchar(); }
                  }
                  else if (type == "File" || type == "file") 
                  {
                    // * NOTE : Assumes geometry "file" specified by scene.json is in  AssetDirPath/, e.g. for AssetDirPath = ~/claymore/Data/, then use ~/claymore/Data/file
                    std::string geometry_file = CheckString(geometry, std::string{"file"}, std::string{"MpmParticles/yoda.sdf"});
                    std::string geometry_fn = std::string(AssetDirPath) + geometry_file;
                    fs::path geometry_file_path{geometry_fn};
                    if (geometry_file_path.empty()) fmt::print(fg(red), "ERROR: Input file[{}] does not exist.\n", geometry_fn);
                    else {
                      std::ifstream istrm(geometry_fn);
                      if (!istrm.is_open())  fmt::print(fg(red), "ERROR: Cannot open file[{}]\n", geometry_fn);
                      istrm.close();
                    }
                    if (operation == "Add" || operation == "add") {
                      if (geometry_file_path.extension() == ".sdf") 
                      {
                        PREC geometry_scaling_factor = CheckDouble(geometry, std::string{"scaling_factor"}, 1);
                        int geometry_padding = CheckInt(geometry, std::string{"padding"}, 1);
                        if (geometry_scaling_factor <= 0) {
                          fmt::print(fg(red), "ERROR: [scaling_factor] must be greater than [0] for SDF file load (e.g. [2] doubles size, [0] erases size). Fix and Retry.\n"); getchar(); }
                        if (geometry_padding < 1) {
                          fmt::print(fg(red), "ERROR: Signed-Distance-Field (.sdf) files require [padding] of atleast [1] (padding is empty exterior cells on sides of model, allows surface definition). Fix and Retry.");fmt::print(fg(yellow), "TIP: Use open-source SDFGen to create *.sdf from *.obj files.\n"); getchar();}
                        mn::read_sdf(geometry_fn, models[gpu_id], materialConfigs.ppc,
                            (PREC)dx, mn::config::g_domain_size, geometry_offset_updated, l,
                            partition_start, partition_end, geometry_scaling_factor, geometry_padding);
                      }
                      else if (geometry_file_path.extension() == ".csv") 
                      {
                        load_csv_particles(geometry_fn, ',', 
                                            models[gpu_id], geometry_offset_updated, 
                                            partition_start, partition_end);
                      }
                      else if (geometry_file_path.extension() == ".bgeo" ||
                          geometry_file_path.extension() == ".geo" ||
                          geometry_file_path.extension() == ".pdb" ||
                          geometry_file_path.extension() == ".ptc") 
                      {
                        has_attributes = CheckBool(geometry, std::string{"has_attributes"}, false);
                        input_attribs = CheckStringArray(geometry, std::string{"input_attribs"}, std::vector<std::string> {{"ID"}});
                        if (has_attributes) fmt::print(fg(white),"GPU[{}] Try to read pre-existing particle attributes into model? [{}].\n", gpu_id, has_attributes);
                        if (input_attribs.size() > mn::config::g_max_particle_attribs) {
                          fmt::print(fg(red), "ERROR: GPU[{}] Model suppports max of [{}] input_attribs, but [{}] are specified.\n", gpu_id, mn::config::g_max_particle_attribs, input_attribs.size()); getchar();
                        }
                        attributes.resize(0, std::vector<PREC>(input_attribs.size()));
                        mn::read_partio_general<PREC>(geometry_fn, models[gpu_id], attributes, input_attribs.size(), input_attribs); 
                        fmt::print("Size of attributes after reading in initial data: Particles {}, Attributess {}\n", attributes.size(), attributes[0].size());
                        fmt::print("First element: {} \n", attributes[0][0]);

                        // Scale particle positions to 1x1x1 simulation
                        for (int part=0; part < models[gpu_id].size(); part++) {
                          for (int d = 0; d<3; d++) {
                            models[gpu_id][part][d] = models[gpu_id][part][d] / l + o;
                          }
                          // Scale length based attributes to 1x1x1 simulation (e.g. Velocity)
                          for (int d = 0; d < input_attribs.size(); d++) {
                            if (input_attribs[d] == "Velocity_X" || input_attribs[d] == "Velocity_Y" || input_attribs[d] == "Velocity_Z" )
                              attributes[part][d] = attributes[part][d] / l; 
                          }
                        }
                      }
                    }
                    else if (operation == "Subtract" || operation == "subtract") { fmt::print(fg(red),"Operation not implemented...\n"); }
                    else if (operation == "Union" || operation == "union") {fmt::print(fg(red),"Operation not implemented...\n");}
                    else if (operation == "Intersect" || operation == "intersect") {fmt::print(fg(red),"Operation not implemented...\n");}
                    else if (operation == "Difference" || operation == "difference") {fmt::print(fg(red),"Operation not implemented...\n");}
                    else { fmt::print(fg(red), "ERROR: GPU[{}] geometry operation[{}] invalid! Press ENTER to continue...\n", gpu_id, operation); getchar(); 
                    }
                  }
                  else  { fmt::print(fg(red), "GPU[{}] ERROR: Geometry object[{}] does not exist! Press ENTER to continue...\n", gpu_id, type); getchar();
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
              fmt::print(fg(red), "ERROR: GPU[{}] Model has no geometry object! Neccesary to create particles.\n", gpu_id);
              fmt::print(fg(red), "Press enter to continue...\n"); getchar();
            }
              
            auto positions = models[gpu_id];
            mn::IO::insert_job([&]() {
              mn::write_partio<PREC,3>(std::string{p.stem()} + save_suffix,positions); });              
            mn::IO::flush();
            fmt::print(fg(green), "GPU[{}] Saved particle model to [{}].\n", gpu_id, std::string{p.stem()} + save_suffix);
            
            if (positions.size() > mn::config::g_max_particle_num) {
              fmt::print(fg(red), "ERROR: GPU[{}] Particle count [{}] exceeds g_max_particle_num in settings.h! Increase and recompile to avoid problems. \n", gpu_id, positions.size());
              fmt::print(fg(red), "Press ENTER to continue anyways... \n");
              getchar();
            }

            // * Initialize particle positions in simulator and on GPU
            initModel(positions, velocity);

            // ! Chose to hard-code available attribute count per particle for input, output
            // ! Gives clearer code/better compiler optimization over run-time binding for GPU data-structures, but could definitely be improved.

            // * Initialize particle attributes in simulator and on GPU
            if (!has_attributes) attributes = std::vector<std::vector<PREC> >(positions.size(), std::vector<PREC>(input_attribs.size(), 0.)); //< Zero initial attribs if none
            if (input_attribs.size() == 1){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(1);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 2){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(2);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 3){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(3);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 4){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(4);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 5){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(5);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 6){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(6);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 7){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(7);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 8){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(8);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 9){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(9);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 10){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(10);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 11){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(11);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 12){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(12);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 13){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(13);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 14){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(14);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 15){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(15);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 16){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(16);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() <= 18 && input_attribs.size() > 16){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(18);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() <= 24 && input_attribs.size() > 18){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(24);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() <= 32 && input_attribs.size() > 24){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(32);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            } else if (input_attribs.size() > 32){
              fmt::print("More than [{}] input_attribs not implemented. You requested [{}].", mn::config::g_max_particle_attribs, input_attribs.size());
              getchar();
            } else {
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(1);
              benchmark->initInitialAttribs<N>(gpu_id, attributes, has_attributes); 
            }
            
            // * Initialize output particle attributes in simulator and on GPU
            attributes = std::vector<std::vector<PREC> >(positions.size(), std::vector<PREC>(output_attribs.size(), 0.));
            if (output_attribs.size() == 1){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(1);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 2){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(2);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 3){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(3);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 4){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(4);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 5){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(5);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 6){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(6);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 7){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(7);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 8){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(8);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 9){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(9);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 10){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(10);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 11){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(11);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 12){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(12);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 13){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(13);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 14){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(14);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 15){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(15);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() == 16){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(16);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() <= 18 && output_attribs.size() > 16){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(18);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() <= 24 && output_attribs.size() > 18){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(24);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() <= 32 && output_attribs.size() > 24){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(32);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else if (output_attribs.size() > 32){
              fmt::print(fg(red), "ERROR: GPU[{}] More than [{}] output_attribs not valid. Requested: [{}]. Truncating...", gpu_id, mn::config::g_max_particle_attribs, output_attribs.size()); 
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(32);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            } else {
              fmt::print(fg(orange), "WARNING: GPU[{}] output_attribs not found. Using [1] element default", gpu_id);
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(1);
              benchmark->initOutputAttribs<N>(gpu_id, attributes); 
            }
            fmt::print(fmt::emphasis::bold,
                      "-----------------------------------------------------------"
                      "-----\n");
          }
        }
      }
    } ///< end models parsing
    {
      auto it = doc.FindMember("grid-targets");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(cyan),"Scene has [{}] grid-targets.\n", it->value.Size());
          int target_ID = 0;
          for (auto &model : it->value.GetArray()) {
          
            std::vector<std::array<PREC_G, mn::config::g_target_attribs>> h_gridTarget(mn::config::g_target_cells, std::array<PREC_G, mn::config::g_target_attribs> {0,0,0,0,0,0,0,0,0,0});
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
                fmt::print(fg(red), "ERROR: gridTarget[{}] has invalid attribute[{}].\n", target_ID, attribute);
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
              fmt::print(fg(red), "ERROR: gridTarget[{}] has invalid direction[{}].\n", target_ID, direction);
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

            mn::config::GridTargetConfigs gridTargetConfigs((int)target[6], (int)target[6], (int)target[6], make_float3((float)target[1], (float)target[2], (float)target[3]), make_float3((float)target[4], (float)target[5], (float)target[6]), (float)freq);

            // * Loop through GPU devices to initialzie
            for (int did = 0; did < mn::config::g_device_cnt; ++did) {
              benchmark->initGridTarget(did, h_gridTarget, target, 
                freq); // TODO : Allow more than one frequency for grid-targets
            fmt::print(fg(green), "GPU[{}] gridTarget[{}] Initialized.\n", did, target_ID);
            }
            target_ID += 1;
            fmt::print(fmt::emphasis::bold,
                      "-----------------------------------------------------------"
                      "-----\n");
          }
        }
      }
    } ///< End grid-target parsing
    {
      auto it = doc.FindMember("particle-targets");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(cyan),"Scene has [{}] particle-targets.\n", it->value.Size());
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
              fmt::print(fg(red), "ERROR: particleTarget[{}] has invalid operation[{}].\n", target_ID, operation);
              getchar();
            }
            // Load and scale target domain to 1 x 1 x 1 domain + off-by-2 offset
            for (int d = 0; d < 3; ++d) 
            {
              target[d+1] = model["domain_start"].GetArray()[d].GetFloat() / l + o;
              target[d+4] = model["domain_end"].GetArray()[d].GetFloat() / l + o;
            }

            PREC freq = model["output_frequency"].GetDouble();

            mn::config::ParticleTargetConfigs particleTargetConfigs((int)target[6], (int)target[6], (int)target[6], {(float)target[1], (float)target[2], (float)target[3]}, {(float)target[4], (float)target[5], (float)target[6]}, (float)freq);


            // Initialize on GPUs
            for (int did = 0; did < mn::config::g_device_cnt; ++did) {
              benchmark->initParticleTarget(did, h_particleTarget, target, 
                freq);
              fmt::print(fg(green), "GPU[{}] particleTarget[{}] Initialized.\n", did, target_ID);
            }
            target_ID += 1; // TODO : Count targets using static variable in a structure
            fmt::print(fmt::emphasis::bold,
                      "-----------------------------------------------------------"
                      "-----\n");
          }
        }
      }
    } ///< End particle-target parsing
    {
      auto it = doc.FindMember("grid-boundaries");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(cyan), "Scene has [{}] grid-boundaries.\n", it->value.Size());
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
              fmt::print(fg(red), "ERROR: gridBoundary[{}] object[{}] or contact[{}] is not valid! \n", boundary_ID, object, contact);
              h_boundary[6] = -1;
            }

            // Set up moving grid-boundary if applicable
            auto motion_file = model.FindMember("file"); // Check for motion file
            auto motion_velocity = model.FindMember("velocity"); // Check for velocity
            if (motion_file != model.MemberEnd()) 
            {
              fmt::print(fg(blue),"Found motion file for grid-boundary[{}]. Loading... \n", boundary_ID);
              MotionHolder motionPath;
              std::string motion_fn = std::string(AssetDirPath) + model["file"].GetString();
              fs::path motion_file_path{motion_fn};
              if (motion_file_path.empty()) fmt::print(fg(red), "ERROR: Input file[{}] does not exist.\n", motion_fn);
              else {
                std::ifstream istrm(motion_fn);
                if (!istrm.is_open())  fmt::print(fg(red), "ERROR: Cannot open file[{}]\n", motion_fn);
                istrm.close();
              }

              load_motionPath(motion_fn, ',', motionPath);
              
              PREC_G gb_freq = 1;
              auto motion_freq = model.FindMember("output_frequency");
              if (motion_freq != model.MemberEnd()) gb_freq = model["output_frequency"].GetFloat();

              for (int did = 0; did < mn::config::g_device_cnt; ++did) {
                benchmark->initMotionPath(did, motionPath, gb_freq);
                fmt::print(fg(green),"GPU[{}] gridBoundary[{}] motion file[{}] initialized with frequency[{}].\n", did, boundary_ID, model["file"].GetString(), gb_freq);
              }
            }
            else if (motion_velocity != model.MemberEnd() && motion_file == model.MemberEnd())
            {
              fmt::print(fg(blue),"Found velocity for grid-boundary[{}]. Loading...", boundary_ID);
              mn::vec<PREC_G, 3> velocity;
              for (int d=0; d<3; d++) velocity[d] = model["velocity"].GetArray()[d].GetDouble() / l;
            }
            else 
              fmt::print(fg(orange),"No motion file or velocity specified for grid-boundary. Assuming static. \n");
            
            // ----------------  Initialize grid-boundaries ---------------- 
            benchmark->initGridBoundaries(0, h_boundary, boundary_ID);
            fmt::print(fg(green), "Initialized gridBoundary[{}]: object[{}], contact[{}].\n", boundary_ID, object, contact);
            boundary_ID += 1;
            fmt::print(fmt::emphasis::bold,
                      "-----------------------------------------------------------"
                      "-----\n");
          }
        }
      }
    }
  }
} ///< End scene file parsing

#endif