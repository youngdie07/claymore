#ifndef __PARTICLE_IO_HPP_
#define __PARTICLE_IO_HPP_
#include "PoissonDisk/SampleGenerator.h"
#include <MnBase/Math/Vec.h>
#include <Partio.h>
#include <array>
#include <string>
#include <vector>
#include <math.h>

namespace mn {

template <typename T, std::size_t dim>
void write_partio(std::string filename,
                  const std::vector<std::array<T, dim>> &data,
                  std::string tag = std::string{"position"}) {
  Partio::ParticlesDataMutable *parts = Partio::create();

  Partio::ParticleAttribute attrib =
      parts->addAttribute(tag.c_str(), Partio::VECTOR, dim);

  parts->addParticles(data.size());
  for (int idx = 0; idx < (int)data.size(); ++idx) {
    float *val = parts->dataWrite<float>(attrib, idx);
    for (int k = 0; k < dim; k++)
      val[k] = data[idx][k];
  }
  Partio::write(filename.c_str(), *parts);
  parts->release();
}

// Write combined particle position (x,y,z) and attribute (...) data (JB)
template <typename T, std::size_t dim>
void write_partio_particles(std::string filename,
                  const std::vector<std::array<T, 3>>  &positions, 
                  const std::vector<std::array<T,dim>> &attributes) {
  // Create a mutable Partio structure pointer
  Partio::ParticlesDataMutable*       parts = Partio::create();

  // Add positions and attributes to the pointer by arrow operator
  Partio::ParticleAttribute pos     = parts->addAttribute("position", Partio::VECTOR, 3);
  Partio::ParticleAttribute attrib  = parts->addAttribute("attributes", Partio::FLOAT, (int)dim);

  for(int i=0; i < (int)positions.size(); ++i)
  {
    // Create new particle with two write-input vectors/arrays
    int idx  = parts->addParticle();
    float* p = parts->dataWrite<float>(pos,    idx);
    float* a = parts->dataWrite<float>(attrib, idx);

    // Add position data for particle
    for(int k=0; k<3; ++k)
    {
      p[k] = positions[i][k];
    }

    // Add extra attributes for particle
    for(int k=0; k<(int)dim; ++k)
    {
      a[k] = attributes[i][k];
    }
  }

  // Write
  Partio::write(filename.c_str(), *parts);

  // Release (scope-dependent)
  parts->release();
}

/// Write grid data (m, mvx, mvy, mvz) on host to disk as *.bgeo (JB) 
template <typename T, std::size_t dim>
void write_partio_grid(std::string filename,
		       const std::vector<std::array<T, dim>> &data) {
  /// Set mutable particle structure, add attributes
  Partio::ParticlesDataMutable* parts = Partio::create();
  Partio::ParticleAttribute position  = parts->addAttribute("position", Partio::VECTOR, 3); /// Block ID
  Partio::ParticleAttribute mass      = parts->addAttribute("mass",     Partio::FLOAT, 1);  /// Mass
  Partio::ParticleAttribute momentum  = parts->addAttribute("momentum", Partio::VECTOR, 3); /// Momentum
    
  /// Loop over grid-blocks, set values in Partio structure
  for(int i=0; i < (int)data.size(); ++i)
    {
      int idx   = parts->addParticle();
      float* p  = parts->dataWrite<float>(position,idx);
      float* m  = parts->dataWrite<float>(mass,idx);
      float* mv = parts->dataWrite<float>(momentum,idx);

      p[0]  = data[i][0];
      p[1]  = data[i][1];
      p[2]  = data[i][2];
      m[0]  = data[i][3];
      mv[0] = data[i][4];
      mv[1] = data[i][5];
      mv[2] = data[i][6];
    }
  /// Output as *.bgeo
  Partio::write(filename.c_str(), *parts);
  parts->release();
}


/// Write grid data (m, mvx, mvy, mvz) on host to disk as *.bgeo (JB) 
template <typename T, std::size_t dim>
void write_partio_gridTarget(std::string filename,
		       const std::vector<std::array<T, dim>> &data) {
  /// Set mutable particle structure, add attributes
  Partio::ParticlesDataMutable* parts = Partio::create();
  Partio::ParticleAttribute position  = parts->addAttribute("position", Partio::VECTOR, 3); /// Block ID
  Partio::ParticleAttribute mass      = parts->addAttribute("mass",     Partio::FLOAT, 1);  /// Mass
  Partio::ParticleAttribute momentum  = parts->addAttribute("momentum", Partio::VECTOR, 3); /// Momentum
  Partio::ParticleAttribute force     = parts->addAttribute("force",    Partio::VECTOR, 3); /// Force

  /// Loop over grid-blocks, set values in Partio structure
  for(int i=0; i < (int)data.size(); ++i)
    {
      int idx   = parts->addParticle();
      float* p  = parts->dataWrite<float>(position,idx);
      float* m  = parts->dataWrite<float>(mass,idx);
      float* mv = parts->dataWrite<float>(momentum,idx);
      float* f  = parts->dataWrite<float>(force,idx);

      p[0]  = data[i][0];
      p[1]  = data[i][1];
      p[2]  = data[i][2];
      m[0]  = data[i][3];
      mv[0] = data[i][4];
      mv[1] = data[i][5];
      mv[2] = data[i][6];
      f[0]  = data[i][7];
      f[1]  = data[i][8];
      f[2]  = data[i][9];
    }
  /// Output as *.bgeo
  Partio::write(filename.c_str(), *parts);
  parts->release();
}

/// have issues
auto read_sdf(std::string fn, float ppc, float dx, vec<float, 3> offset,
              vec<float, 3> lengths) {
  std::vector<std::array<float, 3>> data;
  std::string fileName = std::string(AssetDirPath) + "MpmParticles/" + fn;

  float levelsetDx;
  SampleGenerator pd;
  std::vector<float> samples;
  vec<float, 3> mins, maxs, scales;
  vec<int, 3> maxns;
  pd.LoadSDF(fileName, levelsetDx, mins[0], mins[1], mins[2], maxns[0],
             maxns[1], maxns[2]);
  maxs = maxns.cast<float>() * levelsetDx;
  scales = lengths / (maxs - mins);
  float scale = scales[0] < scales[1] ? scales[0] : scales[1];
  scale = scales[2] < scale ? scales[2] : scale;

  float samplePerLevelsetCell = ppc * levelsetDx / dx * scale;

  if (0) pd.GenerateUniformSamples(samplePerLevelsetCell, samples);
  if (1) pd.GenerateCartesianSamples(samplePerLevelsetCell, samples);

  for (int i = 0, size = samples.size() / 3; i < size; i++) {
    vec<float, 3> p{samples[i * 3 + 0], samples[i * 3 + 1], samples[i * 3 + 2]};
    p = (p - mins) * scale + offset;
    // particle[0] = ((samples[i * 3 + 0]) + offset[0]);
    // particle[1] = ((samples[i * 3 + 1]) + offset[1]);
    // particle[2] = ((samples[i * 3 + 2]) + offset[2]);
    data.push_back(std::array<float, 3>{p[0], p[1], p[2]});
  }
  printf("[%f, %f, %f] - [%f, %f, %f], scale %f, parcnt %d, lsdx %f, dx %f\n",
         mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2], scale,
         (int)data.size(), levelsetDx, dx);
  return data;
}

// Read SDF file, uniformly sample position data into an array (JB)
// Controlled by scene.json - span & offset (per model)
// span = [0.5, 0.5, 0.5] 
// offset = [0.1, 0.2, 0.2]
// Means longest model dim covers 0.5 of the domain length, max
// Offset by 10% of domain in x, 20% y, 20% z
auto read_sdf(std::string fn, float ppc, float dx, int domainsize,
              vec<float, 3> offset, vec<float, 3> lengths) {
  std::vector<std::array<float, 3>> data;
  std::string fileName = std::string(AssetDirPath) + "MpmParticles/" + fn;

  // Create SampleGenerator class
  SampleGenerator pd;
  int pad = 1;
  float fpad = 1.f;
  float levelsetDx;
  std::vector<float> samples;
  vec<float, 3> mins, maxs, scales;
  vec<float, 3> lengthRatios;
  vec<int, 3> maxns;

  // Load sdf into pd, update levelsetDx, mins, maxns
  pd.LoadSDF(fileName, levelsetDx, mins[0], mins[1], mins[2], maxns[0],
             maxns[1], maxns[2]);
  maxs = maxns.cast<float>() * levelsetDx;

  // Adjust *.sdf extents to simulation domainsize, select smallest ratio
  //scales = maxns.cast<float>() / domainsize;
  //mn::vec<float,3> lengths_vol = lengths.cast<float> * lengths.cast<float> * lengths.cast<float>;
  //vec<float,3> maxns_vol = maxns.cast<float> * maxns.cast<float> * maxns.cast<float>;
  //float domainsize_vol = domainsize * domainsize * domainsize;

  //scales = pow(lengths.cast<float>(),3) / pow(maxns.cast<float>(),3) / pow(domainsize,3);
  //scales = lengths_vol / maxns_vol / domainsize_vol;
  lengthRatios[0] = lengths[0] / (maxs[0] - fpad*levelsetDx);
  lengthRatios[1] = lengths[1] / (maxs[1] - fpad*levelsetDx);
  lengthRatios[2] = lengths[2] / (maxs[2] - fpad*levelsetDx);
  float lengthRatio = lengthRatios[0] < lengthRatios[1] ? lengthRatios[0] : lengthRatios[1];
  lengthRatio = lengthRatios[2] < lengthRatio ? lengthRatios[2] : lengthRatio;

  scales[0] = pow((dx / lengthRatios[0]), 3.f);
  scales[1] = pow((dx / lengthRatios[1]), 3.f);
  scales[2] = pow((dx / lengthRatios[2]), 3.f);

  // scales[0] = pow(lengths[0], 3.f) * pow((float)domainsize, 3.f) / pow((float)maxns[0] - (float)pad, 3.f);
  // scales[1] = pow(lengths[1], 3.f) * pow((float)domainsize, 3.f) / pow((float)maxns[1] - (float)pad, 3.f);
  // scales[2] = pow(lengths[2], 3.f) * pow((float)domainsize, 3.f) / pow((float)maxns[2] - (float)pad, 3.f);


  float scale = scales[0] < scales[1] ? scales[0] : scales[1];
  scale = scales[2] < scale ? scales[2] : scale;
  printf("scale %f %f %f %f %f %f %f %f\n", scale, lengths[0],lengths[1],lengths[2],(float)maxns[0],(float)maxns[1],(float)maxns[2], (float)domainsize);

  float samplePerLevelsetCell;
  if (1) samplePerLevelsetCell = ppc * scale;
  if (0) samplePerLevelsetCell = 1.f * (int)(ppc * scale + 0.5f);


  // Output uniformly sampled sdf into samples
  if (0) pd.GenerateUniformSamples(samplePerLevelsetCell, samples);
  if (1) pd.GenerateCartesianSamples(samplePerLevelsetCell, samples);

  // Adjust lengths to extents of the *.sdf, select smallest ratio
  //scales = lengths / (maxs - mins) / maxns.cast<float>();
  //scales = lengths /  maxns.cast<float>();
  // scales[0] = lengths[0] / ((float)maxns[0] - 1.f);
  // scales[1] = lengths[1] / ((float)maxns[1] - 1.f);
  // scales[2] = lengths[2] / ((float)maxns[2] - 1.f);
  scales[0] = lengths[0] / ((float)maxns[0] - fpad);
  scales[1] = lengths[1] / ((float)maxns[1] - fpad);
  scales[2] = lengths[2] / ((float)maxns[2] - fpad);
  scale = scales[0] < scales[1] ? scales[0] : scales[1];
  scale = scales[2] < scale ? scales[2] : scale;

  // offset[0] -= fpad * lengths[0] / ((float)maxns[0] - fpad);
  // offset[1] -= fpad * lengths[1] / ((float)maxns[1] - fpad);
  // offset[2] -= fpad * lengths[2] / ((float)maxns[2] - fpad);

  // offset[0] -= fpad * scale;
  // offset[1] -= fpad * scale;
  // offset[2] -= fpad * scale;


  // Loop through samples
  for (int i = 0, size = samples.size() / 3; i < size; i++) {
    // Group x,y,z position data
    vec<float, 3> p{samples[i * 3 + 0], samples[i * 3 + 1], samples[i * 3 + 2]};
    
    // Scale positions, add-in offset from JSON
    //p = (p - mins) * scale + offset;
    p = (p - fpad) * scale + offset;

    // Add (x,y,z) to data
    data.push_back(std::array<float, 3>{p[0], p[1], p[2]});
  }
  printf("[%f, %f, %f] - [%f, %f, %f], scale %f, parcnt %d, lsdx %f, dx %f\n",
         mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2], scale,
         (int)data.size(), levelsetDx, dx);
  return data;
}

} // namespace mn

#endif
