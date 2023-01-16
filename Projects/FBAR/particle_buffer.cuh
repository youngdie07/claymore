#ifndef __PARTICLE_BUFFER_CUH_
#define __PARTICLE_BUFFER_CUH_
#include "settings.h"
#include <MnBase/Meta/Polymorphism.h>
namespace mn {

using ParticleBinDomain = aligned_domain<char, config::g_bin_capacity>;
using ParticleBufferDomain = compact_domain<int, config::g_max_particle_bin>;
using ParticleArrayDomain = compact_domain<int, config::g_max_particle_num>;
using ParticleTargetDomain = compact_domain<int, config::g_max_particle_target_nodes>;


using particle_bin4_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f32_, f32_, f32_,
               f32_>; ///< pos, J
using particle_bin4_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_,
               f_>; ///< pos, J
using particle_bin7_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_,
               f_, f_, f_, f_>; ///< pos, J / ID, vel
using particle_bin9_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, f_, 
               f_, f_, f_, 
               f_, f_>; ///< pos, J, vel, vol JBar
using particle_bin11_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_,
               f_, f_, f_, f_, f_,
               f_, f_, f_>; ///< pos, ID, forces, restVolume, normals
using particle_bin12_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_>; ///< pos, F
using particle_bin13_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, f_,
               f_, f_, f_, f_, f_, f_, f_, f_,
               f_>; ///< pos, F, logJp
using particle_bin15_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, f_,
               f_, f_, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_>; ///< pos, F, vel
using particle_bin16_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, f_,
               f_, f_, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_>; ///< pos, F, logJp, vel
using particle_bin17_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_,
               f_, f_>; ///< pos, F, vel, vol_Bar, J_Bar
using particle_bin18_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_,
               f_, f_,
               f_>; ///< pos, F, vel, vol_Bar, J_Bar, ID

template <material_e mt> struct particle_bin_;
template <> struct particle_bin_<material_e::JFluid> : particle_bin4_f_ {};
template <> struct particle_bin_<material_e::JFluid_ASFLIP> : particle_bin7_f_ {};
template <> struct particle_bin_<material_e::JBarFluid> : particle_bin9_f_ {};
template <> struct particle_bin_<material_e::FixedCorotated> : particle_bin13_f_ {};
template <> struct particle_bin_<material_e::FixedCorotated_ASFLIP> : particle_bin16_f_ {};
template <> struct particle_bin_<material_e::FixedCorotated_ASFLIP_FBAR> : particle_bin18_f_ {};
template <> struct particle_bin_<material_e::Sand> : particle_bin17_f_ {};
template <> struct particle_bin_<material_e::NACC> : particle_bin17_f_ {};
template <> struct particle_bin_<material_e::Meshed> : particle_bin11_f_ {};


template <typename ParticleBin>
using particle_buffer_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleBufferDomain, attrib_layout::aos, ParticleBin>;



template <material_e mt>
struct ParticleBufferImpl : Instance<particle_buffer_<particle_bin_<mt>>> {
  static constexpr material_e materialType = mt;
  using base_t = Instance<particle_buffer_<particle_bin_<mt>>>;

  template <typename Allocator>
  ParticleBufferImpl(Allocator allocator)
      : base_t{spawn<particle_buffer_<particle_bin_<mt>>, orphan_signature>(
            allocator)} {}
  template <typename Allocator>
  void checkCapacity(Allocator allocator, std::size_t capacity) {
    if (capacity > this->_capacity)
      this->resize(allocator, capacity);
  }

  /// @brief Maps run-time string labels for any MPM particle attributes to int indices for GPU kernel output use
  /// @param n is a std::string containing a particle attribute label  
  /// @return Integer index assigned to particle attribute n, used in GPU kernels
  int mapAttributeStringToIndex(std::string n) {
        if      (n == "ID") return 0; 
        else if (n == "Mass") return 1;
        else if (n == "Volume") return 2;
        else if (n == "Position_X") return 3; 
        else if (n == "Position_Y") return 4;
        else if (n == "Position_Z") return 5;
        else if (n == "Velocity_Y") return 7;
        else if (n == "Velocity_X") return 6;
        else if (n == "Velocity_Z") return 8;
        else if (n == "DefGrad_XX") return 9;
        else if (n == "DefGrad_XY") return 10;
        else if (n == "DefGrad_XZ") return 11;
        else if (n == "DefGrad_YX") return 12;
        else if (n == "DefGrad_YY") return 13;
        else if (n == "DefGrad_YZ") return 14;
        else if (n == "DefGrad_ZX") return 15;
        else if (n == "DefGrad_ZY") return 16;
        else if (n == "DefGrad_ZZ") return 17;
        else if (n == "J")          return 18;
        else if (n == "JBar")       return 19;
        else if (n == "StressCauchy_XX") return 20;
        else if (n == "StressCauchy_XY") return 21;
        else if (n == "StressCauchy_XZ") return 22;
        else if (n == "StressCauchy_YX") return 23;
        else if (n == "StressCauchy_YY") return 24;
        else if (n == "StressCauchy_YZ") return 25;
        else if (n == "StressCauchy_ZX") return 26;
        else if (n == "StressCauchy_ZY") return 27;
        else if (n == "StressCauchy_ZZ") return 28;
        else if (n == "Pressure")        return 29;
        else if (n == "VonMisesStress")  return 30;
        else if (n == "DefGrad_Invariant1") return 31;
        else if (n == "DefGrad_Invariant2") return 32;
        else if (n == "DefGrad_Invariant3") return 33;
        else if (n == "DefGrad_1") return 33;
        else if (n == "DefGrad_2") return 34;
        else if (n == "DefGrad_3") return 35;
        else if (n == "StressCauchy_Invariant1") return 36;
        else if (n == "StressCauchy_Invariant2") return 37;
        else if (n == "StressCauchy_Invariant3") return 38;
        else if (n == "StressCauchy_1") return 39;
        else if (n == "StressCauchy_2") return 40;
        else if (n == "StressCauchy_3") return 41;
        else if (n == "StressPK1_XX") return 42;
        else if (n == "StressPK1_XY") return 43;
        else if (n == "StressPK1_XZ") return 44;
        else if (n == "StressPK1_YX") return 45;
        else if (n == "StressPK1_YY") return 46;
        else if (n == "StressPK1_YZ") return 47;
        else if (n == "StressPK1_ZX") return 48;
        else if (n == "StressPK1_ZY") return 49;
        else if (n == "StressPK1_ZZ") return 50;
        else if (n == "StressPK1_Invariant1") return 51;
        else if (n == "StressPK1_Invariant2") return 52;
        else if (n == "StressPK1_Invariant3") return 53;
        else if (n == "StressPK1_1") return 54;
        else if (n == "StressPK1_2") return 55;
        else if (n == "StressPK1_3") return 56;
        else if (n == "StrainSmall_XX") return 57;
        else if (n == "StrainSmall_XY") return 58;
        else if (n == "StrainSmall_XZ") return 59;
        else if (n == "StrainSmall_YX") return 60;
        else if (n == "StrainSmall_YY") return 61;
        else if (n == "StrainSmall_YZ") return 62;
        else if (n == "StrainSmall_ZX") return 63;
        else if (n == "StrainSmall_ZY") return 64;
        else if (n == "StrainSmall_ZZ") return 65;
        else if (n == "StrainSmall_Invariant1") return 66;
        else if (n == "StrainSmall_Invariant2") return 67;
        else if (n == "StrainSmall_Invariant3") return 68;
        else if (n == "StrainSmall_1") return 69;
        else if (n == "StrainSmall_2") return 70;
        else if (n == "StrainSmall_3") return 71;
        else return -1;
  }

  int track_ID = 0;
  vec<int, 1> track_attribs;
  std::vector<std::string> track_labels;   
  void updateTrack(std::vector<std::string> names, int trackID=0) {
    track_ID = trackID;
    int i = 0;
    for (auto n : names)
    {
    track_labels.emplace_back(n);
        track_attribs[i] = mapAttributeStringToIndex(n);
        i = i+1;
    }
  }

  vec<int, 3> output_attribs;
  std::vector<std::string> output_labels;   
  void updateOutputs(std::vector<std::string> names) {
    int i = 0;
    for (auto n : names)
    {
      output_labels.emplace_back(n);
      output_attribs[i] = mapAttributeStringToIndex(n);
      i++;
    }
  }

  vec<int, 1> target_attribs;
  std::vector<std::string> target_labels;   
  void updateTargets(std::vector<std::string> names) {
    int i = 0;
    for (auto n : names)
    {
      target_labels.emplace_back(n);
      target_attribs[i] = mapAttributeStringToIndex(n);
      i++;
    }
  }

};


template <material_e mt> struct ParticleBuffer;
template <>
struct ParticleBuffer<material_e::JFluid>
    : ParticleBufferImpl<material_e::JFluid> {
  using base_t = ParticleBufferImpl<material_e::JFluid>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY; // Density [kg/m3]
  PREC volume = DOMAIN_VOLUME * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC); // Volume of Particle [m3]
  PREC mass = (volume * DENSITY); // Mass of particle [kg]
  PREC bulk = 5e6; //< Bulk Modulus [Pa]
  PREC gamma = 7.1f; //< Derivative Bulk w.r.t. Pressure
  PREC visco = 0.001f; //< Dynamic Viscosity, [Pa * s]
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, PREC density, PREC ppc, PREC b, PREC g, PREC v,
                        bool ASFLIP=false, bool FEM=false, bool FBAR=false) {
    length = l;
    rho = density;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / ppc);
    mass = volume * density;
    bulk = b;
    gamma = g;
    visco = v;
    use_ASFLIP = ASFLIP;
    use_FEM = FEM;
    use_FBAR = FBAR;
  }
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};

template <>
struct ParticleBuffer<material_e::JFluid_ASFLIP>
    : ParticleBufferImpl<material_e::JFluid_ASFLIP> {
  using base_t = ParticleBufferImpl<material_e::JFluid_ASFLIP>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY; // Density [kg/m3]
  PREC volume = DOMAIN_VOLUME * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC); // Volume of Particle [m3]
  PREC mass = (volume * DENSITY); // Mass of particle [kg]
  PREC bulk = 2.2e9; //< Bulk Modulus [Pa]
  PREC gamma = 7.1; //< Derivative Bulk w.r.t. Pressure
  PREC visco = 0.001; //< Dynamic Viscosity, [Pa * s]
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, PREC density, PREC ppc, PREC b, PREC g, PREC v, 
                        PREC a, PREC bmin, PREC bmax, 
                        bool ASFLIP=false, bool FEM=false, bool FBAR=false) {
    length = l;
    rho = density;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / ppc);
    mass = volume * density;
    bulk = b;
    gamma = g;
    visco = v;
    alpha = a;
    beta_min = bmin;
    beta_max = bmax;
    use_ASFLIP = ASFLIP;
    use_FEM = FEM;
    use_FBAR = FBAR;
  }
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};


template <>
struct ParticleBuffer<material_e::JBarFluid>
    : ParticleBufferImpl<material_e::JBarFluid> {
  using base_t = ParticleBufferImpl<material_e::JBarFluid>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY; // Density [kg/m3]
  PREC volume = DOMAIN_VOLUME * ( 1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC); // Volume of Particle [m3]
  PREC mass = (volume * DENSITY); // Mass of particle [kg]
  PREC bulk = 2.2e9; //< Bulk Modulus [Pa]
  PREC gamma = 7.1; //< Derivative Bulk w.r.t. Pressure
  PREC visco = 0.001; //< Dynamic Viscosity, [Pa * s]
  
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.

  void updateParameters(PREC l, PREC density, PREC ppc, PREC b, PREC g, PREC v, 
                        PREC a, PREC bmin, PREC bmax, PREC fbar_ratio,
                        bool ASFLIP=false, bool FEM=false, bool FBAR=false) {
    length = l;
    rho = density;
    volume = length*length*length * (1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS)) / ppc;
    mass = volume * density;
    bulk = b;
    gamma = g;
    visco = v;
    alpha = a;
    beta_min = bmin;
    beta_max = bmax;
    FBAR_ratio = fbar_ratio;
    use_ASFLIP = ASFLIP;
    use_FEM = FEM;
    use_FBAR = FBAR;
  }  
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};

template <>
struct ParticleBuffer<material_e::FixedCorotated>
    : ParticleBufferImpl<material_e::FixedCorotated> {
  using base_t = ParticleBufferImpl<material_e::FixedCorotated>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, PREC density, PREC ppc, PREC E, PREC nu,
                        bool ASFLIP=false, bool FEM=false, bool FBAR=false) {
    length = l;
    rho = density;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / ppc);
    mass = volume * density;
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    mu = E / (2 * (1 + nu));
    use_ASFLIP = ASFLIP;
    use_FEM = FEM;
    use_FBAR = FBAR;
  }
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};

template <>
struct ParticleBuffer<material_e::FixedCorotated_ASFLIP>
    : ParticleBufferImpl<material_e::FixedCorotated_ASFLIP> {
  using base_t = ParticleBufferImpl<material_e::FixedCorotated_ASFLIP>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, PREC density, PREC ppc, PREC E, PREC nu, 
                        PREC a, PREC bmin, PREC bmax,
                        PREC ASFLIP=false, bool FEM=false, bool FBAR=false) {
    length = l;
    rho = density;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / ppc);
    mass = volume * density;
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    mu = E / (2 * (1 + nu));
    alpha = a;
    beta_min = bmin;
    beta_max = bmax;
    use_ASFLIP = ASFLIP;
    use_FEM = FEM;
    use_FBAR = FBAR;
  }
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};


template <>
struct ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR>
    : ParticleBufferImpl<material_e::FixedCorotated_ASFLIP_FBAR> {
  using base_t = ParticleBufferImpl<material_e::FixedCorotated_ASFLIP_FBAR>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, PREC density, PREC ppc, PREC E, PREC nu, 
                        PREC a, PREC bmin, PREC bmax, PREC fbar_ratio,
                        PREC ASFLIP=false, bool FEM=false, bool FBAR=false) {
    length = l;
    rho = density;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / ppc);
    mass = volume * density;
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    mu = E / (2 * (1 + nu));
    alpha = a;
    beta_min = bmin;
    beta_max = bmax;
    FBAR_ratio = fbar_ratio;
    use_ASFLIP = ASFLIP;
    use_FEM = FEM;
    use_FBAR = FBAR;
  }
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};

template <>
struct ParticleBuffer<material_e::Sand> : ParticleBufferImpl<material_e::Sand> {
  using base_t = ParticleBufferImpl<material_e::Sand>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * 
      (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
       MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda =
      YOUNGS_MODULUS * POISSON_RATIO /
      ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));

  PREC logJp0 = 0.f;
  PREC frictionAngle = 30.f;
  PREC cohesion = 0.f;
  PREC beta = 1.f;
  // std::sqrt(2.f/3.f) * 2.f * std::sin(30.f/180.f*3.141592741f)
  // 						/ (3.f -
  // std::sin(30.f/180.f*3.141592741f))
  PREC yieldSurface =
      0.816496580927726f * 2.f * 0.5f / (3.f - 0.5f);
  bool volumeCorrection = true;
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, PREC density, PREC ppc, PREC E, PREC nu,
                        PREC logJ, PREC friction_angle, PREC c, PREC b, bool volCorrection=true, 
                        bool ASFLIP=false, bool FEM=false, bool FBAR=false) {
    length = l;
    rho = density;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / ppc);
    mass = volume * density;
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    mu = E / (2 * (1 + nu));
    logJp0 = logJ;
    frictionAngle = friction_angle;
    yieldSurface = 0.816496580927726 * 2.0 * std::sin(frictionAngle / 180.0 * 3.141592741) / (3.0 - std::sin(frictionAngle / 180.0 * 3.141592741));
    cohesion = c;
    beta = b;
    volumeCorrection = volCorrection;
    use_ASFLIP = ASFLIP;
    use_FEM = FEM;
    use_FBAR = FBAR;
  }
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};

template <>
struct ParticleBuffer<material_e::NACC> : ParticleBufferImpl<material_e::NACC> {
  using base_t = ParticleBufferImpl<material_e::NACC>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));

  PREC frictionAngle = 45.f;
  PREC bm = 2.f / 3.f * (YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO))) +
             (YOUNGS_MODULUS * POISSON_RATIO /
              ((1 + POISSON_RATIO) *
               (1 - 2 * POISSON_RATIO))); ///< bulk modulus, kappa
  PREC xi = 0.8f;                        ///< hardening factor
  static constexpr PREC logJp0 = -0.01f;
  PREC beta = 0.5f;
  static constexpr PREC mohrColumbFriction =
      0.503599787772409; //< sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 -
                         // sin_phi);
  static constexpr PREC M =
      1.850343771924453; ///< mohrColumbFriction * (T)dim / sqrt((T)2 / ((T)6
                         ///< - dim));
  static constexpr PREC Msqr = 3.423772074299613;
  static constexpr bool hardeningOn = true;
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, PREC density, PREC ppc, PREC E, PREC nu, 
                        PREC be, PREC x,
                        bool ASFLIP=false, bool FEM=false, bool FBAR=false) {
    length = l;
    rho = density;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / ppc);
    mass = volume * density;
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    mu = E / (2 * (1 + nu));
    bm =
        2.f / 3.f * (E / (2 * (1 + nu))) + (E * nu / ((1 + nu) * (1 - 2 * nu)));
    beta = be;
    xi = x;
    use_ASFLIP = ASFLIP;
    use_FEM = FEM;
    use_FBAR = FBAR;
  }
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};

template <>
struct ParticleBuffer<material_e::Meshed>
    : ParticleBufferImpl<material_e::Meshed> {
  using base_t = ParticleBufferImpl<material_e::Meshed>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, PREC density, PREC ppc, PREC ym, PREC pr, PREC a,
                        PREC bmin, PREC bmax, PREC fbar_ratio,
                        bool ASFLIP=false, bool FEM=false, bool FBAR = false) {
    length = l;
    rho = density;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / ppc);
    E = ym;
    nu = pr;
    mass = volume * density;
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    mu = E / (2 * (1 + nu));
    alpha = a;
    beta_min = bmin;
    beta_max = bmax;
    FBAR_ratio = fbar_ratio;
    use_ASFLIP = ASFLIP;
    use_FEM = FEM;
    use_FBAR = FBAR;
  }
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};

/// Reference: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0608r3.html
/// * Make sure to add new materials to this
using particle_buffer_t =
    variant<ParticleBuffer<material_e::JFluid>,
            ParticleBuffer<material_e::JFluid_ASFLIP>,
            ParticleBuffer<material_e::JBarFluid>,
            ParticleBuffer<material_e::FixedCorotated>,
            ParticleBuffer<material_e::FixedCorotated_ASFLIP>,
            ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR>,
            ParticleBuffer<material_e::Sand>, 
            ParticleBuffer<material_e::NACC>,
            ParticleBuffer<material_e::Meshed>>;

using particle_array_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_>;
using particle_array_3_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_>;
using particle_array_6_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_>;
using particle_array_9_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, f_, f_, f_>;

struct ParticleArray : Instance<particle_array_> {
  using base_t = Instance<particle_array_>;
  ParticleArray &operator=(base_t &&instance) {
    static_cast<base_t &>(*this) = instance;
    return *this;
  }
};

//template<int N=3>
struct ParticleAttrib: Instance<particle_array_> {
  using base_t = Instance<particle_array_>;
  ParticleAttrib &operator=(base_t &&instance) {
    static_cast<base_t &>(*this) = instance;
    return *this;
  }
  //static constexpr int number_outputs = N;
};

using particle_target_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleTargetDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, f_, f_, f_, f_>;

/// * ParticleTarget structure for device instantiation 
struct ParticleTarget : Instance<particle_target_> {
  using base_t = Instance<particle_target_>;
  ParticleTarget &operator=(base_t &&instance) {
    static_cast<base_t &>(*this) = instance;
    return *this;
  }
  ParticleTarget(base_t &&instance) { static_cast<base_t &>(*this) = instance; }
  // template <typename Allocator>
  // ParticleTarget(Allocator allocator) : base_t{allocator} {}
};

} // namespace mn

#endif