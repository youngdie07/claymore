#ifndef __PARTICLE_BUFFER_CUH_
#define __PARTICLE_BUFFER_CUH_
#include "settings.h"
#include "constitutive_models.cuh"
#include <MnBase/Meta/Polymorphism.h>
#include "MnBase/Meta/TypeMeta.h"
namespace mn {

using ParticleBinDomain = aligned_domain<char, config::g_bin_capacity>;
using ParticleBufferDomain = compact_domain<int, config::g_max_particle_bin>;
using ParticleArrayDomain = compact_domain<int, config::g_max_particle_num>;
using ParticleTargetDomain = compact_domain<int, config::g_max_particle_target_nodes>;

// * All  particle attributes available for ouput.
// * Not all materials will support every output.
enum class particle_output_attribs_e : int {
        EMPTY=-3, // Empty attribute request 
        INVALID_CT=-2, // Invalid compile-time request, e.g. deprecated variable (below END)
        INVALID_RT=-1, // Invalid run-time request e.g. "vel X" instead of "Velocity_X"
        START=0,
        ID = 0, Mass, Volume,
        Position_X, Position_Y, Position_Z,
        Velocity_X, Velocity_Y, Velocity_Z,
        DefGrad_XX, DefGrad_XY, DefGrad_XZ,
        DefGrad_YX, DefGrad_YY, DefGrad_YZ,
        DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
        J, DefGrad_Determinant=J, JBar, DefGrad_Determinant_FBAR=JBar, 
        StressCauchy_XX, StressCauchy_XY, StressCauchy_XZ,
        StressCauchy_YX, StressCauchy_YY, StressCauchy_YZ,
        StressCauchy_ZX, StressCauchy_ZY, StressCauchy_ZZ,
        Pressure, VonMisesStress,
        DefGrad_Invariant1, DefGrad_Invariant2, DefGrad_Invariant3,
        DefGrad_1, DefGrad_2, DefGrad_3,
        StressCauchy_Invariant1, StressCauchy_Invariant2, StressCauchy_Invariant3,
        StressCauchy_1, StressCauchy_2, StressCauchy_3,
        StressPK1_XX, StressPK1_XY, StressPK1_XZ,
        StressPK1_YX, StressPK1_YY, StressPK1_YZ,
        StressPK1_ZX, StressPK1_ZY, StressPK1_ZZ,
        StressPK1_Invariant1, StressPK1_Invariant2, StressPK1_Invariant3,
        StressPK1_1, StressPK1_2, StressPK1_3,
        StressPK2_XX, StressPK2_XY, StressPK2_XZ,
        StressPK2_YX, StressPK2_YY, StressPK2_YZ,
        StressPK2_ZX, StressPK2_ZY, StressPK2_ZZ,
        StressPK2_Invariant1, StressPK2_Invariant2, StressPK2_Invariant3,
        StressPK2_1, StressPK2_2, StressPK2_3,
        StrainSmall_XX, StrainSmall_XY, StrainSmall_XZ,
        StrainSmall_YX, StrainSmall_YY, StrainSmall_YZ,
        StrainSmall_ZX, StrainSmall_ZY, StrainSmall_ZZ,
        StrainSmall_Invariant1, StrainSmall_Invariant2, StrainSmall_Invariant3,
        Dilation = StrainSmall_Invariant1, StrainSmall_Determinant = StrainSmall_Invariant3,
        StrainSmall_1,  StrainSmall_2, StrainSmall_3,
        logJp=100,
        END,
        ExampleDeprecatedVariable //< Will give INVALID_CT output of -2
};

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
using particle_bin6_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_,
               f_, f_, f_>; ///< pos, J, JBar, ID
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
template <> struct particle_bin_<material_e::JFluid_FBAR> : particle_bin6_f_ {};
template <> struct particle_bin_<material_e::JBarFluid> : particle_bin9_f_ {};
template <> struct particle_bin_<material_e::FixedCorotated> : particle_bin13_f_ {};
template <> struct particle_bin_<material_e::FixedCorotated_ASFLIP> : particle_bin16_f_ {};
template <> struct particle_bin_<material_e::FixedCorotated_ASFLIP_FBAR> : particle_bin18_f_ {};
template <> struct particle_bin_<material_e::NeoHookean_ASFLIP_FBAR> : particle_bin18_f_ {};
template <> struct particle_bin_<material_e::Sand> : particle_bin18_f_ {};
template <> struct particle_bin_<material_e::NACC> : particle_bin18_f_ {};
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
        else if (n == "Velocity_X") return 6;
        else if (n == "Velocity_Y") return 7;
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
        else if (n == "logJp") return 100;
        else return -1;
  }
 int queryAttributeIndex(int idx) {
        switch(idx)
        {
          case -1:
              break;
          case 0:
              break;
          default:
              break;
        } 
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
  vec<int, mn::config::g_max_particle_attribs> output_attribs_dyn;
  std::vector<std::string> output_labels;   
  void updateOutputs(std::vector<std::string> names) {
    int i = 0;
    for (auto n : names)
    {
      if (i>=mn::config::g_max_particle_attribs) continue;
      output_labels.emplace_back(n);
      output_attribs_dyn[i] = mapAttributeStringToIndex(n);
      if (i < 3) output_attribs[i] = mapAttributeStringToIndex(n);
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
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    bulk = mat.bulk;
    gamma = mat.gamma;
    visco = mat.visco;
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          J, DefGrad_Determinant=J, 
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Velocity_X, Velocity_Y, Velocity_Z,
          Volume_FBAR, 
          JBar, DefGrad_Determinant_FBAR=JBar, 
          ID,
          logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
  __forceinline__ __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }

  template <typename T = PREC>
  __forceinline__ __device__ void
  getPressure(T J, T& pressure){
    compute_pressure_jfluid(volume, bulk, gamma, J, pressure);
  }

  template <typename T = PREC>
  __forceinline__ __device__ void
  getStrainEnergy(T J, T& strain_energy){
    compute_energy_jfluid(volume, bulk, gamma, J, strain_energy);
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
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    bulk = mat.bulk;
    gamma = mat.gamma;
    visco = mat.visco;
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          J, DefGrad_Determinant=J, 
          Velocity_X, Velocity_Y, Velocity_Z,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Volume_FBAR, 
          JBar, DefGrad_Determinant_FBAR=JBar, 
          ID,
          logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
  __forceinline__ __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};


template <>
struct ParticleBuffer<material_e::JFluid_FBAR>
    : ParticleBufferImpl<material_e::JFluid_FBAR> {
  using base_t = ParticleBufferImpl<material_e::JFluid_FBAR>;
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

  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * (1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS)) / mat.ppc;
    mass = volume * mat.rho;
    bulk = mat.bulk;
    gamma = mat.gamma;
    visco = mat.visco;
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }  

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          J, DefGrad_Determinant=J, 
          JBar, DefGrad_Determinant_FBAR=JBar, 
          ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Velocity_X, Velocity_Y, Velocity_Z,
          Volume_FBAR, 
          logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
  __forceinline__ __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
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

  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    volume = length*length*length * (1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS)) / mat.ppc;
    rho = mat.rho;
    mass = volume * mat.rho;
    bulk = mat.bulk;
    gamma = mat.gamma;
    visco = mat.visco;
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }  

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          J, DefGrad_Determinant=J, 
          Velocity_X, Velocity_Y, Velocity_Z,
          ID,
          JBar, DefGrad_Determinant_FBAR=JBar, 
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Volume_FBAR, 
          logJp
  };

  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
  __forceinline__ __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
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
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          J, DefGrad_Determinant=J, 
          Velocity_X, Velocity_Y, Velocity_Z,
          Volume_FBAR, JBar, DefGrad_Determinant_FBAR=JBar, 
          logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
  __forceinline__ __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
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
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Velocity_X, Velocity_Y, Velocity_Z,
          ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          J, DefGrad_Determinant=J, Volume_FBAR, JBar, DefGrad_Determinant_FBAR=JBar, logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
  __forceinline__ __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
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
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Velocity_X, Velocity_Y, Velocity_Z,
          Volume_FBAR, JBar, DefGrad_Determinant_FBAR=JBar, ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          J, DefGrad_Determinant=J, logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
  __forceinline__ __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};


template <>
struct ParticleBuffer<material_e::NeoHookean_ASFLIP_FBAR>
    : ParticleBufferImpl<material_e::NeoHookean_ASFLIP_FBAR> {
  using base_t = ParticleBufferImpl<material_e::NeoHookean_ASFLIP_FBAR>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * (1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
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
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    rho = mat.rho;
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Velocity_X, Velocity_Y, Velocity_Z,
          Volume_FBAR, JBar, DefGrad_Determinant_FBAR=JBar, ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          J, DefGrad_Determinant=J, logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
  __forceinline__ __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }

  template <typename T = PREC>
  __forceinline__ __device__ void
  getVelocity(const T bin, const T particle_id_in_bin, PREC * velocity) {
    velocity = {
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_X>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Y>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Z>{}, particle_id_in_bin)
    };
  }

  template <typename T = PREC>
  __forceinline__ __device__ void
  getDefGrad(const T bin, const T particle_id_in_bin, PREC * DefGrad) {
    DefGrad[0] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XX>{}, particle_id_in_bin);
    DefGrad[1] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XY>{}, particle_id_in_bin);
    DefGrad[2] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XZ>{}, particle_id_in_bin);
    DefGrad[3] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YX>{}, particle_id_in_bin);
    DefGrad[4] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YY>{}, particle_id_in_bin);
    DefGrad[5] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YZ>{}, particle_id_in_bin);
    DefGrad[6] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZX>{}, particle_id_in_bin);
    DefGrad[7] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZY>{}, particle_id_in_bin);
    DefGrad[8] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZZ>{}, particle_id_in_bin);
  }



  template <typename T = PREC>
  __forceinline__ __device__ void
  getStress_Cauchy(const vec<T,9>& F, vec<T,9>& PF){
    compute_stress_neohookean(volume, mu, lambda, F, PF);
  }
  template <typename T = PREC>
  __forceinline__ __device__ void
  getStress_Cauchy(T vol, const vec<T,9>& F, vec<T,9>& PF){
    compute_stress_neohookean(vol, mu, lambda, F, PF);
  }
  
  template <typename T = PREC>
  __forceinline__ __device__ void
  getStress_PK1(const vec<T,9>& F, vec<T,9>& P){
    compute_stress_PK1_neohookean(volume, mu, lambda, F, P);
  }

  template <typename T = PREC>
  __forceinline__ __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy, T vol){
    compute_energy_neohookean(vol, mu, lambda, F, strain_energy);
  }
  template <typename T = PREC>
  __forceinline__ __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy){
    compute_energy_neohookean(volume, mu, lambda, F, strain_energy);
  }
  template <typename T = PREC>
  __forceinline__ __device__ void
  getEnergy_Kinetic(const vec<T,3>& velocity, T &kinetic_energy){  
    kinetic_energy = 0.5 * mass * __fma_rn(velocity[0], velocity[0], __fma_rn(velocity[1], velocity[1], (velocity[2], velocity[2])));
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
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    logJp0 = mat.logJp0;
    frictionAngle = mat.frictionAngle;
    yieldSurface = 0.816496580927726 * 2.0 * std::sin(mat.frictionAngle / 180.0 * 3.141592741) / (3.0 - std::sin(mat.frictionAngle / 180.0 * 3.141592741));
    cohesion = mat.cohesion;
    beta = mat.beta;
    volumeCorrection = mat.volumeCorrection;
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          logJp,
          Velocity_X, Velocity_Y, Velocity_Z,
          JBar, DefGrad_Determinant_FBAR=JBar, 
          ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          J, DefGrad_Determinant=J, 
          Volume_FBAR 
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
  __forceinline__ __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
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
  PREC logJp0 = -0.01f;
  PREC beta = 0.5f;
  static constexpr PREC mohrColumbFriction =
      0.503599787772409; //< sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 -
                         // sin_phi);
  static constexpr PREC M =
      1.850343771924453; ///< mohrColumbFriction * (T)dim / sqrt((T)2 / ((T)6
                         ///< - dim));
  static constexpr PREC Msqr = 3.423772074299613;
  bool hardeningOn = true;
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    bm =
        2.f / 3.f * (mat.E / (2 * (1 + mat.nu))) + (mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu)));
    logJp0 = mat.logJp0;
    frictionAngle = mat.frictionAngle;
    beta = mat.beta;
    xi = mat.xi;
    hardeningOn = mat.hardeningOn;
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          logJp,
          Velocity_X, Velocity_Y, Velocity_Z,
          JBar, DefGrad_Determinant_FBAR=JBar, 
          ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          J, DefGrad_Determinant=J, 
          Volume_FBAR 
  };

  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
  __forceinline__ __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
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
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    E = mat.E;
    nu = mat.nu;
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};

/// Reference: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0608r3.html
/// * Make sure to add new materials to this
using particle_buffer_t =
    variant<ParticleBuffer<material_e::JFluid>,
            ParticleBuffer<material_e::JFluid_ASFLIP>,
            ParticleBuffer<material_e::JFluid_FBAR>,
            ParticleBuffer<material_e::JBarFluid>,
            ParticleBuffer<material_e::FixedCorotated>,
            ParticleBuffer<material_e::FixedCorotated_ASFLIP>,
            ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR>,
            ParticleBuffer<material_e::NeoHookean_ASFLIP_FBAR>,
            ParticleBuffer<material_e::Sand>, 
            ParticleBuffer<material_e::NACC>,
            ParticleBuffer<material_e::Meshed>>;

using particle_array_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_>;
using particle_array_1_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_>;
using particle_array_2_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_>;
using particle_array_3_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_>;
using particle_array_4_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_>;
using particle_array_5_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_>;
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
using particle_array_12_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_>;
using particle_array_15_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_>;
using particle_array_24_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_,
               f_, f_, f_, f_, f_, f_, f_, f_, f_>;
using particle_array_32_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_,
               f_, f_, f_, f_, f_, f_, f_, f_, f_,
               f_, f_, f_, f_, f_, f_, f_, f_, f_>;

struct ParticleArray : Instance<particle_array_> {
  using base_t = Instance<particle_array_>;
  ParticleArray &operator=(base_t &&instance) {
    static_cast<base_t &>(*this) = instance;
    return *this;
  }
};

template <num_attribs_e N> struct particle_attrib_;
template <> struct particle_attrib_<num_attribs_e::Three> : particle_array_3_ {};
template <> struct particle_attrib_<num_attribs_e::Four> : particle_array_4_ {}; //
template <> struct particle_attrib_<num_attribs_e::Five> : particle_array_5_ {}; //
template <> struct particle_attrib_<num_attribs_e::Six> : particle_array_6_ {};
// // template <> struct particle_attrib_<7> : particle_array_9_ {}; //
// // template <> struct particle_attrib_<8> : particle_array_9_ {}; //
// template <> struct particle_attrib_<9> : particle_array_9_ {};
// // template <> struct particle_attrib_<10> : particle_array_12_ {}; //
// // template <> struct particle_attrib_<11> : particle_array_12_ {}; //
// template <> struct particle_attrib_<12> : particle_array_12_ {};
// template <> struct particle_attrib_<15> : particle_array_15_ {};
// template <> struct particle_attrib_<24> : particle_array_24_ {};

template<num_attribs_e N=num_attribs_e::Three>
struct ParticleAttrib: Instance<particle_attrib_<N>> {
  static constexpr unsigned numAttributes = static_cast<unsigned>(N);
  using base_t = Instance<particle_attrib_<N>>;
  // ParticleAttrib &operator=(base_t &&instance) {
  //   static_cast<base_t &>(*this) = instance;
  //   return *this;
  // }
  template <typename Allocator>
  ParticleAttrib(Allocator allocator)
      : base_t{spawn<particle_attrib_<N>, orphan_signature>(
            allocator)} {}
};

using particle_attrib_t =
    variant<ParticleAttrib<num_attribs_e::Three>,
            ParticleAttrib<num_attribs_e::Four>,
            ParticleAttrib<num_attribs_e::Five>,
            ParticleAttrib<num_attribs_e::Six> >;


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
};

} // namespace mn

#endif