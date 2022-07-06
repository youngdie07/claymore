#ifndef __FEM_BUFFER_CUH_
#define __FEM_BUFFER_CUH_
#include "settings.h"
#include <MnBase/Meta/Polymorphism.h>

namespace mn {

/// FEM Vertices
/// Basically particles, but ordered
using VerticeArrayDomain = compact_domain<int, config::g_max_fem_vertice_num>;

using vertice_array_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               VerticeArrayDomain, attrib_layout::aos, f32_, f32_, f32_,
               f32_, f32_, f32_, f32_, f32_, f32_, f32_,
               f32_>; //< pos (x,y,z), normals (x,y,z), mass, force internal (x,y,z)

/// FEM Elements
/// Hold vertice IDs and material info for force computation
using ElementBinDomain = aligned_domain<char, config::g_fem_element_bin_capacity>;
using ElementBufferDomain = compact_domain<int, config::g_max_fem_element_bin>;
using ElementArrayDomain = compact_domain<int, config::g_max_fem_element_num>;

using element_bin4_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ElementBinDomain, attrib_layout::soa, i32_, i32_, i32_,
               i32_>; ///< ID.a, ID.b, ID.c, ID.d
using element_bin8_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ElementBinDomain, attrib_layout::soa, i32_, i32_, i32_,
               i32_>; ///< ID.a, ID.b, ID.c, ID.d, ID.e, ID.f, ID.g, ID.h
using element_bin4_10_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ElementBinDomain, attrib_layout::soa, i32_, i32_, i32_,
               i32_, 
               f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, 
               f32_>; ///< ID.a, ID.b, ID.c, ID.d, B[9], restVolume

template <fem_e ft> struct element_bin_;
template <> struct element_bin_<fem_e::Tetrahedron> : element_bin4_10_ {};
template <> struct element_bin_<fem_e::Brick> : element_bin8_ {};

template <typename ElementBin>
using element_buffer_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ElementBufferDomain, attrib_layout::aos, ElementBin>;

// More basic array for elements, no material wrapper or element switches
using element_array_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ElementArrayDomain, attrib_layout::aos, i32_, i32_, i32_,
               i32_>; //< a, b, c, d

template <fem_e ft>
struct ElementBufferImpl : Instance<element_buffer_<element_bin_<ft>>> {
  static constexpr fem_e elementType = ft;
  using base_t = Instance<element_buffer_<element_bin_<ft>>>;

  template <typename Allocator>
  ElementBufferImpl(Allocator allocator)
      : base_t{spawn<element_buffer_<element_bin_<ft>>, orphan_signature>(
            allocator)} {}

  template <typename Allocator>
  void checkCapacity(Allocator allocator, std::size_t capacity) {
    if (capacity > this->_capacity)
      this->resize(allocator, capacity);
  }
};


template <fem_e ft> struct ElementBuffer;
template <>
struct ElementBuffer<fem_e::Tetrahedron>
    : ElementBufferImpl<fem_e::Tetrahedron> {
  using base_t = ElementBufferImpl<fem_e::Tetrahedron>;
  float rho = DENSITY;
  float volume = DOMAIN_VOLUME * (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC_FC);
  float mass = (volume * DENSITY);
  float E = YOUNGS_MODULUS;
  float nu = POISSON_RATIO;
  float lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  float mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));

  void updateParameters(float density, float vol, float ym, float pr) {
    rho = density;
    //volume = vol;
    E = ym;
    nu = pr;
    mass = volume * density;
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    mu = E / (2 * (1 + nu));
  }

  template <typename Allocator>
  ElementBuffer(Allocator allocator) : base_t{allocator} {}
};



template <>
struct ElementBuffer<fem_e::Brick> : ElementBufferImpl<fem_e::Brick> {
  using base_t = ElementBufferImpl<fem_e::Brick>;
  float rho = DENSITY;
  float volume = DOMAIN_VOLUME * (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  float mass = (volume * DENSITY);
  float E = YOUNGS_MODULUS;
  float nu = POISSON_RATIO;
  float lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  float mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));

  float frictionAngle = 45.f;
  float bm = 2.f / 3.f * (YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO))) +
             (YOUNGS_MODULUS * POISSON_RATIO /
              ((1 + POISSON_RATIO) *
               (1 - 2 * POISSON_RATIO))); ///< bulk modulus, kappa
  float xi = 0.8f;                        ///< hardening factor
  static constexpr float logJp0 = -0.01f;
  float beta = 0.5f;
  static constexpr float mohrColumbFriction =
      0.503599787772409; //< sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 -
                         // sin_phi);
  static constexpr float M =
      1.850343771924453; ///< mohrColumbFriction * (T)dim / sqrt((T)2 / ((T)6
                         ///< - dim));
  static constexpr float Msqr = 3.423772074299613;
  static constexpr bool hardeningOn = true;

  template <typename Allocator>
  ElementBuffer(Allocator allocator) : base_t{allocator} {}
};

/// conversion
/// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0608r3.html
using element_buffer_t =
    variant<ElementBuffer<fem_e::Tetrahedron>,
            ElementBuffer<fem_e::Brick>>;


struct VerticeArray : Instance<vertice_array_> {
  using base_t = Instance<vertice_array_>;
  VerticeArray &operator=(base_t &&instance) {
    static_cast<base_t &>(*this) = instance;
    return *this;
  }
};

struct ElementArray : Instance<element_array_> {
  using base_t = Instance<element_array_>;
  ElementArray &operator=(base_t &&instance) {
    static_cast<base_t &>(*this) = instance;
    return *this;
  }
};

} // namespace mn

#endif