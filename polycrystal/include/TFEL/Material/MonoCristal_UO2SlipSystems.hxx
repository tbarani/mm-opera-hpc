/*!
* \file   include/TFEL/Material/MonoCristal_UO2SlipSystems.hxx
* \brief  this file decares the MonoCristal_UO2SlipSystems class.
*         File generated by tfel version 4.2.0-dev
* \author Luc Portelette / Thomas Helfer / Etienne Castelier
 */

#ifndef LIB_TFEL_MATERIAL_MONOCRISTAL_UO2SLIPSYSTEMS_HXX
#define LIB_TFEL_MATERIAL_MONOCRISTAL_UO2SLIPSYSTEMS_HXX

#if (defined _WIN32 || defined _WIN64)
#ifdef min
#undef min
#endif /* min */
#ifdef max
#undef max
#endif /* max */
#ifdef small
#undef small
#endif /* small */
#endif /* (defined _WIN32 || defined _WIN64) */

#include"TFEL/Raise.hxx"
#include"TFEL/Math/tvector.hxx"
#include"TFEL/Math/stensor.hxx"
#include"TFEL/Math/tensor.hxx"

namespace tfel::material{

template<typename real>
struct MonoCristal_UO2SlipSystems
{
//! a simple alias
using tensor = tfel::math::tensor<3u,real>;
//! a simple alias
using vector = tfel::math::tvector<3u,real>;
//! a simple alias
using stensor = tfel::math::stensor<3u,real>;
//! number of sliding systems
static constexpr unsigned short Nss0 = 6;
//! number of sliding systems
static constexpr unsigned short Nss1 = 6;
//! number of sliding systems
static constexpr unsigned short Nss2 = 12;
static constexpr unsigned short Nss = Nss0+Nss1+Nss2;
//! tensor of directional sense
tfel::math::tvector<Nss,tensor> mu;
//! symmetric tensor of directional sense
tfel::math::tvector<Nss,stensor> mus;
//! normal to slip plane
tfel::math::tvector<Nss,vector> np;
//! unit vector in the slip direction
tfel::math::tvector<Nss,vector> ns;
//! glimb tensors
tfel::math::tvector<Nss, stensor> climb_tensors;
//! tensor of directional sense
tfel::math::tvector<Nss0,tensor> mu0;
//! symmetric tensor of directional sense
tfel::math::tvector<Nss0,stensor> mus0;
//! normal to slip plane
tfel::math::tvector<Nss0,vector> np0;
//! glimb tensors
tfel::math::tvector<Nss0, stensor> climb_tensors0;
//! unit vector in the slip direction
tfel::math::tvector<Nss0,vector> ns0;
//! tensor of directional sense
tfel::math::tvector<Nss1,tensor> mu1;
//! symmetric tensor of directional sense
tfel::math::tvector<Nss1,stensor> mus1;
//! normal to slip plane
tfel::math::tvector<Nss1,vector> np1;
//! glimb tensors
tfel::math::tvector<Nss1, stensor> climb_tensors1;
//! unit vector in the slip direction
tfel::math::tvector<Nss1,vector> ns1;
//! tensor of directional sense
tfel::math::tvector<Nss2,tensor> mu2;
//! symmetric tensor of directional sense
tfel::math::tvector<Nss2,stensor> mus2;
//! normal to slip plane
tfel::math::tvector<Nss2,vector> np2;
//! glimb tensors
tfel::math::tvector<Nss2, stensor> climb_tensors2;
//! unit vector in the slip direction
tfel::math::tvector<Nss2,vector> ns2;
/*!
 * \return the gobal index of the jth system of ith family
 * \param[in] i: slip system family
 * \param[in] j: local slip system index
 */
constexpr unsigned short offset(const unsigned short,
const unsigned short) const;
/*!
 * \return the gobal index of the ith system of 0th family
 * \param[in] i: local slip system index
 */
constexpr unsigned short offset0(const unsigned short) const;
/*!
 * \return the gobal index of the ith system of 1th family
 * \param[in] i: local slip system index
 */
constexpr unsigned short offset1(const unsigned short) const;
/*!
 * \return the gobal index of the ith system of 2th family
 * \param[in] i: local slip system index
 */
constexpr unsigned short offset2(const unsigned short) const;
/*!
 * \return true if two systems are coplanar
 * \param[in] i: first slip system index
 * \param[in] j: second slip system index
 */
bool areCoplanar(const unsigned short,
                 const unsigned short) const;
/*!
 * \return an interaction matrix
 * \param[in] m: coefficients of the interaction matrix
 */
constexpr tfel::math::tmatrix<Nss, Nss, real>
buildInteractionMatrix(const tfel::math::fsarray<35, real>&) const;
//! return the unique instance of the class
static const MonoCristal_UO2SlipSystems&
getSlidingSystems();
//! return the unique instance of the class
static const MonoCristal_UO2SlipSystems&
getSlipSystems();
//! return the unique instance of the class
static const MonoCristal_UO2SlipSystems&
getGlidingSystems();
private:
//! Constructor
MonoCristal_UO2SlipSystems();
//! move constructor (disabled)
MonoCristal_UO2SlipSystems(MonoCristal_UO2SlipSystems&&) = delete;
//! copy constructor (disabled)
MonoCristal_UO2SlipSystems(const MonoCristal_UO2SlipSystems&) = delete;
//! move operator (disabled)
MonoCristal_UO2SlipSystems&
operator=(MonoCristal_UO2SlipSystems&&) = delete;
//! copy constructor (disabled)
MonoCristal_UO2SlipSystems&
operator=(const MonoCristal_UO2SlipSystems&) = delete;
}; // end of struct MonoCristal_UO2SlipSystems

//! a simple alias
template<typename real>
using MonoCristal_UO2SlidingSystems = MonoCristal_UO2SlipSystems<real>;

//! a simple alias
template<typename real>
using MonoCristal_UO2GlidingSystems = MonoCristal_UO2SlipSystems<real>;

} // end of namespace tfel::material

#include"TFEL/Material/MonoCristal_UO2SlipSystems.ixx"

#endif /* LIB_TFEL_MATERIAL_MONOCRISTAL_UO2SLIPSYSTEMS_HXX */
