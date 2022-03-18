#ifndef MFEM_ELASTICITY_MAT_LIN_ELAST_HPP
#define MFEM_ELASTICITY_MAT_LIN_ELAST_HPP

#include "mfem.hpp"
#include "linalg/tensor.hpp"

using namespace mfem::internal;

/** @brief Linear elastic material.
 *
 * Defines a linear elastic material response. It satisfies the material_type
 * interface for ElasticityOperator::SetMaterial.
 */
template <int dim> struct LinearElasticMaterial
{
   /**
    * @brief Compute the stress response.
    *
    * @param[in] dudx derivative of the displacement
    * @return tensor<double, dim, dim>
    */
   tensor<double, dim, dim>
   MFEM_HOST_DEVICE stress(const tensor<double, dim, dim> &dudx) const
   {
      constexpr auto I = IsotropicIdentity<dim>();
      auto epsilon = sym(dudx);
      return lambda * tr(epsilon) * I + 2.0 * mu * epsilon;
   }

   /**
    * @brief Apply the gradient of the stress.
    *
    */
   tensor<double, dim, dim> MFEM_HOST_DEVICE
   action_of_gradient(const tensor<double, dim, dim> & /* dudx */,
                      const tensor<double, dim, dim> &ddudx) const
   {
      return stress(ddudx);
   }

   /**
    * @brief Compute the gradient.
    *
    * This method is used in the ElasticityDiagonalPreconditioner type to
    * compute the gradient matrix entries of the current quadrature point,
    * instead of the action.
    *
    * @param[in] dudx
    * @return tensor<double, dim, dim, dim, dim>
    */
   tensor<double, dim, dim, dim, dim>
   MFEM_HOST_DEVICE gradient(tensor<double, dim, dim> /* dudx */) const
   {
      return make_tensor<dim, dim, dim, dim>([&](int i, int j, int k, int l)
      {
         return lambda * (i == j) * (k == l) +
                mu * ((i == l) * (j == k) + (i == k) * (j == l));
      });
   }

   /// First Lame parameter
   double lambda = 100;
   /// Second Lame parameter
   double mu = 50;
};

#endif
