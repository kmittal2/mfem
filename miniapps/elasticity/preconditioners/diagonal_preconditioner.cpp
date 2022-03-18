#include "diagonal_preconditioner.hpp"
#include "general/forall.hpp"
#include "linalg/tensor.hpp"

namespace mfem
{

using namespace mfem::internal;

void ElasticityDiagonalPreconditioner::SetOperator(const Operator &op)
{
   gradient_operator_ = dynamic_cast<const ElasticityGradientOperator *>(&op);
   MFEM_ASSERT(gradient_operator_ != nullptr,
               "Operator is not ElasticityGradientOperator");

   width = height = op.Height();

   gradient_operator_->AssembleGradientDiagonal(Ke_diag_, K_diag_local_,
                                                K_diag_);

   submat_height_ = gradient_operator_->elasticity_op_.h1_fes_.GetVDim();
   num_submats_ = gradient_operator_->elasticity_op_.h1_fes_.GetTrueVSize() /
                  gradient_operator_->elasticity_op_.h1_fes_.GetVDim();
}

void ElasticityDiagonalPreconditioner::Mult(const Vector &x,
                                            Vector &y) const
{
   const int ns = num_submats_, sh = submat_height_, nsh = ns * sh;

   const auto K_diag_submats = Reshape(K_diag_.Read(), ns, sh, sh);
   const auto X = Reshape(x.Read(), ns, dim);

   auto Y = Reshape(y.Write(), ns, dim);

   if (type_ == Type::Diagonal)
   {
      // Assuming Y and X are ordered byNODES. K_diag is ordered byVDIM.
      MFEM_FORALL(si, nsh,
      {
         const int s = si / sh;
         const int i = si % sh;
         Y(s,i) = X(s,i) / K_diag_submats(s, i, i);
      });
   }
   else if (type_ == Type::BlockDiagonal)
   {
      MFEM_FORALL(s, ns,
      {
         const auto submat = make_tensor<dim, dim>([&](int i, int j)
         {
            return K_diag_submats(s, i, j);
         });

         const auto submat_inv = inv(submat);

         const auto x_block = make_tensor<dim>([&](int i) { return X(s,i); });

         tensor<double, dim> y_block = submat_inv * x_block;

         for (int i = 0; i < dim; i++) { Y(s,i) = y_block(i); }
      });
   }
   else
   {
      MFEM_ABORT("Unknwon ElasticityDiagonalPreconditioner::Type");
   }
}
}