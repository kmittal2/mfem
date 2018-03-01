// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_PLINEARFORM
#define MFEM_PLINEARFORM

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pgridfunc.hpp"
#include "linearform.hpp"
#include <complex>

namespace mfem
{

/// Class for parallel linear form
class ParLinearForm : public LinearForm
{
protected:
   ParFiniteElementSpace *pfes;

public:
   ParLinearForm() : LinearForm() { pfes = NULL; }

   ParLinearForm(ParFiniteElementSpace *pf) : LinearForm(pf) { pfes = pf; }

   /// Construct a ParLinearForm using previously allocated array @a data.
   /** The ParLinearForm does not assume ownership of @a data which is assumed
       to be of size at least `pf->GetVSize()`. Similar to the LinearForm and
       Vector constructors for externally allocated array, the pointer @a data
       can be NULL. The data array can be replaced later using the method
       SetData().
    */
   ParLinearForm(ParFiniteElementSpace *pf, double *data) :
      LinearForm(pf, data), pfes(pf) { }

   ParFiniteElementSpace *ParFESpace() const { return pfes; }

   void Update(ParFiniteElementSpace *pf = NULL);

   void Update(ParFiniteElementSpace *pf, Vector &v, int v_offset);

   /// Assemble the vector on the true dofs, i.e. P^t v.
   void ParallelAssemble(Vector &tv);

   /// Returns the vector assembled on the true dofs, i.e. P^t v.
   HypreParVector *ParallelAssemble();

   /// Return the action of the ParLinearForm as a linear mapping.
   /** Linear forms are linear functionals which map ParGridFunction%s to
       the real numbers.  This method performs this mapping which in
       this case is equivalent as an inner product of the ParLinearForm
       and ParGridFunction. */
   double operator()(const ParGridFunction &gf) const
   {
      return InnerProduct(pfes->GetComm(), *this, gf);
   }
};

class ParComplexLinearForm : public Vector
{
private:
   ComplexOperator::Convention conv_;

protected:
   ParLinearForm * plfr_;
   ParLinearForm * plfi_;

   HYPRE_Int * tdof_offsets_;

public:

   ParComplexLinearForm(ParFiniteElementSpace *pf,
                        const ComplexOperator::Convention &
                        convention = ComplexOperator::HERMITIAN);

   virtual ~ParComplexLinearForm();

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                            LinearFormIntegrator *lfi_imag);

   ParFiniteElementSpace *ParFESpace() const { return plfr_->ParFESpace(); }

   ParLinearForm & real() { return *plfr_; }
   ParLinearForm & imag() { return *plfi_; }
   const ParLinearForm & real() const { return *plfr_; }
   const ParLinearForm & imag() const { return *plfi_; }

   void Update(ParFiniteElementSpace *pf = NULL);

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   /// Assemble the vector on the true dofs, i.e. P^t v.
   void ParallelAssemble(Vector &tv);

   /// Returns the vector assembled on the true dofs, i.e. P^t v.
   HypreParVector *ParallelAssemble();

   std::complex<double> operator()(const ParComplexGridFunction &gf) const;

};

}

#endif // MFEM_USE_MPI

#endif
