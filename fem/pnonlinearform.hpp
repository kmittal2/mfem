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

#ifndef MFEM_PNONLINEARFORM
#define MFEM_PNONLINEARFORM

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pgridfunc.hpp"
#include "nonlinearform.hpp"

namespace mfem
{

/// Parallel non-linear operator on the true dofs
class ParNonlinearForm : public NonlinearForm
{
protected:
   mutable ParGridFunction X, Y;
   mutable OperatorHandle pGrad;

public:
   ParNonlinearForm(ParFiniteElementSpace *pf)
      : NonlinearForm(pf), X(pf), Y(pf), pGrad(Operator::Hypre_ParCSR)
   { height = width = pf->TrueVSize(); }

   ParFiniteElementSpace *ParFESpace() const
   { return (ParFiniteElementSpace *)fes; }

   // Here, rhs is a true dof vector
   virtual void SetEssentialBC(const Array<int> &bdr_attr_is_ess,
                               Vector *rhs = NULL);

   /// Compute the energy of a ParGridFunction
   virtual double GetEnergy(const ParGridFunction &x) const;

   /// Compute the energy of a true-dof vector 'x'
   virtual double GetEnergy(const Vector &x) const;

   virtual void Mult(const Vector &x, Vector &y) const;

   /// Return the local gradient matrix for the given true-dof vector x
   const SparseMatrix &GetLocalGradient(const Vector &x) const;

   virtual Operator &GetGradient(const Vector &x) const;

   /// Set the operator type id for the parallel gradient matrix/operator.
   void SetGradientType(Operator::Type tid) { pGrad.SetType(tid); }

   /// Get the parallel finite element space prolongation matrix
   virtual const Operator *GetProlongation() const
   { return ParFESpace()->GetProlongationMatrix(); }
   /// Get the parallel finite element space restriction matrix
   virtual const Operator *GetRestriction() const
   { return ParFESpace()->GetRestrictionMatrix(); }

   virtual ~ParNonlinearForm() { }
};


/** @brief A class representing a general parallel block nonlinear operator
    defined on the Cartesian product of multiple ParFiniteElementSpace%s. */
/** The ParBlockNonlinearForm takes as input, and returns as output, vectors on
    the true dofs. */
class ParBlockNonlinearForm : public BlockNonlinearForm
{
protected:
   mutable BlockVector xs_true, ys_true;
   mutable Array2D<OperatorHandle *> phBlockGrad;
   mutable BlockOperator *pBlockGrad;

public:
   /// Construct an empty ParBlockNonlinearForm. Initialize with SetParSpaces().
   ParBlockNonlinearForm() : pBlockGrad(NULL) { }

   /** @brief Construct a ParBlockNonlinearForm on the given set of
       ParFiniteElementSpace%s. */
   ParBlockNonlinearForm(Array<ParFiniteElementSpace *> &pf);

   /// Return the @a k-th parallel FE space of the ParBlockNonlinearForm.
   ParFiniteElementSpace *ParFESpace(int k);
   /** @brief Return the @a k-th parallel FE space of the ParBlockNonlinearForm
       (const version). */
   const ParFiniteElementSpace *ParFESpace(int k) const;

   /** @brief After a call to SetParSpaces(), the essential b.c. and the
       gradient-type (if different from the default) must be set again. */
   void SetParSpaces(Array<ParFiniteElementSpace *> &pf);

   // Here, rhs is a true dof vector
   virtual void SetEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                               Array<Vector *> &rhs);

   virtual void Mult(const Vector &x, Vector &y) const;

   /// Return the local block gradient matrix for the given true-dof vector x
   const BlockOperator &GetLocalGradient(const Vector &x) const;

   virtual BlockOperator &GetGradient(const Vector &x) const;

   /** @brief Set the operator type id for the blocks of the parallel gradient
       matrix/operator. The default type is Operator::Hypre_ParCSR. */
   void SetGradientType(Operator::Type tid);

   /// Destructor.
   virtual ~ParBlockNonlinearForm();
};

}

#endif // MFEM_USE_MPI

#endif
