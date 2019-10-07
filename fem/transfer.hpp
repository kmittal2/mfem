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

#ifndef MFEM_TRANSFER_HPP
#define MFEM_TRANSFER_HPP

#include "../linalg/linalg.hpp"
#include "fespace.hpp"

namespace mfem
{

/// Matrix-free transfer operator between finite element spaces
class TransferOperator : public Operator
{
 private:
   Operator* opr;

 public:
   /// Constructs a transfer operator from \p lFESpace to \p hFESpace. No
   /// matrices are assembled, only the action to a vector is being computed. If
   /// both spaces' FE collection pointers are pointing to the same collection
   /// we assume that the grid was refined while keeping the order constant.
   TransferOperator(const FiniteElementSpace& lFESpace,
                    const FiniteElementSpace& hFESpace);

   /// Destructor
   virtual ~TransferOperator();

   /// Interpolation or prolongation of a vector \p x corresponding to the
   /// coarse space to the vector \p y corresponding to the fine space.
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method. The vector \p x
   /// corresponding to the fine space is restricted to the vector \p y
   /// corresponding to the coarse space.
   virtual void MultTranspose(const Vector& x, Vector& y) const override;
};

/// Matrix-free transfer operator between finite element spaces on the same mesh
class OrderTransferOperator : public Operator
{
 private:
   const FiniteElementSpace& lFESpace;
   const FiniteElementSpace& hFESpace;

 public:
   /// Constructs a transfer operator from \p lFESpace to \p hFESpace which
   /// have different FE collections. No matrices are assembled, only the action
   /// to a vector is being computed. The underlying finite elements need to
   /// implement the GetTransferMatrix methods.
   OrderTransferOperator(const FiniteElementSpace& lFESpace_,
                         const FiniteElementSpace& hFESpace_);

   /// Destructor
   virtual ~OrderTransferOperator();

   /// Interpolation or prolongation of a vector \p x corresponding to the
   /// coarse space to the vector \p y corresponding to the fine space.
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method. The vector \p x
   /// corresponding to the fine space is restricted to the vector \p y
   /// corresponding to the coarse space.
   virtual void MultTranspose(const Vector& x, Vector& y) const override;

 private:
   /// Helper function to decode encoded dofs
   static inline int DecodeDof(int dof)
   {
      return (dof >= 0) ? dof : (-1 - dof);
   }
};

/// Matrix-free transfer operator between finite element spaces working on true
/// degrees of freedom
class TrueTransferOperator : public Operator
{
 private:
   TransferOperator* localTransferOperator;
   TripleProductOperator* opr;

 public:
   /// Constructs a transfer operator working on true degrees of freedom from \p
   /// lFESpace to \p hFESpace
   TrueTransferOperator(const FiniteElementSpace& lFESpace_,
                        const FiniteElementSpace& hFESpace_);

   /// Destructor
   ~TrueTransferOperator();

   /// Interpolation or prolongation of a true dof vector \p x corresponding to
   /// the coarse space to the true dof vector \p y corresponding to the fine
   /// space.
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method. The true dof
   /// vector \p x corresponding to the fine space is restricted to the true dof
   /// vector \p y corresponding to the coarse space.
   virtual void MultTranspose(const Vector& x, Vector& y) const override;
};

} // namespace mfem
#endif