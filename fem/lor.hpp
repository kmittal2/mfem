// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LOR
#define MFEM_LOR

#include "bilinearform.hpp"

namespace mfem
{

/// @brief Abstract base class for LOR and ParLOR classes, which construct
/// low-order refined versions of bilinear forms.
class LORBase
{
private:
   using GetIntegratorsFn = Array<BilinearFormIntegrator*> *(BilinearForm::*)();
   using GetMarkersFn = Array<Array<int>*> *(BilinearForm::*)();
   using AddIntegratorFn = void (BilinearForm::*)(BilinearFormIntegrator*);
   using AddIntegratorMarkersFn =
      void (BilinearForm::*)(BilinearFormIntegrator*, Array<int>&);

   IntegrationRules irs;
   const IntegrationRule *ir;
   std::map<BilinearFormIntegrator*, const IntegrationRule*> ir_map;

   void AddIntegrators(BilinearForm &a_to,
                       GetIntegratorsFn get_integrators,
                       AddIntegratorFn add_integrator);

   void AddIntegratorsAndMarkers(BilinearForm &a_to,
                                 GetIntegratorsFn get_integrators,
                                 GetMarkersFn get_markers,
                                 AddIntegratorMarkersFn add_integrator);


   void ResetIntegrationRules(GetIntegratorsFn get_integrators);

   static inline int absdof(int i) { return i < 0 ? -1-i : i; }

protected:
   enum FESpaceType { H1, ND, RT, L2, INVALID };

   BilinearForm &a_ho;
   Mesh *mesh;
   FiniteElementCollection *fec;
   FiniteElementSpace *fes;
   BilinearForm *a;
   OperatorHandle A;
   mutable Array<int> perm;

   /// Adds all the integrators from the member data @a a to the BilinearForm
   /// @a a_to. Marks @a a_to as using external integrators. If the mesh
   /// consists of tensor product elements, temporarily changes the integration
   /// rules of the integrators to use collocated quadrature for better
   /// conditioning of the LOR system.
   void AddIntegrators(BilinearForm &a_to);

   /// Resets the integration rules of the integrators of @a a to their original
   /// values.
   void ResetIntegrationRules();

   /// Construct the permutation that maps LOR DOFs to high-order DOFs. See
   /// GetDofPermutation
   void ConstructDofPermutation() const;

   /// Return the type of finite element space: H1, ND, RT or L2.
   FESpaceType GetFESpaceType() const;

   /// Return the order of the LOR space. 1 for H1 or ND, 0 for L2 or RT.
   int GetLOROrder() const;

   /// Assemble the LOR system.
   void AssembleSystem(const Array<int> &ess_tdof_list);

   LORBase(BilinearForm &a_);

public:
   /// Return the assembled LOR system
   const OperatorHandle &GetAssembledSystem() const;

   /// Returns the permutation that maps LOR DOFs to high-order DOFs. This
   /// permutation is constructed the first time it is requested. For H1 finite
   /// element spaces, this is the identity. For vector finite element spaces,
   /// this permutation is nontrivial. Returns an array @a perm such that, given
   /// an index @a i of a LOR dof, @a perm[i] is the index of the corresponding
   /// HO dof.
   const Array<int> &GetDofPermutation() const;

   /// Returns true if the LOR spaces requires a DOF permutation (if the
   /// corresponding LOR and HO DOFs are numbered differently), false otherwise.
   bool RequiresDofPermutation() const;

   /// Returns the low-order refined finite element space.
   FiniteElementSpace &GetFESpace() const { return *fes; }

   ~LORBase();
};

/// Create and assemble a low-order refined version of a BilinearForm.
class LOR : public LORBase
{
public:
   /// Construct the low-order refined version of @a a_ho using the given list
   /// of essential DOFs. The mesh is refined using the refinement type
   /// specified by @a ref_type (see Mesh::MakeRefined).
   LOR(BilinearForm &a_ho, const Array<int> &ess_tdof_list,
       int ref_type=BasisType::GaussLobatto);

   /// Return the assembled LOR operator as a SparseMatrix
   SparseMatrix &GetAssembledMatrix() const;
};

#ifdef MFEM_USE_MPI

/// Create and assemble a low-order refined version of a ParBilinearForm.
class ParLOR : public LORBase
{
public:
   /// Construct the low-order refined version of @a a_ho using the given list
   /// of essential DOFs. The mesh is refined using the refinement type
   /// specified by @a ref_type (see ParMesh::MakeRefined).
   ParLOR(ParBilinearForm &a_ho, const Array<int> &ess_tdof_list,
          int ref_type=BasisType::GaussLobatto);

   /// Return the assembled LOR operator as a HypreParMatrix.
   HypreParMatrix &GetAssembledMatrix() const;

   ParFiniteElementSpace &GetParFESpace() const;
};

#endif

/// @brief Represents a solver of type @a SolverType created using the low-order
/// refined version of the given BilinearForm or ParBilinearForm.
template <typename SolverType>
class LORSolver : public Solver
{
protected:
   LORBase *lor;
   bool own_lor = true;
   bool use_permutation = true;
   SolverType solver;
   mutable Vector px, py;
public:
   /// Create a solver of type @a SolverType, formed using the assembled
   /// SparseMatrix of the LOR version of @a a_ho. @see LOR
   LORSolver(BilinearForm &a_ho, const Array<int> &ess_tdof_list,
             int ref_type=BasisType::GaussLobatto)
   {
      lor = new LOR(a_ho, ess_tdof_list, ref_type);
      solver.SetOperator(*lor->GetAssembledSystem());
   }

#ifdef MFEM_USE_MPI
   /// Create a solver of type @a SolverType, formed using the assembled
   /// HypreParMatrix of the LOR version of @a a_ho. @see ParLOR
   LORSolver(ParBilinearForm &a_ho, const Array<int> &ess_tdof_list,
             int ref_type=BasisType::GaussLobatto)
   {
      lor = new ParLOR(a_ho, ess_tdof_list, ref_type);
      solver.SetOperator(*lor->GetAssembledSystem());
   }
#endif

   template <typename... Args>
   LORSolver(LORBase &lor_, Args... args) : solver(args...)
   {
      lor = &lor_;
      own_lor = false;
      solver.SetOperator(*lor->GetAssembledSystem());
   }

   /// Not supported.
   void SetOperator(const Operator &op)
   {
      MFEM_ABORT("LORSolver::SetOperator not supported.")
   }

   void Mult(const Vector &x, Vector &y) const
   {
      if (use_permutation && lor->RequiresDofPermutation())
      {
         const Array<int> &p = lor->GetDofPermutation();
         px.SetSize(x.Size());
         py.SetSize(y.Size());
         for (int i=0; i<x.Size(); ++i)
         { px[i] = p[i] < 0 ? -x[-1-p[i]] : x[p[i]]; }

         solver.Mult(px, py);

         for (int i=0; i<y.Size(); ++i)
         {
            int pi = p[i];
            int s = pi < 0 ? -1 : 1;
            y[pi < 0 ? -1-pi : pi] = s*py[i];
         }
      }
      else
      {
         solver.Mult(x, y);
      }
   }

   /** @brief Enable or disable the DOF permutation (enabled by default).

       The corresponding LOR and high-order DOFs may not have the same numbering
       (for example, when using ND or RT spaces), and so a permutation is
       required when applying the LOR solver as a preconditioner for the
       high-order problem. This permutation can be disabled (for example, in
       order to precondition the low-order problem directly). */
   void UsePermutation(bool use_permutation_)
   {
      use_permutation = use_permutation_;
   }

   /// Support the use of -> to call methods of the underlying solver.
   SolverType *operator->() const { return solver; }

   /// Access the underlying solver.
   SolverType &operator*() { return solver; }

   ~LORSolver() { if (own_lor) { delete lor; } }
};

} // namespace mfem

#endif
