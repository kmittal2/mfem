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

#ifndef MFEM_PBILINEARFORM
#define MFEM_PBILINEARFORM

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <mpi.h>
#include "pfespace.hpp"
#include "pgridfunc.hpp"
#include "bilinearform.hpp"

namespace mfem
{
/*
// *****************************************************************************
// * ParAbstractBilinearForm
// *****************************************************************************
class AbstractParBilinearForm : public Operator
{
public:
   const ParFiniteElementSpace *pfes;
public:
   AbstractParBilinearForm(ParFiniteElementSpace *f) : Operator(f?f->GetVSize():0),
      pfes(f) { }
   virtual ~AbstractParBilinearForm() { }
   virtual void AddDomainIntegrator(AbstractBilinearFormIntegrator*) = 0;
   virtual void Assemble(int skip_zeros = 1) = 0;
   virtual void FormOperator(const Array<int> &ess_tdof_list,
                             Operator &A) = 0;
   virtual void FormLinearSystem(const Array<int> &ess_tdof_list,
                                 Vector &x, Vector &b,
                                 Operator *&A, Vector &X, Vector &B,
                                 int copy_interior=0) = 0;
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b,
                                   Vector &x) = 0;
   virtual void EnableStaticCondensation() =0;
   virtual void Mult(const Vector &x, Vector &y) const = 0;
   };*/

// ***************************************************************************
// * Par PA BilinearForm
// ***************************************************************************
class ParPABilinearForm : public PABilinearForm
{
protected:
   const Mesh *mesh;
   const ParFiniteElementSpace *trialFes;
   const ParFiniteElementSpace *testFes;
   Array<BilinearPAFormIntegrator*> integrators;
   mutable Vector localX, localY;
   kFiniteElementSpace *kfes;
public:
   ParPABilinearForm(ParFiniteElementSpace*);
   ~ParPABilinearForm();
   // *************************************************************************
   virtual void EnableStaticCondensation();
   virtual void AddDomainIntegrator(AbstractBilinearFormIntegrator*);
   void AddBoundaryIntegrator(AbstractBilinearFormIntegrator*);
   void AddInteriorFaceIntegrator(AbstractBilinearFormIntegrator*);
   void AddBoundaryFaceIntegrator(AbstractBilinearFormIntegrator*);
   // *************************************************************************
   virtual void Assemble(int skip_zeros = 1);
   virtual void FormOperator(const Array<int> &ess_tdof_list, Operator &A);
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         Operator *&A, Vector &X, Vector &B,
                         int copy_interior = 0);
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b,
                                   Vector &x);
   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void MultTranspose(const Vector &x, Vector &y) const;
};

// *****************************************************************************
// * Par FA BilinearForm
// * Class for parallel bilinear form
// *****************************************************************************
class ParFABilinearForm : public FABilinearForm
{
protected:
   ParFiniteElementSpace *pfes;
   mutable ParGridFunction X, Y; // used in TrueAddMult

   OperatorHandle p_mat, p_mat_e;

   bool keep_nbr_block;

   // Allocate mat - called when (mat == NULL && fbfi.Size() > 0)
   void pAllocMat();

   void AssembleSharedFaces(int skip_zeros = 1);

public:
   ParFABilinearForm(ParFiniteElementSpace *pf)
      : FABilinearForm(pf), pfes(pf),
        p_mat(Operator::Hypre_ParCSR), p_mat_e(Operator::Hypre_ParCSR)
   { keep_nbr_block = false; }

   ParFABilinearForm(ParFiniteElementSpace *pf, ParFABilinearForm *bf)
      : FABilinearForm(pf, bf), pfes(pf),
        p_mat(Operator::Hypre_ParCSR), p_mat_e(Operator::Hypre_ParCSR)
   { keep_nbr_block = false; }

   /** When set to true and the ParFABilinearForm has interior face integrators,
       the local SparseMatrix will include the rows (in addition to the columns)
       corresponding to face-neighbor dofs. The default behavior is to disregard
       those rows. Must be called before the first Assemble call. */
   void KeepNbrBlock(bool knb = true) { keep_nbr_block = knb; }

   /// Set the operator type id for the parallel matrix/operator.
   /** If using static condensation or hybridization, call this method *after*
       enabling it. */
   void SetOperatorType(Operator::Type tid)
   {
      p_mat.SetType(tid); p_mat_e.SetType(tid);
      if (hybridization) { hybridization->SetOperatorType(tid); }
      if (static_cond) { static_cond->SetOperatorType(tid); }
   }

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   HypreParMatrix *ParallelAssemble() { return ParallelAssemble(mat); }

   /// Returns the eliminated matrix assembled on the true dofs, i.e. P^t A_e P.
   /** The returned matrix has to be deleted by the caller. */
   HypreParMatrix *ParallelAssembleElim() { return ParallelAssemble(mat_e); }

   /// Return the matrix @a m assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   HypreParMatrix *ParallelAssemble(SparseMatrix *m);

   /** @brief Returns the matrix assembled on the true dofs, i.e.
       @a A = P^t A_local P, in the format (type id) specified by @a A. */
   void ParallelAssemble(OperatorHandle &A) { ParallelAssemble(A, mat); }

   /** Returns the eliminated matrix assembled on the true dofs, i.e.
       @a A_elim = P^t A_elim_local P in the format (type id) specified by @a A.
    */
   void ParallelAssembleElim(OperatorHandle &A_elim)
   { ParallelAssemble(A_elim, mat_e); }

   /** Returns the matrix @a A_local assembled on the true dofs, i.e.
       @a A = P^t A_local P in the format (type id) specified by @a A. */
   void ParallelAssemble(OperatorHandle &A, SparseMatrix *A_local);

   /// Eliminate essential boundary DOFs from a parallel assembled system.
   /** The array @a bdr_attr_is_ess marks boundary attributes that constitute
       the essential part of the boundary. */
   void ParallelEliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                                     HypreParMatrix &A,
                                     const HypreParVector &X,
                                     HypreParVector &B) const;

   /// Eliminate essential boundary DOFs from a parallel assembled matrix @a A.
   /** The array @a bdr_attr_is_ess marks boundary attributes that constitute
       the essential part of the boundary. The eliminated part is stored in a
       matrix A_elim such that A_original = A_new + A_elim. Returns a pointer to
       the newly allocated matrix A_elim which should be deleted by the caller.
       The matrices @a A and A_elim can be used to eliminate boundary conditions
       in multiple right-hand sides, by calling the function EliminateBC() (from
       hypre.hpp). */
   HypreParMatrix *ParallelEliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                                                HypreParMatrix &A) const;

   /// Eliminate essential true DOFs from a parallel assembled matrix @a A.
   /** Given a list of essential true dofs and the parallel assembled matrix
       @a A, eliminate the true dofs from the matrix, storing the eliminated
       part in a matrix A_elim such that A_original = A_new + A_elim. Returns a
       pointer to the newly allocated matrix A_elim which should be deleted by
       the caller. The matrices @a A and A_elim can be used to eliminate
       boundary conditions in multiple right-hand sides, by calling the function
       EliminateBC() (from hypre.hpp). */
   HypreParMatrix *ParallelEliminateTDofs(const Array<int> &tdofs_list,
                                          HypreParMatrix &A) const
   { return A.EliminateRowsCols(tdofs_list); }

   /** @brief Compute @a y += @a a (P^t A P) @a x, where @a x and @a y are
       vectors on the true dofs. */
   void TrueAddMult(const Vector &x, Vector &y, const double a = 1.0) const;

   /// Return the parallel FE space associated with the ParFABilinearForm.
   ParFiniteElementSpace *ParFESpace() const { return pfes; }

   /// Return the parallel trace FE space associated with static condensation.
   ParFiniteElementSpace *SCParFESpace() const
   { return static_cond ? static_cond->GetParTraceFESpace() : NULL; }

   /// Get the parallel finite element space prolongation matrix
   virtual const Operator *GetProlongation() const
   { return pfes->GetProlongationMatrix(); }
   /// Get the parallel finite element space restriction matrix
   virtual const Operator *GetRestriction() const
   { return pfes->GetRestrictionMatrix(); }

   /** Form the linear system A X = B, corresponding to the current bilinear
       form and b(.), by applying any necessary transformations such as:
       eliminating boundary conditions; applying conforming constraints for
       non-conforming AMR; parallel assembly; static condensation;
       hybridization.

       The ParGridFunction-size vector x must contain the essential b.c. The
       ParFABilinearForm and the ParLinearForm-size vector b must be assembled.

       The vector X is initialized with a suitable initial guess: when using
       hybridization, the vector X is set to zero; otherwise, the essential
       entries of X are set to the corresponding b.c. and all other entries are
       set to zero (copy_interior == 0) or copied from x (copy_interior != 0).

       This method can be called multiple times (with the same ess_tdof_list
       array) to initialize different right-hand sides and boundary condition
       values.

       After solving the linear system, the finite element solution x can be
       recovered by calling RecoverFEMSolution (with the same vectors X, b, and
       x). */
   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   /** Version of the method FormLinearSystem() where the system matrix is
       returned in the variable @a A, of type OpType, holding a *reference* to
       the system matrix (created with the method OpType::MakeRef()). The
       reference will be invalidated when SetOperatorType(), Update(), or the
       destructor is called. */
   template <typename OpType>
   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         OpType &A, Vector &X, Vector &B,
                         int copy_interior = 0)
   {
      OperatorHandle Ah;
      FormLinearSystem(ess_tdof_list, x, b, Ah, X, B, copy_interior);
      OpType *A_ptr = Ah.Is<OpType>();
      MFEM_VERIFY(A_ptr, "invalid OpType used");
      A.MakeRef(*A_ptr);
   }

   /// Form the linear system matrix @a A, see FormLinearSystem() for details.
   void FormSystemMatrix(const Array<int> &ess_tdof_list, OperatorHandle &A);

   /** Version of the method FormSystemMatrix() where the system matrix is
       returned in the variable @a A, of type OpType, holding a *reference* to
       the system matrix (created with the method OpType::MakeRef()). The
       reference will be invalidated when SetOperatorType(), Update(), or the
       destructor is called. */
   template <typename OpType>
   void FormSystemMatrix(const Array<int> &ess_tdof_list, OpType &A)
   {
      OperatorHandle Ah;
      FormSystemMatrix(ess_tdof_list, Ah);
      OpType *A_ptr = Ah.Is<OpType>();
      MFEM_VERIFY(A_ptr, "invalid OpType used");
      A.MakeRef(*A_ptr);
   }

   /** Call this method after solving a linear system constructed using the
       FormLinearSystem method to recover the solution as a ParGridFunction-size
       vector in x. Use the same arguments as in the FormLinearSystem call. */
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   virtual void Update(FiniteElementSpace *nfes = NULL);

   virtual ~ParFABilinearForm() { }
};

// *****************************************************************************
// * BilinearForm ⇒ (PA|FA) BilinearForm
// *****************************************************************************
class ParBilinearForm
{
private:
   const bool FA = true;
   ParPABilinearForm *ppabf;
   ParFABilinearForm *pfabf;
public:
   ParBilinearForm(ParFiniteElementSpace *f):
      FA(config::Get().PA()==false),
      ppabf(FA?NULL:new ParPABilinearForm(f)),
      pfabf(FA?new ParFABilinearForm(f):NULL)
   { }
   virtual ~ParBilinearForm() {}
   // **************************************************************************
   void EnableStaticCondensation() {assert(false);}
   void AddDomainIntegrator(AbstractBilinearFormIntegrator *i)
   {
      (FA?
       pfabf->AddDomainIntegrator(i):
       ppabf->AddDomainIntegrator(i));
   }
   // **************************************************************************
   virtual void Assemble()
   {
      (FA?
       pfabf->Assemble():
       ppabf->Assemble());
   }
   virtual void FormOperator(const Array<int> &ess_tdof_list,
                             Operator &A)
   {
      (FA?
       pfabf->FormOperator(ess_tdof_list,A):
       ppabf->FormOperator(ess_tdof_list,A));
   }

   template <typename OpType>
   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         OpType &A, Vector &X, Vector &B,
                         int copy_interior = 0)
   {
      assert(FA);
      pfabf->FormLinearSystem(ess_tdof_list,x,b,A,X,B,copy_interior);
   }

   virtual void FormLinearSystem(const Array<int> &ess_tdof_list,
                                 Vector &x, Vector &b,
                                 Operator *&A, Vector &X, Vector &B,
                                 int copy_interior =0)
   {
      (FA? pfabf->FormLinearSystem(ess_tdof_list,x,b,*static_cast<HypreParMatrix*>(A),
                                   X,B,copy_interior) :
       ppabf->FormLinearSystem(ess_tdof_list,x,b,A,X,B,copy_interior));
   }
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b,
                                   Vector &x)
   {
      (FA?
       pfabf->RecoverFEMSolution(X,b,x):
       ppabf->RecoverFEMSolution(X,b,x));
   }
   HypreParMatrix *ParallelAssemble() { assert(false); return NULL; }
   virtual void Finalize(int skip_zeros = 1) {assert(false);}
   virtual void Mult(const Vector &x, Vector &y) const {assert(false);}
   virtual void MultTranspose(const Vector &x, Vector &y) const {assert(false);}
};

// *****************************************************************************
/// Class for parallel bilinear form using different test and trial FE spaces.
// *****************************************************************************
class ParMixedBilinearForm : public MixedBilinearForm
{
protected:
   ParFiniteElementSpace *trial_pfes;
   ParFiniteElementSpace *test_pfes;
   mutable ParGridFunction X, Y; // used in TrueAddMult

public:
   ParMixedBilinearForm(ParFiniteElementSpace *trial_fes,
                        ParFiniteElementSpace *test_fes)
      : MixedBilinearForm(trial_fes, test_fes)
   {
      trial_pfes = trial_fes;
      test_pfes  = test_fes;
   }

   /// Returns the matrix assembled on the true dofs, i.e. P_test^t A P_trial.
   HypreParMatrix *ParallelAssemble();

   /** @brief Returns the matrix assembled on the true dofs, i.e.
       @a A = P_test^t A_local P_trial, in the format (type id) specified by
       @a A. */
   void ParallelAssemble(OperatorHandle &A);

   /// Compute y += a (P^t A P) x, where x and y are vectors on the true dofs
   void TrueAddMult(const Vector &x, Vector &y, const double a = 1.0) const;

   virtual ~ParMixedBilinearForm() { }
};

/** The parallel matrix representation a linear operator between parallel finite
    element spaces */
class ParDiscreteLinearOperator : public DiscreteLinearOperator
{
protected:
   ParFiniteElementSpace *domain_fes;
   ParFiniteElementSpace *range_fes;

public:
   ParDiscreteLinearOperator(ParFiniteElementSpace *dfes,
                             ParFiniteElementSpace *rfes)
      : DiscreteLinearOperator(dfes, rfes) { domain_fes=dfes; range_fes=rfes; }

   /// Returns the matrix "assembled" on the true dofs
   HypreParMatrix *ParallelAssemble() const;

   /** Extract the parallel blocks corresponding to the vector dimensions of the
       domain and range parallel finite element spaces */
   void GetParBlocks(Array2D<HypreParMatrix *> &blocks) const;

   virtual ~ParDiscreteLinearOperator() { }
};

}

#endif // MFEM_USE_MPI

#endif
