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

#ifndef MFEM_CONSTRAINED
#define MFEM_CONSTRAINED

#include "solvers.hpp"
#include "blockoperator.hpp"
#include "sparsemat.hpp"
#include "hypre.hpp"

namespace mfem
{

/** @brief An abstract class to solve the constrained system \f$ Ax = f \f$
    subject to the constraint \f$ B x = r \f$.

    Although implementations may not use the below formulation, for
    understanding some of its methods and notation you can think of
    it as solving the saddle-point system

     (  A   B^T  )  ( x )         (  f  )
     (  B        )  ( lambda)  =  (  r  )

    Not to be confused with ConstrainedOperator, which is totally
    different.

    This abstract object unifies handling of the "dual" rhs \f$ r \f$
    and the Lagrange multiplier solution \f$ \lambda \f$, so that derived
    classses can reuse that code. */
class ConstrainedSolver : public IterativeSolver
{
public:
   ConstrainedSolver(MPI_Comm comm, Operator& A_, Operator& B_);
   virtual ~ConstrainedSolver();

   virtual void SetOperator(const Operator& op) { }

   /** @brief Set the right-hand side r for the constraint B x = r

       (r defaults to zero if you don't call this) */
   virtual void SetConstraintRHS(const Vector& r);

   /** @brief Return the Lagrange multiplier solution in lambda

       Does not make sense unless you've already solved the constrained
       system with Mult() */
   void GetMultiplierSolution(Vector& lambda) const { lambda = multiplier_sol; }

   /** @brief Solve for \f$ x \f$ given \f$ f \f$.

       If you want to set \f$ r \f$, call SetConstraintRHS() before this.

       If you want to get \f$ \lambda \f$, call GetMultiplierSolution() after
       this.

       The implementation for the base class calls SaddleMult(), so
       a derived class must implement either Mult() or SaddleMult() */
   virtual void Mult(const Vector& f, Vector& x) const override;

protected:
   /** @brief Solve for (x, lambda) given (f, r)

       Derived classes must implement either this or Mult(). */
   virtual void SaddleMult(const Vector& f_and_r, Vector& x_and_lambda) const
   {
      mfem_error("Not Implemented!");
   }

   Operator& A;
   Operator& B;

   Vector constraint_rhs;
   mutable Vector multiplier_sol;
   mutable Vector workb;
   mutable Vector workx;
};


/** @brief Perform elimination of a single constraint.

    See EliminatinoProjection, EliminationCGSolver

    This keeps track of primary / secondary tdofs and does small dense block
    solves to eliminate constraints from a global system.

    \f$ B_s^{-1} \f$ maps lagrange space into secondary displacements,
    \f$ -B_s^{-1} B_p \f$ maps primary displacements to secondary displacements

    @todo should interface operate on small vectors (as below), or on
    large vectors, so that this object does its own GetSubVector()
    SetSubVector() with the primary/secondary dof Arrays that it owns? */
class Eliminator
{
public:
   Eliminator(const SparseMatrix& B, const Array<int>& lagrange_dofs,
              const Array<int>& primary_tdofs,
              const Array<int>& secondary_tdofs);

   const Array<int>& LagrangeDofs() const { return lagrange_tdofs_; }
   const Array<int>& PrimaryDofs() const { return primary_tdofs_; }
   const Array<int>& SecondaryDofs() const { return secondary_tdofs_; }

   /// Given primary displacements, return secondary displacements
   /// This applies \f$ -B_s^{-1} B_p \f$.
   void Eliminate(const Vector& in, Vector& out) const;

   /// Transpose of Eliminate(), applies \f$ -B_p^T B_s^{-T} \f$
   void EliminateTranspose(const Vector& in, Vector& out) const;

   /// Maps Lagrange multipliers to secondary displacements,
   /// applies \f$ B_s^{-1} \f$
   void LagrangeSecondary(const Vector& in, Vector& out) const;

   /// Transpose of LagrangeSecondary()
   void LagrangeSecondaryTranspose(const Vector& in, Vector& out) const;

   /// Return \f$ -B_s^{-1} B_p \f$ explicitly assembled in mat
   void ExplicitAssembly(DenseMatrix& mat) const;

private:
   Array<int> lagrange_tdofs_;
   Array<int> primary_tdofs_; // in original displacement ordering
   Array<int> secondary_tdofs_;

   DenseMatrix Bp_;
   DenseMatrix Bs_;  // gets inverted in place
   LUFactors Bsinverse_;
   /// @todo there is probably a better way to handle the B_s^{-T}
   DenseMatrix BsT_;   // gets inverted in place
   LUFactors BsTinverse_;
   Array<int> ipiv_;
   Array<int> ipivT_;
};

/** Collects action of several Eliminator objects to perform elimination of
    constraints.

    Works in parallel, but each Eliminator must be processor local, and must
    operate on disjoint degrees of freedom (ie, the primary and secondary dofs
    for one Eliminator must have no overlap with any dofs from a different
    Eliminator). */
class EliminationProjection : public Operator
{
public:
   EliminationProjection(const Operator& A, Array<Eliminator*>& eliminators);

   void Mult(const Vector& x, Vector& y) const;

   void MultTranspose(const Vector& x, Vector& y) const;

   /** @brief Assemble this projector as a (processor-local) SparseMatrix.

       Some day we may also want to try approximate variants. */
   SparseMatrix * AssembleExact() const;

   /** Given Lagrange multiplier right-hand-side \f$ g \f$, return
       \f$ \tilde{g} \f$ */
   void BuildGTilde(const Vector& g, Vector& gtilde) const;

   /** After a solve, recover the Lagrange multiplier. */
   void RecoverMultiplier(const Vector& disprhs,
                          const Vector& disp, Vector& lm) const;

private:
   const Operator& A_;
   Array<Eliminator*> eliminators_;
};

#ifdef MFEM_USE_MPI

/** @brief Solve constrained system by eliminating the constraint; see
    ConstrainedSolver

    Solves the system with the operator \f$ P^T A P + Z_P \f$, where P is
    EliminationProjection and Z_P is the identity on the eliminated dofs. */
class EliminationCGSolver : public ConstrainedSolver
{
public:
   /** @brief Constructor, with explicit splitting into primary/secondary dofs.

       This constructor uses a single elimination block (per processor), which
       provides the most general algorithm but is also not scalable

       The secondary_dofs are eliminated from the system in this algorithm,
       as they can be written in terms of the primary_dofs. */
   EliminationCGSolver(HypreParMatrix& A, SparseMatrix& B,
                       Array<int>& primary_dofs,
                       Array<int>& secondary_dofs,
                       int dimension=0);

   /** @brief Constructor, elimination is by blocks.

       The nonzeros in B are assumed to be in disjoint rows and columns; the
       rows are identified with the lagrange_rowstarts array, the secondary
       dofs are assumed to be the first nonzeros in the rows. */
   EliminationCGSolver(HypreParMatrix& A, SparseMatrix& B,
                       Array<int>& lagrange_rowstarts,
                       int dimension=0);

   ~EliminationCGSolver();

   void Mult(const Vector& x, Vector& y) const override;

private:
   /// Utility routine for constructors
   void BuildPreconditioner(int dimension);

   HypreParMatrix& hA_;
   Array<Eliminator*> elims_;
   EliminationProjection * projector_;
   HypreParMatrix * h_explicit_operator_;
   HypreBoomerAMG * prec_;
};

/** @brief Solve constrained system with penalty method; see ConstrainedSolver.

    Uses a HypreBoomerAMG preconditioner for the penalized system. Only
    approximates the solution, better approximation with higher penalty,
    but with higher penalty the preconditioner is less effective. */
class PenaltyConstrainedSolver : public ConstrainedSolver
{
public:
   PenaltyConstrainedSolver(MPI_Comm comm, HypreParMatrix& A,
                            SparseMatrix& B, double penalty_,
                            int dimension=0);

   PenaltyConstrainedSolver(MPI_Comm comm, HypreParMatrix& A,
                            HypreParMatrix& B, double penalty_,
                            int dimension=0);

   ~PenaltyConstrainedSolver();

   void Mult(const Vector& x, Vector& y) const override;

private:
   void Initialize(HypreParMatrix& A, HypreParMatrix& B, int dimension);

   double penalty;
   Operator& constraintB;
   HypreParMatrix * penalized_mat;
   HypreBoomerAMG * prec;
};

#endif

/** @brief Solve constrained system by solving original mixed sysetm;
    see ConstrainedSolver.

    Solves the saddle-point problem with a block-diagonal preconditioner, with
    user-provided preconditioner in the top-left block and (by default) an
    identity matrix in the bottom-right.

    This is the most general ConstrainedSolver, needing only Operator objects
    to function. But in general it is not very efficient or scalable. */
class SchurConstrainedSolver : public ConstrainedSolver
{
public:
   /// Setup constrained system, with primal_pc a user-provided preconditioner
   /// for the top-left block.
   SchurConstrainedSolver(MPI_Comm comm, Operator& A_, Operator& B_,
                          Solver& primal_pc_);
   virtual ~SchurConstrainedSolver();

   virtual void SaddleMult(const Vector& x, Vector& y) const override;

protected:
   SchurConstrainedSolver(MPI_Comm comm, Operator& A_, Operator& B_);

   Array<int> offsets;
   BlockOperator * block_op;  // owned
   TransposeOperator * tr_B;  // owned
   Solver * primal_pc; // NOT owned
   BlockDiagonalPreconditioner * block_pc;  // owned
   Solver * dual_pc;  // owned

private:
   void Initialize();
};


/** @brief Basic saddle-point solver with assembled blocks (ie, the
    operators are assembled HypreParMatrix objects.)

    This uses a block-diagonal preconditioner that approximates
    \f$ [ A^{-1} 0; 0 (B A^{-1} B^T)^{-1} ] \f$.

    In the top-left block, we approximate \f$ A^{-1} \f$ with HypreBoomerAMG.
    In the bottom-right, we approximate \f$ A^{-1} \f$ with the inverse of the
    diagonal of \f$ A \f$, assemble \f$ B D^{-1} B^T \f$, and use
    HypreBoomerAMG on that assembled matrix. */
class SchurConstrainedHypreSolver : public SchurConstrainedSolver
{
public:
   SchurConstrainedHypreSolver(MPI_Comm comm, HypreParMatrix& hA_,
                               HypreParMatrix& hB_, int dimension=0);
   virtual ~SchurConstrainedHypreSolver();

private:
   HypreParMatrix& hA;
   HypreParMatrix& hB;
   HypreParMatrix * schur_mat;
};

}

#endif
