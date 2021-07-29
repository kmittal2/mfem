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

#ifndef MFEM_MESH_OPERATORS
#define MFEM_MESH_OPERATORS

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "mesh.hpp"
#include "../fem/estimators.hpp"

#include <limits>

namespace mfem
{

/** @brief The MeshOperator class serves as base for mesh manipulation classes.

    The purpose of the class is to provide a common abstraction for various
    AMR mesh control schemes. The typical use in an AMR loop is illustrated
    in examples 6/6p and 15/15p.

    A more general loop that also supports sequences of mesh operators with
    multiple updates looks like this:
    \code
       for (...)
       {
          // computations on the current mesh ...
          while (mesh_operator->Apply(mesh))
          {
             // update FiniteElementSpaces and interpolate GridFunctions ...
             if (mesh_operator->Continue()) { break; }
          }
          if (mesh_operator->Stop()) { break; }
       }
    \endcode
 */
class MeshOperator
{
private:
   int mod;

protected:
   friend class MeshOperatorSequence;

   /** @brief Implementation of the mesh operation. Invoked by the Apply()
              public method.
       @return Combination of ActionInfo constants. */
   virtual int ApplyImpl(Mesh &mesh) = 0;

   /// Constructor to be used by derived classes.
   MeshOperator() : mod(NONE) { }

public:
   /** @brief Action and information constants and masks.

       Combinations of constants are returned by the Apply() virtual method and
       can be accessed directly with GetActionInfo() or indirectly with methods
       like Stop(), Continue(), etc. The information bits (MASK_INFO) can be set
       only when the update bit is set (see MASK_UPDATE). */
   enum Action
   {
      NONE        = 0, /**< continue with computations without updating spaces
                            or grid-functions, i.e. the mesh was not modified */
      CONTINUE    = 1, /**< update spaces and grid-functions and continue
                            computations with the new mesh */
      STOP        = 2, ///< a stopping criterion was satisfied
      REPEAT      = 3, /**< update spaces and grid-functions and call the
                            operator Apply() method again */
      MASK_UPDATE = 1, ///< bit mask for the "update" bit
      MASK_ACTION = 3  ///< bit mask for all "action" bits
   };

   enum Info
   {
      REFINED     = 4*1, ///< the mesh was refined
      DEREFINED   = 4*2, ///< the mesh was de-refined
      REBALANCED  = 4*3, ///< the mesh was rebalanced
      MASK_INFO   = ~3   ///< bit mask for all "info" bits
   };

   /** @brief Perform the mesh operation.
       @return true if FiniteElementSpaces and GridFunctions need to be updated.
   */
   bool Apply(Mesh &mesh) { return ((mod = ApplyImpl(mesh)) & MASK_UPDATE); }

   /** @brief Check if STOP action is requested, e.g. stopping criterion is
       satisfied. */
   bool Stop() const { return ((mod & MASK_ACTION) == STOP); }
   /** @brief Check if REPEAT action is requested, i.e. FiniteElementSpaces and
       GridFunctions need to be updated, and Apply() must be called again. */
   bool Repeat() const { return ((mod & MASK_ACTION) == REPEAT); }
   /** @brief Check if CONTINUE action is requested, i.e. FiniteElementSpaces
       and GridFunctions need to be updated and computations should continue. */
   bool Continue() const { return ((mod & MASK_ACTION) == CONTINUE); }

   /// Check if the mesh was refined.
   bool Refined() const { return ((mod & MASK_INFO) == REFINED); }
   /// Check if the mesh was de-refined.
   bool Derefined() const { return ((mod & MASK_INFO) == DEREFINED); }
   /// Check if the mesh was rebalanced.
   bool Rebalanced() const { return ((mod & MASK_INFO) == REBALANCED); }

   /** @brief Get the full ActionInfo value generated by the last call to
       Apply(). */
   int GetActionInfo() const { return mod; }

   /// Reset the MeshOperator.
   virtual void Reset() = 0;

   /// The destructor is virtual.
   virtual ~MeshOperator() { }
};


/** Composition of MeshOperators into a sequence. Use the Append() method to
    create the sequence. */
class MeshOperatorSequence : public MeshOperator
{
protected:
   int step;
   Array<MeshOperator*> sequence; ///< MeshOperators sequence, owned by us.

   /// Do not allow copy construction, due to assumed ownership.
   MeshOperatorSequence(const MeshOperatorSequence &) { }

   /** @brief Apply the MeshOperatorSequence.
       @return ActionInfo value corresponding to the last applied operator from
       the sequence. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /// Constructor. Use the Append() method to create the sequence.
   MeshOperatorSequence() : step(-1) { }

   /// Delete all operators from the sequence.
   virtual ~MeshOperatorSequence();

   /** @brief Add an operator to the end of the sequence.
       The MeshOperatorSequence assumes ownership of the operator. */
   void Append(MeshOperator *mc) { sequence.Append(mc); }

   /// Access the underlying sequence.
   Array<MeshOperator*> &GetSequence() { return sequence; }

   /// Reset all MeshOperators in the sequence.
   virtual void Reset();
};


/** @brief Mesh refinement operator using an error threshold.

    This class uses the given ErrorEstimator to estimate local element errors
    and then marks for refinement all elements i such that loc_err_i > threshold.
    The threshold is computed as
    \code
       threshold = max(total_err * total_fraction * pow(num_elements,-1.0/p),
                       local_err_goal);
    \endcode
    where p (=total_norm_p), total_fraction, and local_err_goal are settable
    parameters, total_err = (sum_i local_err_i^p)^{1/p}, when p < inf,
    or total_err = max_i local_err_i, when p = inf.
*/
class ThresholdRefiner : public MeshOperator
{
protected:
   ErrorEstimator &estimator;
   AnisotropicErrorEstimator *aniso_estimator;

   double total_norm_p;
   double total_err_goal;
   double total_fraction;
   double local_err_goal;
   long   max_elements;

   double threshold;
   long num_marked_elements;

   Array<Refinement> marked_elements;
   long current_sequence;

   int non_conforming;
   int nc_limit;

   double GetNorm(const Vector &local_err, Mesh &mesh) const;

   /** @brief Apply the operator to the mesh.
       @return STOP if a stopping criterion is satisfied or no elements were
       marked for refinement; REFINED + CONTINUE otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /// Construct a ThresholdRefiner using the given ErrorEstimator.
   ThresholdRefiner(ErrorEstimator &est);

   // default destructor (virtual)

   /** @brief Set the exponent, p, of the discrete p-norm used to compute the
       total error from the local element errors. */
   void SetTotalErrorNormP(double norm_p = infinity())
   { total_norm_p = norm_p; }

   /** @brief Set the total error stopping criterion: stop when
       total_err <= total_err_goal. The default value is zero. */
   void SetTotalErrorGoal(double err_goal) { total_err_goal = err_goal; }

   /** @brief Set the total fraction used in the computation of the threshold.
       The default value is 1/2.
       @note If fraction == 0, total_err is essentially ignored in the threshold
       computation, i.e. threshold = local error goal. */
   void SetTotalErrorFraction(double fraction) { total_fraction = fraction; }

   /** @brief Set the local stopping criterion: stop when
       local_err_i <= local_err_goal. The default value is zero.
       @note If local_err_goal == 0, it is essentially ignored in the threshold
       computation. */
   void SetLocalErrorGoal(double err_goal) { local_err_goal = err_goal; }

   /** @brief Set the maximum number of elements stopping criterion: stop when
       the input mesh has num_elements >= max_elem. The default value is
       LONG_MAX. */
   void SetMaxElements(long max_elem) { max_elements = max_elem; }

   /// Use nonconforming refinement, if possible (triangles, quads, hexes).
   void PreferNonconformingRefinement() { non_conforming = 1; }

   /** @brief Use conforming refinement, if possible (triangles, tetrahedra)
       -- this is the default. */
   void PreferConformingRefinement() { non_conforming = -1; }

   /** @brief Set the maximum ratio of refinement levels of adjacent elements
       (0 = unlimited). */
   void SetNCLimit(int nc_limit)
   {
      MFEM_ASSERT(nc_limit >= 0, "Invalid NC limit");
      this->nc_limit = nc_limit;
   }

   /// Get the number of marked elements in the last Apply() call.
   long GetNumMarkedElements() const { return num_marked_elements; }

   /// Get the threshold used in the last Apply() call.
   double GetThreshold() const { return threshold; }

   /// Reset the associated estimator.
   virtual void Reset();
};

// TODO: BulkRefiner to refine a portion of the global error


/** @brief De-refinement operator using an error threshold.

    This de-refinement operator marks elements in the hierarchy whose children
    are leaves and their combined error is below a given threshold. The
    errors of the children are combined by one of the following operations:
    - op = 0: minimum of the errors
    - op = 1: sum of the errors (default)
    - op = 2: maximum of the errors. */
class ThresholdDerefiner : public MeshOperator
{
protected:
   ErrorEstimator &estimator;

   double threshold;
   int nc_limit, op;

   /** @brief Apply the operator to the mesh.
       @return DEREFINED + CONTINUE if some elements were de-refined; NONE
       otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /// Construct a ThresholdDerefiner using the given ErrorEstimator.
   ThresholdDerefiner(ErrorEstimator &est)
      : estimator(est)
   {
      threshold = 0.0;
      nc_limit = 0;
      op = 1;
   }

   // default destructor (virtual)

   /// Set the de-refinement threshold. The default value is zero.
   void SetThreshold(double thresh) { threshold = thresh; }

   void SetOp(int op) { this->op = op; }

   /** @brief Set the maximum ratio of refinement levels of adjacent elements
       (0 = unlimited). */
   void SetNCLimit(int nc_limit)
   {
      MFEM_ASSERT(nc_limit >= 0, "Invalid NC limit");
      this->nc_limit = nc_limit;
   }

   /// Reset the associated estimator.
   virtual void Reset() { estimator.Reset(); }
};

/** @brief Refinement operator to control data oscillation.

    This class uses the given computes osc_K(f) := \| h \cdot (I - \Pi) f \|_K at
    each element K. Here, \Pi is the L2-projection and \| \cdot \|_K is the
    L2-norm, restricted to the element K. All elements satisfying the inequality
    \code
       osc_K(f) > threshold \cdot \| f \| / sqrt(n_el) ,
    \endcode
    are refined. Here, threshold is a postive parameter, \| \cdot \| is the
    L2-norm over the entire \Omega, and n_el is the number of elements in the
    mesh.

    Note that if osc(f) = threshold \cdot \| f \| / sqrt(n_el) for each K,
    then
    \code
       osc(f) = sqrt( sum_K osc_K^2(f)) = threshold \cdot \| f \| .
    \endcode
    This is the reason for the 1/sqrt(n_el) factor. */
class CoefficientRefiner : public MeshOperator
{
protected:
   int nc_limit = 1;
   int nonconforming = -1;
   int order;
   double threshold = 1.0e-3;
   double relative_osc = 0.0;
   Array<int> mesh_refinements;
   Coefficient *coeff = NULL;
   GridFunction *gf;
   const IntegrationRule *ir_default[Geometry::NumGeom];
   const IntegrationRule **irs = NULL;

   /** @brief Apply the operator to the mesh once.
       @return STOP if a stopping criterion is satisfied or no elements were
       marked for refinement; REFINED + CONTINUE otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /// Constructor
   CoefficientRefiner(int order_) : order(order_) { }

   /** @brief Apply the operator to the mesh max_it times or until tolerance
    *  achieved.
       @return STOP if a stopping criterion is satisfied or no elements were
       marked for refinement; REFINED + CONTINUE otherwise. */
   virtual int PreprocessMesh(Mesh &mesh, int max_it);

   int PreprocessMesh(Mesh &mesh)
   {
      int max_it = 10;
      return PreprocessMesh(mesh, max_it);
   }

   /// Set the de-refinement threshold. The default value is zero.
   void SetThreshold(double threshold_) { threshold = threshold_; }

   /// Set the de-refinement threshold. The default value is zero.
   void SetCoefficient(Coefficient &coeff_) { coeff = &coeff_; }

   /// Reset the oscillation order
   void SetOrder(double order_) { order = order_; }

   /** @brief Set the maximum ratio of refinement levels of adjacent elements
       (0 = unlimited). */
   void SetNCLimit(int nc_limit_)
   {
      MFEM_ASSERT(nc_limit_ >= 0, "Invalid NC limit");
      nc_limit = nc_limit_;
   }

   // Set a custom integration rule
   void SetIntRule(const IntegrationRule *irs_[]) { irs = irs_; }
   
   // Return data oscillation value
   double GetOsc() { return relative_osc; }

   /// Reset
   virtual void Reset();
};

/** @brief ParMesh rebalancing operator.

    If the mesh is a parallel mesh, perform rebalancing; otherwise, do nothing.
*/
class Rebalancer : public MeshOperator
{
protected:
   /** @brief Rebalance a parallel mesh (only non-conforming parallel meshes are
       supported).
       @return CONTINUE + REBALANCE on success, NONE otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /// Empty.
   virtual void Reset() { }
};

} // namespace mfem

#endif // MFEM_MESH_OPERATORS
