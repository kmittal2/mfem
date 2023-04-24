// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DOFTRANSFORM
#define MFEM_DOFTRANSFORM

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "intrules.hpp"

namespace mfem
{

class StatelessDofTransformation
{
protected:
   int size_;

   StatelessDofTransformation(int size)
      : size_(size) {}

public:
   inline int Size() const { return size_; }
   inline int Height() const { return size_; }
   inline int NumRows() const { return size_; }
   inline int Width() const { return size_; }
   inline int NumCols() const { return size_; }

   virtual void TransformPrimal(const Array<int> & face_orientation,
                                double *v) const = 0;
   virtual void TransformPrimal(const Array<int> & face_orientation,
                                Vector &v) const
   { TransformPrimal(face_orientation, v.GetData()); }

   virtual void InvTransformPrimal(const Array<int> & face_orientation,
                                   double *v) const = 0;
   virtual void InvTransformPrimal(const Array<int> & face_orientation,
                                   Vector &v) const
   { InvTransformPrimal(face_orientation, v.GetData()); }

   virtual void TransformDual(const Array<int> & face_orientation,
                              double *v) const = 0;
   virtual void TransformDual(const Array<int> & face_orientation,
                              Vector &v) const
   { TransformDual(face_orientation, v.GetData()); }

   virtual void InvTransformDual(const Array<int> & face_orientation,
                                 double *v) const = 0;
   virtual void InvTransformDual(const Array<int> & face_orientation,
                                 Vector &v) const
   { InvTransformDual(face_orientation, v.GetData()); }
};

/** The DofTransformation class is an abstract base class for a family of
    transformations that map local degrees of freedom (DoFs), contained within
    individual elements, to global degrees of freedom, stored within
    GridFunction objects. These transformations are necessary to ensure that
    basis functions in neighboring elements align correctly. Closely related but
    complementary transformations are required for the entries stored in
    LinearForm and BilinearForm objects. The DofTransformation class is designed
    to apply the action of both of these types of DoF transformations.

    Let the "primal transformation" be given by the operator T. This means that
    given a local element vector v the data that must be placed into a
    GridFunction object is v_t = T * v.

    We also need the inverse of the primal transformation T^{-1} so that we can
    recover the local element vector from data read out of a GridFunction
    e.g. v = T^{-1} * v_t.

    We need to preserve the action of our linear forms applied to primal
    vectors. In other words, if f is the local vector computed by a linear
    form then f * v = f_t * v_t (where "*" represents an inner product of
    vectors). This requires that f_t = T^{-T} * f i.e. the "dual transform" is
    given by the transpose of the inverse of the primal transformation.

    For bilinear forms we require that v^T * A * v = v_t^T * A_t * v_t. This
    implies that A_t = T^{-T} * A * T^{-1}. This can be accomplished by
    performing dual transformations of the rows and columns of the matrix A.

    For discrete linear operators the range must be modified with the primal
    transformation rather than the dual transformation because the result is a
    primal vector rather than a dual vector. This leads to the transformation
    D_t = T * D * T^{-1}. This can be accomplished by using a primal
    transformation on the columns of D and a dual transformation on its rows.
*/
class DofTransformation : virtual public StatelessDofTransformation
{
protected:
   Array<int> Fo;

   DofTransformation(int size)
      : StatelessDofTransformation(size) {}

public:

   /** @brief Configure the transformation using face orientations for the
       current element. */
   /// The face_orientation array can be obtained from Mesh::GetElementFaces.
   inline void SetFaceOrientations(const Array<int> & face_orientation)
   { Fo = face_orientation; }

   inline const Array<int> & GetFaceOrientations() const { return Fo; }

   using StatelessDofTransformation::TransformPrimal;
   using StatelessDofTransformation::InvTransformPrimal;
   using StatelessDofTransformation::TransformDual;
   using StatelessDofTransformation::InvTransformDual;

   /** Transform local DoFs to align with the global DoFs. For example, this
       transformation can be used to map the local vector computed by
       FiniteElement::Project() to the transformed vector stored within a
       GridFunction object. */
   virtual void TransformPrimal(double *v) const
   { TransformPrimal(Fo, v); }
   virtual void TransformPrimal(Vector &v) const
   { TransformPrimal(v.GetData()); }

   /// Transform groups of DoFs stored as dense matrices
   virtual void TransformPrimalCols(DenseMatrix &V) const;

   /** Inverse transform local DoFs. Used to transform DoFs from a global vector
       back to their element-local form. For example, this must be used to
       transform the vector obtained using GridFunction::GetSubVector before it
       can be used to compute a local interpolation.
   */
   virtual void InvTransformPrimal(double *v) const
   { InvTransformPrimal(Fo, v); }
   virtual void InvTransformPrimal(Vector &v) const
   { InvTransformPrimal(v.GetData()); }

   /** Transform dual DoFs as computed by a LinearFormIntegrator before summing
       into a LinearForm object. */
   virtual void TransformDual(double *v) const
   { TransformDual(Fo, v); }
   virtual void TransformDual(Vector &v) const
   { TransformDual(v.GetData()); }

   /** Inverse Transform dual DoFs */
   virtual void InvTransformDual(double *v) const
   { InvTransformDual(Fo, v); }
   virtual void InvTransformDual(Vector &v) const
   { InvTransformDual(v.GetData()); }

   /** Transform a matrix of dual DoFs entries as computed by a
       BilinearFormIntegrator before summing into a BilinearForm object. */
   virtual void TransformDual(DenseMatrix &V) const;

   /// Transform groups of dual DoFs stored as dense matrices
   virtual void TransformDualRows(DenseMatrix &V) const;
   virtual void TransformDualCols(DenseMatrix &V) const;

   virtual ~DofTransformation() {}
};

/** Transform a matrix of DoFs entries from different finite element spaces as
    computed by a DiscreteInterpolator before copying into a
    DiscreteLinearOperator.
*/
void TransformPrimal(const DofTransformation *ran_dof_trans,
                     const DofTransformation *dom_dof_trans,
                     DenseMatrix &elmat);

/** Transform a matrix of dual DoFs entries from different finite element spaces
    as computed by a BilinearFormIntegrator before summing into a
    MixedBilinearForm object.
*/
void TransformDual(const DofTransformation *ran_dof_trans,
                   const DofTransformation *dom_dof_trans,
                   DenseMatrix &elmat);

class StatelessVDofTransformation : virtual public StatelessDofTransformation
{
protected:
   int vdim_;
   int ordering_;
   StatelessDofTransformation * sdoftrans_;

public:
   /** @brief Default constructor which requires that SetDofTransformation be
       called before use. */
   StatelessVDofTransformation(int vdim = 1, int ordering = 0)
      : StatelessDofTransformation(0)
      , vdim_(vdim)
      , ordering_(ordering)
      , sdoftrans_(NULL)
   {}

   /// Constructor with a known StatelessDofTransformation
   StatelessVDofTransformation(StatelessDofTransformation & doftrans,
                               int vdim = 1,
                               int ordering = 0)
      : StatelessDofTransformation(vdim * doftrans.Size())
      , vdim_(vdim)
      , ordering_(ordering)
      , sdoftrans_(&doftrans)
   {}

   /// Set or change the vdim parameter
   inline void SetVDim(int vdim)
   {
      vdim_ = vdim;
      if (sdoftrans_)
      {
         size_ = vdim_ * sdoftrans_->Size();
      }
   }

   /// Return the current vdim value
   inline int GetVDim() const { return vdim_; }

   /// Set or change the nested StatelessDofTransformation object
   virtual void SetDofTransformation(StatelessDofTransformation & doftrans)
   {
      size_ = vdim_ * doftrans.Size();
      sdoftrans_ = &doftrans;
   }

   /// Return the nested StatelessDofTransformation object
   inline StatelessDofTransformation * GetDofTransformation() const
   { return sdoftrans_; }

   using StatelessDofTransformation::TransformPrimal;
   using StatelessDofTransformation::InvTransformPrimal;
   using StatelessDofTransformation::TransformDual;
   using StatelessDofTransformation::InvTransformDual;

   void TransformPrimal(const Array<int> & face_ori, double *v) const;
   void InvTransformPrimal(const Array<int> & face_ori, double *v) const;
   void TransformDual(const Array<int> & face_ori, double *v) const;
   void InvTransformDual(const Array<int> & face_ori, double *v) const;
};

/** The VDofTransformation class implements a nested transformation where an
    arbitrary DofTransformation is replicated with a vdim >= 1.
*/
class VDofTransformation : public StatelessVDofTransformation,
   public DofTransformation
{
protected:
   DofTransformation * doftrans_;

public:
   /** @brief Default constructor which requires that SetDofTransformation be
       called before use. */
   VDofTransformation(int vdim = 1, int ordering = 0)
      : StatelessDofTransformation(0)
      , StatelessVDofTransformation(vdim, ordering)
      , DofTransformation(0)
      , doftrans_(NULL)
   {}

   /// Constructor with a known DofTransformation
   /// @note The face orientations in @a doftrans will be copied into the
   /// new VDofTransformation object.
   VDofTransformation(DofTransformation & doftrans, int vdim = 1,
                      int ordering = 0)
      : StatelessDofTransformation(vdim * doftrans.Size())
      , StatelessVDofTransformation(doftrans, vdim, ordering)
      , DofTransformation(vdim * doftrans.Size())
      , doftrans_(&doftrans)
   {
      DofTransformation::SetFaceOrientations(doftrans.GetFaceOrientations());
   }

   using StatelessVDofTransformation::SetDofTransformation;

   /// Set or change the nested DofTransformation object
   /// @note The face orientations in @a doftrans will be copied into the
   /// VDofTransformation object.
   void SetDofTransformation(DofTransformation & doftrans)
   {
      doftrans_ = &doftrans;
      StatelessVDofTransformation::SetDofTransformation(doftrans);
      DofTransformation::SetFaceOrientations(doftrans.GetFaceOrientations());
   }

   /// Return the nested DofTransformation object
   inline DofTransformation * GetDofTransformation() const { return doftrans_; }

   inline void SetFaceOrientations(const Array<int> & face_orientation)
   {
      DofTransformation::SetFaceOrientations(face_orientation);
      if (doftrans_) { doftrans_->SetFaceOrientations(face_orientation); }
   }

   using DofTransformation::TransformPrimal;
   using DofTransformation::InvTransformPrimal;
   using DofTransformation::TransformDual;
   using DofTransformation::InvTransformDual;

   inline void TransformPrimal(double *v) const
   { TransformPrimal(Fo, v); }
   inline void InvTransformPrimal(double *v) const
   { InvTransformPrimal(Fo, v); }
   inline void TransformDual(double *v) const
   { TransformDual(Fo, v); }
   inline void InvTransformDual(double *v) const
   { InvTransformDual(Fo, v); }
};

/** Abstract base class for high-order Nedelec spaces on elements with
    triangular faces.

    The Nedelec DoFs on the interior of triangular faces come in pairs which
    share an interpolation point but have different vector directions. These
    directions depend on the orientation of the face and can therefore differ in
    neighboring elements. The mapping required to transform these DoFs can be
    implemented as series of 2x2 linear transformations. The raw data for these
    linear transformations is stored in the T_data and TInv_data arrays and can
    be accessed as DenseMatrices using the GetFaceTransform() and
    GetFaceInverseTransform() methods.
*/
class ND_StatelessDofTransformation : virtual public StatelessDofTransformation
{
private:
   static const double T_data[24];
   static const double TInv_data[24];
   static const DenseTensor T, TInv;

protected:
   const int order;  // basis function order
   const int nedofs; // number of DoFs per edge
   const int nfdofs; // number of DoFs per face
   const int nedges; // number of edges per element
   const int nfaces; // number of triangular faces per element

   ND_StatelessDofTransformation(int size, int order,
                                 int num_edges, int num_tri_faces);

public:
   // Return the 2x2 transformation operator for the given face orientation
   static const DenseMatrix & GetFaceTransform(int ori) { return T(ori); }

   // Return the 2x2 inverse transformation operator
   static const DenseMatrix & GetFaceInverseTransform(int ori)
   { return TInv(ori); }
   /*
    using StatelessDofTransformation::TransformPrimal;
    using StatelessDofTransformation::InvTransformPrimal;
    using StatelessDofTransformation::TransformDual;
    using StatelessDofTransformation::InvTransformDual;
   */
   void TransformPrimal(const Array<int> & face_orientation,
                        double *v) const;

   void InvTransformPrimal(const Array<int> & face_orientation,
                           double *v) const;

   void TransformDual(const Array<int> & face_orientation,
                      double *v) const;

   void InvTransformDual(const Array<int> & face_orientation,
                         double *v) const;
};

/// Stateless DoF transformation implementation for the Nedelec basis on
/// triangles
class ND_TriStatelessDofTransformation : public ND_StatelessDofTransformation
{
public:
   ND_TriStatelessDofTransformation(int order)
      : StatelessDofTransformation(order*(order + 2))
      , ND_StatelessDofTransformation(order*(order + 2), order, 3, 1)
   {}
};

/// DoF transformation implementation for the Nedelec basis on triangles
class ND_TriDofTransformation : public DofTransformation,
   public ND_TriStatelessDofTransformation
{
public:
   ND_TriDofTransformation(int order)
      : StatelessDofTransformation(order*(order + 2))
      , DofTransformation(order*(order + 2))
      , ND_TriStatelessDofTransformation(order)
   {}

   using DofTransformation::TransformPrimal;
   using DofTransformation::InvTransformPrimal;
   using DofTransformation::TransformDual;
   using DofTransformation::InvTransformDual;

   using ND_TriStatelessDofTransformation::TransformPrimal;
   using ND_TriStatelessDofTransformation::InvTransformPrimal;
   using ND_TriStatelessDofTransformation::TransformDual;
   using ND_TriStatelessDofTransformation::InvTransformDual;
};

/// DoF transformation implementation for the Nedelec basis on tetrahedra
class ND_TetStatelessDofTransformation : public ND_StatelessDofTransformation
{
public:
   ND_TetStatelessDofTransformation(int order)
      : StatelessDofTransformation(order*(order + 2)*(order + 3)/2)
      , ND_StatelessDofTransformation(order*(order + 2)*(order + 3)/2, order,
                                      6, 4)
   {}
};

/// DoF transformation implementation for the Nedelec basis on tetrahedra
class ND_TetDofTransformation : public DofTransformation,
   public ND_TetStatelessDofTransformation
{
public:
   ND_TetDofTransformation(int order)
      : StatelessDofTransformation(order*(order + 2)*(order + 3)/2)
      , DofTransformation(order*(order + 2)*(order + 3)/2)
      , ND_TetStatelessDofTransformation(order)
   {}

   using DofTransformation::TransformPrimal;
   using DofTransformation::InvTransformPrimal;
   using DofTransformation::TransformDual;
   using DofTransformation::InvTransformDual;

   using ND_TetStatelessDofTransformation::TransformPrimal;
   using ND_TetStatelessDofTransformation::InvTransformPrimal;
   using ND_TetStatelessDofTransformation::TransformDual;
   using ND_TetStatelessDofTransformation::InvTransformDual;
};

/// DoF transformation implementation for the Nedelec basis on wedge elements
class ND_WedgeStatelessDofTransformation : public ND_StatelessDofTransformation
{
public:
   ND_WedgeStatelessDofTransformation(int order)
      : StatelessDofTransformation(3 * order * ((order + 1) * (order + 2))/2)
      , ND_StatelessDofTransformation(3 * order * ((order + 1) * (order + 2))/2,
                                      order, 9, 2)
   {}
};

/// DoF transformation implementation for the Nedelec basis on wedge elements
class ND_WedgeDofTransformation : public DofTransformation,
   public ND_WedgeStatelessDofTransformation
{
public:
   ND_WedgeDofTransformation(int order)
      : StatelessDofTransformation(3 * order * ((order + 1) * (order + 2))/2)
      , DofTransformation(3 * order * ((order + 1) * (order + 2))/2)
      , ND_WedgeStatelessDofTransformation(order)
   {}

   using DofTransformation::TransformPrimal;
   using DofTransformation::InvTransformPrimal;
   using DofTransformation::TransformDual;
   using DofTransformation::InvTransformDual;

   using ND_WedgeStatelessDofTransformation::TransformPrimal;
   using ND_WedgeStatelessDofTransformation::InvTransformPrimal;
   using ND_WedgeStatelessDofTransformation::TransformDual;
   using ND_WedgeStatelessDofTransformation::InvTransformDual;
};

} // namespace mfem

#endif // MFEM_DOFTRANSFORM
