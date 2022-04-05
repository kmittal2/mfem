// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_QFUNCTION
#define MFEM_QFUNCTION

#include "../config/config.hpp"
#include "qspace.hpp"
#include "gridfunc.hpp"

namespace mfem
{

/** @brief Class representing a function through its values (scalar or vector)
    at quadrature points. */
class QuadratureFunction : public Vector
{
protected:
   QuadratureSpace *qspace; ///< Associated QuadratureSpace
   int vdim;                ///< Vector dimension
   bool own_qspace;         ///< QuadratureSpace ownership flag

public:
   /// Create an empty QuadratureFunction.
   /** The object can be initialized later using the SetSpace() methods. */
   QuadratureFunction()
      : qspace(NULL), vdim(0), own_qspace(false) { }

   /** @brief Copy constructor. The QuadratureSpace ownership flag, #own_qspace,
       in the new object is set to false. */
   QuadratureFunction(const QuadratureFunction &orig)
      : Vector(orig),
        qspace(orig.qspace), vdim(orig.vdim), own_qspace(false) { }

   /// Create a QuadratureFunction based on the given QuadratureSpace.
   /** The QuadratureFunction does not assume ownership of the QuadratureSpace.
       @note The Vector data is not initialized. */
   QuadratureFunction(QuadratureSpace *qspace_, int vdim_ = 1)
      : Vector(vdim_*qspace_->GetSize()),
        qspace(qspace_), vdim(vdim_), own_qspace(false) { }

   /** @brief Create a QuadratureFunction based on the given QuadratureSpace,
       using the external data, @a qf_data. */
   /** The QuadratureFunction does not assume ownership of neither the
       QuadratureSpace nor the external data. */
   QuadratureFunction(QuadratureSpace *qspace_, double *qf_data, int vdim_ = 1)
      : Vector(qf_data, vdim_*qspace_->GetSize()),
        qspace(qspace_), vdim(vdim_), own_qspace(false) { }

   /// Read a QuadratureFunction from the stream @a in.
   /** The QuadratureFunction assumes ownership of the read QuadratureSpace. */
   QuadratureFunction(Mesh *mesh, std::istream &in);

   virtual ~QuadratureFunction() { if (own_qspace) { delete qspace; } }

   /// Get the associated QuadratureSpace.
   QuadratureSpace *GetSpace() const { return qspace; }

   /// Change the QuadratureSpace and optionally the vector dimension.
   /** If the new QuadratureSpace is different from the current one, the
       QuadratureFunction will not assume ownership of the new space; otherwise,
       the ownership flag remains the same.

       If the new vector dimension @a vdim_ < 0, the vector dimension remains
       the same.

       The data size is updated by calling Vector::SetSize(). */
   inline void SetSpace(QuadratureSpace *qspace_, int vdim_ = -1);

   /** @brief Change the QuadratureSpace, the data array, and optionally the
       vector dimension. */
   /** If the new QuadratureSpace is different from the current one, the
       QuadratureFunction will not assume ownership of the new space; otherwise,
       the ownership flag remains the same.

       If the new vector dimension @a vdim_ < 0, the vector dimension remains
       the same.

       The data array is replaced by calling Vector::NewDataAndSize(). */
   inline void SetSpace(QuadratureSpace *qspace_, double *qf_data,
                        int vdim_ = -1);

   /// Get the vector dimension.
   int GetVDim() const { return vdim; }

   /// Set the vector dimension, updating the size by calling Vector::SetSize().
   void SetVDim(int vdim_)
   { vdim = vdim_; SetSize(vdim*qspace->GetSize()); }

   /// Get the QuadratureSpace ownership flag.
   bool OwnsSpace() { return own_qspace; }

   /// Set the QuadratureSpace ownership flag.
   void SetOwnsSpace(bool own) { own_qspace = own; }

   /// Redefine '=' for QuadratureFunction = constant.
   QuadratureFunction &operator=(double value);

   /// Copy the data from @a v.
   /** The size of @a v must be equal to the size of the associated
       QuadratureSpace #qspace times the QuadratureFunction dimension
       i.e. QuadratureFunction::Size(). */
   QuadratureFunction &operator=(const Vector &v);

   /// Copy assignment. Only the data of the base class Vector is copied.
   /** The QuadratureFunctions @a v and @a *this must have QuadratureSpaces with
       the same size.

       @note Defining this method overwrites the implicitly defined copy
       assignment operator. */
   QuadratureFunction &operator=(const QuadratureFunction &v);

   /// Get the IntegrationRule associated with mesh element @a idx.
   const IntegrationRule &GetElementIntRule(int idx) const
   { return qspace->GetElementIntRule(idx); }

   /// Return all values associated with mesh element @a idx in a Vector.
   /** The result is stored in the Vector @a values as a reference to the
       global values.

       Inside the Vector @a values, the index `i+vdim*j` corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   inline void GetElementValues(int idx, Vector &values);

   /// Return all values associated with mesh element @a idx in a Vector.
   /** The result is stored in the Vector @a values as a copy of the
       global values.

       Inside the Vector @a values, the index `i+vdim*j` corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   inline void GetElementValues(int idx, Vector &values) const;

   /// Return the quadrature function values at an integration point.
   /** The result is stored in the Vector @a values as a reference to the
       global values. */
   inline void GetElementValues(int idx, const int ip_num, Vector &values);

   /// Return the quadrature function values at an integration point.
   /** The result is stored in the Vector @a values as a copy to the
       global values. */
   inline void GetElementValues(int idx, const int ip_num, Vector &values) const;

   /// Return all values associated with mesh element @a idx in a DenseMatrix.
   /** The result is stored in the DenseMatrix @a values as a reference to the
       global values.

       Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   inline void GetElementValues(int idx, DenseMatrix &values);

   /// Return all values associated with mesh element @a idx in a const DenseMatrix.
   /** The result is stored in the DenseMatrix @a values as a copy of the
       global values.

       Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   inline void GetElementValues(int idx, DenseMatrix &values) const;

   /// Write the QuadratureFunction to the stream @a out.
   void Save(std::ostream &out) const;

   /// @brief Write the QuadratureFunction to @a out in VTU (ParaView) format.
   ///
   /// The data will be uncompressed if @a compression_level is zero, or if the
   /// format is VTKFormat::ASCII. Otherwise, zlib compression will be used for
   /// binary data.
   void SaveVTU(std::ostream &out, VTKFormat format=VTKFormat::ASCII,
                int compression_level=0) const;

   /// @brief Save the QuadratureFunction to a VTU (ParaView) file.
   ///
   /// The extension ".vtu" will be appended to @a filename.
   /// @sa SaveVTU(std::ostream &out, VTKFormat format=VTKFormat::ASCII,
   ///             int compression_level=0)
   void SaveVTU(const std::string &filename, VTKFormat format=VTKFormat::ASCII,
                int compression_level=0) const;
};

// Inline methods

inline void QuadratureFunction::SetSpace(QuadratureSpace *qspace_, int vdim_)
{
   if (qspace_ != qspace)
   {
      if (own_qspace) { delete qspace; }
      qspace = qspace_;
      own_qspace = false;
   }
   vdim = (vdim_ < 0) ? vdim : vdim_;
   SetSize(vdim*qspace->GetSize());
}

inline void QuadratureFunction::SetSpace(QuadratureSpace *qspace_,
                                         double *qf_data, int vdim_)
{
   if (qspace_ != qspace)
   {
      if (own_qspace) { delete qspace; }
      qspace = qspace_;
      own_qspace = false;
   }
   vdim = (vdim_ < 0) ? vdim : vdim_;
   NewDataAndSize(qf_data, vdim*qspace->GetSize());
}

inline void QuadratureFunction::GetElementValues(int idx, Vector &values)
{
   const int s_offset = qspace->element_offsets[idx];
   const int sl_size = qspace->element_offsets[idx+1] - s_offset;
   values.NewDataAndSize(data + vdim*s_offset, vdim*sl_size);
}

inline void QuadratureFunction::GetElementValues(int idx, Vector &values) const
{
   const int s_offset = qspace->element_offsets[idx];
   const int sl_size = qspace->element_offsets[idx+1] - s_offset;
   values.SetSize(vdim*sl_size);
   const double *q = data + vdim*s_offset;
   for (int i = 0; i<values.Size(); i++)
   {
      values(i) = *(q++);
   }
}

inline void QuadratureFunction::GetElementValues(int idx, const int ip_num,
                                                 Vector &values)
{
   const int s_offset = qspace->element_offsets[idx] * vdim + ip_num * vdim;
   values.NewDataAndSize(data + s_offset, vdim);
}

inline void QuadratureFunction::GetElementValues(int idx, const int ip_num,
                                                 Vector &values) const
{
   const int s_offset = qspace->element_offsets[idx] * vdim + ip_num * vdim;
   values.SetSize(vdim);
   const double *q = data + s_offset;
   for (int i = 0; i < values.Size(); i++)
   {
      values(i) = *(q++);
   }
}

inline void QuadratureFunction::GetElementValues(int idx, DenseMatrix &values)
{
   const int s_offset = qspace->element_offsets[idx];
   const int sl_size = qspace->element_offsets[idx+1] - s_offset;
   values.Reset(data + vdim*s_offset, vdim, sl_size);
}

inline void QuadratureFunction::GetElementValues(int idx,
                                                 DenseMatrix &values) const
{
   const int s_offset = qspace->element_offsets[idx];
   const int sl_size = qspace->element_offsets[idx+1] - s_offset;
   values.SetSize(vdim, sl_size);
   const double *q = data + vdim*s_offset;
   for (int j = 0; j<sl_size; j++)
   {
      for (int i = 0; i<vdim; i++)
      {
         values(i,j) = *(q++);
      }
   }
}

} // namespace mfem

#endif
