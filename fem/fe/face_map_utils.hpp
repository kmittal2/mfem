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

#ifndef MFEM_FACE_MAP_UTILS_HPP
#define MFEM_FACE_MAP_UTILS_HPP

#include "../../general/array.hpp"
#include "../../general/backends.hpp"
#include <utility> // std::pair
#include <vector>

namespace mfem
{

namespace internal
{

/// Each face of a hexahedron is given by a level set x_i = l, where x_i is one
/// of x, y, or z (corresponding to i=0, i=1, i=2), and l is either 0 or 1.
/// Returns i and level.
std::pair<int,int> GetFaceNormal3D(const int face_id);

/// @brief Fills in the entries of the lexicographic face_map.
///
/// For use in FiniteElement::GetFaceMap.
///
/// n_face_dofs_per_component is the number of DOFs for each vector component
/// on the face (there is only one vector component in all cases except for 3D
/// Nedelec elements, where the face DOFs have two components to span the
/// tangent space).
///
/// The DOFs for the i-th vector component begin at offsets[i] (i.e. the number
/// of vector components is given by offsets.size()).
///
/// The DOFs for each vector component are arranged in a Cartesian grid defined
/// by strides and n_dofs_per_dim.
void FillFaceMap(const int n_face_dofs_per_component,
                 const std::vector<int> &offsets,
                 const std::vector<int> &strides,
                 const std::vector<int> &n_dofs_per_dim,
                 Array<int> &face_map);

/// Return the face map for nodal tensor elements (H1, L2, and Bernstein basis).
void GetTensorFaceMap(const int dim, const int order, const int face_id,
                      Array<int> &face_map);

// maps face index (in counter-clockwise order) to Lexocographic index
MFEM_HOST_DEVICE
inline int ToLexOrdering2D(const int face_id, const int size1d, const int i)
{
   if (face_id==2 || face_id==3)
   {
      return size1d-1-i;
   }
   else
   {
      return i;
   }
}

// permutes face index from native ordering to lexocographic
MFEM_HOST_DEVICE
inline int PermuteFace2D(const int face_id1, const int face_id2,
                         const int orientation,
                         const int size1d, const int index)
{
   int new_index;
   // Convert from lex ordering
   if (face_id1==2 || face_id1==3)
   {
      new_index = size1d-1-index;
   }
   else
   {
      new_index = index;
   }
   // Permute based on face orientations
   if (orientation==1)
   {
      new_index = size1d-1-new_index;
   }
   return ToLexOrdering2D(face_id2, size1d, new_index);
}

MFEM_HOST_DEVICE
inline int ToLexOrdering3D(const int face_id, const int size1d, const int i,
                           const int j)
{
   if (face_id==2 || face_id==1 || face_id==5)
   {
      return i + j*size1d;
   }
   else if (face_id==3 || face_id==4)
   {
      return (size1d-1-i) + j*size1d;
   }
   else // face_id==0
   {
      return i + (size1d-1-j)*size1d;
   }
}

MFEM_HOST_DEVICE
inline int PermuteFace3D(const int face_id1, const int face_id2,
                         const int orientation,
                         const int size1d, const int index)
{
   int i=0, j=0, new_i=0, new_j=0;
   i = index%size1d;
   j = index/size1d;
   // Convert from lex ordering
   if (face_id1==3 || face_id1==4)
   {
      i = size1d-1-i;
   }
   else if (face_id1==0)
   {
      j = size1d-1-j;
   }
   // Permute based on face orientations
   switch (orientation)
   {
      case 0:
         new_i = i;
         new_j = j;
         break;
      case 1:
         new_i = j;
         new_j = i;
         break;
      case 2:
         new_i = j;
         new_j = (size1d-1-i);
         break;
      case 3:
         new_i = (size1d-1-i);
         new_j = j;
         break;
      case 4:
         new_i = (size1d-1-i);
         new_j = (size1d-1-j);
         break;
      case 5:
         new_i = (size1d-1-j);
         new_j = (size1d-1-i);
         break;
      case 6:
         new_i = (size1d-1-j);
         new_j = i;
         break;
      case 7:
         new_i = i;
         new_j = (size1d-1-j);
         break;
   }
   return ToLexOrdering3D(face_id2, size1d, new_i, new_j);
}

// maps quadrature index on face to (row, col) index pair
MFEM_HOST_DEVICE
inline void FaceQuad2Lex2D(const int qi, const int nq,
                           const int face_id0, const int face_id1, const int side,
                           int &i, int &j)
{
   const int face_id = (side == 0) ? face_id0 : face_id1;
   const int edge_idx = (side == 0) ? qi : PermuteFace2D(face_id0, face_id1, side,
                                                         nq, qi);
   if (face_id == 0 || face_id == 2)
   {
      i = edge_idx;
      j = (face_id == 0) ? 0 : (nq-1);
   }
   else
   {
      j = edge_idx;
      i = (face_id == 3) ? 0 : (nq-1);
   }
}

MFEM_HOST_DEVICE
inline void FaceQuad2Lex3D(const int index, const int size1d, const int face_id0, const int face_id1, const int side, const int orientation, int& i, int& j, int& k)
{
   MFEM_VERIFY(face_id1 >= 0 || side == 0, "accessing second side but face_id1 is not valid.");

   const int size2d = size1d * size1d;
   const int face_id = (side == 0) ? face_id0 : face_id1;
   int fidx = (side == 0) ? index : PermuteFace3D(face_id0, face_id1, side, size1d, index);

   const bool xy_plane = (face_id == 0 || face_id == 5); // is this face parallel to the x,y plane in reference coo
   const bool xz_plane = (face_id == 1 || face_id == 3);
   const bool yz_plane = (face_id == 2 || face_id == 4);

   const int level = (face_id == 0 || face_id == 1 || face_id == 4) ? 0 : 1;

   int idx3d, strides[2];
   if (yz_plane)
   {
      idx3d = level ? (size1d-1) : 0;
      strides[0] = size1d;
      strides[1] = size2d;
   }
   else if (xz_plane)
   {
      idx3d = level ? (size1d-1)*size1d : 0;
      strides[0] = 1;
      strides[1] = size2d;
   }
   else // xy_plane
   {
      idx3d = level ? (size1d-1)*size2d : 0;
      strides[0] = 1;
      strides[1] = size1d;
   }
   
   idx3d += strides[0] * (fidx % size1d);
   fidx /= size1d;
   idx3d += strides[1] * (fidx % size1d);

   k = idx3d / size2d;
   j = (idx3d - k * size2d) / size1d;
   i = idx3d - k * size2d - j * size1d;
}

} // namespace internal

} // namespace mfem

#endif
