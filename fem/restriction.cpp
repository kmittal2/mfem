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

#include "restriction.hpp"
#include "gridfunc.hpp"
#include "fespace.hpp"
#include "../mesh/mesh_headers.hpp"
#include "../general/forall.hpp"

namespace mfem
{

ElementRestriction::ElementRestriction(const FiniteElementSpace &f,
                                       ElementDofOrdering e_ordering)
   : fes(f),
     ne(fes.GetNE()),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndofs(fes.GetNDofs()),
     dof(ne > 0 ? fes.GetFE(0)->GetDof() : 0),
     nedofs(ne*dof),
     offsets(ndofs+1),
     indices(ne*dof),
     gatherMap(ne*dof)
{
   // Assuming all finite elements are the same.
   height = vdim*ne*dof;
   width = fes.GetVSize();
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   const int *dof_map = NULL;
   if (dof_reorder && ne > 0)
   {
      for (int e = 0; e < ne; ++e)
      {
         const FiniteElement *fe = fes.GetFE(e);
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
      const FiniteElement *fe = fes.GetFE(0);
      const TensorBasisElement* el =
         dynamic_cast<const TensorBasisElement*>(fe);
      const Array<int> &fe_dof_map = el->GetDofMap();
      MFEM_VERIFY(fe_dof_map.Size() > 0, "invalid dof map");
      dof_map = fe_dof_map.GetData();
   }
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   // We will be keeping a count of how many local nodes point to its global dof
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }
   for (int e = 0; e < ne; ++e)
   {
      for (int d = 0; d < dof; ++d)
      {
         const int sgid = elementMap[dof*e + d];  // signed
         const int gid = (sgid >= 0) ? sgid : -1 - sgid;
         ++offsets[gid + 1];
      }
   }
   // Aggregate to find offsets for each global dof
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   // For each global dof, fill in all local nodes that point to it
   for (int e = 0; e < ne; ++e)
   {
      for (int d = 0; d < dof; ++d)
      {
         const int sdid = dof_reorder ? dof_map[d] : 0;  // signed
         const int did = (!dof_reorder)?d:(sdid >= 0 ? sdid : -1-sdid);
         const int sgid = elementMap[dof*e + did];  // signed
         const int gid = (sgid >= 0) ? sgid : -1-sgid;
         const int lid = dof*e + d;
         const bool plus = (sgid >= 0 && sdid >= 0) || (sgid < 0 && sdid < 0);
         gatherMap[lid] = plus ? gid : -1-gid;
         indices[offsets[gid]++] = plus ? lid : -1-lid;
      }
   }
   // We shifted the offsets vector by 1 by using it as a counter.
   // Now we shift it back.
   for (int i = ndofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
}

void ElementRestriction::Mult(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nd, vd, ne);
   auto d_gatherMap = gatherMap.Read();
   MFEM_FORALL(i, dof*ne,
   {
      const int gid = d_gatherMap[i];
      const bool plus = gid >= 0;
      const int j = plus ? gid : -1-gid;
      for (int c = 0; c < vd; ++c)
      {
         const double dofValue = d_x(t?c:j, t?j:c);
         d_y(i % nd, c, i / nd) = plus ? dofValue : -dofValue;
      }
   });
}

void ElementRestriction::MultUnsigned(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nd, vd, ne);
   auto d_gatherMap = gatherMap.Read();

   MFEM_FORALL(i, dof*ne,
   {
      const int gid = d_gatherMap[i];
      const int j = gid >= 0 ? gid : -1-gid;
      for (int c = 0; c < vd; ++c)
      {
         d_y(i % nd, c, i / nd) = d_x(t?c:j, t?j:c);
      }
   });
}

void ElementRestriction::MultTranspose(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_x = Reshape(x.Read(), nd, vd, ne);
   auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dofValue = 0;
         for (int j = offset; j < nextOffset; ++j)
         {
            const int idx_j = (d_indices[j] >= 0) ? d_indices[j] : -1 - d_indices[j];
            dofValue += (d_indices[j] >= 0) ? d_x(idx_j % nd, c,
            idx_j / nd) : -d_x(idx_j % nd, c, idx_j / nd);
         }
         d_y(t?c:i,t?i:c) = dofValue;
      }
   });
}

void ElementRestriction::MultTransposeUnsigned(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_x = Reshape(x.Read(), nd, vd, ne);
   auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dofValue = 0;
         for (int j = offset; j < nextOffset; ++j)
         {
            const int idx_j = (d_indices[j] >= 0) ? d_indices[j] : -1 - d_indices[j];
            dofValue += d_x(idx_j % nd, c, idx_j / nd);
         }
         d_y(t?c:i,t?i:c) = dofValue;
      }
   });
}

void ElementRestriction::MultLeftInverse(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_x = Reshape(x.Read(), nd, vd, ne);
   auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int nextOffset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dofValue = 0;
         const int j = nextOffset - 1;
         const int idx_j = (d_indices[j] >= 0) ? d_indices[j] : -1 - d_indices[j];
         dofValue = (d_indices[j] >= 0) ? d_x(idx_j % nd, c, idx_j / nd) :
         -d_x(idx_j % nd, c, idx_j / nd);
         d_y(t?c:i,t?i:c) = dofValue;
      }
   });
}

void ElementRestriction::BooleanMask(Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;

   Array<char> processed(vd * ndofs);
   processed = 0;

   auto d_offsets = offsets.HostRead();
   auto d_indices = indices.HostRead();
   auto d_x = Reshape(processed.HostReadWrite(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.HostWrite(), nd, vd, ne);
   for (int i = 0; i < ndofs; ++i)
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i+1];
      for (int c = 0; c < vd; ++c)
      {
         for (int j = offset; j < nextOffset; ++j)
         {
            const int idx_j = d_indices[j];
            if (d_x(t?c:i,t?i:c))
            {
               d_y(idx_j % nd, c, idx_j / nd) = 0.0;
            }
            else
            {
               d_y(idx_j % nd, c, idx_j / nd) = 1.0;
               d_x(t?c:i,t?i:c) = 1;
            }
         }
      }
   }
}

void ElementRestriction::FillSparseMatrix(const Vector &mat_ea,
                                          SparseMatrix &mat) const
{
   mat.GetMemoryI().New(mat.Height()+1, mat.GetMemoryI().GetMemoryType());
   const int nnz = FillI(mat);
   mat.GetMemoryJ().New(nnz, mat.GetMemoryJ().GetMemoryType());
   mat.GetMemoryData().New(nnz, mat.GetMemoryData().GetMemoryType());
   FillJAndData(mat_ea, mat);
}

template <int MaxNbNbr>
static MFEM_HOST_DEVICE int GetMinElt(const int *my_elts, const int nbElts,
                                      const int *nbr_elts, const int nbrNbElts)
{
   // Building the intersection
   int inter[MaxNbNbr];
   int cpt = 0;
   for (int i = 0; i < nbElts; i++)
   {
      const int e_i = my_elts[i];
      for (int j = 0; j < nbrNbElts; j++)
      {
         if (e_i==nbr_elts[j])
         {
            inter[cpt] = e_i;
            cpt++;
         }
      }
   }
   // Finding the minimum
   int min = inter[0];
   for (int i = 1; i < cpt; i++)
   {
      if (inter[i] < min)
      {
         min = inter[i];
      }
   }
   return min;
}

/** Returns the index where a non-zero entry should be added and increment the
    number of non-zeros for the row i_L. */
static MFEM_HOST_DEVICE int GetAndIncrementNnzIndex(const int i_L, int* I)
{
   int ind = AtomicAdd(I[i_L],1);
   return ind;
}

int ElementRestriction::FillI(SparseMatrix &mat) const
{
   static constexpr int Max = MaxNbNbr;
   const int all_dofs = ndofs;
   const int vd = vdim;
   const int elt_dofs = dof;
   auto I = mat.ReadWriteI();
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_gatherMap = gatherMap.Read();
   MFEM_FORALL(i_L, vd*all_dofs+1,
   {
      I[i_L] = 0;
   });
   MFEM_FORALL(e, ne,
   {
      for (int i = 0; i < elt_dofs; i++)
      {
         int i_elts[Max];
         const int i_E = e*elt_dofs + i;
         const int i_L = d_gatherMap[i_E];
         const int i_offset = d_offsets[i_L];
         const int i_nextOffset = d_offsets[i_L+1];
         const int i_nbElts = i_nextOffset - i_offset;
         for (int e_i = 0; e_i < i_nbElts; ++e_i)
         {
            const int i_E = d_indices[i_offset+e_i];
            i_elts[e_i] = i_E/elt_dofs;
         }
         for (int j = 0; j < elt_dofs; j++)
         {
            const int j_E = e*elt_dofs + j;
            const int j_L = d_gatherMap[j_E];
            const int j_offset = d_offsets[j_L];
            const int j_nextOffset = d_offsets[j_L+1];
            const int j_nbElts = j_nextOffset - j_offset;
            if (i_nbElts == 1 || j_nbElts == 1) // no assembly required
            {
               GetAndIncrementNnzIndex(i_L, I);
            }
            else // assembly required
            {
               int j_elts[Max];
               for (int e_j = 0; e_j < j_nbElts; ++e_j)
               {
                  const int j_E = d_indices[j_offset+e_j];
                  const int elt = j_E/elt_dofs;
                  j_elts[e_j] = elt;
               }
               int min_e = GetMinElt<Max>(i_elts, i_nbElts, j_elts, j_nbElts);
               if (e == min_e) // add the nnz only once
               {
                  GetAndIncrementNnzIndex(i_L, I);
               }
            }
         }
      }
   });
   // We need to sum the entries of I, we do it on CPU as it is very sequential.
   auto h_I = mat.HostReadWriteI();
   const int nTdofs = vd*all_dofs;
   int sum = 0;
   for (int i = 0; i < nTdofs; i++)
   {
      const int nnz = h_I[i];
      h_I[i] = sum;
      sum+=nnz;
   }
   h_I[nTdofs] = sum;
   // We return the number of nnz
   return h_I[nTdofs];
}

void ElementRestriction::FillJAndData(const Vector &ea_data,
                                      SparseMatrix &mat) const
{
   static constexpr int Max = MaxNbNbr;
   const int all_dofs = ndofs;
   const int vd = vdim;
   const int elt_dofs = dof;
   auto I = mat.ReadWriteI();
   auto J = mat.WriteJ();
   auto Data = mat.WriteData();
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_gatherMap = gatherMap.Read();
   auto mat_ea = Reshape(ea_data.Read(), elt_dofs, elt_dofs, ne);
   MFEM_FORALL(e, ne,
   {
      for (int i = 0; i < elt_dofs; i++)
      {
         int i_elts[Max];
         int i_B[Max];
         const int i_E = e*elt_dofs + i;
         const int i_L = d_gatherMap[i_E];
         const int i_offset = d_offsets[i_L];
         const int i_nextOffset = d_offsets[i_L+1];
         const int i_nbElts = i_nextOffset - i_offset;
         for (int e_i = 0; e_i < i_nbElts; ++e_i)
         {
            const int i_E = d_indices[i_offset+e_i];
            i_elts[e_i] = i_E/elt_dofs;
            i_B[e_i]    = i_E%elt_dofs;
         }
         for (int j = 0; j < elt_dofs; j++)
         {
            const int j_E = e*elt_dofs + j;
            const int j_L = d_gatherMap[j_E];
            const int j_offset = d_offsets[j_L];
            const int j_nextOffset = d_offsets[j_L+1];
            const int j_nbElts = j_nextOffset - j_offset;
            if (i_nbElts == 1 || j_nbElts == 1) // no assembly required
            {
               const int nnz = GetAndIncrementNnzIndex(i_L, I);
               J[nnz] = j_L;
               Data[nnz] = mat_ea(j,i,e);
            }
            else // assembly required
            {
               int j_elts[Max];
               int j_B[Max];
               for (int e_j = 0; e_j < j_nbElts; ++e_j)
               {
                  const int j_E = d_indices[j_offset+e_j];
                  const int elt = j_E/elt_dofs;
                  j_elts[e_j] = elt;
                  j_B[e_j]    = j_E%elt_dofs;
               }
               int min_e = GetMinElt<Max>(i_elts, i_nbElts, j_elts, j_nbElts);
               if (e == min_e) // add the nnz only once
               {
                  double val = 0.0;
                  for (int i = 0; i < i_nbElts; i++)
                  {
                     const int e_i = i_elts[i];
                     const int i_Bloc = i_B[i];
                     for (int j = 0; j < j_nbElts; j++)
                     {
                        const int e_j = j_elts[j];
                        const int j_Bloc = j_B[j];
                        if (e_i == e_j)
                        {
                           val += mat_ea(j_Bloc, i_Bloc, e_i);
                        }
                     }
                  }
                  const int nnz = GetAndIncrementNnzIndex(i_L, I);
                  J[nnz] = j_L;
                  Data[nnz] = val;
               }
            }
         }
      }
   });
   // We need to shift again the entries of I, we do it on CPU as it is very
   // sequential.
   auto h_I = mat.HostReadWriteI();
   const int size = vd*all_dofs;
   for (int i = 0; i < size; i++)
   {
      h_I[size-i] = h_I[size-(i+1)];
   }
   h_I[0] = 0;
}

L2ElementRestriction::L2ElementRestriction(const FiniteElementSpace &fes)
   : ne(fes.GetNE()),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndof(ne > 0 ? fes.GetFE(0)->GetDof() : 0),
     ndofs(fes.GetNDofs())
{
   height = vdim*ne*ndof;
   width = vdim*ne*ndof;
}

void L2ElementRestriction::Mult(const Vector &x, Vector &y) const
{
   const int nd = ndof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nd, vd, ne);
   MFEM_FORALL(i, ndofs,
   {
      const int idx = i;
      const int dof = idx % nd;
      const int e = idx / nd;
      for (int c = 0; c < vd; ++c)
      {
         d_y(dof, c, e) = d_x(t?c:idx, t?idx:c);
      }
   });
}

void L2ElementRestriction::MultTranspose(const Vector &x, Vector &y) const
{
   const int nd = ndof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_x = Reshape(x.Read(), nd, vd, ne);
   auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int idx = i;
      const int dof = idx % nd;
      const int e = idx / nd;
      for (int c = 0; c < vd; ++c)
      {
         d_y(t?c:idx,t?idx:c) = d_x(dof, c, e);
      }
   });
}

void L2ElementRestriction::FillI(SparseMatrix &mat) const
{
   const int elem_dofs = ndof;
   const int vd = vdim;
   auto I = mat.WriteI();
   const int isize = mat.Height() + 1;
   const int interior_dofs = ne*elem_dofs*vd;
   MFEM_FORALL(dof, isize,
   {
      I[dof] = dof<interior_dofs ? elem_dofs : 0;
   });
}

static MFEM_HOST_DEVICE int AddNnz(const int iE, int *I, const int dofs)
{
   int val = AtomicAdd(I[iE],dofs);
   return val;
}

void L2ElementRestriction::FillJAndData(const Vector &ea_data,
                                        SparseMatrix &mat) const
{
   const int elem_dofs = ndof;
   const int vd = vdim;
   auto I = mat.ReadWriteI();
   auto J = mat.WriteJ();
   auto Data = mat.WriteData();
   auto mat_ea = Reshape(ea_data.Read(), elem_dofs, elem_dofs, ne);
   MFEM_FORALL(iE, ne*elem_dofs*vd,
   {
      const int offset = AddNnz(iE,I,elem_dofs);
      const int e = iE/elem_dofs;
      const int i = iE%elem_dofs;
      for (int j = 0; j < elem_dofs; j++)
      {
         J[offset+j] = e*elem_dofs+j;
         Data[offset+j] = mat_ea(j,i,e);
      }
   });
}

// Return the face degrees of freedom returned in Lexicographic order.
void GetFaceDofs(const int dim, const int face_id,
                 const int dof1d, Array<int> &faceMap)
{
   switch (dim)
   {
      case 1:
         switch (face_id)
         {
            case 0: // WEST
               faceMap[0] = 0;
               break;
            case 1: // EAST
               faceMap[0] = dof1d-1;
               break;
         }
         break;
      case 2:
         switch (face_id)
         {
            case 0: // SOUTH
               for (int i = 0; i < dof1d; ++i)
               {
                  faceMap[i] = i;
               }
               break;
            case 1: // EAST
               for (int i = 0; i < dof1d; ++i)
               {
                  faceMap[i] = dof1d-1 + i*dof1d;
               }
               break;
            case 2: // NORTH
               for (int i = 0; i < dof1d; ++i)
               {
                  faceMap[i] = (dof1d-1)*dof1d + i;
               }
               break;
            case 3: // WEST
               for (int i = 0; i < dof1d; ++i)
               {
                  faceMap[i] = i*dof1d;
               }
               break;
         }
         break;
      case 3:
         switch (face_id)
         {
            case 0: // BOTTOM
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     faceMap[i+j*dof1d] = i + j*dof1d;
                  }
               }
               break;
            case 1: // SOUTH
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     faceMap[i+j*dof1d] = i + j*dof1d*dof1d;
                  }
               }
               break;
            case 2: // EAST
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     faceMap[i+j*dof1d] = dof1d-1 + i*dof1d + j*dof1d*dof1d;
                  }
               }
               break;
            case 3: // NORTH
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     faceMap[i+j*dof1d] = (dof1d-1)*dof1d + i + j*dof1d*dof1d;
                  }
               }
               break;
            case 4: // WEST
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     faceMap[i+j*dof1d] = i*dof1d + j*dof1d*dof1d;
                  }
               }
               break;
            case 5: // TOP
               for (int i = 0; i < dof1d; ++i)
               {
                  for (int j = 0; j < dof1d; ++j)
                  {
                     faceMap[i+j*dof1d] = (dof1d-1)*dof1d*dof1d + i + j*dof1d;
                  }
               }
               break;
         }
         break;
   }
}

H1FaceRestriction::H1FaceRestriction(const FiniteElementSpace &fes,
                                     const ElementDofOrdering e_ordering,
                                     const FaceType type)
   : fes(fes),
     nf(fes.GetNFbyType(type)),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndofs(fes.GetNDofs()),
     dof(nf > 0 ? fes.GetFaceElement(0)->GetDof() : 0),
     nfdofs(nf*dof),
     scatter_indices(nf*dof),
     offsets(ndofs+1),
     gather_indices(nf*dof)
{
   if (nf==0) { return; }
   // If fespace == H1
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "H1FaceRestriction.");
   // Assuming all finite elements are using Gauss-Lobatto.
   height = vdim*nf*dof;
   width = fes.GetVSize();
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const FiniteElement *fe = fes.GetFaceElement(f);
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
      const FiniteElement *fe = fes.GetFaceElement(0);
      const TensorBasisElement* el =
         dynamic_cast<const TensorBasisElement*>(fe);
      const Array<int> &fe_dof_map = el->GetDofMap();
      MFEM_VERIFY(fe_dof_map.Size() > 0, "invalid dof map");
   }
   const TensorBasisElement* el =
      dynamic_cast<const TensorBasisElement*>(fe);
   const int *dof_map = el->GetDofMap().GetData();
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   Array<int> faceMap(dof);
   int e1, e2;
   int inf1, inf2;
   int face_id;
   int orientation;
   int ncface;
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   const int elem_dofs = fes.GetFE(0)->GetDof();
   const int dim = fes.GetMesh()->SpaceDimension();
   // Computation of scatter_indices
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2, &ncface);
      orientation = inf1 % 64;
      face_id = inf1 / 64;
      if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0 && ncface==-1))) ||
          (type==FaceType::Boundary && e2<0 && inf2<0 && ncface==-1) )
      {
         // Assumes Gauss-Lobatto basis
         if (dof_reorder)
         {
            if (orientation != 0)
            {
               MFEM_ABORT("FaceRestriction used on degenerated mesh.");
            }
            GetFaceDofs(dim, face_id, dof1d, faceMap); // Only for hex
         }
         else
         {
            MFEM_ABORT("FaceRestriction not yet implemented for this type of "
                       "element.");
            // TODO Something with GetFaceDofs?
         }
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap[d];
            const int did = (!dof_reorder)?face_dof:dof_map[face_dof];
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            scatter_indices[lid] = gid;
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
   // Computation of gather_indices
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }
   f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2, &ncface);
      orientation = inf1 % 64;
      face_id = inf1 / 64;
      if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0 && ncface==-1))) ||
          (type==FaceType::Boundary && e2<0 && inf2<0 && ncface==-1) )
      {
         GetFaceDofs(dim, face_id, dof1d, faceMap);
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap[d];
            const int did = (!dof_reorder)?face_dof:dof_map[face_dof];
            const int gid = elementMap[e1*elem_dofs + did];
            ++offsets[gid + 1];
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2, &ncface);
      orientation = inf1 % 64;
      face_id = inf1 / 64;
      if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0 && ncface==-1))) ||
          (type==FaceType::Boundary && e2<0 && inf2<0 && ncface==-1) )
      {
         GetFaceDofs(dim, face_id, dof1d, faceMap);
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap[d];
            const int did = (!dof_reorder)?face_dof:dof_map[face_dof];
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            gather_indices[offsets[gid]++] = lid;
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
   for (int i = ndofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
}

void H1FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_indices = scatter_indices.Read();
   auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
   auto d_y = Reshape(y.Write(), nd, vd, nf);
   MFEM_FORALL(i, nfdofs,
   {
      const int idx = d_indices[i];
      const int dof = i % nd;
      const int face = i / nd;
      for (int c = 0; c < vd; ++c)
      {
         d_y(dof, c, face) = d_x(t?c:idx, t?idx:c);
      }
   });
}

void H1FaceRestriction::MultTranspose(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   auto d_offsets = offsets.Read();
   auto d_indices = gather_indices.Read();
   auto d_x = Reshape(x.Read(), nd, vd, nf);
   auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dofValue = 0;
         for (int j = offset; j < nextOffset; ++j)
         {
            const int idx_j = d_indices[j];
            dofValue +=  d_x(idx_j % nd, c, idx_j / nd);
         }
         d_y(t?c:i,t?i:c) += dofValue;
      }
   });
}

static int ToLexOrdering2D(const int face_id, const int size1d, const int i)
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

static int PermuteFace2D(const int face_id1, const int face_id2,
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

static int ToLexOrdering3D(const int face_id, const int size1d, const int i,
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

static int PermuteFace3D(const int face_id1, const int face_id2,
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

// Permute dofs or quads on a face for e2 to match with the ordering of e1
int PermuteFaceL2(const int dim, const int face_id1,
                  const int face_id2, const int orientation,
                  const int size1d, const int index)
{
   switch (dim)
   {
      case 1:
         return 0;
      case 2:
         return PermuteFace2D(face_id1, face_id2, orientation, size1d, index);
      case 3:
         return PermuteFace3D(face_id1, face_id2, orientation, size1d, index);
      default:
         MFEM_ABORT("Unsupported dimension.");
         return 0;
   }
}

L2FaceRestriction::L2FaceRestriction(const FiniteElementSpace &fes,
                                     const FaceType type,
                                     const L2FaceValues m)
   : fes(fes),
     nf(fes.GetNFbyType(type)),
     ne(fes.GetNE()),
     vdim(fes.GetVDim()),
     byvdim(fes.GetOrdering() == Ordering::byVDIM),
     ndofs(fes.GetNDofs()),
     dof(nf > 0 ?
         fes.GetTraceElement(0, fes.GetMesh()->GetFaceBaseGeometry(0))->GetDof()
         : 0),
     elemDofs(fes.GetFE(0)->GetDof()),
     type(type),
     m(m),
     nfdofs(nf*dof),
     scatter_indices1(nf*dof),
     scatter_indices2(m==L2FaceValues::DoubleValued?nf*dof:0),
     offsets(ndofs+1),
     gather_indices((m==L2FaceValues::DoubleValued? 2 : 1)*nf*dof)
{
}

L2FaceRestriction::L2FaceRestriction(const FiniteElementSpace &fes,
                                     const ElementDofOrdering e_ordering,
                                     const FaceType type,
                                     const L2FaceValues m)
   : L2FaceRestriction(fes, type, m)
{
   // If fespace == L2
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "L2FaceRestriction.");
   MFEM_VERIFY(fes.GetMesh()->Conforming(),
               "Non-conforming meshes not yet supported with partial assembly.");
   if (nf==0) { return; }
   height = (m==L2FaceValues::DoubleValued? 2 : 1)*vdim*nf*dof;
   width = fes.GetVSize();
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (!dof_reorder)
   {
      MFEM_ABORT("Non-Tensor L2FaceRestriction not yet implemented.");
   }
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const FiniteElement *fe =
            fes.GetTraceElement(f, fes.GetMesh()->GetFaceBaseGeometry(f));
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
   }
   // End of verifications
   Mesh &mesh = *fes.GetMesh();
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   Array<int> faceMap1(dof), faceMap2(dof);
   int e1, e2;
   int face_id1 = -1, face_id2 = -1;
   int orientation = -1;
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   const int elem_dofs = fes.GetFE(0)->GetDof();
   const int dim = fes.GetMesh()->SpaceDimension();
   // Computation of scatter indices
   int f_ind=0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      e1 = info.elem_1_index;
      e2 = info.elem_2_index;
      face_id1 = info.elem_1_local_face;
      face_id2 = info.elem_2_local_face;
      orientation = info.elem_2_orientation;
      if (dof_reorder)
      {
         GetFaceDofs(dim, face_id1, dof1d, faceMap1); // Only for hex
         GetFaceDofs(dim, face_id2, dof1d, faceMap2); // Only for hex
      }
      else
      {
         MFEM_ABORT("FaceRestriction not yet implemented for this type of "
                    "element.");
         // TODO Something with GetFaceDofs?
      }
      if (info.location==Mesh::FaceLocation::Shared)
      {
         MFEM_ABORT("Unexpected shared face in L2FaceRestriction.");
      }
      if ((type==FaceType::Interior &&
           info.location==Mesh::FaceLocation::Interior) ||
          (type==FaceType::Boundary &&
           info.location==Mesh::FaceLocation::Boundary))
      {
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap1[d];
            const int did = face_dof;
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            scatter_indices1[lid] = gid;
         }
         if (m==L2FaceValues::DoubleValued)
         {
            if (type==FaceType::Interior &&
                info.location==Mesh::FaceLocation::Interior)
            {
               for (int d = 0; d < dof; ++d)
               {
                  const int pd = PermuteFaceL2(dim, face_id1, face_id2,
                                               orientation, dof1d, d);
                  const int face_dof = faceMap2[pd];
                  const int did = face_dof;
                  const int gid = elementMap[e2*elem_dofs + did];
                  const int lid = dof*f_ind + d;
                  scatter_indices2[lid] = gid;
               }
            }
            else if (type==FaceType::Boundary &&
                     info.location==Mesh::FaceLocation::Boundary)
            {
               for (int d = 0; d < dof; ++d)
               {
                  const int lid = dof*f_ind + d;
                  scatter_indices2[lid] = -1;
               }
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
   // Computation of gather_indices
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }
   f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      e1 = info.elem_1_index;
      e2 = info.elem_2_index;
      face_id1 = info.elem_1_local_face;
      face_id2 = info.elem_2_local_face;
      orientation = info.elem_2_orientation;
      if (info.location==Mesh::FaceLocation::Shared)
      {
         MFEM_ABORT("Unexpected shared face in NCL2FaceRestriction.");
      }
      if ((type==FaceType::Interior &&
           info.location==Mesh::FaceLocation::Interior) ||
          (type==FaceType::Boundary &&
           info.location==Mesh::FaceLocation::Boundary))
      {
         GetFaceDofs(dim, face_id1, dof1d, faceMap1);
         GetFaceDofs(dim, face_id2, dof1d, faceMap2);
         for (int d = 0; d < dof; ++d)
         {
            const int did = faceMap1[d];
            const int gid = elementMap[e1*elem_dofs + did];
            ++offsets[gid + 1];
         }
         if (m==L2FaceValues::DoubleValued &&
             type==FaceType::Interior &&
             info.location==Mesh::FaceLocation::Interior)
         {
            for (int d = 0; d < dof; ++d)
            {
               const int pd = PermuteFaceL2(dim, face_id1, face_id2,
                                            orientation, dof1d, d);
               const int did = faceMap2[pd];
               const int gid = elementMap[e2*elem_dofs + did];
               ++offsets[gid + 1];
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      e1 = info.elem_1_index;
      e2 = info.elem_2_index;
      face_id1 = info.elem_1_local_face;
      face_id2 = info.elem_2_local_face;
      orientation = info.elem_2_orientation;
      if (info.location==Mesh::FaceLocation::Shared)
      {
         MFEM_ABORT("Unexpected shared face in NCL2FaceRestriction.");
      }
      if ((type==FaceType::Interior &&
           info.location==Mesh::FaceLocation::Interior) ||
          (type==FaceType::Boundary &&
           info.location==Mesh::FaceLocation::Boundary))
      {
         GetFaceDofs(dim, face_id1, dof1d, faceMap1);
         GetFaceDofs(dim, face_id2, dof1d, faceMap2);
         for (int d = 0; d < dof; ++d)
         {
            const int did = faceMap1[d];
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            // We don't shift lid to express that it's e1 of f
            gather_indices[offsets[gid]++] = lid;
         }
         if (m==L2FaceValues::DoubleValued &&
             type==FaceType::Interior &&
             info.location==Mesh::FaceLocation::Interior)
         {
            for (int d = 0; d < dof; ++d)
            {
               const int pd = PermuteFaceL2(dim, face_id1, face_id2,
                                            orientation, dof1d, d);
               const int did = faceMap2[pd];
               const int gid = elementMap[e2*elem_dofs + did];
               const int lid = dof*f_ind + d;
               // We shift lid to express that it's e2 of f
               gather_indices[offsets[gid]++] = nfdofs + lid;
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of faces.");
   for (int i = ndofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
}

void L2FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;

   if (m==L2FaceValues::DoubleValued)
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_indices2 = scatter_indices2.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, 2, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nd;
         const int face = i / nd;
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, 0, face) = d_x(t?c:idx1, t?idx1:c);
         }
         const int idx2 = d_indices2[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, 1, face) = idx2==-1 ? 0.0 : d_x(t?c:idx2, t?idx2:c);
         }
      });
   }
   else
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nd;
         const int face = i / nd;
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, face) = d_x(t?c:idx1, t?idx1:c);
         }
      });
   }
}

void L2FaceRestriction::MultTranspose(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   const int dofs = nfdofs;
   auto d_offsets = offsets.Read();
   auto d_indices = gather_indices.Read();

   if (m == L2FaceValues::DoubleValued)
   {
      auto d_x = Reshape(x.Read(), nd, vd, 2, nf);
      auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
      MFEM_FORALL(i, ndofs,
      {
         const int offset = d_offsets[i];
         const int nextOffset = d_offsets[i + 1];
         for (int c = 0; c < vd; ++c)
         {
            double dofValue = 0;
            for (int j = offset; j < nextOffset; ++j)
            {
               int idx_j = d_indices[j];
               bool isE1 = idx_j < dofs;
               idx_j = isE1 ? idx_j : idx_j - dofs;
               dofValue +=  isE1 ?
               d_x(idx_j % nd, c, 0, idx_j / nd)
               :d_x(idx_j % nd, c, 1, idx_j / nd);
            }
            d_y(t?c:i,t?i:c) += dofValue;
         }
      });
   }
   else
   {
      auto d_x = Reshape(x.Read(), nd, vd, nf);
      auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
      MFEM_FORALL(i, ndofs,
      {
         const int offset = d_offsets[i];
         const int nextOffset = d_offsets[i + 1];
         for (int c = 0; c < vd; ++c)
         {
            double dofValue = 0;
            for (int j = offset; j < nextOffset; ++j)
            {
               int idx_j = d_indices[j];
               dofValue +=  d_x(idx_j % nd, c, idx_j / nd);
            }
            d_y(t?c:i,t?i:c) += dofValue;
         }
      });
   }
}

void L2FaceRestriction::FillI(SparseMatrix &mat,
                              const bool keep_nbr_block) const
{
   const int face_dofs = dof;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto I = mat.ReadWriteI();
   MFEM_FORALL(fdof, nf*face_dofs,
   {
      const int iE1 = d_indices1[fdof];
      const int iE2 = d_indices2[fdof];
      AddNnz(iE1,I,face_dofs);
      AddNnz(iE2,I,face_dofs);
   });
}

void L2FaceRestriction::FillJAndData(const Vector &fea_data,
                                     SparseMatrix &mat,
                                     const bool keep_nbr_block) const
{
   const int face_dofs = dof;
   auto d_indices1 = scatter_indices1.Read();
   auto d_indices2 = scatter_indices2.Read();
   auto I = mat.ReadWriteI();
   auto mat_fea = Reshape(fea_data.Read(), face_dofs, face_dofs, 2, nf);
   auto J = mat.WriteJ();
   auto Data = mat.WriteData();
   MFEM_FORALL(fdof, nf*face_dofs,
   {
      const int f  = fdof/face_dofs;
      const int iF = fdof%face_dofs;
      const int iE1 = d_indices1[f*face_dofs+iF];
      const int iE2 = d_indices2[f*face_dofs+iF];
      const int offset1 = AddNnz(iE1,I,face_dofs);
      const int offset2 = AddNnz(iE2,I,face_dofs);
      for (int jF = 0; jF < face_dofs; jF++)
      {
         const int jE1 = d_indices1[f*face_dofs+jF];
         const int jE2 = d_indices2[f*face_dofs+jF];
         J[offset2+jF] = jE1;
         J[offset1+jF] = jE2;
         Data[offset2+jF] = mat_fea(jF,iF,0,f);
         Data[offset1+jF] = mat_fea(jF,iF,1,f);
      }
   });
}

void L2FaceRestriction::AddFaceMatricesToElementMatrices(const Vector &fea_data,
                                                         Vector &ea_data) const
{
   const int face_dofs = dof;
   const int elem_dofs = elemDofs;
   const int NE = ne;
   if (m==L2FaceValues::DoubleValued)
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_indices2 = scatter_indices2.Read();
      auto mat_fea = Reshape(fea_data.Read(), face_dofs, face_dofs, 2, nf);
      auto mat_ea = Reshape(ea_data.ReadWrite(), elem_dofs, elem_dofs, ne);
      MFEM_FORALL(f, nf,
      {
         const int e1 = d_indices1[f*face_dofs]/elem_dofs;
         const int e2 = d_indices2[f*face_dofs]/elem_dofs;
         for (int j = 0; j < face_dofs; j++)
         {
            const int jB1 = d_indices1[f*face_dofs+j]%elem_dofs;
            for (int i = 0; i < face_dofs; i++)
            {
               const int iB1 = d_indices1[f*face_dofs+i]%elem_dofs;
               AtomicAdd(mat_ea(iB1,jB1,e1), mat_fea(i,j,0,f));
            }
         }
         if (e2 < NE)
         {
            for (int j = 0; j < face_dofs; j++)
            {
               const int jB2 = d_indices2[f*face_dofs+j]%elem_dofs;
               for (int i = 0; i < face_dofs; i++)
               {
                  const int iB2 = d_indices2[f*face_dofs+i]%elem_dofs;
                  AtomicAdd(mat_ea(iB2,jB2,e2), mat_fea(i,j,1,f));
               }
            }
         }
      });
   }
   else
   {
      auto d_indices = scatter_indices1.Read();
      auto mat_fea = Reshape(fea_data.Read(), face_dofs, face_dofs, nf);
      auto mat_ea = Reshape(ea_data.ReadWrite(), elem_dofs, elem_dofs, ne);
      MFEM_FORALL(f, nf,
      {
         const int e = d_indices[f*face_dofs]/elem_dofs;
         for (int j = 0; j < face_dofs; j++)
         {
            const int jE = d_indices[f*face_dofs+j]%elem_dofs;
            for (int i = 0; i < face_dofs; i++)
            {
               const int iE = d_indices[f*face_dofs+i]%elem_dofs;
               AtomicAdd(mat_ea(iE,jE,e), mat_fea(i,j,f));
            }
         }
      });
   }
}

NCL2FaceRestriction::NCL2FaceRestriction(const FiniteElementSpace &fes,
                                         const FaceType type,
                                         const L2FaceValues m)
   : L2FaceRestriction(fes, type, m),
     interp_config(
        (m==L2FaceValues::DoubleValued && type==FaceType::Interior) ? nf : 0)
{
}

NCL2FaceRestriction::NCL2FaceRestriction(const FiniteElementSpace &fes,
                                         const ElementDofOrdering e_ordering,
                                         const FaceType type,
                                         const L2FaceValues m)
   : NCL2FaceRestriction(fes, type, m)
{
   // If fespace == L2
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "NCL2FaceRestriction.");
   if (nf==0) { return; }
   height = (m==L2FaceValues::DoubleValued? 2 : 1)*vdim*nf*dof;
   width = fes.GetVSize();
   const bool dof_reorder = (e_ordering == ElementDofOrdering::LEXICOGRAPHIC);
   if (!dof_reorder)
   {
      MFEM_ABORT("Non-Tensor NCL2FaceRestriction not yet implemented.");
   }
   if (dof_reorder && nf > 0)
   {
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const FiniteElement *fe =
            fes.GetTraceElement(f, fes.GetMesh()->GetFaceBaseGeometry(f));
         const TensorBasisElement* el =
            dynamic_cast<const TensorBasisElement*>(fe);
         if (el) { continue; }
         MFEM_ABORT("Finite element not suitable for lexicographic ordering");
      }
   }
   // End of verifications
   Mesh &mesh = *fes.GetMesh();
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   Array<int> faceMap1(dof), faceMap2(dof);
   int e1, e2;
   int face_id1, face_id2;
   int orientation;
   const int dof1d = fes.GetFE(0)->GetOrder()+1;
   const int elem_dofs = fes.GetFE(0)->GetDof();
   const int dim = fes.GetMesh()->SpaceDimension();
   int nc_cpt = 0;
   using Key = std::pair<const DenseMatrix*,int>;
   std::map<Key, std::pair<int,const DenseMatrix*>> interp_map;
   // Computation of scatter and offsets indices
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }
   int f_ind=0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      e1 = info.elem_1_index;
      e2 = info.elem_2_index;
      face_id1 = info.elem_1_local_face;
      face_id2 = info.elem_2_local_face;
      orientation = info.elem_2_orientation;
      if (dof_reorder)
      {
         GetFaceDofs(dim, face_id1, dof1d, faceMap1); // Only for quad and hex
         GetFaceDofs(dim, face_id2, dof1d, faceMap2); // Only for quad and hex
      }
      else
      {
         MFEM_ABORT("FaceRestriction not yet implemented for this type of "
                    "element.");
         // TODO Something with GetFaceDofs?
      }
      // We skip non-conforming master faces, as they will be treated by the
      // slave faces.
      if (info.conformity==Mesh::FaceConformity::NonConformingMaster)
      {
         continue;
      }
      if ( type==FaceType::Interior &&
           info.location==Mesh::FaceLocation::Interior )
      {
         // Extract scatter indices for elem1
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap1[d];
            const int gid = elementMap[e1*elem_dofs + face_dof];
            const int lid = dof*f_ind + d;
            scatter_indices1[lid] = gid;
            ++offsets[gid + 1];
         }
         if ( m==L2FaceValues::DoubleValued )
         {
            if ( info.conformity==Mesh::FaceConformity::Conforming )
            {
               interp_config[f_ind] = conforming;
               // Extract scatter indices for elem2
               for (int d = 0; d < dof; ++d)
               {
                  int pd = PermuteFaceL2(dim, face_id1, face_id2,
                                         orientation, dof1d, d);
                  const int face_dof = faceMap2[pd];
                  const int gid = elementMap[e2*elem_dofs + face_dof];
                  const int lid = dof*f_ind + d;
                  scatter_indices2[lid] = gid;
                  ++offsets[gid + 1];
               }
            }
            else // Non-conforming face
            {
               MFEM_ASSERT(e_ordering == ElementDofOrdering::LEXICOGRAPHIC,
                           "The following interpolation operator is "
                           "lexicographic.");
               const DenseMatrix* ptMat = mesh.GetNCFacesPtMat(info.ncface);
               Key key(ptMat, face_id1 + 6*face_id2);
               auto itr = interp_map.find(key);
               if (itr == interp_map.end())
               {
                  const DenseMatrix* interpolator =
                     ComputeCoarseToFineInterpolation(ptMat,face_id1,face_id2,
                                                      orientation);
                  interp_map[key] = {nc_cpt, interpolator};
                  interp_config[f_ind] = nc_cpt;
                  nc_cpt++;
               }
               else
               {
                  interp_config[f_ind] = itr->second.first;
               }
               // Extract scatter indices for elem2
               // Contrary to the conforming case, there is no need to call
               // PermuteFaceL2, the permutation is achieved by the
               // interpolation operator for simplicity.
               for (int d = 0; d < dof; ++d)
               {
                  const int face_dof = faceMap2[d];
                  const int gid = elementMap[e2*elem_dofs + face_dof];
                  const int lid = dof*f_ind + d;
                  scatter_indices2[lid] = gid;
                  ++offsets[gid + 1];
               }
            }
         }
         f_ind++;
      }
      else if ( type==FaceType::Boundary &&
                info.location==Mesh::FaceLocation::Boundary )
      {
         // Extract scatter indices for elem1
         for (int d = 0; d < dof; ++d)
         {
            const int face_dof = faceMap1[d];
            const int gid = elementMap[e1*elem_dofs + face_dof];
            const int lid = dof*f_ind + d;
            scatter_indices1[lid] = gid;
            ++offsets[gid + 1];
         }
         if ( m==L2FaceValues::DoubleValued )
         {
            // Extract scatter indices for elem2
            for (int d = 0; d < dof; ++d)
            {
               const int lid = dof*f_ind + d;
               scatter_indices2[lid] = -1;
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of " <<
               (type==FaceType::Interior? "interior" : "boundary") <<
               " faces: " << f_ind << " vs " << nf );
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   // Computation of gather_indices
   f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      e1 = info.elem_1_index;
      e2 = info.elem_2_index;
      face_id1 = info.elem_1_local_face;
      face_id2 = info.elem_2_local_face;
      orientation = info.elem_2_orientation;
      if (info.conformity==Mesh::FaceConformity::NonConformingMaster)
      {
         continue;
      }
      if (info.location==Mesh::FaceLocation::Shared)
      {
         MFEM_ABORT("Unexpected shared face in NCL2FaceRestriction. Use "
                    "ParNCL2FaceRestriction");
      }
      if ((type==FaceType::Interior &&
           info.location==Mesh::FaceLocation::Interior) ||
          (type==FaceType::Boundary &&
           info.location==Mesh::FaceLocation::Boundary))
      {
         GetFaceDofs(dim, face_id1, dof1d, faceMap1);
         GetFaceDofs(dim, face_id2, dof1d, faceMap2);
         // Extract gather indices for elem1
         for (int d = 0; d < dof; ++d)
         {
            const int did = faceMap1[d];
            const int gid = elementMap[e1*elem_dofs + did];
            const int lid = dof*f_ind + d;
            // We don't shift lid to express that it's e1 of f
            gather_indices[offsets[gid]++] = lid;
         }
         if (m==L2FaceValues::DoubleValued &&
             type==FaceType::Interior &&
             info.location==Mesh::FaceLocation::Interior)
         {
            if (info.conformity==Mesh::FaceConformity::Conforming)
            {
               // Extract gather indices for elem2
               for (int d = 0; d < dof; ++d)
               {
                  int pd = PermuteFaceL2(dim, face_id1, face_id2,
                                         orientation, dof1d, d);
                  const int did = faceMap2[pd];
                  const int gid = elementMap[e2*elem_dofs + did];
                  const int lid = dof*f_ind + d;
                  // We shift lid to express that it's e2 of f
                  gather_indices[offsets[gid]++] = nfdofs + lid;
               }
            }
            else
            {
               // Extract gather indices for elem2
               for (int d = 0; d < dof; ++d)
               {
                  // The permutation is handled by the interpolation operator.
                  const int did = faceMap2[d];
                  const int gid = elementMap[e2*elem_dofs + did];
                  const int lid = dof*f_ind + d;
                  // We shift lid to express that it's e2 of f
                  gather_indices[offsets[gid]++] = nfdofs + lid;
               }
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Unexpected number of " <<
               (type==FaceType::Interior? "interior" : "boundary") <<
               " faces: " << f_ind << " vs " << nf );
   // Switch back offsets to their correct value
   for (int i = ndofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
   // Transform the interpolation matrix map into a contiguous memory structure.
   nc_size = interp_map.size();
   interpolators.SetSize(dof*dof*nc_size);
   auto interp = Reshape(interpolators.HostWrite(),dof,dof,nc_size);
   for (auto val : interp_map)
   {
      const int idx = val.second.first;
      for (int i = 0; i < dof; i++)
      {
         for (int j = 0; j < dof; j++)
         {
            interp(i,j,idx) = (*val.second.second)(i,j);
         }
      }
      delete val.second.second;
   }
}

void NCL2FaceRestriction::Mult(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;

   if ( type==FaceType::Interior && m==L2FaceValues::DoubleValued )
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_indices2 = scatter_indices2.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, 2, nf);
      auto interp_config_ptr = interp_config.Read();
      auto interp = Reshape(interpolators.Read(), nd, nd, nc_size);
      static constexpr int max_nd = 16*16;
      MFEM_VERIFY(nd<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nd, 1, 1,
      {
         MFEM_SHARED double dofs[max_nd];
         for (int side = 0; side < 2; side++)
         {
            const int config = side==0 ? conforming : interp_config_ptr[face];
            if ( config==conforming ) // No interpolation needed
            {
               MFEM_FOREACH_THREAD(dof,x,nd)
               {
                  const int i = face*nd + dof;
                  const int idx = side==0 ? d_indices1[i] : d_indices2[i];
                  for (int c = 0; c < vd; ++c)
                  {
                     d_y(dof, c, side, face) = d_x(t?c:idx, t?idx:c);
                  }
               }
            }
            else // Interpolation from coarse to fine
            {
               for (int c = 0; c < vd; ++c)
               {
                  MFEM_FOREACH_THREAD(dof,x,nd)
                  {
                     const int i = face*nd + dof;
                     const int idx = side==0 ? d_indices1[i] : d_indices2[i];
                     dofs[dof] = d_x(t?c:idx, t?idx:c);
                  }
                  MFEM_SYNC_THREAD;
                  MFEM_FOREACH_THREAD(dofOut,x,nd)
                  {
                     double res = 0.0;
                     for (int dofIn = 0; dofIn<nd; dofIn++)
                     {
                        res += interp(dofOut, dofIn, config)*dofs[dofIn];
                     }
                     d_y(dofOut, c, side, face) = res;
                  }
                  MFEM_SYNC_THREAD;
               }
            }
         }
      });
   }
   else if ( type==FaceType::Boundary && m==L2FaceValues::DoubleValued )
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_indices2 = scatter_indices2.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, 2, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nd;
         const int face = i / nd;
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, 0, face) = d_x(t?c:idx1, t?idx1:c);
         }
         const int idx2 = d_indices2[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, 1, face) = idx2==-1 ? 0.0 : d_x(t?c:idx2, t?idx2:c);
         }
      });
   }
   else // Single valued
   {
      auto d_indices1 = scatter_indices1.Read();
      auto d_x = Reshape(x.Read(), t?vd:ndofs, t?ndofs:vd);
      auto d_y = Reshape(y.Write(), nd, vd, nf);
      MFEM_FORALL(i, nfdofs,
      {
         const int dof = i % nd;
         const int face = i / nd;
         const int idx1 = d_indices1[i];
         for (int c = 0; c < vd; ++c)
         {
            d_y(dof, c, face) = d_x(t?c:idx1, t?idx1:c);
         }
      });
   }
}

void NCL2FaceRestriction::MultTranspose(const Vector& x, Vector& y) const
{
   // Assumes all elements have the same number of dofs
   const int nd = dof;
   const int vd = vdim;
   const bool t = byvdim;
   // Interpolation from slave to master face dofs
   if ( type==FaceType::Interior && m==L2FaceValues::DoubleValued )
   {
      auto d_x = Reshape(const_cast<Vector&>(x).ReadWrite(), nd, vd, 2, nf);
      auto interp_config_ptr = interp_config.Read();
      auto interp = Reshape(interpolators.Read(), nd, nd, nc_size);
      static constexpr int max_nd = 16*16;
      MFEM_VERIFY(nd<=max_nd, "Too many degrees of freedom.");
      MFEM_FORALL_3D(face, nf, nd, 1, 1,
      {
         MFEM_SHARED double dofs[max_nd];
         const int side = 1;
         const int config = interp_config_ptr[face];
         if ( config!=conforming ) // Interpolation from fine to coarse
         {
            for (int c = 0; c < vd; ++c)
            {
               MFEM_FOREACH_THREAD(dof,x,nd)
               {
                  dofs[dof] = d_x(dof, c, side, face);
               }
               MFEM_SYNC_THREAD;
               MFEM_FOREACH_THREAD(dofOut,x,nd)
               {
                  double res = 0.0;
                  for (int dofIn = 0; dofIn<nd; dofIn++)
                  {
                     res += interp(dofIn, dofOut, config)*dofs[dofIn];
                  }
                  d_x(dofOut, c, side, face) = res;
               }
               MFEM_SYNC_THREAD;
            }
         }
      });
   }
   const int dofs = nfdofs;
   auto d_offsets = offsets.Read();
   auto d_indices = gather_indices.Read();
   // Gathering of face dofs into element dofs
   if ( m==L2FaceValues::DoubleValued )
   {
      auto d_x = Reshape(x.Read(), nd, vd, 2, nf);
      auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
      MFEM_FORALL(i, ndofs,
      {
         const int offset = d_offsets[i];
         const int nextOffset = d_offsets[i + 1];
         for (int c = 0; c < vd; ++c)
         {
            double dofValue = 0;
            for (int j = offset; j < nextOffset; ++j)
            {
               int idx_j = d_indices[j];
               bool isE1 = idx_j < dofs;
               idx_j = isE1 ? idx_j : idx_j - dofs;
               dofValue +=  isE1 ?
               d_x(idx_j % nd, c, 0, idx_j / nd)
               :d_x(idx_j % nd, c, 1, idx_j / nd);
            }
            d_y(t?c:i,t?i:c) += dofValue;
         }
      });
   }
   else // Single valued
   {
      auto d_x = Reshape(x.Read(), nd, vd, nf);
      auto d_y = Reshape(y.Write(), t?vd:ndofs, t?ndofs:vd);
      MFEM_FORALL(i, ndofs,
      {
         const int offset = d_offsets[i];
         const int nextOffset = d_offsets[i + 1];
         for (int c = 0; c < vd; ++c)
         {
            double dofValue = 0;
            for (int j = offset; j < nextOffset; ++j)
            {
               int idx_j = d_indices[j];
               dofValue +=  d_x(idx_j % nd, c, idx_j / nd);
            }
            d_y(t?c:i,t?i:c) += dofValue;
         }
      });
   }
}

void NCL2FaceRestriction::FillI(SparseMatrix &mat,
                                const bool keep_nbr_block) const
{
   MFEM_ABORT("Not yet implemented.");
}

void NCL2FaceRestriction::FillJAndData(const Vector &ea_data,
                                       SparseMatrix &mat,
                                       const bool keep_nbr_block) const
{
   MFEM_ABORT("Not yet implemented.");
}

void NCL2FaceRestriction::AddFaceMatricesToElementMatrices(Vector &fea_data,
                                                           Vector &ea_data)
const
{
   MFEM_ABORT("Not yet implemented.");
}

int ToLexOrdering(const int dim, const int face_id, const int size1d,
                  const int index)
{
   switch (dim)
   {
      case 1:
         return 0;
      case 2:
         return ToLexOrdering2D(face_id, size1d, index);
      case 3:
         return ToLexOrdering3D(face_id, size1d, index%size1d, index/size1d);
      default:
         MFEM_ABORT("Unsupported dimension.");
         return 0;
   }
}

const DenseMatrix* NCL2FaceRestriction::ComputeCoarseToFineInterpolation(
   const DenseMatrix* ptMat,
   const int face_id1,
   const int face_id2,
   const int orientation)
{
   // Computation of the interpolation matrix from master
   // (coarse) face to slave (fine) face.
   // Assumes all trace elements are the same.
   const FiniteElement *trace_fe =
      fes.GetTraceElement(0, fes.GetMesh()->GetFaceBaseGeometry(0));
   DenseMatrix* interpolator = new DenseMatrix(dof,dof);
   Vector shape(dof);
   IntegrationPoint f_ip;

   switch (trace_fe->GetGeomType())
   {
      case Geometry::SQUARE:
      {
         MFEM_ASSERT(ptMat->Height() == 2, "Unexpected PtMat height.");
         MFEM_ASSERT(ptMat->Width() == 4, "Unexpected PtMat width.");
         // Reference coordinates (xi, eta) in [0,1]^2
         // Fine face coordinates (x, y)
         // x = a0*(1-xi)*(1-eta) + a1*xi*(1-eta) + a2*xi*eta + a3*(1-xi)*eta
         // y = b0*(1-xi)*(1-eta) + b1*xi*(1-eta) + b2*xi*eta + b3*(1-xi)*eta
         double a0, a1, a2, a3, b0, b1, b2, b3;
         a0 = (*ptMat)(0,0);
         a1 = (*ptMat)(0,1);
         a2 = (*ptMat)(0,2);
         a3 = (*ptMat)(0,3);

         b0 = (*ptMat)(1,0);
         b1 = (*ptMat)(1,1);
         b2 = (*ptMat)(1,2);
         b3 = (*ptMat)(1,3);
         const IntegrationRule & nodes = trace_fe->GetNodes();
         const int dof1d = fes.GetFE(0)->GetOrder()+1;
         const int dim = fes.GetMesh()->SpaceDimension();
         for (int i = 0; i < dof; i++)
         {
            const IntegrationPoint &ip = nodes[i];
            double xi = ip.x, eta = ip.y;
            f_ip.x = a0*(1-xi)*(1-eta) + a1*xi*(1-eta) + a2*xi*eta + a3*(1-xi)*eta;
            f_ip.y = b0*(1-xi)*(1-eta) + b1*xi*(1-eta) + b2*xi*eta + b3*(1-xi)*eta;
            trace_fe->CalcShape(f_ip, shape);
            int li = ToLexOrdering(dim, face_id1, dof1d, i);
            for (int j = 0; j < dof; j++)
            {
               const int lj = ToLexOrdering(dim, face_id2, dof1d, j);
               (*interpolator)(li,lj) = shape(j);
            }
         }
      }
      break;
      case Geometry::SEGMENT:
      {
         MFEM_ASSERT(ptMat->Height() == 1, "Unexpected PtMat height.");
         MFEM_ASSERT(ptMat->Width() == 2, "Unexpected PtMat width.");
         double x_min, x_max;
         x_min = (*ptMat)(0,0);
         x_max = (*ptMat)(0,1);
         const IntegrationRule & nodes = trace_fe->GetNodes();
         const int dof1d = fes.GetFE(0)->GetOrder()+1;
         const int dim = fes.GetMesh()->SpaceDimension();
         for (int i = 0; i < dof; i++)
         {
            const IntegrationPoint &ip = nodes[i];
            f_ip.x = x_min + (x_max - x_min) * ip.x;
            trace_fe->CalcShape(f_ip, shape);
            int li =  ToLexOrdering(dim, face_id2, dof1d, i);
            // We need to permute since the orientation is not
            // in the PointMatrix in 2D.
            li = PermuteFaceL2(dim, face_id2, face_id1,
                               orientation, dof1d, li);
            for (int j = 0; j < dof; j++)
            {
               const int lj = ToLexOrdering(dim, face_id2, dof1d, j);
               (*interpolator)(li,lj) = shape(j);
            }
         }
      }
      break;
      default: MFEM_ABORT("unsupported geometry");
   }
   return interpolator;
}

} // namespace mfem
