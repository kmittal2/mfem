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

#include "../../../fem/gridfunc.hpp"
#include "ceed.hpp"

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED

static void InitNonTensorRestriction(const mfem::FiniteElementSpace &fes,
                                     Ceed ceed, CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = fes.GetFE(0);
   const int P = fe->GetDof();
   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   const mfem::Table &el_dof = fes.GetElementToDofTable();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
   const mfem::TensorBasisElement * tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   const int stride = compstride == 1 ? fes.GetVDim() : 1;
   if (tfe) // Lexicographic ordering using dof_map
   {
      const mfem::Array<int>& dof_map = tfe->GetDofMap();
      for (int i = 0; i < fes.GetNE(); i++)
      {
         const int el_offset = P * i;
         for (int j = 0; j < P; j++)
         {
            tp_el_dof[j+el_offset] = stride*el_dof.GetJ()[dof_map[j]+el_offset];
         }
      }
   }
   else  // Native ordering
   {
      for (int e = 0; e < fes.GetNE(); e++)
      {
         for (int i = 0; i < P; i++)
         {
            tp_el_dof[i + e*P] = stride*el_dof.GetJ()[i + e*P];
         }
      }
   }
   CeedElemRestrictionCreate(ceed, fes.GetNE(), P, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitNonTensorRestrictionWithIndices(
   const mfem::FiniteElementSpace &fes,
   int nelem,
   const int* indices,
   Ceed ceed, CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = fes.GetFE(indices[0]);
   const int P = fe->GetDof();
   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   mfem::Array<int> tp_el_dof(nelem*P);
   const mfem::TensorBasisElement * tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   Array<int> dofs;
   const int stride = compstride == 1 ? fes.GetVDim() : 1;
   if (tfe) // Lexicographic ordering using dof_map
   {
      const mfem::Array<int>& dof_map = tfe->GetDofMap();
      for (int i = 0; i < nelem; i++)
      {
         const int elem_index = indices[i];
         fes.GetElementDofs(elem_index, dofs);
         const int el_offset = P * i;
         for (int j = 0; j < P; j++)
         {
            tp_el_dof[j + el_offset] = stride*dofs[dof_map[j]];
         }
      }
   }
   else  // Native ordering
   {
      for (int i = 0; i < nelem; i++)
      {
         const int elem_index = indices[i];
         fes.GetElementDofs(elem_index, dofs);
         const int el_offset = P * i;
         for (int j = 0; j < P; j++)
         {
            tp_el_dof[j + el_offset] = stride*dofs[j];
         }
      }
   }
   CeedElemRestrictionCreate(ceed, nelem, P, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

// TODO fuse Tensor and NonTensor Restriction
void InitTensorRestriction(const mfem::FiniteElementSpace &fes,
                           Ceed ceed, CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = fes.GetFE(0);
   const mfem::TensorBasisElement * tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   MFEM_VERIFY(tfe, "invalid FE");
   const mfem::Array<int>& dof_map = tfe->GetDofMap();

   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   const mfem::Table &el_dof = fes.GetElementToDofTable();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
   const int dof = fe->GetDof();
   const int stride = compstride == 1 ? fes.GetVDim() : 1;
   if (dof_map.Size()>0)
   {
      for (int i = 0; i < fes.GetNE(); i++)
      {
         const int el_offset = dof * i;
         for (int j = 0; j < dof; j++)
         {
            tp_el_dof[j+el_offset] = stride*el_dof.GetJ()[dof_map[j]+el_offset];
         }
      }
   }
   else // dof_map.Size == 0, means dof_map[j]==j;
   {
      for (int i = 0; i < fes.GetNE(); i++)
      {
         const int el_offset = dof * i;
         for (int j = 0; j < dof; j++)
         {
            tp_el_dof[j+el_offset] = stride*el_dof.GetJ()[j+el_offset];
         }
      }
   }
   CeedElemRestrictionCreate(ceed, fes.GetNE(), dof, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

void InitTensorRestrictionWithIndices(const mfem::FiniteElementSpace &fes,
                                      int nelem,
                                      const int* indices,
                                      Ceed ceed, CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = fes.GetFE(indices[0]);
   const mfem::TensorBasisElement * tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   MFEM_VERIFY(tfe, "invalid FE");
   const mfem::Array<int>& dof_map = tfe->GetDofMap();

   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   const int dof = fe->GetDof();
   mfem::Array<int> tp_el_dof(nelem*dof);
   const int stride = compstride == 1 ? fes.GetVDim() : 1;
   Array<int> dofs;
   if (dof_map.Size()>0)
   {
      for (int i = 0; i < nelem; i++)
      {
         const int elem_index = indices[i];
         fes.GetElementDofs(elem_index, dofs);
         const int el_offset = dof * i;
         for (int j = 0; j < dof; j++)
         {
            tp_el_dof[j+el_offset] = stride*dofs[dof_map[j]];
         }
      }
   }
   else // dof_map.Size == 0, means dof_map[j]==j;
   {
      for (int i = 0; i < nelem; i++)
      {
         const int elem_index = indices[i];
         fes.GetElementDofs(elem_index, dofs);
         const int el_offset = dof * i;
         for (int j = 0; j < dof; j++)
         {
            tp_el_dof[j+el_offset] = stride*dofs[j];
         }
      }
   }
   CeedElemRestrictionCreate(ceed, nelem, dof, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

void InitStridedRestriction(const mfem::FiniteElementSpace &fes,
                            CeedInt nelem, CeedInt nqpts, CeedInt qdatasize,
                            const CeedInt *strides,
                            CeedElemRestriction *restr)
{
   RestrKey restr_key(&fes, nelem, nqpts, qdatasize, restr_type::Strided);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      CeedElemRestrictionCreateStrided(mfem::internal::ceed, nelem, nqpts, qdatasize,
                                       nelem*nqpts*qdatasize,
                                       strides,
                                       restr);
      // Will be automatically destroyed when @a fes gets destroyed.
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

void InitRestriction(const FiniteElementSpace &fes,
                     const IntegrationRule &irm,
                     Ceed ceed,
                     CeedElemRestriction *restr)
{
   // Check for FES -> basis, restriction in hash tables
   const mfem::FiniteElement *fe = fes.GetFE(0);
   const int P = fe->GetDof();
   const int nelem = fes.GetNE();
   const int ncomp = fes.GetVDim();
   RestrKey restr_key(&fes, nelem, P, ncomp, restr_type::Standard);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);

   // Init or retreive key values
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      if (UsesTensorBasis(fes))
      {
         InitTensorRestriction(fes, ceed, restr);
      }
      else
      {
         InitNonTensorRestriction(fes, ceed, restr);
      }
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

void InitRestrictionWithIndices(const FiniteElementSpace &fes,
                                int nelem,
                                const int* indices,
                                Ceed ceed,
                                CeedElemRestriction *restr)
{
   // Check for FES -> basis, restriction in hash tables
   const mfem::FiniteElement *fe = fes.GetFE(indices[0]);
   const int P = fe->GetDof();
   const int ncomp = fes.GetVDim();
   RestrKey restr_key(&fes, nelem, P, ncomp, restr_type::Standard);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);
   const bool tensor = dynamic_cast<const mfem::TensorBasisElement *>
                       (fe) != nullptr;

   // Init or retreive key values
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      if (tensor)
      {
         InitTensorRestrictionWithIndices(fes, nelem, indices, ceed, restr);
      }
      else
      {
         InitNonTensorRestrictionWithIndices(fes, nelem, indices, ceed, restr);
      }
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

#endif

} // namespace ceed

} // namespace mfem
