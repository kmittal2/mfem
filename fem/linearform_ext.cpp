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

#include "linearform.hpp"
#include "../general/forall.hpp"

// Implementations of FullLinearFormExtension.

namespace mfem
{

FullLinearFormExtension::FullLinearFormExtension(LinearForm *lf):
   LinearFormExtension(lf)
{
   const int ne = lf->FESpace()->GetNE();
   const Mesh &mesh = *lf->FESpace()->GetMesh();

   markers.SetSize(ne);
   attributes.SetSize(ne);

   // Fill the attributes on the host
   for (int i = 0; i < ne; ++i) { attributes[i] = mesh.GetAttribute(i); }
}

void FullLinearFormExtension::Assemble()
{
   // This operation is executed on device
   lf->Vector::operator=(0.0);

   // Filter out the unsupported integrators
   MFEM_VERIFY(lf->GetBLFI()->Size() == 0,
               "Integrators added with AddBoundaryIntegrator() "
               "are not supported!");

   MFEM_VERIFY(lf->GetDLFI_Delta()->Size() == 0, ""
               "Integrators added with AddDomainIntegrator() which are "
               "DeltaLFIntegrators with delta coefficients "
               "are not supported!");

   MFEM_VERIFY(lf->GetIFLFI()->Size() == 0,
               "Integrators added with AddInteriorFaceIntegrator() "
               "are not supported!");

   MFEM_VERIFY(lf->GetFLFI()->Size() == 0,
               "Integrators added with AddBdrFaceIntegrator() "
               " are not supported!");

   const FiniteElementSpace &fes = *lf->FESpace();
   const Array<Array<int>*> &domain_integs_marker = *lf->GetDLFIM();
   const int mesh_attributes_size = fes.GetMesh()->attributes.Size();
   const Array<LinearFormIntegrator*> &domain_integs = *lf->GetDLFI();

   for (int k = 0; k < domain_integs.Size(); ++k)
   {
      // Get the markers for this integrator
      const Array<int> *domain_integs_marker_k = domain_integs_marker[k];

      // check if there are markers for this integrator
      const bool has_markers_k = domain_integs_marker_k != nullptr;

      if (has_markers_k)
      {
         // Element attribute marker should be of length mesh->attributes
         MFEM_VERIFY(mesh_attributes_size == domain_integs_marker_k->Size(),
                     "invalid element marker for domain linear form "
                     "integrator #" << k << ", counting from zero");
      }

      // if there are no markers, just use the whole linear form (1)
      if (!has_markers_k) { markers = 1; }
      else
      {
         // otherwise, scan the attributes to set the markers to 0 or 1
         const int NE = fes.GetNE();
         const auto attr = attributes.Read();
         const auto dimk = domain_integs_marker_k->Read();
         auto markers_w = markers.Write();
         MFEM_FORALL(e, NE, markers_w[e] = dimk[attr[e]-1] == 1;);
      }

      domain_integs[k]->AssembleFull(fes, markers, *lf);
   }
}

} // namespace mfem
