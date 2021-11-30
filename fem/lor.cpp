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

#include "lor.hpp"
#include "lor_assembly.hpp"
#include "restriction.hpp"
#include "pbilinearform.hpp"

#include "../mfem-performance.hpp"

#include "../general/forall.hpp"
#include "../general/nvvp.hpp"

#define MFEM_DEBUG_COLOR 220
#include "../general/debug.hpp"

namespace mfem
{

void LORBase::AddIntegrators(BilinearForm &a_from,
                             BilinearForm &a_to,
                             GetIntegratorsFn get_integrators,
                             AddIntegratorFn add_integrator,
                             const IntegrationRule *ir)
{
   Array<BilinearFormIntegrator*> *integrators = (a_from.*get_integrators)();
   for (int i=0; i<integrators->Size(); ++i)
   {
      BilinearFormIntegrator *integrator = (*integrators)[i];
      if (!integrator->SupportsBatchedLOR()) { supports_batched_assembly = false; }
      (a_to.*add_integrator)(integrator);
      ir_map[integrator] = integrator->GetIntegrationRule();
      if (ir) { integrator->SetIntegrationRule(*ir); }
   }
}

void LORBase::AddIntegratorsAndMarkers(BilinearForm &a_from,
                                       BilinearForm &a_to,
                                       GetIntegratorsFn get_integrators,
                                       GetMarkersFn get_markers,
                                       AddIntegratorMarkersFn add_integrator_marker,
                                       AddIntegratorFn add_integrator,
                                       const IntegrationRule *ir)
{
   Array<BilinearFormIntegrator*> *integrators = (a_from.*get_integrators)();
   Array<Array<int>*> *markers = (a_from.*get_markers)();

   for (int i=0; i<integrators->Size(); ++i)
   {
      BilinearFormIntegrator *integrator = (*integrators)[i];
      if (*markers[i])
      {
         (a_to.*add_integrator_marker)(integrator, *(*markers[i]));
      }
      else
      {
         (a_to.*add_integrator)(integrator);
      }
      if (!integrator->SupportsBatchedLOR()) { supports_batched_assembly = false; }
      ir_map[integrator] = integrator->GetIntegrationRule();
      if (ir) { integrator->SetIntegrationRule(*ir); }
   }
}

void LORBase::ResetIntegrationRules(GetIntegratorsFn get_integrators)
{
   Array<BilinearFormIntegrator*> *integrators = (a->*get_integrators)();
   for (int i=0; i<integrators->Size(); ++i)
   {
      ((*integrators)[i])->SetIntegrationRule(*ir_map[(*integrators)[i]]);
   }
}

LORBase::FESpaceType LORBase::GetFESpaceType() const
{
   const FiniteElementCollection *fec = fes_ho.FEColl();
   if (dynamic_cast<const H1_FECollection*>(fec)) { return H1; }
   else if (dynamic_cast<const ND_FECollection*>(fec)) { return ND; }
   else if (dynamic_cast<const RT_FECollection*>(fec)) { return RT; }
   else if (dynamic_cast<const L2_FECollection*>(fec)) { return L2; }
   else { MFEM_ABORT("Bad LOR space type."); }
   return INVALID;
}

int LORBase::GetLOROrder() const
{
   FESpaceType type = GetFESpaceType();
   return (type == L2 || type == RT) ? 0 : 1;
}

void LORBase::ConstructLocalDofPermutation(Array<int> &perm_) const
{
   FESpaceType type = GetFESpaceType();
   MFEM_VERIFY(type != H1 && type != L2, "");

   auto get_dof_map = [](FiniteElementSpace &fes, int i)
   {
      const FiniteElement *fe = fes.GetFE(i);
      auto tfe = dynamic_cast<const TensorBasisElement*>(fe);
      MFEM_ASSERT(tfe != nullptr, "");
      return tfe->GetDofMap();
   };

   FiniteElementSpace &fes_lor = *fes;
   Mesh &mesh_lor = *fes_lor.GetMesh();
   int dim = mesh_lor.Dimension();
   const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();

   using GeomRef = std::pair<Geometry::Type, int>;
   std::map<GeomRef, int> point_matrices_offsets;
   perm_.SetSize(fes_lor.GetVSize());

   Array<int> vdof_ho, vdof_lor;
   for (int ilor=0; ilor<mesh_lor.GetNE(); ++ilor)
   {
      int iho = cf_tr.embeddings[ilor].parent;
      int p = fes_ho.GetOrder(iho);
      int lor_index = cf_tr.embeddings[ilor].matrix;
      // We use the point matrix index to identify the local LOR element index
      // within the high-order coarse element.
      //
      // In variable-order spaces, the point matrices for each order are
      // concatenated sequentially, so for the given element order, we need to
      // find the offset that will give us the point matrix index relative to
      // the current element order only.
      GeomRef id(mesh_lor.GetElementBaseGeometry(ilor), p);
      if (point_matrices_offsets.find(id) == point_matrices_offsets.end())
      {
         point_matrices_offsets[id] = lor_index;
      }
      lor_index -= point_matrices_offsets[id];

      fes_ho.GetElementVDofs(iho, vdof_ho);
      fes_lor.GetElementVDofs(ilor, vdof_lor);

      if (type == L2)
      {
         perm_[vdof_lor[0]] = vdof_ho[lor_index];
         continue;
      }

      int p1 = p+1;
      int ndof_per_dim = (dim == 2) ? p*p1 : type == ND ? p*p1*p1 : p*p*p1;

      const Array<int> &dofmap_ho = get_dof_map(fes_ho, iho);
      const Array<int> &dofmap_lor = get_dof_map(fes_lor, ilor);

      int off_x = lor_index % p;
      int off_y = (lor_index / p) % p;
      int off_z = (lor_index / p) / p;

      auto set_perm = [&](int off_lor, int off_ho, int n1, int n2)
      {
         for (int i1=0; i1<2; ++i1)
         {
            int m = (dim == 2 || type == RT) ? 1 : 2;
            for (int i2=0; i2<m; ++i2)
            {
               int i;
               i = dofmap_lor[off_lor + i1 + i2*2];
               int s1 = i < 0 ? -1 : 1;
               int idof_lor = vdof_lor[absdof(i)];
               i = dofmap_ho[off_ho + i1*n1 + i2*n2];
               int s2 = i < 0 ? -1 : 1;
               int idof_ho = vdof_ho[absdof(i)];
               int s3 = idof_lor < 0 ? -1 : 1;
               int s4 = idof_ho < 0 ? -1 : 1;
               int s = s1*s2*s3*s4;
               i = absdof(idof_ho);
               perm_[absdof(idof_lor)] = s < 0 ? -1-absdof(i) : absdof(i);
            }
         }
      };

      int offset;

      if (type == ND)
      {
         // x
         offset = off_x + off_y*p + off_z*p*p1;
         set_perm(0, offset, p, p*p1);
         // y
         offset = ndof_per_dim + off_x + off_y*(p1) + off_z*p1*p;
         set_perm(dim == 2 ? 2 : 4, offset, 1, p*p1);
         // z
         if (dim == 3)
         {
            offset = 2*ndof_per_dim + off_x + off_y*p1 + off_z*p1*p1;
            set_perm(8, offset, 1, p+1);
         }
      }
      else if (type == RT)
      {
         // x
         offset = off_x + off_y*p1 + off_z*p*p1;
         set_perm(0, offset, 1, 0);
         // y
         offset = ndof_per_dim + off_x + off_y*p + off_z*p1*p;
         set_perm(2, offset, p, 0);
         // z
         if (dim == 3)
         {
            offset = 2*ndof_per_dim + off_x + off_y*p + off_z*p*p;
            set_perm(4, offset, p*p, 0);
         }
      }
   }
}

void LORBase::ConstructDofPermutation() const
{
   FESpaceType type = GetFESpaceType();
   if (type == H1 || type == L2)
   {
      // H1 and L2: no permutation necessary, return identity
      dof_perm.SetSize(fes->GetVSize());
      for (int i=0; i<dof_perm.Size(); ++i) { dof_perm[i] = i; }
      tdof_perm.SetSize(fes->GetTrueVSize());
      for (int i=0; i<tdof_perm.Size(); ++i) { tdof_perm[i] = i; }
      return;
   }

#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes_ho
      = dynamic_cast<ParFiniteElementSpace*>(&fes_ho);
   ParFiniteElementSpace *pfes_lor = dynamic_cast<ParFiniteElementSpace*>(fes);
   if (pfes_ho && pfes_lor)
   {
      ConstructLocalDofPermutation(dof_perm);
      tdof_perm.SetSize(pfes_lor->GetTrueVSize());
      for (int i=0; i<dof_perm.Size(); ++i)
      {
         int j = dof_perm[i];
         int s = j < 0 ? -1 : 1;
         int t_i = pfes_lor->GetLocalTDofNumber(i);
         int t_j = pfes_ho->GetLocalTDofNumber(absdof(j));
         // Either t_i and t_j both -1, or both non-negative
         if ((t_i < 0 && t_j >=0) || (t_j < 0 && t_i >= 0))
         {
            MFEM_ABORT("Inconsistent DOF numbering");
         }
         if (t_i < 0) { continue; }
         tdof_perm[t_i] = s < 0 ? -1 - t_j : t_j;
      }
   }
   else
#endif
   {
      ConstructLocalDofPermutation(dof_perm);
      tdof_perm.MakeRef(dof_perm);
   }
}

const Array<int> &LORBase::GetDofPermutation() const
{
   if (dof_perm.Size() == 0) { ConstructDofPermutation(); }
   return dof_perm;
}

const Array<int> &LORBase::GetTrueDofPermutation() const
{
   if (tdof_perm.Size() == 0) { ConstructDofPermutation(); }
   return tdof_perm;
}

bool LORBase::HasSameDofNumbering() const
{
   FESpaceType type = GetFESpaceType();
   return type == H1 || type == L2;
}

const OperatorHandle &LORBase::GetAssembledSystem() const
{
   MFEM_VERIFY(a != nullptr && A.Ptr() != nullptr, "No LOR system assembled");
   return A;
}

const Operator *LORBase::GetLORRestriction() const
{
   if (R_lor.Ptr() == NULL)
   {
      R_lor.Reset(new LORRestriction(*fes, fes_ho));
   }
   return R_lor.Ptr();
}

void LORBase::AssembleSystem_(BilinearForm &a_ho,
                              const Array<int> &ess_dofs_ho)
{
   dbg();
   // By default, we want to use "batched assembly", however this is only
   // supported for certain integrators. We set it to true here, and then when
   // we loop through the integrators, if we encounter unsupported integrators,
   // we set it to false.
   supports_batched_assembly = true;
   a->UseExternalIntegrators();
   AddIntegrators(a_ho, *a, &BilinearForm::GetDBFI,
                  &BilinearForm::AddDomainIntegrator, ir_el);
   AddIntegrators(a_ho, *a, &BilinearForm::GetFBFI,
                  &BilinearForm::AddInteriorFaceIntegrator, ir_face);
   AddIntegratorsAndMarkers(a_ho, *a, &BilinearForm::GetBBFI,
                            &BilinearForm::GetBBFI_Marker,
                            &BilinearForm::AddBoundaryIntegrator,
                            &BilinearForm::AddBoundaryIntegrator, ir_face);
   AddIntegratorsAndMarkers(a_ho, *a, &BilinearForm::GetBFBFI,
                            &BilinearForm::GetBFBFI_Marker,
                            &BilinearForm::AddBdrFaceIntegrator,
                            &BilinearForm::AddBdrFaceIntegrator, ir_face);

   //OperatorHandle Ad;
   Array<int> ess_tdof_list_lo;
#warning ess_tdof_list_lo

   if (supports_batched_assembly)
   {
      dbg("supports_batched_assembly");
      fes->GetMesh()->EnsureNodes();

      Array<int> ess_bdr(fes->GetMesh()->bdr_attributes.Max());
      ess_bdr = 1;
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list_lo);


      ParFiniteElementSpace *pfes_ho = dynamic_cast<ParFiniteElementSpace*>(&fes_ho);
#ifdef MFEM_USE_MPI
      if (pfes_ho)
      {
         dbg("=> PARALLEL AssembleBatchedLOR");
         ParAssembleBatchedLOR(*this, *a, fes_ho, ess_tdof_list_lo, A);
      }
      else
      {
         dbg("=> SEQUENTIAL AssembleBatchedLOR");
         AssembleBatchedLOR(*this, *a, fes_ho, ess_tdof_list_lo, A);
      }
#else
      AssembleBatchedLOR(*this, *a, fes_ho, ess_dofs, A);
#endif
   }
   else
   {
      dbg("NOT supports_batched_assembly");
      a->Assemble();
      a->FormSystemMatrix(ess_tdof_list_lo, A);
   }

   /*if (true)
   {
      #warning Ad is A for device
      dbg("Checks");
      const int dofs = fes_ho.GetTrueVSize();
      Vector x(dofs), y(dofs);
      x.Randomize(1);
      y.Randomize(1);

      A.As<SparseMatrix>()->HostReadWriteI();
      A.As<SparseMatrix>()->HostReadWriteJ();
      A.As<SparseMatrix>()->HostReadWriteData();
      const double dot_legacy = A.As<SparseMatrix>()->InnerProduct(x,y);
      dbg("dot_legacy: %.8e", dot_legacy);

      Ad.As<SparseMatrix>()->HostReadWriteI();
      Ad.As<SparseMatrix>()->HostReadWriteJ();
      Ad.As<SparseMatrix>()->HostReadWriteData();
      const double dot_device = Ad.As<SparseMatrix>()->InnerProduct(x,y);
      dbg("dot_device: %.8e", dot_device);
      dbg("fabs(dot_legacy-dot_device): %.8e", fabs(dot_legacy-dot_device));
      MFEM_VERIFY(fabs(dot_legacy-dot_device)<1e-15, "dot_device error!");
      Ad.As<SparseMatrix>()->Add(-1.0, *A.As<SparseMatrix>());
      const double max_norm_deviced = Ad.As<SparseMatrix>()->MaxNorm();
      dbg("max_norm_deviced: %.8e", max_norm_deviced);
      MFEM_VERIFY(max_norm_deviced < 1e-15, "max_norm_deviced");
   }*/

   ResetIntegrationRules(&BilinearForm::GetDBFI);
   ResetIntegrationRules(&BilinearForm::GetFBFI);
   ResetIntegrationRules(&BilinearForm::GetBBFI);
   ResetIntegrationRules(&BilinearForm::GetBFBFI);
}

void LORBase::SetupProlongationAndRestriction()
{
   if (!HasSameDofNumbering())
   {
      Array<int> p;
      ConstructLocalDofPermutation(p);
      fes->CopyProlongationAndRestriction(fes_ho, &p);
   }
   else
   {
      fes->CopyProlongationAndRestriction(fes_ho, nullptr);
   }
}

template <typename FEC>
void CheckScalarBasisType(const FiniteElementSpace &fes)
{
   const FEC *fec = dynamic_cast<const FEC*>(fes.FEColl());
   if (fec)
   {
      int btype = fec->GetBasisType();
      if (btype != BasisType::GaussLobatto)
      {
         mfem::err << "\nWARNING: Constructing low-order refined "
                   << "discretization with basis type\n"
                   << BasisType::Name(btype) << ". "
                   << "The LOR discretization is only spectrally equivalent\n"
                   << "with Gauss-Lobatto basis.\n" << std::endl;
      }
   }
}

template <typename FEC>
void CheckVectorBasisType(const FiniteElementSpace &fes)
{
   const FEC *fec = dynamic_cast<const FEC*>(fes.FEColl());
   if (fec)
   {
      int cbtype = fec->GetClosedBasisType();
      int obtype = fec->GetOpenBasisType();
      if (cbtype != BasisType::GaussLobatto || obtype != BasisType::IntegratedGLL)
      {
         mfem::err << "\nWARNING: Constructing vector low-order refined "
                   << "discretization with basis type \npair ("
                   << BasisType::Name(cbtype) << ", "
                   << BasisType::Name(obtype) << "). "
                   << "The LOR discretization is only spectrally\nequivalent "
                   << "with basis types (Gauss-Lobatto, IntegratedGLL).\n"
                   << std::endl;
      }
   }
}

void CheckBasisType(const FiniteElementSpace &fes)
{
   CheckScalarBasisType<H1_FECollection>(fes);
   CheckVectorBasisType<ND_FECollection>(fes);
   CheckVectorBasisType<RT_FECollection>(fes);
   // L2 is a bit more complicated, for now don't verify basis type
}

LORBase::LORBase(FiniteElementSpace &fes_ho_)
   : irs(0, Quadrature1D::GaussLobatto), fes_ho(fes_ho_)
{
   Mesh &mesh_ = *fes_ho_.GetMesh();
   int dim = mesh_.Dimension();
   Array<Geometry::Type> geoms;
   mesh_.GetGeometries(dim, geoms);
   if (geoms.Size() == 1 && Geometry::IsTensorProduct(geoms[0]))
   {
      ir_el = &irs.Get(geoms[0], 1);
      ir_face = &irs.Get(Geometry::TensorProductGeometry(dim-1), 1);
   }
   else
   {
      ir_el = nullptr;
      ir_face = nullptr;
   }
   a = nullptr;
   supports_batched_assembly = true;
}

LORBase::~LORBase()
{
   delete a;
   delete fes;
   delete fec;
   delete mesh;
}

LORDiscretization::LORDiscretization(BilinearForm &a_ho_,
                                     const Array<int> &ess_tdof_list,
                                     int ref_type)
   : LORDiscretization(*a_ho_.FESpace(), ref_type)
{
   AssembleSystem(a_ho_, ess_tdof_list);
}

LORDiscretization::LORDiscretization(FiniteElementSpace &fes_ho,
                                     int ref_type) : LORBase(fes_ho)
{
   CheckBasisType(fes_ho);

   Mesh &mesh_ho = *fes_ho.GetMesh();
   // For H1, ND and RT spaces, use refinement = element order, for DG spaces,
   // use refinement = element order + 1 (since LOR is p = 0 in this case).
   int increment = (GetFESpaceType() == L2) ? 1 : 0;
   Array<int> refinements(mesh_ho.GetNE());
   for (int i=0; i<refinements.Size(); ++i)
   {
      refinements[i] = fes_ho.GetOrder(i) + increment;
   }
   mesh = new Mesh(Mesh::MakeRefined(mesh_ho, refinements, ref_type));

   fec = fes_ho.FEColl()->Clone(GetLOROrder());
   fes = new FiniteElementSpace(mesh, fec);
   SetupProlongationAndRestriction();

   A.SetType(Operator::MFEM_SPARSEMAT);
}

void LORDiscretization::AssembleSystem(BilinearForm &a_ho,
                                       const Array<int> &ess_dofs)
{
   dbg();
   delete a;
   a = new BilinearForm(&GetFESpace());
   AssembleSystem_(a_ho, ess_dofs);
}

SparseMatrix &LORDiscretization::GetAssembledMatrix() const
{
   dbg();
   MFEM_VERIFY(a != nullptr && A.Ptr() != nullptr, "No LOR system assembled");
   return *A.As<SparseMatrix>();
}

#ifdef MFEM_USE_MPI

ParLORDiscretization::ParLORDiscretization(ParBilinearForm &a_ho_,
                                           const Array<int> &ess_tdof_list,
                                           int ref_type)
   : ParLORDiscretization(*a_ho_.ParFESpace(), ref_type)
{
   dbg();
   AssembleSystem(a_ho_, ess_tdof_list);
}

ParLORDiscretization::ParLORDiscretization(ParFiniteElementSpace &fes_ho,
                                           int ref_type) : LORBase(fes_ho)
{
   dbg();
   if (fes_ho.GetMyRank() == 0) { CheckBasisType(fes_ho); }
   // TODO: support variable-order spaces in parallel
   MFEM_VERIFY(!fes_ho.IsVariableOrder(),
               "Cannot construct LOR operators on variable-order spaces");

   int order = fes_ho.GetMaxElementOrder();
   if (GetFESpaceType() == L2) { ++order; }

   ParMesh &mesh_ho = *fes_ho.GetParMesh();
   ParMesh *pmesh = new ParMesh(ParMesh::MakeRefined(mesh_ho, order, ref_type));
   mesh = pmesh;

   fec = fes_ho.FEColl()->Clone(GetLOROrder());
   ParFiniteElementSpace *pfes = new ParFiniteElementSpace(pmesh, fec);
   fes = pfes;
   SetupProlongationAndRestriction();

   A.SetType(Operator::Hypre_ParCSR);
}

void ParLORDiscretization::AssembleSystem(ParBilinearForm &a_ho,
                                          const Array<int> &ess_dofs)
{
   dbg();
   delete a;
   a = new ParBilinearForm(&GetParFESpace());
   AssembleSystem_(a_ho, ess_dofs);
}

HypreParMatrix &ParLORDiscretization::GetAssembledMatrix() const
{
   dbg();
   MFEM_VERIFY(a != nullptr && A.Ptr() != nullptr, "No LOR system assembled");
   return *A.As<HypreParMatrix>();
}

ParFiniteElementSpace &ParLORDiscretization::GetParFESpace() const
{
   dbg();
   return static_cast<ParFiniteElementSpace&>(*fes);
}

#endif // MFEM_USE_MPI

LORRestriction::LORRestriction(const FiniteElementSpace &fes_lo,
                               const FiniteElementSpace &fes_ho)
   : fes_lo(fes_lo),
     fes_ho(fes_ho),

     ne(fes_lo.GetNE()),
     vdim(fes_lo.GetVDim()),
     byvdim(fes_lo.GetOrdering() == Ordering::byVDIM),
     ndofs(fes_lo.GetNDofs()),
     dof(ne > 0 ? fes_lo.GetFE(0)->GetDof() : 0),

     offsets(ndofs+1),
     indices(ne*dof),
     gatherMap(ne*dof),

     dof_glob2loc(),
     dof_glob2loc_offsets(),
     el_dof_lex()
{
   SetupLocalToElement();
   SetupGlobalToLocal();

   NvtxPush(EnsureNodes,Chocolate);
   // nodes will be ordered byVDIM but won't use SetCurvature each time
   fes_lo.GetMesh()->EnsureNodes();
   NvtxPop();
}

void LORRestriction::Mult(const Vector&, Vector&) const { assert(false); }
void LORRestriction::MultTranspose(const Vector&, Vector&) const { assert(false); }

void LORRestriction::SetupLocalToElement()
{
   NvtxPush(Setup,Chocolate);

   NvtxPush(Ini,LightBlue);
   MFEM_VERIFY(ne>0, "ne==0 not supported");

   const FiniteElement *fe = fes_lo.GetFE(0);
   const TensorBasisElement* el =
      dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(el, "!TensorBasisElement");

   const Array<int> &fe_dof_map = el->GetDofMap();
   MFEM_VERIFY(fe_dof_map.Size() > 0, "invalid dof map");
   NvtxPop(Ini);

   const Table& e2dTable = fes_lo.GetElementToDofTable();

   auto d_offsets = offsets.Write();
   const int NDOFS = ndofs;
   NvtxPush(Flush,DarkSalmon);
   MFEM_FORALL(i, NDOFS+1, d_offsets[i] = 0;);
   NvtxPop(Flush);

   NvtxPush(offsets,IndianRed);
   const Memory<int> &J = e2dTable.GetJMemory();
   const MemoryClass mc = Device::GetDeviceMemoryClass();
   const int *d_elementMap = J.Read(mc, J.Capacity());
   const int DOF = dof;
   MFEM_FORALL(e, ne,
   {
      for (int d = 0; d < DOF; ++d)
      {
         const int sgid = d_elementMap[DOF*e + d];  // signed
         const int gid = (sgid >= 0) ? sgid : -1 - sgid;
         AtomicAdd(d_offsets[gid+1], 1);
      }
   });
   NvtxPop(offsets);

   NvtxPush(Aggregate,Moccasin);
   // Aggregate to find offsets for each global dof
   offsets.HostReadWrite();
   for (int i = 1; i <= ndofs; ++i) { offsets[i] += offsets[i - 1]; }
   NvtxPop(Aggregate);

   NvtxPush(Fill,DarkOrange);
   // For each global dof, fill in all local nodes that point to it
   auto d_gather = gatherMap.Write();
   auto d_indices = indices.Write();
   auto drw_offsets = offsets.ReadWrite();
   const auto dof_map_mem = fe_dof_map.GetMemory();
   const auto d_dof_map = fe_dof_map.GetMemory().Read(mc,dof_map_mem.Capacity());
   MFEM_FORALL(e, ne,
   {
      for (int d = 0; d < DOF; ++d)
      {
         const int sdid = d_dof_map[d];  // signed
         const int did = d;
         const int sgid = d_elementMap[DOF*e + did];  // signed
         const int gid = (sgid >= 0) ? sgid : -1-sgid;
         const int lid = DOF*e + d;
         const bool plus = (sgid >= 0 && sdid >= 0) || (sgid < 0 && sdid < 0);
         d_gather[lid] = plus ? gid : -1-gid;
         d_indices[AtomicAdd(drw_offsets[gid], 1)] = plus ? lid : -1-lid;
      }
   });
   NvtxPop(Fill);

   NvtxPush(Shift,YellowGreen);
   offsets.HostReadWrite();
   for (int i = ndofs; i > 0; --i) { offsets[i] = offsets[i - 1]; }
   offsets[0] = 0;
   NvtxPop(Shift);

   NvtxPop(Setup);
}

void LORRestriction::SetupGlobalToLocal()
{
   const int ndof = fes_ho.GetVSize();
   const int nel_ho = fes_ho.GetMesh()->GetNE();
   const int order = fes_ho.GetMaxElementOrder();
   const int dim = fes_ho.GetMesh()->Dimension();
   MFEM_VERIFY(dim==3, "Not supported");
   const int nd1d = order + 1;
   const int ndof_per_el = nd1d*nd1d*nd1d;

   dof_glob2loc.SetSize(2*ndof_per_el*nel_ho);
   dof_glob2loc_offsets.SetSize(ndof+1);
   el_dof_lex.SetSize(ndof_per_el*nel_ho);

   NvtxPush(BlockMapping, Olive);
   Array<int> dofs;

   const Array<int> &lex_map =
      dynamic_cast<const NodalFiniteElement&>
      (*fes_ho.GetFE(0)).GetLexicographicOrdering();

   dof_glob2loc_offsets = 0;

   for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
   {
      fes_ho.GetElementDofs(iel_ho, dofs);
      for (int i=0; i<ndof_per_el; ++i)
      {
         const int dof = dofs[lex_map[i]];
         el_dof_lex[i + iel_ho*ndof_per_el] = dof;
         dof_glob2loc_offsets[dof+1] += 2;
      }
   }

   dof_glob2loc_offsets.PartialSum();

   // Sanity check
   MFEM_VERIFY(dof_glob2loc_offsets[ndof] == dof_glob2loc.Size(), "");

   Array<int> dof_ptr(ndof);

   for (int i=0; i<ndof; ++i) { dof_ptr[i] = dof_glob2loc_offsets[i]; }

   for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
   {
      fes_ho.GetElementDofs(iel_ho, dofs);
      for (int i=0; i<ndof_per_el; ++i)
      {
         const int dof = dofs[lex_map[i]];
         dof_glob2loc[dof_ptr[dof]++] = iel_ho;
         dof_glob2loc[dof_ptr[dof]++] = i;
      }
   }
   NvtxPop(BlockMapping);
}

static MFEM_HOST_DEVICE int GetMinElt(const int *my_elts, const int nbElts,
                                      const int *nbr_elts, const int nbrNbElts)
{
   // Find the minimal element index found in both my_elts[] and nbr_elts[]
   int min_el = INT_MAX;
   for (int i = 0; i < nbElts; i++)
   {
      const int e_i = my_elts[i];
      if (e_i >= min_el) { continue; }
      for (int j = 0; j < nbrNbElts; j++)
      {
         if (e_i==nbr_elts[j])
         {
            min_el = e_i; // we already know e_i < min_el
            break;
         }
      }
   }
   return min_el;
}

int LORRestriction::FillI(SparseMatrix &mat) const
{
   static constexpr int Max = 16;
   const int all_dofs = ndofs;
   const int vd = vdim;
   const int elt_dofs = dof;
   auto I = mat.ReadWriteI();
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_gatherMap = gatherMap.Read();
   MFEM_FORALL(i_L, vd*all_dofs+1, { I[i_L] = 0; });
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
               AtomicAdd(I[i_L],1);
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
               const int min_e = GetMinElt(i_elts, i_nbElts, j_elts, j_nbElts);
               if (e == min_e) // add the nnz only once
               {
                  AtomicAdd(I[i_L],1);
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

void LORRestriction::FillJAndZeroData(SparseMatrix &mat) const
{
   NvtxPush(J,Chartreuse);
   static constexpr int Max = 8;
   const int all_dofs = ndofs;
   const int vd = fes_lo.GetVDim();
   const int elt_dofs = fes_lo.GetFE(0)->GetDof();
   auto I = mat.ReadWriteI();
   auto J = mat.WriteJ();
   auto Data = mat.WriteData();
   const int NE = fes_lo.GetNE();
   auto d_offsets = offsets.Read();
   auto d_indices = indices.Read();
   auto d_gatherMap = gatherMap.Read();

   MFEM_FORALL(e, NE,
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
               const int nnz = AtomicAdd(I[i_L],1);
               J[nnz] = j_L;
               Data[nnz] = 0.0;
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
               const int min_e = GetMinElt(i_elts, i_nbElts, j_elts, j_nbElts);
               if (e == min_e) // add the nnz only once
               {
                  const int nnz = AtomicAdd(I[i_L],1);
                  J[nnz] = j_L;
                  Data[nnz] = 0.0;
               }
            }
         }
      }
   });
   NvtxPop(J);
   NvtxPush(Shift,LightCyan);
   // We need to shift again the entries of I, we do it on CPU as it is very
   // sequential.
   auto h_I = mat.HostReadWriteI();
   const int size = vd*all_dofs;
   for (int i = 0; i < size; i++) { h_I[size-i] = h_I[size-(i+1)]; }
   h_I[0] = 0;
   NvtxPop(Shift);
}

} // namespace mfem
