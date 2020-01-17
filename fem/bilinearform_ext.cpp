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

// Implementations of classes FABilinearFormExtension, EABilinearFormExtension,
// PABilinearFormExtension and MFBilinearFormExtension.

#include "../general/forall.hpp"
#include "bilinearform.hpp"
#include "libceed/ceed.hpp"

namespace mfem
{

BilinearFormExtension::BilinearFormExtension(BilinearForm *form)
   : Operator(form->Size()), a(form)
{
   // empty
}

const Operator *BilinearFormExtension::GetProlongation() const
{
   return a->GetProlongation();
}

const Operator *BilinearFormExtension::GetRestriction() const
{
   return a->GetRestriction();
}


// Data and methods for partially-assembled bilinear forms
PABilinearFormExtension::PABilinearFormExtension(BilinearForm *form)
   : BilinearFormExtension(form),
     trialFes(a->FESpace()), testFes(a->FESpace())
{
   elem_restrict_lex = trialFes->GetElementRestriction(
                          ElementDofOrdering::LEXICOGRAPHIC);
   if (elem_restrict_lex)
   {
      localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
   }
}

void PABilinearFormExtension::Assemble()
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->AssemblePA(*a->FESpace());
   }
}

void PABilinearFormExtension::AssembleDiagonal(Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();

   const int iSz = integrators.Size();
   if (elem_restrict_lex)
   {
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AssembleDiagonalPA(localY);
      }
      elem_restrict_lex->MultTranspose(localY, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AssembleDiagonalPA(y);
      }
   }

   for (int i = 0; i < y.Size(); ++i)
   {
      y[i] = fabs(
                y[i]); // necessary due to asymmetric signs applied by elem_restrict_lex->MultTranspose.
   }
}

void PABilinearFormExtension::Update()
{
   FiniteElementSpace *fes = a->FESpace();
   height = width = fes->GetVSize();
   trialFes = fes;
   testFes = fes;
   elem_restrict_lex = trialFes->GetElementRestriction(
                          ElementDofOrdering::LEXICOGRAPHIC);
   if (elem_restrict_lex)
   {
      localX.SetSize(elem_restrict_lex->Height());
      localY.SetSize(elem_restrict_lex->Height());
   }
}

void PABilinearFormExtension::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                               OperatorHandle &A)
{
   const Operator* trialP = trialFes->GetProlongationMatrix();
   const Operator* testP  = testFes->GetProlongationMatrix();
   Operator *rap = this;
   if (trialP) { rap = new RAPOperator(*testP, *this, *trialP); }
   const bool own_A = (rap!=this);
   A.Reset(new ConstrainedOperator(rap, ess_tdof_list, own_A));
}

void PABilinearFormExtension::FormLinearSystem(const Array<int> &ess_tdof_list,
                                               Vector &x, Vector &b,
                                               OperatorHandle &A,
                                               Vector &X, Vector &B,
                                               int copy_interior)
{
   Operator *oper;
   Operator::FormLinearSystem(ess_tdof_list, x, b, oper, X, B, copy_interior);
   A.Reset(oper); // A will own oper
}

void PABilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();

   const int iSz = integrators.Size();
   if (DeviceCanUseCeed() || !elem_restrict_lex)
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultPA(x, y);
      }
   }
   else
   {
      elem_restrict_lex->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultPA(localX, localY);
      }
      elem_restrict_lex->MultTranspose(localY, y);
   }
}

void PABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int iSz = integrators.Size();
   if (elem_restrict_lex)
   {
      elem_restrict_lex->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultTransposePA(localX, localY);
      }
      elem_restrict_lex->MultTranspose(localY, y);
   }
   else
   {
      y.UseDevice(true);
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultTransposePA(x, y);
      }
   }
}

MixedBilinearFormExtension::MixedBilinearFormExtension(MixedBilinearForm *form)
   : Operator(form->Height(), form->Width()), a(form)
{
   // empty
}

// Data and methods for partially-assembled bilinear forms
PAMixedBilinearFormExtension::PAMixedBilinearFormExtension(
   MixedBilinearForm *form)
   : MixedBilinearFormExtension(form),
     trialFes(a->TrialFESpace()), testFes(a->TestFESpace())
{
   trial_elem_restrict_lex = trialFes->GetElementRestriction(
                                ElementDofOrdering::LEXICOGRAPHIC);
   test_elem_restrict_lex = testFes->GetElementRestriction(
                               ElementDofOrdering::LEXICOGRAPHIC);

   if (trial_elem_restrict_lex && test_elem_restrict_lex)
   {
      localX.SetSize(trial_elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(test_elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
   }
}

void PAMixedBilinearFormExtension::Assemble()
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->AssemblePA(*a->TrialFESpace(), *a->TestFESpace());
   }
}

void PAMixedBilinearFormExtension::Update()
{
   trialFes = a->TrialFESpace();
   testFes = a->TestFESpace();
   height = testFes->GetVSize();
   width = trialFes->GetVSize();
   trial_elem_restrict_lex = trialFes->GetElementRestriction(
                                ElementDofOrdering::LEXICOGRAPHIC);
   test_elem_restrict_lex = testFes->GetElementRestriction(
                               ElementDofOrdering::LEXICOGRAPHIC);

   if (trial_elem_restrict_lex && test_elem_restrict_lex)
   {
      localX.SetSize(trial_elem_restrict_lex->Height());
      localY.SetSize(test_elem_restrict_lex->Height());
   }
}

// TODO: Generalize with ess_tdof_list for trial and test spaces.
void PAMixedBilinearFormExtension::FormSystemMatrix(const Array<int>
                                                    &ess_tdof_list,
                                                    OperatorHandle &A)
{
   mfem_error("FormSystemMatrix not supported for mixed bilinear forms.");
}

// TODO: Generalize with ess_tdof_list for trial and test spaces.
void PAMixedBilinearFormExtension::FormLinearSystem(const Array<int>
                                                    &ess_tdof_list,
                                                    Vector &x, Vector &b,
                                                    OperatorHandle &A,
                                                    Vector &X, Vector &B,
                                                    int copy_interior)
{
   mfem_error("FormLinearSystem not supported for mixed bilinear forms.");
}

void PAMixedBilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();

   const int iSz = integrators.Size();
   if (trial_elem_restrict_lex && test_elem_restrict_lex)
   {
      trial_elem_restrict_lex->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultPA(localX, localY);
      }
      test_elem_restrict_lex->MultTranspose(localY, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultPA(x, y);
      }
   }
}

void PAMixedBilinearFormExtension::MultTranspose(const Vector &x,
                                                 Vector &y) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int iSz = integrators.Size();
   if (trial_elem_restrict_lex && test_elem_restrict_lex)
   {
      test_elem_restrict_lex->Mult(x, localY);
      localX = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultTransposePA(localY, localX);
      }
      trial_elem_restrict_lex->MultTranspose(localX, y);
   }
   else
   {
      y.UseDevice(true);
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultTransposePA(x, y);
      }
   }
}

} // namespace mfem
