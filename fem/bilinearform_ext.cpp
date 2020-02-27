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
     trialFes(a->FESpace()),
     testFes(a->FESpace())
{
   elem_restrict_lex = NULL;
   int_face_restrict_lex = NULL;
   bound_face_restrict_lex = NULL;
}

void PABilinearFormExtension::SetupRestrictionOperators()
{
   if (elem_restrict_lex == NULL)
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

   // Construct face restriction operators only if the bilinear form has
   // interior or boundary face integrators
   if (int_face_restrict_lex == NULL && a->GetFBFI()->Size() > 0)
   {
      int_face_restrict_lex = trialFes->GetFaceRestriction(
                                 ElementDofOrdering::LEXICOGRAPHIC, FaceType::Interior);
      if (int_face_restrict_lex)
      {
         faceIntX.SetSize(int_face_restrict_lex->Height(), Device::GetMemoryType());
         faceIntY.SetSize(int_face_restrict_lex->Height(), Device::GetMemoryType());
         faceIntY.UseDevice(true); // ensure 'faceIntY = 0.0' is done on device
      }
   }

   if (bound_face_restrict_lex == NULL && a->GetBFBFI()->Size() > 0)
   {
      bound_face_restrict_lex = trialFes->GetFaceRestriction(
                                   ElementDofOrdering::LEXICOGRAPHIC, FaceType::Boundary);
      if (bound_face_restrict_lex)
      {
         faceBdrX.SetSize(bound_face_restrict_lex->Height(), Device::GetMemoryType());
         faceBdrY.SetSize(bound_face_restrict_lex->Height(), Device::GetMemoryType());
         faceBdrY.UseDevice(true); // ensure 'faceBoundY = 0.0' is done on device
      }
   }
}

void PABilinearFormExtension::Assemble()
{
   SetupRestrictionOperators();

   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->AssemblePA(*a->FESpace());
   }

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int intFaceIntegratorCount = intFaceIntegrators.Size();
   for (int i = 0; i < intFaceIntegratorCount; ++i)
   {
      intFaceIntegrators[i]->AssemblePAInteriorFaces(*a->FESpace());
   }

   Array<BilinearFormIntegrator*> &boundFaceIntegrators = *a->GetBFBFI();
   const int boundFaceIntegratorCount = boundFaceIntegrators.Size();
   for (int i = 0; i < boundFaceIntegratorCount; ++i)
   {
      boundFaceIntegrators[i]->AssemblePABoundaryFaces(*a->FESpace());
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

   const int iFISz = a->GetFBFI()->Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      int_face_restrict_lex = trialFes->GetFaceRestriction(
                                 ElementDofOrdering::LEXICOGRAPHIC, FaceType::Interior);
      if (int_face_restrict_lex)
      {
         faceIntX.SetSize(int_face_restrict_lex->Height());
         faceIntY.SetSize(int_face_restrict_lex->Height());
      }
   }

   const int bFISz = a->GetBFBFI()->Size();
   if (bound_face_restrict_lex && bFISz>0)
   {
      bound_face_restrict_lex = trialFes->GetFaceRestriction(
                                   ElementDofOrdering::LEXICOGRAPHIC, FaceType::Boundary);
      if (bound_face_restrict_lex)
      {
         faceBdrX.SetSize(bound_face_restrict_lex->Height());
         faceBdrY.SetSize(bound_face_restrict_lex->Height());
      }
   }
}

void PABilinearFormExtension::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                               OperatorHandle &A)
{
   Operator *oper;
   Operator::FormSystemOperator(ess_tdof_list, oper);
   A.Reset(oper); // A will own oper
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

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      int_face_restrict_lex->Mult(x, faceIntX);
      if (faceIntX.Size()>0)
      {
         faceIntY = 0.0;
         for (int i = 0; i < iFISz; ++i)
         {
            intFaceIntegrators[i]->AddMultPA(faceIntX, faceIntY);
         }
         int_face_restrict_lex->MultTranspose(faceIntY, y);
      }
   }

   Array<BilinearFormIntegrator*> &boundFaceIntegrators = *a->GetBFBFI();
   const int bFISz = boundFaceIntegrators.Size();
   if (bound_face_restrict_lex && bFISz>0)
   {
      bound_face_restrict_lex->Mult(x, faceBdrX);
      if (faceBdrX.Size()>0)
      {
         faceBdrY = 0.0;
         for (int i = 0; i < bFISz; ++i)
         {
            boundFaceIntegrators[i]->AddMultPA(faceBdrX, faceBdrY);
         }
         bound_face_restrict_lex->MultTranspose(faceBdrY, y);
      }
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

   Array<BilinearFormIntegrator*> &intFaceIntegrators = *a->GetFBFI();
   const int iFISz = intFaceIntegrators.Size();
   if (int_face_restrict_lex && iFISz>0)
   {
      int_face_restrict_lex->Mult(x, faceIntX);
      if (faceIntX.Size()>0)
      {
         faceIntY = 0.0;
         for (int i = 0; i < iFISz; ++i)
         {
            intFaceIntegrators[i]->AddMultTransposePA(faceIntX, faceIntY);
         }
         int_face_restrict_lex->MultTranspose(faceIntY, y);
      }
   }

   Array<BilinearFormIntegrator*> &boundFaceIntegrators = *a->GetBFBFI();
   const int bFISz = boundFaceIntegrators.Size();
   if (bound_face_restrict_lex && bFISz>0)
   {
      bound_face_restrict_lex->Mult(x, faceBdrX);
      if (faceBdrX.Size()>0)
      {
         faceBdrY = 0.0;
         for (int i = 0; i < bFISz; ++i)
         {
            boundFaceIntegrators[i]->AddMultTransposePA(faceBdrX, faceBdrY);
         }
         bound_face_restrict_lex->MultTranspose(faceBdrY, y);
      }
   }
}


MixedBilinearFormExtension::MixedBilinearFormExtension(MixedBilinearForm *form)
   : Operator(form->Height(), form->Width()), a(form)
{
   // empty
}

const Operator *MixedBilinearFormExtension::GetProlongation() const
{
   return a->GetProlongation();
}

const Operator *MixedBilinearFormExtension::GetRestriction() const
{
   return a->GetRestriction();
}

const Operator *MixedBilinearFormExtension::GetOutputProlongation() const
{
   return a->GetOutputProlongation();
}

const Operator *MixedBilinearFormExtension::GetOutputRestriction() const
{
   return a->GetOutputRestriction();
}

// Data and methods for partially-assembled bilinear forms

PAMixedBilinearFormExtension::PAMixedBilinearFormExtension(
   MixedBilinearForm *form)
   : MixedBilinearFormExtension(form),
     trialFes(form->TrialFESpace()),
     testFes(form->TestFESpace()),
     elem_restrict_trial(NULL),
     elem_restrict_test(NULL)
{
   Update();
}

void PAMixedBilinearFormExtension::Assemble()
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->AssemblePA(*trialFes, *testFes);
   }
}

void PAMixedBilinearFormExtension::Update()
{
   trialFes = a->TrialFESpace();
   testFes  = a->TestFESpace();
   height = testFes->GetVSize();
   width = trialFes->GetVSize();
   elem_restrict_trial = trialFes->GetElementRestriction(
                            ElementDofOrdering::LEXICOGRAPHIC);
   elem_restrict_test  =  testFes->GetElementRestriction(
                             ElementDofOrdering::LEXICOGRAPHIC);
   if (elem_restrict_trial)
   {
      localTrial.UseDevice(true);
      localTrial.SetSize(elem_restrict_trial->Height(), Device::GetMemoryType());

   }
   if (elem_restrict_test)
   {
      localTest.UseDevice(true); // ensure 'localY = 0.0' is done on device
      localTest.SetSize(elem_restrict_test->Height(), Device::GetMemoryType());
   }
}

void PAMixedBilinearFormExtension::FormRectangularSystemOperator(
   const Array<int> &trial_tdof_list,
   const Array<int> &test_tdof_list,
   OperatorHandle &A)
{
   Operator * oper;
   Operator::FormRectangularSystemOperator(trial_tdof_list, test_tdof_list, oper);
   A.Reset(oper); // A will own oper
}

void PAMixedBilinearFormExtension::FormRectangularLinearSystem(
   const Array<int> &trial_tdof_list,
   const Array<int> &test_tdof_list,
   Vector &x, Vector &b,
   OperatorHandle &A,
   Vector &X, Vector &B)
{
   Operator *oper;
   Operator::FormRectangularLinearSystem(trial_tdof_list, test_tdof_list, x, b,
                                         oper, X, B);
   A.Reset(oper); // A will own oper
}

void PAMixedBilinearFormExtension::SetupMultInputs(const Operator
                                                   *elem_restrict_x,
                                                   const Vector &x,
                                                   Vector &localX,
                                                   const Operator *elem_restrict_y,
                                                   Vector &y,
                                                   Vector &localY,
                                                   const double c) const
{
   // * G operation: localX = c*local(x)
   if (elem_restrict_x)
   {
      elem_restrict_x->Mult(x, localX);
      if (c != 1.0)
      {
         localX *= c;
      }
   }
   else
   {
      if (c == 1.0)
      {
         localX.SyncAliasMemory(x);
      }
      else
      {
         localX.Set(c, x);
      }
   }
   if (elem_restrict_y)
   {
      localY = 0.0;
   }
   else
   {
      y.UseDevice(true);
      localY.SyncAliasMemory(y);
   }
}

void PAMixedBilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   AddMult(x, y);
}

void PAMixedBilinearFormExtension::AddMult(const Vector &x, Vector &y,
                                           const double c) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int iSz = integrators.Size();

   // * G operation
   SetupMultInputs(elem_restrict_trial, x, localTrial,
                   elem_restrict_test, y, localTest, c);

   // * B^TDB operation
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->AddMultPA(localTrial, localTest);
   }

   // * G^T operation
   if (elem_restrict_test)
   {
      tempY.SetSize(y.Size());
      elem_restrict_test->MultTranspose(localTest, tempY);
      y += tempY;
   }
}

void PAMixedBilinearFormExtension::MultTranspose(const Vector &x,
                                                 Vector &y) const
{
   y = 0.0;
   AddMultTranspose(x, y);
}

void PAMixedBilinearFormExtension::AddMultTranspose(const Vector &x, Vector &y,
                                                    const double c) const
{
   Array<BilinearFormIntegrator*> &integrators = *a->GetDBFI();
   const int iSz = integrators.Size();

   // * G operation
   SetupMultInputs(elem_restrict_test, x, localTest,
                   elem_restrict_trial, y, localTrial, c);

   // * B^TD^TB operation
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->AddMultTransposePA(localTest, localTrial);
   }

   // * G^T operation
   if (elem_restrict_trial)
   {
      tempY.SetSize(y.Size());
      elem_restrict_trial->MultTranspose(localTrial, tempY);
      y += tempY;
   }
}

} // namespace mfem
