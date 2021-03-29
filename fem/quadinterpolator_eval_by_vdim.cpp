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

#include "quadinterpolator.hpp"
#include "quadinterpolator_eval.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

template<>
void QuadratureInterpolator::Values<QVectorLayout::byVDIM>(
   const Vector &e_vec, Vector &q_val) const
{
   const int NE = fespace->GetNE();
   if (NE == 0) { return; }
   const int vdim = fespace->GetVDim();
   const int dim = fespace->GetMesh()->Dimension();
   const FiniteElement *fe = fespace->GetFE(0);
   const IntegrationRule *ir =
      IntRule ? IntRule : &qspace->GetElementIntRule(0);
   const DofToQuad &maps = fe->GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const double *B = maps.B.Read();
   const double *X = e_vec.Read();
   double *Y = q_val.Write();

   constexpr QVectorLayout L = QVectorLayout::byVDIM;

   const int id = (dim<<12) | (vdim<<8) | (D1D<<4) | Q1D;

   switch (id)
   {
      case 0x2124: return Eval2D<L,1,2,4,8>(NE,B,X,Y);
      case 0x2136: return Eval2D<L,1,3,6,4>(NE,B,X,Y);
      case 0x2148: return Eval2D<L,1,4,8,2>(NE,B,X,Y);

      case 0x2224: return Eval2D<L,2,2,4,8>(NE,B,X,Y);
      case 0x2234: return Eval2D<L,2,3,4,8>(NE,B,X,Y);
      case 0x2236: return Eval2D<L,2,3,6,4>(NE,B,X,Y);
      case 0x2248: return Eval2D<L,2,4,8,2>(NE,B,X,Y);

      case 0x3124: return Eval3D<L,1,2,4>(NE,B,X,Y);
      case 0x3136: return Eval3D<L,1,3,6>(NE,B,X,Y);
      case 0x3148: return Eval3D<L,1,4,8>(NE,B,X,Y);

      case 0x3324: return Eval3D<L,3,2,4>(NE,B,X,Y);
      case 0x3336: return Eval3D<L,3,3,6>(NE,B,X,Y);
      case 0x3348: return Eval3D<L,3,4,8>(NE,B,X,Y);

      default:
      {
         constexpr int MD1 = 8;
         constexpr int MQ1 = 8;
         MFEM_VERIFY(D1D <= MD1, "Orders higher than " << MD1-1
                     << " are not supported!");
         MFEM_VERIFY(Q1D <= MQ1, "Quadrature rules with more than "
                     << MQ1 << " 1D points are not supported!");
         if (dim == 2) { Eval2D<L,0,0,0,0,MD1,MQ1>(NE,B,X,Y,vdim,D1D,Q1D); }
         if (dim == 3) { Eval3D<L,0,0,0,MD1,MQ1>(NE,B,X,Y,vdim,D1D,Q1D); }
         return;
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Kernel not supported yet");
}

} // namespace mfem
