// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "tmop.hpp"
#include "linearform.hpp"
#include "pgridfunc.hpp"
#include "tmop_tools.hpp"
#define MFEM_DBG_COLOR 221
#include "../general/dbg.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

// *****************************************************************************
double TMOP_Integrator::GetGridFunctionEnergyPA(const FiniteElementSpace &fes,
                                                const Vector &x) const
{
   dbg("");
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      dbg("");
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   const int dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2, "");
   const int NE = fes.GetMesh()->GetNE();
   const int NQ = ir->GetNPoints();
   const int Q1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();

   DenseTensor Jtr_E(dim, dim, NQ*NE);
   DenseTensor Jpt_E(dim, dim, NQ*NE);

   x.HostRead();
   for (int e = 0; e < NE; e++) // NonlinearForm::GetGridFunctionEnergy
   {
      Vector el_x;
      Array<int> vdofs;
      const FiniteElement *fe = fes.GetFE(e);
      fes.GetElementVDofs(e, vdofs);
      ElementTransformation &T = *fes.GetElementTransformation(e);
      x.GetSubVector(vdofs, el_x);
      {
         // TMOP_Integrator::GetElementEnergy
         // ... fe => el, el_x => elfun
         const FiniteElement &el = *fe;
         const Vector &elfun = el_x;
         const int dof = el.GetDof(), dim = el.GetDim();

         DSh.SetSize(dof, dim);
         Jrt.SetSize(dim);
         Jpr.SetSize(dim);
         Jpt.SetSize(dim);
         PMatI.UseExternalData(elfun.GetData(), dof, dim);
         DenseTensor Jtr(dim, dim, NQ);
         targetC->ComputeElementTargets(T.ElementNo, el, *ir, elfun, Jtr);
         for (int i = 0; i < NQ; i++) { Jtr_E(e*NQ+i) = Jtr(i); }
         for (int i = 0; i < NQ; i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);
            const DenseMatrix &Jtr_i = Jtr(i);
            metric->SetTargetJacobian(Jtr_i);
            CalcInverse(Jtr_i, Jrt);
            el.CalcDShape(ip, DSh);
            MultAtB(PMatI, DSh, Jpr);
            Mult(Jpr, Jrt, Jpt);
            Jpt_E(e*NQ+i) = Jpt;
         }
      }
   }

   const auto W = ir->GetWeights().Read();
   const auto Jtr = Reshape(Jtr_E.Read(), dim, dim, NE*NQ);
   const auto Jpt = Reshape(Jpt_E.Read(), dim, dim, NE*NQ);
   MFEM_VERIFY(NQ == Q1D*Q1D, "");
   Vector energy(NE*NQ), one(NE*NQ);
   auto E = Reshape(energy.Write(), Q1D, Q1D, NE);
   auto O = Reshape(one.Write(), Q1D, Q1D, NE);
   const double metric_normal_d = metric_normal;
   MFEM_VERIFY(metric_normal == 1.0, "");
   //InvariantsEvaluator2D<double> ie;
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const int i = qx + qy * Q1D;
            //const IntegrationPoint &ip = ir->IntPoint(i);
            const double J11 = Jtr(0,0,e*NQ+i);
            const double J12 = Jtr(1,0,e*NQ+i);
            const double J21 = Jtr(0,1,e*NQ+i);
            const double J22 = Jtr(1,1,e*NQ+i);
            const double Jtr_i_Det = (J11*J22)-(J21*J12);
            const double weight = W[i]* Jtr_i_Det;
            double JPT[4];
            DenseMatrix Jpt_a(dim);
            {
               JPT[0] = Jpt(0,0,e*NQ+i);
               JPT[1] = Jpt(1,0,e*NQ+i);
               JPT[2] = Jpt(0,1,e*NQ+i);
               JPT[3] = Jpt(1,1,e*NQ+i);
               Jpt_a.UseExternalData(JPT, dim, dim);
            }
            const double val = metric_normal_d * metric->EvalW(Jpt_a);
            // TMOP_Metric_002::EvalW: 0.5 * ie.Get_I1b() - 1.0;
            // Eval_I1b() // det(J)^{-2/3}*I_1 = I_1/I_3^{1/3}
            //ie.SetJacobian(Jpt.GetData());
            //const double metric_EvalW = 0.5 * ie.Get_I1b() - 1.0;
            //const double val = metric_normal_d * metric_EvalW;
            E(qx,qy,e) = weight * val;
            O(qx,qy,e) = 1.0;
         }
      }
   });
   return energy * one;
}

// *****************************************************************************
// Setup dim, ne, nq, maps, geom & fes
void TMOP_Integrator::AssemblePA(const FiniteElementSpace &fespace)
{
   dbg("");
   fes = &fespace;
   MFEM_ASSERT(fes->GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");
   Mesh *mesh = fes->GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(IntRule,"");
   MFEM_VERIFY(dim == 2, "");
   nq = IntRule->GetNPoints();
   ne = fes->GetMesh()->GetNE();
   const IntegrationRule &ir = *IntRule;
   maps = &fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   geom = mesh->GetGeometricFactors(ir, GeometricFactors::JACOBIANS);
   D.SetSize(dim * dim * nq * ne, Device::GetDeviceMemoryType());
   const int dof = fes->GetFE(0)->GetDof();
   JrtD.SetSize(dof * dim * nq * ne, Device::GetDeviceMemoryType());
}

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void AddMultPA_Kernel_2D(const int NE,
                                const Array<double> &w_,
                                const Array<double> &b_,
                                const Array<double> &g_,
                                const Vector &d_,
                                const Vector &x_,
                                Vector &y_,
                                const int d1d = 0,
                                const int q1d = 0)
{
   constexpr int dim =2;
   constexpr int VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   const auto W = Reshape(w_.Read(), Q1D, Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D, Q1D, VDIM, VDIM, NE);
   auto X = Reshape(x_.Read(), D1D, D1D, VDIM, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, VDIM, NE);
   //dbg("D1D:%d, Q1D:%d", D1D, Q1D);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (sBG+0);
      double (*G)[MD1] = (double (*)[MD1]) (sBG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
      double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);
      MFEM_SHARED double Xz[2][NBZ][MD1*MD1];
      MFEM_SHARED double GD[4][NBZ][MD1*MQ1];
      MFEM_SHARED double GQ[4][NBZ][MQ1*MQ1];
      double (*Xx)[MD1]   = (double (*)[MD1])(Xz[0] + tidz);
      double (*Xy)[MD1]   = (double (*)[MD1])(Xz[1] + tidz);

      double (*DQxB)[MQ1] = (double (*)[MQ1])(GD[0] + tidz);
      double (*DQxG)[MQ1] = (double (*)[MQ1])(GD[1] + tidz);
      double (*DQyB)[MQ1] = (double (*)[MQ1])(GD[2] + tidz);
      double (*DQyG)[MQ1] = (double (*)[MQ1])(GD[3] + tidz);

      double (*QQx0)[MQ1] = (double (*)[MQ1])(GQ[0] + tidz);
      double (*QQx1)[MQ1] = (double (*)[MQ1])(GQ[1] + tidz);
      double (*QQy0)[MQ1] = (double (*)[MQ1])(GQ[2] + tidz);
      double (*QQy1)[MQ1] = (double (*)[MQ1])(GQ[3] + tidz);

      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            Xx[dy][dx] = X(dx,dy,0,e);
            Xy[dy][dx] = X(dx,dy,1,e);
         }
      }
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[2] = {0};
            double v[2] = {0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double cx = Xx[dy][dx];
               const double cy = Xy[dy][dx];
               u[0] += B[qx][dx] * cx;
               v[0] += G[qx][dx] * cx;
               u[1] += B[qx][dx] * cy;
               v[1] += G[qx][dx] * cy;
            }
            DQxB[dy][qx] = u[0];
            DQxG[dy][qx] = v[0];
            DQyB[dy][qx] = u[1];
            DQyG[dy][qx] = v[1];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u[2] = {0};
            double v[2] = {0};
            for (int dy = 0; dy < D1D; ++dy)
            {
               u[0] += DQxG[dy][qx] * B[qy][dy];
               v[0] += DQxB[dy][qx] * G[qy][dy];
               u[1] += DQyG[dy][qx] * B[qy][dy];
               v[1] += DQyB[dy][qx] * G[qy][dy];
            }
            QQx0[qy][qx] = u[0];
            QQx1[qy][qx] = v[0];
            QQy0[qy][qx] = u[1];
            QQy1[qy][qx] = v[1];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double weight = W(qx,qy);

            //  Jtr = targetC->ComputeElementTargets
            const double Jtrx0 = D(qx,qy,0,0,e);
            const double Jtrx1 = D(qx,qy,0,1,e);
            const double Jtry0 = D(qx,qy,1,0,e);
            const double Jtry1 = D(qx,qy,1,1,e);
            double Jtr_p[4] = {Jtrx0, Jtry0, Jtrx1, Jtry1};
            DenseMatrix Jtr(Jtr_p, dim, dim);
            /*{
               dbg("\033[0mdetJtr: %.15e",Jtr.Det());
               dbg("Jtr: %.15e %.15e",Jtr(0,0),Jtr(0,1));
               dbg("Jtr: %.15e %.15e",Jtr(1,0),Jtr(1,1));
            }*/

            const double detJtr = Jtrx0*Jtry1 - Jtrx1*Jtry0;
            const double weight_detJtr = weight * detJtr;

            // Jrt = Jtr^{-1}
            DenseMatrix Jrt(2);
            kernels::CalcInverse<2>(Jtr_p, Jrt.GetData());
            /*{
               dbg("\033[0mdetJrt: %.15e",Jrt.Det());
               dbg("Jrt: %.15e %.15e",Jrt(0,0),Jrt(0,1));
               dbg("Jrt: %.15e %.15e",Jrt(1,0),Jrt(1,1));
            }*/

            // G = X{^T}.DSh
            const double Gx0 = QQx0[qy][qx];
            const double Gx1 = QQx1[qy][qx];
            const double Gy0 = QQy0[qy][qx];
            const double Gy1 = QQy1[qy][qx];
            double G_p[4] = {Gx0, Gy0, Gx1, Gy1};
            DenseMatrix G(G_p, 2, 2);
            /*{
               dbg("\033[0mdetG: %.15e",G.Det());
               dbg("G: %.15e %.15e",G(0,0),G(0,1));
               dbg("G: %.15e %.15e",G(1,0),G(1,1));
            }*/

            // Jpt = X{^T}.DS = (X{^T}.DSh).Jrt = G.Jrt
            DenseMatrix Jpt(2);
            Mult(G,Jrt,Jpt);
            /*{
               dbg("\033[0mdetJpt %.15e",Jpt.Det());
               dbg("Jpt: %.15e %.15e",Jpt(0,0),Jpt(0,1));
               dbg("Jpt: %.15e %.15e",Jpt(1,0),Jpt(1,1));
            }*/

            // metric->EvalP(Jpt, P);
            //const double J[4]= {Jptxx, Jptyx, Jptxy, Jptyy};
            InvariantsEvaluator2D<double> ie;
            ie.SetJacobian(Jpt.GetData());
            DenseMatrix P(2);
            P.Set(0.5, ie.Get_dI1b());

            P *= weight_detJtr;
            /*{
               dbg("\033[0mdetP %.15e",P.Det());
               dbg("P: %.15e %.15e",P(0,0),P(0,1));
               dbg("P: %.15e %.15e",P(1,0),P(1,1));
            }*/

            // PMatO +=  DS . P^t += DSh . (Jrt . (P==Jpt)^t)
            double A_p[4];
            DenseMatrix A(A_p, 2, 2);
            MultABt(Jrt, P, A);
            QQx0[qy][qx] = A(0,0);
            QQy0[qy][qx] = A(0,1);
            QQx1[qy][qx] = A(1,0);
            QQy1[qy][qx] = A(1,1);
            /* {
                dbg("\033[0mdetA: %.15e", A.Det());
                dbg("A: %.15e %.15e",A(0,0), A(0,1));
                dbg("A: %.15e %.15e",A(1,0), A(1,1));
             }*/
         }
      }
      MFEM_SYNC_THREAD;
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bt[d][q] = b(q,d);
               Gt[d][q] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[2] = {0};
            double v[2] = {0};
            for (int qx = 0; qx < Q1D; ++qx)
            {
               u[0] += Gt[dx][qx] * QQx0[qy][qx];
               v[0] += Bt[dx][qx] * QQx1[qy][qx];
               u[1] += Gt[dx][qx] * QQy0[qy][qx];
               v[1] += Bt[dx][qx] * QQy1[qy][qx];
            }
            DQxB[dx][qy] = u[0];
            DQxG[dx][qy] = v[0];
            DQyB[dx][qy] = u[1];
            DQyG[dx][qy] = v[1];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double u[2] = {0};
            double v[2] = {0};
            for (int qy = 0; qy < Q1D; ++qy)
            {
               u[0] += DQxB[dx][qy] * Bt[dy][qy];
               v[0] += DQxG[dx][qy] * Gt[dy][qy];
               u[1] += DQyB[dx][qy] * Bt[dy][qy];
               v[1] += DQyG[dx][qy] * Gt[dy][qy];
            }
            Y(dx,dy,0,e) += u[0] + v[0];
            Y(dx,dy,1,e) += u[1] + v[1];
         }
      }
   });
}

// *****************************************************************************
void TMOP_Integrator::AddMultPA(const Vector &X, Vector &Y) const
{
   MFEM_VERIFY(IntRule,"");
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = maps->B;
   const Array<double> &G = maps->G;
   const int id = (D1D << 4 ) | Q1D;

   // Jtr setup:
   //  - TargetConstructor::target_type == IDEAL_SHAPE_UNIT_SIZE
   //  - Jtr(i) == Wideal
#if 1
   const FiniteElement *fe = fes->GetFE(0);
   const Geometry::Type geom_type = fe->GetGeomType();
   const DenseMatrix Wideal = Geometries.GetGeomToPerfGeomJac(geom_type);
   MFEM_VERIFY(Wideal.Det() == 1.0 ,"");
   {
      MFEM_VERIFY(Wideal(0,0)==1.0 && Wideal(1,1)==1.0 &&
                  Wideal(1,0)==0.0 && Wideal(0,1)==0.0,"");
   }
#else
   DenseMatrix Wideal(dim);
   Wideal(0,0) = 2.0;
   Wideal(0,1) = 0.1;
   Wideal(1,0) = 0.2;
   Wideal(1,1) = -3.0;
#endif
   /*
      Array<int> vdofs;
      DenseTensor Jtr(dim, dim, ir->GetNPoints());
      for (int i = 0; i < fes->GetNE(); i++)
      {
         const FiniteElement *el = fes->GetFE(i);
         fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         px.GetSubVector(vdofs, el_x);
         targetC->ComputeElementTargets(T.ElementNo, el, *ir, elfun, Jtr);
     }*/
   const auto Jtr = Reshape(Wideal.Read(), dim, dim);
   auto J = Reshape(D.Write(), Q1D, Q1D, dim, dim, ne);
   MFEM_FORALL_2D(e, ne, Q1D, Q1D, 1,
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            J(qx,qy,0,0,e) = Jtr(0,0);
            J(qx,qy,0,1,e) = Jtr(0,1);
            J(qx,qy,1,0,e) = Jtr(1,0);
            J(qx,qy,1,1,e) = Jtr(1,1);
         }
      }
   });


   switch (id)
   {
      case 0x21: return AddMultPA_Kernel_2D<2,1,1>(ne,W,B,G,D,X,Y);/*
      case 0x22: return AddMultPA_Kernel_2D<2,2,1>(ne,W,B,G,D,X,Y);
      case 0x23: return AddMultPA_Kernel_2D<2,3,1>(ne,W,B,G,D,X,Y);
      case 0x24: return AddMultPA_Kernel_2D<2,4,1>(ne,W,B,G,D,X,Y);
      case 0x25: return AddMultPA_Kernel_2D<2,5,1>(ne,W,B,G,D,X,Y);

      case 0x31: return AddMultPA_Kernel_2D<3,1,1>(ne,W,B,G,D,X,Y);
      case 0x32: return AddMultPA_Kernel_2D<3,2,1>(ne,W,B,G,D,X,Y);
      case 0x33: return AddMultPA_Kernel_2D<3,3,1>(ne,W,B,G,D,X,Y);
      case 0x34: return AddMultPA_Kernel_2D<3,4,1>(ne,W,B,G,D,X,Y);
      case 0x35: return AddMultPA_Kernel_2D<3,5,1>(ne,W,B,G,D,X,Y);

      case 0x41: return AddMultPA_Kernel_2D<4,1,1>(ne,W,B,G,D,X,Y);
      case 0x42: return AddMultPA_Kernel_2D<4,2,1>(ne,W,B,G,D,X,Y);
      case 0x43: return AddMultPA_Kernel_2D<4,3,1>(ne,W,B,G,D,X,Y);
      case 0x44: return AddMultPA_Kernel_2D<4,4,1>(ne,W,B,G,D,X,Y);
      case 0x45: return AddMultPA_Kernel_2D<4,5,1>(ne,W,B,G,D,X,Y);

      case 0x51: return AddMultPA_Kernel_2D<5,1,1>(ne,W,B,G,D,X,Y);
      case 0x52: return AddMultPA_Kernel_2D<5,2,1>(ne,W,B,G,D,X,Y);
      case 0x53: return AddMultPA_Kernel_2D<5,3,1>(ne,W,B,G,D,X,Y);
      case 0x54: return AddMultPA_Kernel_2D<5,4,1>(ne,W,B,G,D,X,Y);
      case 0x55: return AddMultPA_Kernel_2D<5,5,1>(ne,W,B,G,D,X,Y);*/
      default:  break;
   }
   dbg("kernel id: %x", id);
   MFEM_ABORT("Unknown kernel.");
}


// *****************************************************************************
// dI2_dM = d(det(M))_dM = adj(M)^T.
static void Dim2Invariant2_dM(const DenseMatrix &M, DenseMatrix &dM)
{
   MFEM_ASSERT(M.Height() == 2 && M.Width() == 2, "Incorrect dimensions!");
   dM(0, 0) =  M(1, 1); dM(0, 1) = -M(1, 0);
   dM(1, 0) = -M(0, 1); dM(1, 1) =  M(0, 0);
}

// *****************************************************************************
static
void Dim2Invariant2_dMdM(const DenseMatrix &M, int i, int j,
                         DenseMatrix &dMdM)
{
   MFEM_ASSERT(M.Height() == 2 && M.Width() == 2, "Incorrect dimensions!");
   dMdM = 0.0;
   dMdM(1-i,1-j) = (i == j) ? 1.0 : -1.0;
}

// *****************************************************************************
// (dI1_dM)_d(Mij) = d[(2 det(M) M - |M|^2 adj(M)^T) / det(M)^2]_d[Mij].
static
void Dim2Invariant1_dMdM(const DenseMatrix &M, int i, int j,
                         DenseMatrix &dMdM)
{
   MFEM_ASSERT(M.Height() == 2 && M.Width() == 2, "Incorrect dimensions!");

   // Compute d(det(M))_d(Mij), d(|M|^2)_d(Mij).
   DenseMatrix dI(2);
   Dim2Invariant2_dM(M, dI);
   const double ddet   = dI(i,j);
   const double dfnorm2 = 2.0 * M(i,j);

   const double det    = M.Det();
   const double det2   = det * det;
   const double fnorm2 = M.FNorm2();

   DenseMatrix dM(2); dM = 0.0; dM(i, j) = 1.0;
   DenseMatrix ddI(2);
   Dim2Invariant2_dMdM(M, i, j, ddI);
   for (int r = 0; r < 2; r++)
   {
      for (int c = 0; c < 2; c++)
      {
         dMdM(r,c) =
            (det2 *
             (2.0 * ddet * M(r,c) + 2.0 * det * dM(r,c)
              - dfnorm2 * dI(r,c) - fnorm2 * ddI(r,c))
             - 2.0 * det * ddet *
             (2.0 * det * M(r,c) - fnorm2 * dI(r,c)) ) / (det2 * det2);
      }
   }
}


// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void AddMultGradPA_Kernel_2D(const Vector &xe_,
                                    const int NE,
                                    const Array<double> &w_,
                                    const Array<double> &b_,
                                    const Array<double> &g_,
                                    const Vector &d_,
                                    const Vector &re_,
                                    Vector &ce_,
                                    const int d1d = 0,
                                    const int q1d = 0)
{
   constexpr int dim = 2;
   constexpr int VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   const auto W = Reshape(w_.Read(), Q1D*Q1D);
   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D, Q1D, VDIM, VDIM, NE);
   auto X = Reshape(xe_.Read(), D1D, D1D, VDIM, NE);
   auto R = Reshape(re_.Read(), D1D, D1D, VDIM, NE);
   auto C = Reshape(ce_.ReadWrite(), D1D, D1D, VDIM, NE);
   //dbg("D1D:%d, Q1D:%d, nq:%d", D1D, Q1D, Q1D*Q1D);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      double (*B)[MD1]  = (double (*)[MD1])(s_BG[0]);
      double (*G)[MD1]  = (double (*)[MD1])(s_BG[1]);
      double (*Bt)[MQ1] = (double (*)[MQ1])(s_BG[0]);
      double (*Gt)[MQ1] = (double (*)[MQ1])(s_BG[1]);

      MFEM_SHARED double s_R[2][NBZ][MD1*MD1];
      MFEM_SHARED double s_X[2][NBZ][MD1*MD1];
      double (*Rx)[MD1]  = (double (*)[MD1])(s_R[0] + tidz);
      double (*Ry)[MD1]  = (double (*)[MD1])(s_R[1] + tidz);
      double (*Xx)[MD1]  = (double (*)[MD1])(s_X[0] + tidz);
      double (*Xy)[MD1]  = (double (*)[MD1])(s_X[1] + tidz);

      MFEM_SHARED double s_DQ[3][4][NBZ][MD1*MQ1];
      double (*RxB)[MQ1] = (double (*)[MQ1])(s_DQ[0][0] + tidz);
      double (*RxG)[MQ1] = (double (*)[MQ1])(s_DQ[0][1] + tidz);
      double (*RyB)[MQ1] = (double (*)[MQ1])(s_DQ[0][2] + tidz);
      double (*RyG)[MQ1] = (double (*)[MQ1])(s_DQ[0][3] + tidz);

      double (*XxB)[MQ1] = (double (*)[MQ1])(s_DQ[1][0] + tidz);
      double (*XxG)[MQ1] = (double (*)[MQ1])(s_DQ[1][1] + tidz);
      double (*XyB)[MQ1] = (double (*)[MQ1])(s_DQ[1][2] + tidz);
      double (*XyG)[MQ1] = (double (*)[MQ1])(s_DQ[1][3] + tidz);

      double (*CxB)[MQ1] = (double (*)[MQ1])(s_DQ[2][0] + tidz);
      double (*CxG)[MQ1] = (double (*)[MQ1])(s_DQ[2][1] + tidz);
      double (*CyB)[MQ1] = (double (*)[MQ1])(s_DQ[2][2] + tidz);
      double (*CyG)[MQ1] = (double (*)[MQ1])(s_DQ[2][3] + tidz);

      MFEM_SHARED double s_QQ[3][4][NBZ][MQ1*MQ1];
      double (*Rx0)[MQ1] = (double (*)[MQ1])(s_QQ[0][0] + tidz);
      double (*Rx1)[MQ1] = (double (*)[MQ1])(s_QQ[0][1] + tidz);
      double (*Ry0)[MQ1] = (double (*)[MQ1])(s_QQ[0][2] + tidz);
      double (*Ry1)[MQ1] = (double (*)[MQ1])(s_QQ[0][3] + tidz);

      double (*Xx0)[MQ1] = (double (*)[MQ1])(s_QQ[1][0] + tidz);
      double (*Xx1)[MQ1] = (double (*)[MQ1])(s_QQ[1][1] + tidz);
      double (*Xy0)[MQ1] = (double (*)[MQ1])(s_QQ[1][2] + tidz);
      double (*Xy1)[MQ1] = (double (*)[MQ1])(s_QQ[1][3] + tidz);

      double (*Cx0)[MQ1] = (double (*)[MQ1])(s_QQ[2][0] + tidz);
      double (*Cx1)[MQ1] = (double (*)[MQ1])(s_QQ[2][1] + tidz);
      double (*Cy0)[MQ1] = (double (*)[MQ1])(s_QQ[2][2] + tidz);
      double (*Cy1)[MQ1] = (double (*)[MQ1])(s_QQ[2][3] + tidz);

      // Load R(x,y) and X(x,y)
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            Rx[dy][dx] = R(dx,dy,0,e);
            Ry[dy][dx] = R(dx,dy,1,e);
            Xx[dy][dx] = X(dx,dy,0,e);
            Xy[dy][dx] = X(dx,dy,1,e);
         }
      }
      // Load B1d and G1d matrices
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double ru[2] = {0};
            double rv[2] = {0};
            double xu[2] = {0};
            double xv[2] = {0};
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double rx = Rx[dy][dx];
               const double ry = Ry[dy][dx];
               ru[0] += B[qx][dx] * rx;
               rv[0] += G[qx][dx] * rx;
               ru[1] += B[qx][dx] * ry;
               rv[1] += G[qx][dx] * ry;
               const double xx = Xx[dy][dx];
               const double xy = Xy[dy][dx];
               xu[0] += B[qx][dx] * xx;
               xv[0] += G[qx][dx] * xx;
               xu[1] += B[qx][dx] * xy;
               xv[1] += G[qx][dx] * xy;
            }
            RxB[dy][qx] = ru[0];
            RxG[dy][qx] = rv[0];
            RyB[dy][qx] = ru[1];
            RyG[dy][qx] = rv[1];

            XxB[dy][qx] = xu[0];
            XxG[dy][qx] = xv[0];
            XyB[dy][qx] = xu[1];
            XyG[dy][qx] = xv[1];

         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double ru[2] = {0};
            double rv[2] = {0};
            double xu[2] = {0};
            double xv[2] = {0};
            for (int dy = 0; dy < D1D; ++dy)
            {
               ru[0] += RxG[dy][qx] * B[qy][dy];
               rv[0] += RxB[dy][qx] * G[qy][dy];
               ru[1] += RyG[dy][qx] * B[qy][dy];
               rv[1] += RyB[dy][qx] * G[qy][dy];

               xu[0] += XxG[dy][qx] * B[qy][dy];
               xv[0] += XxB[dy][qx] * G[qy][dy];
               xu[1] += XyG[dy][qx] * B[qy][dy];
               xv[1] += XyB[dy][qx] * G[qy][dy];
            }
            Rx0[qy][qx] = ru[0];
            Rx1[qy][qx] = rv[0];
            Ry0[qy][qx] = ru[1];
            Ry1[qy][qx] = rv[1];

            Xx0[qy][qx] = xu[0];
            Xx1[qy][qx] = xv[0];
            Xy0[qy][qx] = xu[1];
            Xy1[qy][qx] = xv[1];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const int q= qx + qy*Q1D;
            const double weight = W(q);

            //  Jtr = targetC->ComputeElementTargets
            const double Jtrx0 = D(qx,qy,0,0,e);
            const double Jtrx1 = D(qx,qy,0,1,e);
            const double Jtry0 = D(qx,qy,1,0,e);
            const double Jtry1 = D(qx,qy,1,1,e);
            double Jtr_p[4] = {Jtrx0, Jtry0, Jtrx1, Jtry1};

            const double detJtr = Jtrx0*Jtry1 - Jtrx1*Jtry0;
            const double weight_detJtr = weight * detJtr;

            // Jrt = Jtr^{-1}
            DenseMatrix Jrt(dim);
            kernels::CalcInverse<dim>(Jtr_p, Jrt.HostWrite());

            // Compute DSh (dof x dim)
            const int dof = D1D*D1D;
            DenseMatrix DSh(dof, dim);
            for (int i1 = 0; i1 < D1D; ++i1)
            {
               for (int i2 = 0; i2 < D1D; ++i2)
               {
                  const double bg = G[qx][i1] * B[qy][i2];
                  const double gb = B[qx][i1] * G[qy][i2];
                  const int dof = i2 + i1*D1D;
                  DSh(dof, 0) = bg;
                  DSh(dof, 1) = gb;
               }
            }
            //dbg("DSh:"); DSh.Print();

            // Compute DS = DSh Jrt
            DenseMatrix DS(dof, dim);
            Mult(DSh, Jrt, DS);

            // GR = DS.R^T
            // GR = DSh.Jrt.R^T
            // GR = Jrt.(DSh.R^T)
            const double GRx0h = Rx0[qy][qx];
            const double GRx1h = Rx1[qy][qx];
            const double GRy0h = Ry0[qy][qx];
            const double GRy1h = Ry1[qy][qx];
            double hGR_p[4] = {GRx0h, GRy0h, GRx1h, GRy1h};
            DenseMatrix hGR(hGR_p, dim, dim);
            DenseMatrix GR(dim);
            Mult(hGR,Jrt,GR);

            // GX = X^T.DSh
            const double GXx0h = Xx0[qy][qx];
            const double GXx1h = Xx1[qy][qx];
            const double GXy0h = Xy0[qy][qx];
            const double GXy1h = Xy1[qy][qx];
            double GXh_p[4] = {GXx0h, GXy0h, GXx1h, GXy1h};
            DenseMatrix GXh(GXh_p, dim, dim);

            // Jpt = GX^T.DS = (GX^T.DSh).Jrt = GX.Jrt
            DenseMatrix Jpt(dim);
            Mult(GXh,Jrt,Jpt);

            //metric->AssembleH(Jpt, DS, weight_m, elmat);
            InvariantsEvaluator2D<double> ie;
            ie.SetJacobian(Jpt.GetData());
            ie.SetDerivativeMatrix(DS.Height(), DS.GetData());
            DenseMatrix elmat(dof*dim);
            elmat = 0.0;
            ie.Assemble_ddI1b(0.5*weight_detJtr, elmat.GetData());

            DenseMatrix dP(dim);
            DenseMatrix Pelmat(dof*dim);
            Pelmat = 0.0;
            // The first two go over the rows and cols of dP_dJ where P = dW_dJ.
            for (int r = 0; r < dim; r++)
            {
               for (int c = 0; c < dim; c++)
               {
                  Dim2Invariant1_dMdM(Jpt, r, c, dP);
                  // Compute each entry of d(Prc)_dJ.
                  for (int rr = 0; rr < dim; rr++)
                  {
                     for (int cc = 0; cc < dim; cc++)
                     {
                        const double entry_rr_cc = 0.5 * dP(rr,cc);
                        for (int i = 0; i < dof; i++)
                        {
                           for (int j = 0; j < dof; j++)
                           {
                              const double ds = DS(i, c) * DS(j, cc);
                              Pelmat(i+r*dof, j+rr*dof) +=
                                 weight_detJtr * ds * entry_rr_cc;
                           }
                        }
                     }
                  }
               }
            }

            const double EPS = 1.e-4;
            const bool flip = Jpt.Det() < 0.0;
            Pelmat *= flip ? -1.0 : 1.0;
            //dbg("P_ELMAT:"); Pelmat.Print();
            for (int i = 0; i < dim*dof; i++)
            {
               for (int j = 0; j < dim*dof; j++)
               {
                  if (fabs(elmat(i,j)-Pelmat(i,j)) > EPS)
                  {
                     dbg("\033[31m%.15e", elmat(i,j));
                     dbg("\033[31m%.15e", Pelmat(i,j));
                  }
                  MFEM_VERIFY(fabs(elmat(i,j)-Pelmat(i,j)) < EPS,"");
               }
            }

            double GZ_p[4];
            DenseMatrix GZ(GZ_p, dim, dim);
            for (int r = 0; r < dim; r++)
            {
               for (int c = 0; c < dim; c++)
               {
                  GZ(r,c) = 0.0;
                  Dim2Invariant1_dMdM(Jpt, r, c, dP);
                  dP *= 0.5 * weight_detJtr;
                  for (int rr = 0; rr < dim; rr++)
                  {
                     for (int cc = 0; cc < dim; cc++)
                     {
                        GZ(r,c) += dP(rr,cc) * GR(rr,cc);
                     } // cc
                  } // rr
               } // c
            } // r

            GZ *= flip ? -1.0 : 1.0;

            double A_p[4];
            DenseMatrix A(A_p, dim, dim);
            MultABt(Jrt, GZ, A);

            Cx0[qy][qx] = A(0,0);
            Cy0[qy][qx] = A(0,1);
            Cx1[qy][qx] = A(1,0);
            Cy1[qy][qx] = A(1,1);
         } // qx
      } // qy

      MFEM_SYNC_THREAD;
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bt[d][q] = b(q,d);
               Gt[d][q] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double cu[2] = {0};
            double cv[2] = {0};
            for (int qx = 0; qx < Q1D; ++qx)
            {
               cu[0] += Gt[dx][qx] * Cx0[qy][qx];
               cv[0] += Bt[dx][qx] * Cx1[qy][qx];
               cu[1] += Gt[dx][qx] * Cy0[qy][qx];
               cv[1] += Bt[dx][qx] * Cy1[qy][qx];
            }
            CxB[dx][qy] = cu[0];
            CxG[dx][qy] = cv[0];
            CyB[dx][qy] = cu[1];
            CyG[dx][qy] = cv[1];
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            double cu[2] = {0};
            double cv[2] = {0};
            for (int qy = 0; qy < Q1D; ++qy)
            {
               cu[0] += CxB[dx][qy] * Bt[dy][qy];
               cv[0] += CxG[dx][qy] * Gt[dy][qy];
               cu[1] += CyB[dx][qy] * Bt[dy][qy];
               cv[1] += CyG[dx][qy] * Gt[dy][qy];
            }
            C(dx,dy,0,e) += cu[0] + cv[0];
            C(dx,dy,1,e) += cu[1] + cv[1];
         }
      }
   });
}

// *****************************************************************************
void TMOP_Integrator::AddMultGradPA(const Vector &Xe, const Vector &Re,
                                    Vector &Ce) const
{
   //dbg("Xe: %d, Re:%d, Ce:%d", Xe.Size(), Re.Size(), Ce.Size());
   //dbg("Xe: %.15e, Re: %.15e", Xe*Xe, Re*Re);
   MFEM_VERIFY(IntRule,"");
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B1d = maps->B;
   const Array<double> &G1d = maps->G;
   const int id = (D1D << 4 ) | Q1D;

   {
      // Jtr setup:
      //  - TargetConstructor::target_type == IDEAL_SHAPE_UNIT_SIZE
      //  - Jtr(i) == Wideal
      // Get Wideal into Jtr
#if 1
      const FiniteElement *fe = fes->GetFE(0);
      const Geometry::Type geom_type = fe->GetGeomType();
      const DenseMatrix Jtr = Geometries.GetGeomToPerfGeomJac(geom_type);
      MFEM_VERIFY(Jtr.Det() == 1.0 ,"");
      {
         MFEM_VERIFY(Jtr(0,0)==1.0 && Jtr(1,1)==1.0 &&
                     Jtr(1,0)==0.0 && Jtr(0,1)==0.0,"");
      }
#elif 0
      DenseMatrix Jtr(dim);
      Jtr(0,0) = 1.0;
      Jtr(0,1) = 0.0;
      Jtr(1,0) = 0.0;
      Jtr(1,1) = 1.0;
#elif 0
      DenseMatrix Jtr(dim);
      Jtr(0,0) = 2.0;
      Jtr(0,1) = 0.0;
      Jtr(1,0) = 0.0;
      Jtr(1,1) = -1.0;
#else
      DenseMatrix Jtr(dim);
      Jtr(0,0) = 2.0;
      Jtr(0,1) = +1.123;
      Jtr(1,0) = -1.456;
      Jtr(1,1) = 1.0;
#endif
      //dbg("Jtr:"); Jtr.Print();

      /*
         Array<int> vdofs;
         DenseTensor Jtr(dim, dim, ir->GetNPoints());
         for (int i = 0; i < fes->GetNE(); i++)
         {
            const FiniteElement *el = fes->GetFE(i);
            fes->GetElementVDofs(i, vdofs);
            T = fes->GetElementTransformation(i);
            px.GetSubVector(vdofs, el_x);
            targetC->ComputeElementTargets(T.ElementNo, el, *ir, elfun, Jtr);
        }*/
      const auto J = Reshape(Jtr.Read(), dim, dim);
      auto G = Reshape(D.Write(), Q1D, Q1D, dim, dim, ne);
      MFEM_FORALL_2D(e, ne, Q1D, Q1D, 1,
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               G(qx,qy,0,0,e) = J(0,0);
               G(qx,qy,0,1,e) = J(0,1);
               G(qx,qy,1,0,e) = J(1,0);
               G(qx,qy,1,1,e) = J(1,1);
            }
         }
      });
   }

   switch (id)
   {
      case 0x21: return AddMultGradPA_Kernel_2D<2,1,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);/*
      case 0x22: return AddMultGradPA_Kernel_2D<2,2,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x23: return AddMultGradPA_Kernel_2D<2,3,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x24: return AddMultGradPA_Kernel_2D<2,4,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x25: return AddMultGradPA_Kernel_2D<2,5,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);

      case 0x31: return AddMultGradPA_Kernel_2D<3,1,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x32: return AddMultGradPA_Kernel_2D<3,2,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x33: return AddMultGradPA_Kernel_2D<3,3,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x34: return AddMultGradPA_Kernel_2D<3,4,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x35: return AddMultGradPA_Kernel_2D<3,5,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);

      case 0x41: return AddMultGradPA_Kernel_2D<4,1,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x42: return AddMultGradPA_Kernel_2D<4,2,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x43: return AddMultGradPA_Kernel_2D<4,3,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x44: return AddMultGradPA_Kernel_2D<4,4,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x45: return AddMultGradPA_Kernel_2D<4,5,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);

      case 0x51: return AddMultGradPA_Kernel_2D<5,1,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x52: return AddMultGradPA_Kernel_2D<5,2,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x53: return AddMultGradPA_Kernel_2D<5,3,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x54: return AddMultGradPA_Kernel_2D<5,4,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);
      case 0x55: return AddMultGradPA_Kernel_2D<5,5,1>(Xe,ne,W,B1d,G1d,D,Re,Ce);*/
      default:  break;
   }
   dbg("kernel id: %x", id);
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem
