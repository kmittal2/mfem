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
                                                const Vector &x)
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
void TMOP_Integrator::AssemblePA(const FiniteElementSpace &fespace)
{
   dbg("");
   fes = &fespace;

   MFEM_ASSERT(fes->GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");

   Mesh *mesh = fes->GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2, "");

   MFEM_VERIFY(IntRule,"");
   const IntegrationRule &ir = *IntRule;

   ne = fes->GetMesh()->GetNE();
   nq = IntRule->GetNPoints();
   maps = &fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   const int flags = GeometricFactors::COORDINATES|GeometricFactors::JACOBIANS;
   geom = mesh->GetGeometricFactors(ir, flags);

   const int D1D = maps->ndof;
   const FiniteElement *fe = fes->GetFE(0);
   const int dof = fe->GetDof();
   MFEM_VERIFY(D1D*D1D == dof,"");

   const int Q1D = maps->nqpt;
   MFEM_VERIFY(Q1D*Q1D == nq,"");

   //DShT.SetSize(dof, dim, ne); // gradients of reference shape functions
   //DST.SetSize(dof, dim, ne);  // gradients of target shape functions, DS = DSh Jrt
   //JrtT.SetSize(dim, dim, ne); // inverse of the ref->tgt Jacobian, Jrt = Jtr^{-1}.
   //JptT.SetSize(dim, dim, ne); // the tgt->phy T Jacobian, Jpt = Jpr Jrt.
   //PT.SetSize(dim, dim, ne);   // represents dW_d(Jtp) (dim x dim).

   // Setup for TargetConstructor::target_type == IDEAL_SHAPE_UNIT_SIZE
   // Jtr(i) = Wideal
   /*const DenseMatrix &Wideal =
      Geometries.GetGeomToPerfGeomJac(fe->GetGeomType());*/
   //dbg("Wideal:"); Wideal.Print();
   //const DenseMatrix &Jtr = Wideal;
   //const double JTR[4] = {Wideal(0,0), Wideal(0,1), Wideal(1,0), Wideal(1,1)};
   // same for all QPoints

   D.SetSize(dim * dim * nq * ne, Device::GetDeviceMemoryType());

   //const auto W = ir.GetWeights().Read();
   const auto J = Reshape(geom->J.Read(), nq, dim, dim, ne);
   //const auto Jtr = Reshape(JTR, dim, dim, ne);
   //auto Jrt = Reshape(JrtT.Write(), dim, dim, ne);
   auto G = Reshape(D.Write(), Q1D, Q1D, dim, dim, ne);
   MFEM_FORALL_2D(e, ne, Q1D, Q1D, 1,
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const int q = qx + qy * Q1D;
            //dbg("W:%.15e",W[q]);
            //const DenseMatrix &Jtr_i = Jtr;//(e);
            // metric->SetTargetJacobian(Jtr_i); => TMOP_Metric_002
            //kernels::CalcInverse<DIM>(Jtr.GetData(), &Jrt(0,0,e));
            const double J00 = J(q,0,0,e);
            const double J01 = J(q,0,1,e);
            const double J10 = J(q,1,0,e);
            const double J11 = J(q,1,1,e);
            //const double detJ = (J00*J11)-(J01*J10);
            //G(qx,qy,0,0,e) =  detJ * J11;
            //G(qx,qy,0,1,e) = -detJ * J01;
            //G(qx,qy,1,0,e) = -detJ * J10;
            //G(qx,qy,1,1,e) =  detJ * J00;
            G(qx,qy,0,0,e) = J00;
            G(qx,qy,0,1,e) = J01;
            G(qx,qy,1,0,e) = J10;
            G(qx,qy,1,1,e) = J11;
         }
      }
   });

}
// TMOP_Metric_002::EvalW: 0.5 * Get_I1b(Jpt) - 1.0 => weight
// Get_I1b == Get_I1()/Get_I2b();

// TMOP_Metric_002::EvalP: 0.5 * Get_dI1b(Jpt) => DenseMatrix P

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
   dbg("");
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
   dbg("D1D:%d, Q1D:%d, nq:%d", D1D, Q1D, Q1D*Q1D);
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

            // G = PMatI{^T}.DSh
            const double Gx0 = QQx0[qy][qx];
            const double Gx1 = QQx1[qy][qx];
            const double Gy0 = QQy0[qy][qx];
            const double Gy1 = QQy1[qy][qx];
            {
               const double detG = Gx0*Gy1 - Gx1*Gy0;
               dbg("");
               dbg("\033[0mdetG: %.15e",detG);
               dbg("G: %.15e %.15e",Gx0,Gx1);
               dbg("G: %.15e %.15e",Gy0,Gy1);
            }

            //  Jtr = targetC->ComputeElementTargets
            const double Jtrx0 = D(qx,qy,0,0,e);
            const double Jtrx1 = D(qx,qy,0,1,e);
            const double Jtry0 = D(qx,qy,1,0,e);
            const double Jtry1 = D(qx,qy,1,1,e);
            const double detJtr = Jtrx0*Jtry1 - Jtrx1*Jtry0;
            const double w = weight * detJtr;

            // Jrt = Jtr^{-1}
            double Jrt0x =  Jtry1 / detJtr;
            double Jrt0y = -Jtrx1 / detJtr;
            double Jrt1x = -Jtry0 / detJtr;
            double Jrt1y =  Jtrx0 / detJtr;
            {
               const double detJrt = (Jrt0x*Jrt1y)-(Jrt0y*Jrt1x);
               dbg("\033[0mdetJrt: %.15e", detJrt);
               dbg("Jrt: %.15e %.15e",Jrt0x,Jrt0y);
               dbg("Jrt: %.15e %.15e",Jrt1x,Jrt1y);
            }

            // Jpt = PMatI{^T}.DS = (PMatI{^T}.DSh).Jrt = G.Jrt
            //             |Jrt0x Jrt0y|
            //             |Jrt1x Jrt1y|
            //   |Gx0 Gx1| |Jptxx Jptxy|
            //   |Gy0 Gy1| |Jptyx Jptyy|
            double Jptxx = ((Gx0 * Jrt0x) + (Gx1 * Jrt1x));
            double Jptxy = ((Gx0 * Jrt0y) + (Gx1 * Jrt1y));
            double Jptyx = ((Gy0 * Jrt0x) + (Gy1 * Jrt1x));
            double Jptyy = ((Gy0 * Jrt0y) + (Gy1 * Jrt1y));
            {
               const double detJpt = Jptxx*Jptyy - Jptxy*Jptyx;
               dbg("\033[0mdetJpt: %.15e",detJpt);
               dbg("Jpt: %.15e %.15e",Jptxx,Jptxy);
               dbg("Jpt: %.15e %.15e",Jptyx,Jptyy);
            }
            double Pxx = w * Jptxx;
            double Pxy = w * Jptxy;
            double Pyx = w * Jptyx;
            double Pyy = w * Jptyy;
            {
               const double detP = Pxx*Pyy - Pxy*Pyx;
               dbg("\033[0mdetP %.15e",detP);
               dbg("P: %.15e %.15e",Pxx,Pxy);
               dbg("P: %.15e %.15e",Pyx,Pyy);
            }

            // PMatO +=  DS . P^t += DSh . (Jrt . (P==Jpt)^t)
            // Jrt . Jpt^t:
            // |Pxx Pxy|^{T}  => |Pxx Pyx|
            // |Pyx Pyy|         |Pxy Pyy|
            //     |Jrt0x Jrt0y|  A0x A0y
            //     |Jrt1x Jrt1y|  A1x A1y
            const double A0x = Jrt0x*Pxx + Jrt0y*Pxy;
            const double A0y = Jrt0x*Pyx + Jrt0y*Pyy;
            const double A1x = Jrt1x*Pxx + Jrt1y*Pxy;
            const double A1y = Jrt1x*Pyx + Jrt1y*Pyy;
            QQx0[qy][qx] = A0x;
            QQy0[qy][qx] = A0y;
            QQx1[qy][qx] = A1x;
            QQy1[qy][qx] = A1y;
            {
               dbg("\033[0mdetA: %.15e", A0x*A1y - A1x*A0y);
               dbg("A: %.15e %.15e",A0x,A0y);
               dbg("A: %.15e %.15e",A1x,A1y);
            }
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
   dbg("");
   MFEM_VERIFY(IntRule,"");
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B = maps->B;
   const Array<double> &G = maps->G;
   const int id = (D1D << 4 ) | Q1D;

   switch (id)
   {
      case 0x21: return AddMultPA_Kernel_2D<2,1,1>(ne,W,B,G,D,X,Y);
      case 0x23: return AddMultPA_Kernel_2D<2,3,1>(ne,W,B,G,D,X,Y);

      case 0x31: return AddMultPA_Kernel_2D<3,1,1>(ne,W,B,G,D,X,Y);
      case 0x32: return AddMultPA_Kernel_2D<3,2,1>(ne,W,B,G,D,X,Y);
      case 0x33: return AddMultPA_Kernel_2D<3,3,1>(ne,W,B,G,D,X,Y);

      case 0x41: return AddMultPA_Kernel_2D<4,1,1>(ne,W,B,G,D,X,Y);
      case 0x42: return AddMultPA_Kernel_2D<4,2,1>(ne,W,B,G,D,X,Y);
      case 0x43: return AddMultPA_Kernel_2D<4,3,1>(ne,W,B,G,D,X,Y);
      case 0x44: return AddMultPA_Kernel_2D<4,4,1>(ne,W,B,G,D,X,Y);

      case 0x52: return AddMultPA_Kernel_2D<5,2,1>(ne,W,B,G,D,X,Y);
      case 0x55: return AddMultPA_Kernel_2D<5,5,1>(ne,W,B,G,D,X,Y);
      case 0x57: return AddMultPA_Kernel_2D<5,7,1>(ne,W,B,G,D,X,Y);
      default:  break;
   }
   dbg("kernel id: %x", id);
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem
