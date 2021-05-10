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

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "ceed/convection.hpp"

#include "../fem/kernels.hpp"

namespace mfem
{

// PA Convection Integrator

// PA Convection Assemble 2D kernel
static void PAConvectionSetup2D(const int Q1D,
                                const int ne,
                                const Array<double> &w,
                                const Vector &j,
                                const Vector &vel,
                                const double alpha,
                                Vector &op)
{
   const int NE = ne;
   const int NQ = Q1D*Q1D;
   auto W = w.Read();

   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   const bool const_v = vel.Size() == 2;
   auto V =
      const_v ? Reshape(vel.Read(), 2,1,1) : Reshape(vel.Read(), 2,NQ,NE);
   auto y = Reshape(op.Write(), NQ, 2, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double w = alpha * W[q];
         const double v0 = const_v ? V(0,0,0) : V(0,q,e);
         const double v1 = const_v ? V(1,0,0) : V(1,q,e);
         const double wx = w * v0;
         const double wy = w * v1;
         // y = alpha * W * det(J) * J^{-1} . v = adj(J) . { wx, wy }
         y(q,0,e) =  wx * J22 - wy * J12; // 1
         y(q,1,e) = -wx * J21 + wy * J11; // 2
      }
   });
}

// PA Convection Assemble 3D kernel
static void PAConvectionSetup3D(const int Q1D,
                                const int NE,
                                const Array<double> &w,
                                const Vector &j,
                                const Vector &vel,
                                const double alpha,
                                Vector &op)
{
   const auto W = Reshape(w.Read(), Q1D,Q1D,Q1D);
   const auto J = Reshape(j.Read(), Q1D,Q1D,Q1D,3,3,NE);
   const bool const_v = vel.Size() == 3;
   const auto V = const_v ?
                  Reshape(vel.Read(), 3,1,1,1,1) :
                  Reshape(vel.Read(), 3,Q1D,Q1D,Q1D,NE);
   auto y = Reshape(op.Write(), Q1D,Q1D,Q1D,3,NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               const double J11 = J(qx,qy,qz,0,0,e);
               const double J12 = J(qx,qy,qz,0,1,e);
               const double J13 = J(qx,qy,qz,0,2,e);
               const double J21 = J(qx,qy,qz,1,0,e);
               const double J22 = J(qx,qy,qz,1,1,e);
               const double J23 = J(qx,qy,qz,1,2,e);
               const double J31 = J(qx,qy,qz,2,0,e);
               const double J32 = J(qx,qy,qz,2,1,e);
               const double J33 = J(qx,qy,qz,2,2,e);
               const double w = alpha * W(qx,qy,qz);
               const double v0 = const_v ? V(0,0,0,0,0) : V(0,qx,qy,qz,e);
               const double v1 = const_v ? V(1,0,0,0,0) : V(1,qx,qy,qz,e);
               const double v2 = const_v ? V(2,0,0,0,0) : V(2,qx,qy,qz,e);
               const double wx = w * v0;
               const double wy = w * v1;
               const double wz = w * v2;
               // A = adj(J)
               const double A11 = (J22 * J33) - (J23 * J32);
               const double A12 = (J32 * J13) - (J12 * J33);
               const double A13 = (J12 * J23) - (J22 * J13);
               const double A21 = (J31 * J23) - (J21 * J33);
               const double A22 = (J11 * J33) - (J13 * J31);
               const double A23 = (J21 * J13) - (J11 * J23);
               const double A31 = (J21 * J32) - (J31 * J22);
               const double A32 = (J31 * J12) - (J11 * J32);
               const double A33 = (J11 * J22) - (J12 * J21);
               // y = alpha * W * det(J) * J^{-1} . v = adj(J) . { wx, wy, wz }
               y(qx,qy,qz,0,e) = wx * A11 + wy * A12 + wz * A13;
               y(qx,qy,qz,1,e) = wx * A21 + wy * A22 + wz * A23;
               y(qx,qy,qz,2,e) = wx * A31 + wy * A32 + wz * A33;
            }
         }
      }
   });
}

static void PAConvectionSetup(const int dim,
                              const int Q1D,
                              const int NE,
                              const Array<double> &W,
                              const Vector &J,
                              const Vector &coeff,
                              const double alpha,
                              Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PAConvectionSetup"); }
   if (dim == 2)
   {
      PAConvectionSetup2D(Q1D, NE, W, J, coeff, alpha, op);
   }
   if (dim == 3)
   {
      PAConvectionSetup3D(Q1D, NE, W, J, coeff, alpha, op);
   }
}

// PA Convection Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0> static
void PAConvectionApply2D(const int ne,
                         const Array<double> &b,
                         const Array<double> &g,
                         const Array<double> &bt,
                         const Array<double> &gt,
                         const Vector &_op,
                         const Vector &_x,
                         Vector &_y,
                         const int d1d = 0,
                         const int q1d = 0)
{
   const int NE = ne;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 2, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double u[max_D1D][max_D1D];
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            u[dy][dx] = x(dx,dy,e);
         }
      }
      double Bu[max_D1D][max_Q1D];
      double Gu[max_D1D][max_Q1D];
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            Bu[dy][qx] = 0.0;
            Gu[dy][qx] = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double bx = B(qx,dx);
               const double gx = G(qx,dx);
               const double x = u[dy][dx];
               Bu[dy][qx] += bx * x;
               Gu[dy][qx] += gx * x;
            }
         }
      }
      double GBu[max_Q1D][max_Q1D];
      double BGu[max_Q1D][max_Q1D];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            GBu[qy][qx] = 0.0;
            BGu[qy][qx] = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double bx = B(qy,dy);
               const double gx = G(qy,dy);
               GBu[qy][qx] += gx * Bu[dy][qx];
               BGu[qy][qx] += bx * Gu[dy][qx];
            }
         }
      }
      // Calculate Dxy, xDy in plane
      double DGu[max_Q1D][max_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double O1 = op(qx,qy,0,e);
            const double O2 = op(qx,qy,1,e);

            const double gradX = BGu[qy][qx];
            const double gradY = GBu[qy][qx];

            DGu[qy][qx] = (O1 * gradX) + (O2 * gradY);
         }
      }
      double BDGu[max_D1D][max_Q1D];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            BDGu[dy][qx] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double w = Bt(dy,qy);
               BDGu[dy][qx] += w * DGu[qy][qx];
            }
         }
      }
      for (int dx = 0; dx < D1D; ++dx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            double BBDGu = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double w = Bt(dx,qx);
               BBDGu += w * BDGu[dy][qx];
            }
            y(dx,dy,e) += BBDGu;
         }
      }
   });
}

// Optimized PA Convection Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0> static
void SmemPAConvectionApply2D(const int ne,
                             const Array<double> &b,
                             const Array<double> &g,
                             const Array<double> &bt,
                             const Array<double> &gt,
                             const Vector &_op,
                             const Vector &_x,
                             Vector &_y,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int NE = ne;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 2, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      // constexpr int MDQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;
      MFEM_SHARED double u[NBZ][max_D1D][max_D1D];
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            // e is really equal to e+tidz
            u[tidz][dy][dx] = x(dx,dy,e);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_SHARED double Bu[NBZ][max_D1D][max_Q1D];
      MFEM_SHARED double Gu[NBZ][max_D1D][max_Q1D];
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            Bu[tidz][dy][qx] = 0.0;
            Gu[tidz][dy][qx] = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double bx = B(qx,dx);
               const double gx = G(qx,dx);
               const double x = u[tidz][dy][dx];
               Bu[tidz][dy][qx] += bx * x;
               Gu[tidz][dy][qx] += gx * x;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_SHARED double GBu[NBZ][max_Q1D][max_Q1D];
      MFEM_SHARED double BGu[NBZ][max_Q1D][max_Q1D];
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            GBu[tidz][qy][qx] = 0.0;
            BGu[tidz][qy][qx] = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double bx = B(qy,dy);
               const double gx = G(qy,dy);
               GBu[tidz][qy][qx] += gx * Bu[tidz][dy][qx];
               BGu[tidz][qy][qx] += bx * Gu[tidz][dy][qx];
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Calculate Dxy, xDy in plane
      MFEM_SHARED double DGu[NBZ][max_Q1D][max_Q1D];
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double O1 = op(qx,qy,0,e);
            const double O2 = op(qx,qy,1,e);

            const double gradX = BGu[tidz][qy][qx];
            const double gradY = GBu[tidz][qy][qx];

            DGu[tidz][qy][qx] = (O1 * gradX) + (O2 * gradY);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_SHARED double BDGu[NBZ][max_D1D][max_Q1D];
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            BDGu[tidz][dy][qx] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double w = Bt(dy,qy);
               BDGu[tidz][dy][qx] += w * DGu[tidz][qy][qx];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            double BBDGu = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double w = Bt(dx,qx);
               BBDGu += w * BDGu[tidz][dy][qx];
            }
            y(dx,dy,e) += BBDGu;
         }
      }
   });
}

// PA Convection Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0> static
void PAConvectionApply3D(const int ne,
                         const Array<double> &b,
                         const Array<double> &g,
                         const Array<double> &bt,
                         const Array<double> &gt,
                         const Vector &_op,
                         const Vector &_x,
                         Vector &_y,
                         const int d1d = 0,
                         const int q1d = 0)
{
   const int NE = ne;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 3, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double u[max_D1D][max_D1D][max_D1D];
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               u[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
      }
      double Bu[max_D1D][max_D1D][max_Q1D];
      double Gu[max_D1D][max_D1D][max_Q1D];
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Bu[dz][dy][qx] = 0.0;
               Gu[dz][dy][qx] = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double bx = B(qx,dx);
                  const double gx = G(qx,dx);
                  const double x = u[dz][dy][dx];
                  Bu[dz][dy][qx] += bx * x;
                  Gu[dz][dy][qx] += gx * x;
               }
            }
         }
      }
      double BBu[max_D1D][max_Q1D][max_Q1D];
      double GBu[max_D1D][max_Q1D][max_Q1D];
      double BGu[max_D1D][max_Q1D][max_Q1D];
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               BBu[dz][qy][qx] = 0.0;
               GBu[dz][qy][qx] = 0.0;
               BGu[dz][qy][qx] = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  const double bx = B(qy,dy);
                  const double gx = G(qy,dy);
                  BBu[dz][qy][qx] += bx * Bu[dz][dy][qx];
                  GBu[dz][qy][qx] += gx * Bu[dz][dy][qx];
                  BGu[dz][qy][qx] += bx * Gu[dz][dy][qx];
               }
            }
         }
      }
      double GBBu[max_Q1D][max_Q1D][max_Q1D];
      double BGBu[max_Q1D][max_Q1D][max_Q1D];
      double BBGu[max_Q1D][max_Q1D][max_Q1D];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qz = 0; qz < Q1D; ++qz)
            {
               GBBu[qz][qy][qx] = 0.0;
               BGBu[qz][qy][qx] = 0.0;
               BBGu[qz][qy][qx] = 0.0;
               for (int dz = 0; dz < D1D; ++dz)
               {
                  const double bx = B(qz,dz);
                  const double gx = G(qz,dz);
                  GBBu[qz][qy][qx] += gx * BBu[dz][qy][qx];
                  BGBu[qz][qy][qx] += bx * GBu[dz][qy][qx];
                  BBGu[qz][qy][qx] += bx * BGu[dz][qy][qx];
               }
            }
         }
      }
      // Calculate Dxy, xDy in plane
      double DGu[max_Q1D][max_Q1D][max_Q1D];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double O1 = op(qx,qy,qz,0,e);
               const double O2 = op(qx,qy,qz,1,e);
               const double O3 = op(qx,qy,qz,2,e);

               const double gradX = BBGu[qz][qy][qx];
               const double gradY = BGBu[qz][qy][qx];
               const double gradZ = GBBu[qz][qy][qx];

               DGu[qz][qy][qx] = (O1 * gradX) + (O2 * gradY) + (O3 * gradZ);
            }
         }
      }
      double BDGu[max_D1D][max_Q1D][max_Q1D];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int dz = 0; dz < D1D; ++dz)
            {
               BDGu[dz][qy][qx] = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  const double w = Bt(dz,qz);
                  BDGu[dz][qy][qx] += w * DGu[qz][qy][qx];
               }
            }
         }
      }
      double BBDGu[max_D1D][max_D1D][max_Q1D];
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               BBDGu[dz][dy][qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double w = Bt(dy,qy);
                  BBDGu[dz][dy][qx] += w * BDGu[dz][qy][qx];
               }
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               double BBBDGu = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double w = Bt(dx,qx);
                  BBBDGu += w * BBDGu[dz][dy][qx];
               }
               y(dx,dy,dz,e) += BBBDGu;
            }
         }
      }
   });
}

// Optimized PA Convection Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0> static
void SmemPAConvectionApply3D(const int ne,
                             const Array<double> &b,
                             const Array<double> &g,
                             const Array<double> &bt,
                             const Array<double> &gt,
                             const Vector &_op,
                             const Vector &_x,
                             Vector &_y,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int NE = ne;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 3, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;
      MFEM_SHARED double sm0[max_DQ*max_DQ*max_DQ];
      MFEM_SHARED double sm1[max_DQ*max_DQ*max_DQ];
      MFEM_SHARED double sm2[max_DQ*max_DQ*max_DQ];
      MFEM_SHARED double sm3[max_DQ*max_DQ*max_DQ];
      MFEM_SHARED double sm4[max_DQ*max_DQ*max_DQ];
      MFEM_SHARED double sm5[max_DQ*max_DQ*max_DQ];

      double (*u)[max_D1D][max_D1D] = (double (*)[max_D1D][max_D1D]) sm0;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               u[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      double (*Bu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm1;
      double (*Gu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm2;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double Bu_ = 0.0;
               double Gu_ = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double bx = B(qx,dx);
                  const double gx = G(qx,dx);
                  const double x = u[dz][dy][dx];
                  Bu_ += bx * x;
                  Gu_ += gx * x;
               }
               Bu[dz][dy][qx] = Bu_;
               Gu[dz][dy][qx] = Gu_;
            }
         }
      }
      MFEM_SYNC_THREAD;
      double (*BBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;
      double (*GBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4;
      double (*BGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm5;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               double BBu_ = 0.0;
               double GBu_ = 0.0;
               double BGu_ = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  const double bx = B(qy,dy);
                  const double gx = G(qy,dy);
                  BBu_ += bx * Bu[dz][dy][qx];
                  GBu_ += gx * Bu[dz][dy][qx];
                  BGu_ += bx * Gu[dz][dy][qx];
               }
               BBu[dz][qy][qx] = BBu_;
               GBu[dz][qy][qx] = GBu_;
               BGu[dz][qy][qx] = BGu_;
            }
         }
      }
      MFEM_SYNC_THREAD;
      double (*GBBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm0;
      double (*BGBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm1;
      double (*BBGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm2;
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               double GBBu_ = 0.0;
               double BGBu_ = 0.0;
               double BBGu_ = 0.0;
               for (int dz = 0; dz < D1D; ++dz)
               {
                  const double bx = B(qz,dz);
                  const double gx = G(qz,dz);
                  GBBu_ += gx * BBu[dz][qy][qx];
                  BGBu_ += bx * GBu[dz][qy][qx];
                  BBGu_ += bx * BGu[dz][qy][qx];
               }
               GBBu[qz][qy][qx] = GBBu_;
               BGBu[qz][qy][qx] = BGBu_;
               BBGu[qz][qy][qx] = BBGu_;
            }
         }
      }
      MFEM_SYNC_THREAD;
      double (*DGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double O1 = op(qx,qy,qz,0,e);
               const double O2 = op(qx,qy,qz,1,e);
               const double O3 = op(qx,qy,qz,2,e);

               const double gradX = BBGu[qz][qy][qx];
               const double gradY = BGBu[qz][qy][qx];
               const double gradZ = GBBu[qz][qy][qx];

               DGu[qz][qy][qx] = (O1 * gradX) + (O2 * gradY) + (O3 * gradZ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      double (*BDGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4;
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dz,z,D1D)
            {
               double BDGu_ = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  const double w = Bt(dz,qz);
                  BDGu_ += w * DGu[qz][qy][qx];
               }
               BDGu[dz][qy][qx] = BDGu_;
            }
         }
      }
      MFEM_SYNC_THREAD;
      double (*BBDGu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm5;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               double BBDGu_ = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double w = Bt(dy,qy);
                  BBDGu_ += w * BDGu[dz][qy][qx];
               }
               BBDGu[dz][dy][qx] = BBDGu_;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double BBBDGu = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double w = Bt(dx,qx);
                  BBBDGu += w * BBDGu[dz][dy][qx];
               }
               y(dx,dy,dz,e) = BBBDGu;
            }
         }
      }
   });
}

template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0, int T_MAX = 0>
static void QEvalVGF2D(const int NE,
                       const double *b_,
                       const double *x_,
                       double *y_,
                       const int vdim = 1,
                       const int d1d = 0,
                       const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto X = Reshape(x_, D1D, D1D, VDIM, NE);
   auto C = Reshape(y_, VDIM, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
   {
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;

      MFEM_SHARED double B[MQ1*MD1];
      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,b,B);

      MFEM_SHARED double DD[NBZ][MD1*MD1];
      MFEM_SHARED double DQ[NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[NBZ][MQ1*MQ1];

      for (int c = 0; c < VDIM; c++)
      {
         kernels::internal::LoadX<MD1,NBZ>(e,D1D,c,X,DD);
         kernels::internal::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,B,DD,DQ);
         kernels::internal::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,B,DQ,QQ);

         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               double G;
               mfem::kernels::internal::PullEval<MQ1,NBZ>(qx,qy,QQ,G);
               C(c,qx,qy,e) = G;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0, int T_MAX = 0>
static void QEvalVGF3D(const int NE,
                       const double *b_,
                       const double *x_,
                       double *y_,
                       const int vdim = 1,
                       const int d1d = 0,
                       const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto X = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto C = Reshape(y_, VDIM, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED double sB[MQ1*MD1];
      ConstDeviceMatrix B(sB, D1D,Q1D);
      mfem::kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,b,sB);

      MFEM_SHARED double sm0[MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[MDQ*MDQ*MDQ];
      DeviceCube DDD(sm0, MD1,MD1,MD1);
      DeviceCube DDQ(sm1, MD1,MD1,MQ1);
      DeviceCube DQQ(sm0, MD1,MQ1,MQ1);
      DeviceCube QQQ(sm1, MQ1,MQ1,MQ1);

      for (int c = 0; c < VDIM; c++)
      {
         kernels::internal::LoadX<MD1>(e,D1D,c,X,DDD);
         kernels::internal::EvalX<MD1,MQ1>(D1D,Q1D,B,DDD,DDQ);
         kernels::internal::EvalY<MD1,MQ1>(D1D,Q1D,B,DDQ,DQQ);
         kernels::internal::EvalZ<MD1,MQ1>(D1D,Q1D,B,DQQ,QQQ);

         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qz,z,Q1D)
               {
                  C(c,qx,qy,qz,e) = QQQ(qx,qy,qz);
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void ConvectionIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation &Trans = *fes.GetElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, Trans);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      ceedOp = new ceed::PAConvectionIntegrator(fes, *ir, Q, alpha);
      return;
   }
   const int dims = el.GetDim();
   const int symmDims = dims;
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   maps = &el.GetDofToQuad(*ir, mode);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());
   Vector vel;
   if (VectorConstantCoefficient *cQ =
          dynamic_cast<VectorConstantCoefficient*>(Q))
   {
      vel = cQ->GetVec();
   }
   else if (VectorGridFunctionCoefficient *vgfQ =
               dynamic_cast<VectorGridFunctionCoefficient*>(Q))
   {
      Vector xe;
      vel.SetSize(dim * nq * ne);

      const GridFunction *gf = vgfQ->GetGridFunction();
      const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
      const FiniteElementSpace &gf_fes = *gf->FESpace();

      const int vdim = gf_fes.GetVDim();
      const Operator *R = gf_fes.GetElementRestriction(ordering);
      const FiniteElement &el_gf = *gf_fes.GetFE(0);
      const DofToQuad *maps_gf = &el_gf.GetDofToQuad(*ir, mode);
      const int D1D = maps_gf->ndof;
      const int Q1D = maps_gf->nqpt;

      MFEM_VERIFY(R,"");
      MFEM_VERIFY(vdim == dim, "");
      MFEM_VERIFY(dim==2 || dim==3,"");

      xe.SetSize(R->Height(), Device::GetMemoryType());
      xe.UseDevice(true);
      R->Mult(*gf, xe);

      const auto B = maps_gf->B.Read();
      const auto x = xe.Read();
      auto y = vel.Write();

      const int id = (D1D << 4 ) | Q1D;
      if (dim == 2)
      {
         switch (id)
         {
            case 0x22: QEvalVGF2D<2,2,2>(ne,B,x,y); break;
            case 0x33: QEvalVGF2D<2,3,3>(ne,B,x,y); break;
            case 0x34: QEvalVGF2D<2,3,4>(ne,B,x,y); break;
            default:
            {
               constexpr int MAX_DQ = 8;
               MFEM_VERIFY(D1D <= MAX_DQ, "");
               MFEM_VERIFY(Q1D <= MAX_DQ, "");
               QEvalVGF2D<0,0,0,MAX_DQ>(ne,B,x,y,vdim,D1D,Q1D);
            }
         }
      }
      if (dim == 3)
      {
         switch (id)
         {
            case 0x23: QEvalVGF3D<3,2,3>(ne,B,x,y); break;
            case 0x34: QEvalVGF3D<3,3,4>(ne,B,x,y); break;
            case 0x35: QEvalVGF3D<3,3,5>(ne,B,x,y); break;
            case 0x46: QEvalVGF3D<3,4,6>(ne,B,x,y); break;
            case 0x48: QEvalVGF3D<3,4,8>(ne,B,x,y); break;
            default:
            {
               constexpr int MAX_DQ = 7;
               MFEM_VERIFY(D1D <= MAX_DQ, "D1D:"<<D1D<<", MAX_DQ:"<<MAX_DQ);
               MFEM_VERIFY(Q1D <= MAX_DQ, "Q1D:"<<Q1D<<", MAX_DQ:"<<MAX_DQ);
               QEvalVGF3D<0,0,0,MAX_DQ>(ne,B,x,y,vdim,D1D,Q1D);
            }
         }
      }
   }
   else if (VectorQuadratureFunctionCoefficient* cQ =
               dynamic_cast<VectorQuadratureFunctionCoefficient*>(Q))
   {
      const QuadratureFunction &qFun = cQ->GetQuadFunction();
      MFEM_VERIFY(qFun.Size() == dim * nq * ne,
                  "Incompatible QuadratureFunction dimension \n");

      MFEM_VERIFY(ir == &qFun.GetSpace()->GetElementIntRule(0),
                  "IntegrationRule used within integrator and in"
                  " QuadratureFunction appear to be different");

      qFun.Read();
      vel.MakeRef(const_cast<QuadratureFunction &>(qFun),0);
   }
   else
   {
      vel.SetSize(dim * nq * ne);
      auto C = Reshape(vel.HostWrite(), dim, nq, ne);
      DenseMatrix Q_ir;
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation& T = *fes.GetElementTransformation(e);
         Q->Eval(Q_ir, T, *ir);
         for (int q = 0; q < nq; ++q)
         {
            for (int i = 0; i < dim; ++i)
            {
               C(i,q,e) = Q_ir(i,q);
            }
         }
      }
   }
   PAConvectionSetup(dim, quad1D, ne, ir->GetWeights(), geom->J,
                     vel, alpha, pa_data);
}

static void PAConvectionApply(const int dim,
                              const int D1D,
                              const int Q1D,
                              const int NE,
                              const Array<double> &B,
                              const Array<double> &G,
                              const Array<double> &Bt,
                              const Array<double> &Gt,
                              const Vector &op,
                              const Vector &x,
                              Vector &y)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return SmemPAConvectionApply2D<2,2,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x33: return SmemPAConvectionApply2D<3,3,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x34: return SmemPAConvectionApply2D<3,4,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x44: return SmemPAConvectionApply2D<4,4,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x46: return SmemPAConvectionApply2D<4,6,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x55: return SmemPAConvectionApply2D<5,5,2>(NE,B,G,Bt,Gt,op,x,y);
         case 0x58: return SmemPAConvectionApply2D<5,8,2>(NE,B,G,Bt,Gt,op,x,y);
         case 0x66: return SmemPAConvectionApply2D<6,6,1>(NE,B,G,Bt,Gt,op,x,y);
         case 0x77: return SmemPAConvectionApply2D<7,7,1>(NE,B,G,Bt,Gt,op,x,y);
         case 0x88: return SmemPAConvectionApply2D<8,8,1>(NE,B,G,Bt,Gt,op,x,y);
         case 0x99: return SmemPAConvectionApply2D<9,9,1>(NE,B,G,Bt,Gt,op,x,y);
         default:   return PAConvectionApply2D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return SmemPAConvectionApply3D<2,3>(NE,B,G,Bt,Gt,op,x,y);
         case 0x24: return SmemPAConvectionApply3D<2,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x26: return SmemPAConvectionApply3D<2,6>(NE,B,G,Bt,Gt,op,x,y);
         case 0x34: return SmemPAConvectionApply3D<3,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x35: return SmemPAConvectionApply3D<3,5>(NE,B,G,Bt,Gt,op,x,y);
         case 0x45: return SmemPAConvectionApply3D<4,5>(NE,B,G,Bt,Gt,op,x,y);
         case 0x48: return SmemPAConvectionApply3D<4,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x56: return SmemPAConvectionApply3D<5,6>(NE,B,G,Bt,Gt,op,x,y);
         case 0x67: return SmemPAConvectionApply3D<6,7>(NE,B,G,Bt,Gt,op,x,y);
         case 0x78: return SmemPAConvectionApply3D<7,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x89: return SmemPAConvectionApply3D<8,9>(NE,B,G,Bt,Gt,op,x,y);
         default:   return PAConvectionApply3D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

// PA Convection Apply kernel
void ConvectionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      PAConvectionApply(dim, dofs1D, quad1D, ne,
                        maps->B, maps->G, maps->Bt, maps->Gt,
                        pa_data, x, y);
   }
}

void ConvectionIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else
   {
      MFEM_ABORT("AssembleDiagonalPA not yet implemented for"
                 " ConvectionIntegrator.");
   }
}

} // namespace mfem
