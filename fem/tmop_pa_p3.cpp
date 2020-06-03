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
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#define MFEM_DBG_COLOR 82
#include "../general/dbg.hpp"

namespace mfem
{

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0>
static void AddMultPA_Kernel_3D(const int NE,
                                const Array<double> &w_,
                                const Array<double> &b_,
                                const Array<double> &g_,
                                const Vector &d_,
                                const Vector &x_,
                                Vector &y_,
                                const int d1d = 0,
                                const int q1d = 0)
{
   constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto D = Reshape(d_.Read(), Q1D, Q1D, Q1D, VDIM, VDIM, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, D1D, VDIM, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;

      MFEM_SHARED double s_BG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1])(s_BG[0]);
      double (*G)[MD1] = (double (*)[MD1])(s_BG[1]);
      double (*Bt)[MQ1] = (double (*)[MQ1])(s_BG[0]);
      double (*Gt)[MQ1] = (double (*)[MQ1])(s_BG[1]);

      MFEM_SHARED double s_DDD[3][MD1*MD1*MD1];
      double (*Xx)[MD1][MD1] = (double (*)[MD1][MD1])(s_DDD[0]);
      double (*Xy)[MD1][MD1] = (double (*)[MD1][MD1])(s_DDD[1]);
      double (*Xz)[MD1][MD1] = (double (*)[MD1][MD1])(s_DDD[2]);

      MFEM_SHARED double s_DDQ[9][MD1*MD1*MQ1];
      double (*XxB)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[0]);
      double (*XxG)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[1]);
      double (*XyB)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[2]);
      double (*XyG)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[3]);
      double (*XzB)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[4]);
      double (*XzG)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[5]);
      double (*XxC)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[6]);
      double (*XyC)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[7]);
      double (*XzC)[MD1][MQ1] = (double (*)[MD1][MQ1])(s_DDQ[8]);

      MFEM_SHARED double s_DQQ[9][MD1*MQ1*MQ1];
      double (*XxBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[0]);
      double (*XxBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[1]);
      double (*XxGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[2]);
      double (*XyBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[3]);
      double (*XyBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[4]);
      double (*XyGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[5]);
      double (*XzBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[6]);
      double (*XzBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[7]);
      double (*XzGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_DQQ[8]);

      MFEM_SHARED double s_QQQ[9][MQ1*MQ1*MQ1];
      double (*XxBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+0);
      double (*XxBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+1);
      double (*XxGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+2);
      double (*XyBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+3);
      double (*XyBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+4);
      double (*XyGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+5);
      double (*XzBBG)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+6);
      double (*XzBGB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+7);
      double (*XzGBB)[MQ1][MQ1] = (double (*)[MQ1][MQ1])(s_QQQ+8);

      // Load X(x,y,z)
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               Xx[dz][dy][dx] = X(dx,dy,dz,0,e);
               Xy[dz][dy][dx] = X(dx,dy,dz,1,e);
               Xz[dz][dy][dx] = X(dx,dy,dz,2,e);
            }
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
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u[3] {};
               double v[3] {};
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double xx = Xx[dz][dy][dx];
                  const double xy = Xy[dz][dy][dx];
                  const double xz = Xz[dz][dy][dx];
                  const double Bx = B[qx][dx];
                  const double Gx = G[qx][dx];
                  u[0] += Bx * xx;
                  u[1] += Bx * xy;
                  u[2] += Bx * xz;

                  v[0] += Gx * xx;
                  v[1] += Gx * xy;
                  v[2] += Gx * xz;
               }
               XxB[dz][dy][qx] = u[0];
               XyB[dz][dy][qx] = u[1];
               XzB[dz][dy][qx] = u[2];

               XxG[dz][dy][qx] = v[0];
               XyG[dz][dy][qx] = v[1];
               XzG[dz][dy][qx] = v[2];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u[3] {};
               double v[3] {};
               double w[3] {};
               for (int dy = 0; dy < D1D; ++dy)
               {
                  const double By = B[qy][dy];
                  const double Gy = G[qy][dy];

                  u[0] += XxB[dz][dy][qx] * By;
                  u[1] += XyB[dz][dy][qx] * By;
                  u[2] += XzB[dz][dy][qx] * By;

                  v[0] += XxG[dz][dy][qx] * By;
                  v[1] += XyG[dz][dy][qx] * By;
                  v[2] += XzG[dz][dy][qx] * By;

                  w[0] += XxB[dz][dy][qx] * Gy;
                  w[1] += XyB[dz][dy][qx] * Gy;
                  w[2] += XzB[dz][dy][qx] * Gy;
               }
               XxBB[dz][qy][qx] = u[0];
               XyBB[dz][qy][qx] = u[1];
               XzBB[dz][qy][qx] = u[2];

               XxBG[dz][qy][qx] = v[0];
               XyBG[dz][qy][qx] = v[1];
               XzBG[dz][qy][qx] = v[2];

               XxGB[dz][qy][qx] = w[0];
               XyGB[dz][qy][qx] = w[1];
               XzGB[dz][qy][qx] = w[2];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u[3] {};
               double v[3] {};
               double w[3] {};
               for (int dz = 0; dz < D1D; ++dz)
               {
                  const double Bz = B[qz][dz];
                  const double Gz = G[qz][dz];

                  u[0] += XxBG[dz][qy][qx] * Bz;
                  u[1] += XyBG[dz][qy][qx] * Bz;
                  u[2] += XzBG[dz][qy][qx] * Bz;

                  v[0] += XxGB[dz][qy][qx] * Bz;
                  v[1] += XyGB[dz][qy][qx] * Bz;
                  v[2] += XzGB[dz][qy][qx] * Bz;

                  w[0] += XxBB[dz][qy][qx] * Gz;
                  w[1] += XyBB[dz][qy][qx] * Gz;
                  w[2] += XzBB[dz][qy][qx] * Gz;
               }
               XxBBG[qz][qy][qx] = u[0];
               XyBBG[qz][qy][qx] = u[1];
               XzBBG[qz][qy][qx] = u[2];

               XxBGB[qz][qy][qx] = v[0];
               XyBGB[qz][qy][qx] = v[1];
               XzBGB[qz][qy][qx] = v[2];

               XxGBB[qz][qy][qx] = w[0];
               XyGBB[qz][qy][qx] = w[1];
               XzGBB[qz][qy][qx] = w[2];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double weight = W(qx,qy,qz);
               const double Jtrx0 = D(qx,qy,qz,0,0,e);
               const double Jtrx1 = D(qx,qy,qz,0,1,e);
               const double Jtrx2 = D(qx,qy,qz,0,2,e);
               const double Jtry0 = D(qx,qy,qz,1,0,e);
               const double Jtry1 = D(qx,qy,qz,1,1,e);
               const double Jtry2 = D(qx,qy,qz,1,2,e);
               const double Jtrz0 = D(qx,qy,qz,2,0,e);
               const double Jtrz1 = D(qx,qy,qz,2,1,e);
               const double Jtrz2 = D(qx,qy,qz,2,2,e);
               const double Jtr[9] =
               {
                  Jtrx0, Jtry0, Jtrz0,
                  Jtrx1, Jtry1, Jtrz1,
                  Jtrx2, Jtry2, Jtrz2
               };
               const double detJtr = kernels::Det<3>(Jtr);
               const double weight_detJtr = weight * detJtr;

               // Jrt = Jtr^{-1}
               double Jrt[9];
               kernels::CalcInverse<3>(Jtr, Jrt);

               // Jpr = X^T.DSh
               const double JprxBBG = XxBBG[qz][qy][qx];
               const double JprxBGB = XxBGB[qz][qy][qx];
               const double JprxGBB = XxGBB[qz][qy][qx];
               const double JpryBBG = XyBBG[qz][qy][qx];
               const double JpryBGB = XyBGB[qz][qy][qx];
               const double JpryGBB = XyGBB[qz][qy][qx];
               const double JprzBBG = XzBBG[qz][qy][qx];
               const double JprzBGB = XzBGB[qz][qy][qx];
               const double JprzGBB = XzGBB[qz][qy][qx];
               const double Jpr[9] =
               {
                  JprxBBG, JpryBBG, JprzBBG,
                  JprxBGB, JpryBGB, JprzBGB,
                  JprxGBB, JpryGBB, JprzGBB
               };
               // J = Jpt = X^T.DS = (X^T.DSh).Jrt = Jpr.Jrt
               double J[9];
               kernels::Mult(3,3,3, Jpr, Jrt, J);
               // metric->EvalP(Jpt, P);
               // TMOP_Metric_302
               // mu_2 = I1b*I2b/9-1

               // P = (I1b/9)*dI2b + (I2b/9)*dI1b
               const double detJ = J[0] * (J[4]*J[8] - J[7]*J[5]) -
                                   J[1] * (J[3]*J[8] - J[5]*J[6]) +
                                   J[2] * (J[3]*J[7] - J[4]*J[6]);
               const double sign_detJ = ScalarOps<double>::sign(detJ);
               const double I3b = sign_detJ * detJ;
               const double I3b_p = sign_detJ*std::pow(I3b, -2./3.);
               const double B0 = J[0]*J[0] + J[3]*J[3] + J[6]*J[6];
               const double B1 = J[1]*J[1] + J[4]*J[4] + J[7]*J[7];
               const double B2 = J[2]*J[2] + J[5]*J[5] + J[8]*J[8];
               const double I1 = B0 + B1 + B2;
               const double I1b = I1 * I3b_p;
               const double alpha = I1b/9.0;
               const double B3 = J[0]*J[1] + J[3]*J[4] + J[6]*J[7];
               const double B4 = J[0]*J[2] + J[3]*J[5] + J[6]*J[8];
               const double B5 = J[1]*J[2] + J[4]*J[5] + J[7]*J[8];
               const double BF2 = B0*B0+B1*B1+B2*B2+2*(B3*B3+B4*B4+B5*B5);
               const double I2 = (I1*I1 - BF2)/2;
               const double I2b = I2 * I3b_p * I3b_p;
               const double beta = I2b/9.0;
               // dI1b = 2*I3b^{-2/3}*(J - (1/3)*I1/I3b*dI3b)
               double dI1b[9], dI3b[9];
               {
                  const double c1 = 2*I3b_p;
                  const double c2 = I1/(3*I3b);
                  dI3b[0] = sign_detJ*(J[4]*J[8] - J[5]*J[7]);
                  dI3b[1] = sign_detJ*(J[5]*J[6] - J[3]*J[8]);
                  dI3b[2] = sign_detJ*(J[3]*J[7] - J[4]*J[6]);
                  dI3b[3] = sign_detJ*(J[2]*J[7] - J[1]*J[8]);
                  dI3b[4] = sign_detJ*(J[0]*J[8] - J[2]*J[6]);
                  dI3b[5] = sign_detJ*(J[1]*J[6] - J[0]*J[7]);
                  dI3b[6] = sign_detJ*(J[1]*J[5] - J[2]*J[4]);
                  dI3b[7] = sign_detJ*(J[2]*J[3] - J[0]*J[5]);
                  dI3b[8] = sign_detJ*(J[0]*J[4] - J[1]*J[3]);
                  for (int i = 0; i < 9; i++)
                  {
                     dI1b[i] = c1*(J[i] - c2*dI3b[i]);
                  }
               }
               double dI2b[9], dI2[9];
               {
                  const double c1 = I3b_p*I3b_p;
                  const double c2 = (4*I2/I3b)/3;
                  const double C[6] =
                  { 2*(I1-B0), 2*(I1-B1), 2*(I1-B2), -2*B3, -2*B4, -2*B5 };
                  dI2[0] = C[0]*J[0] + C[3]*J[1] + C[4]*J[2];
                  dI2[1] = C[3]*J[0] + C[1]*J[1] + C[5]*J[2];
                  dI2[2] = C[4]*J[0] + C[5]*J[1] + C[2]*J[2];
                  dI2[3] = C[0]*J[3] + C[3]*J[4] + C[4]*J[5];
                  dI2[4] = C[3]*J[3] + C[1]*J[4] + C[5]*J[5];
                  dI2[5] = C[4]*J[3] + C[5]*J[4] + C[2]*J[5];
                  dI2[6] = C[0]*J[6] + C[3]*J[7] + C[4]*J[8];
                  dI2[7] = C[3]*J[6] + C[1]*J[7] + C[5]*J[8];
                  dI2[8] = C[4]*J[6] + C[5]*J[7] + C[2]*J[8];
                  for (int i = 0; i < 9; i++)
                  {
                     dI2b[i] = c1*(dI2[i] - c2*dI3b[i]);
                  }
               }
               // P = alpha * dI2b + beta * dI1b
               double P[9];
               for (int i = 0; i < 9; i++)
               {
                  P[i] = alpha * dI2b[i] + beta * dI1b[i];
                  P[i] *= weight_detJtr;
               }

               // Y +=  DS . P^t += DSh . (Jrt . (P==Jpt)^t)
               double A[9];
               kernels::MultABt(3,3,3, Jrt, P, A);
               XxBBG[qz][qy][qx] = A[0];
               XxBGB[qz][qy][qx] = A[1];
               XxGBB[qz][qy][qx] = A[2];

               XyBBG[qz][qy][qx] = A[3];
               XyBGB[qz][qy][qx] = A[4];
               XyGBB[qz][qy][qx] = A[5];

               XzBBG[qz][qy][qx] = A[6];
               XzBGB[qz][qy][qx] = A[7];
               XzGBB[qz][qy][qx] = A[8];
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
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double u[3] {};
               double v[3] {};
               double w[3] {};
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double Btx = Bt[dx][qx];
                  const double Gtx = Gt[dx][qx];

                  u[0] += XxBBG[qz][qy][qx] * Gtx;
                  v[0] += XxBGB[qz][qy][qx] * Btx;
                  w[0] += XxGBB[qz][qy][qx] * Btx;

                  u[1] += XyBBG[qz][qy][qx] * Gtx;
                  v[1] += XyBGB[qz][qy][qx] * Btx;
                  w[1] += XyGBB[qz][qy][qx] * Btx;

                  u[2] += XzBBG[qz][qy][qx] * Gtx;
                  v[2] += XzBGB[qz][qy][qx] * Btx;
                  w[2] += XzGBB[qz][qy][qx] * Btx;
               }
               XxBB[dx][qy][qz] = u[0];
               XxBG[dx][qy][qz] = v[0];
               XxGB[dx][qy][qz] = w[0];

               XyBB[dx][qy][qz] = u[1];
               XyBG[dx][qy][qz] = v[1];
               XyGB[dx][qy][qz] = w[1];

               XzBB[dx][qy][qz] = u[2];
               XzBG[dx][qy][qz] = v[2];
               XzGB[dx][qy][qz] = w[2];
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double u[3] {};
               double v[3] {};
               double w[3] {};
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double Bty = Bt[dy][qy];
                  const double Gty = Gt[dy][qy];

                  u[0] += XxBB[dx][qy][qz] * Bty;
                  v[0] += XxBG[dx][qy][qz] * Gty;
                  w[0] += XxGB[dx][qy][qz] * Bty;

                  u[1] += XyBB[dx][qy][qz] * Bty;
                  v[1] += XyBG[dx][qy][qz] * Gty;
                  w[1] += XyGB[dx][qy][qz] * Bty;

                  u[2] += XzBB[dx][qy][qz] * Bty;
                  v[2] += XzBG[dx][qy][qz] * Gty;
                  w[2] += XzGB[dx][qy][qz] * Bty;

               }
               XxB[dx][dy][qz] = u[0];
               XxC[dx][dy][qz] = v[0];
               XxG[dx][dy][qz] = w[0];

               XyB[dx][dy][qz] = u[1];
               XyC[dx][dy][qz] = v[1];
               XyG[dx][dy][qz] = w[1];

               XzB[dx][dy][qz] = u[2];
               XzC[dx][dy][qz] = v[2];
               XzG[dx][dy][qz] = w[2];
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
               double u[3] {};
               double v[3] {};
               double w[3] {};
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  const double Btz = Bt[dz][qz];
                  const double Gtz = Gt[dz][qz];

                  u[0] += XxB[dx][dy][qz] * Btz;
                  v[0] += XxC[dx][dy][qz] * Btz;
                  w[0] += XxG[dx][dy][qz] * Gtz;

                  u[1] += XyB[dx][dy][qz] * Btz;
                  v[1] += XyC[dx][dy][qz] * Btz;
                  w[1] += XyG[dx][dy][qz] * Gtz;

                  u[2] += XzB[dx][dy][qz] * Btz;
                  v[2] += XzC[dx][dy][qz] * Btz;
                  w[2] += XzG[dx][dy][qz] * Gtz;
               }
               Y(dx,dy,dz,0,e) += u[0] + v[0] + w[0];
               Y(dx,dy,dz,1,e) += u[1] + v[1] + w[1];
               Y(dx,dy,dz,2,e) += u[2] + v[2] + w[2];
            }
         }
      }
   });
}

// *****************************************************************************
void TMOP_Integrator::AddMultPA_3D(const Vector &X, Vector &Y) const
{
   MFEM_VERIFY(IntRule,"");
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B1d = maps->B;
   const Array<double> &G1d = maps->G;
   const int id = (D1D << 4 ) | Q1D;

   // Jtr setup:
   //  - TargetConstructor::target_type == IDEAL_SHAPE_UNIT_SIZE
   //  - Jtr(i) == Wideal
   DenseMatrix Wid(dim);

   const FiniteElement *fe = fes->GetFE(0);
   const Geometry::Type geom_type = fe->GetGeomType();
   const DenseMatrix &GeomJac = Geometries.GetGeomToPerfGeomJac(geom_type);
   //dbg("GeomJac:"); GeomJac.Print();
   Wid = GeomJac;

   MFEM_VERIFY(Wid.Det() == 1.0 ,"");
   MFEM_VERIFY(Wid(0,0)==1.0 && Wid(1,1)==1.0 && Wid(2,2)==1.0 &&
               Wid(1,0)==0.0 && Wid(0,1)==0.0 && Wid(1,2)==0.0 &&
               Wid(2,0)==0.0 && Wid(0,2)==0.0 && Wid(2,1)==0.0,  "");

   const auto Jtr = Reshape(Wid.Read(), dim, dim);
   auto J = Reshape(Dpa.Write(), Q1D, Q1D, Q1D, dim, dim, ne);
   MFEM_FORALL_3D(e, ne, Q1D, Q1D, Q1D,
   {
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               J(qx,qy,qz,0,0,e) = Jtr(0,0);
               J(qx,qy,qz,0,1,e) = Jtr(0,1);
               J(qx,qy,qz,0,2,e) = Jtr(0,2);

               J(qx,qy,qz,1,0,e) = Jtr(1,0);
               J(qx,qy,qz,1,1,e) = Jtr(1,1);
               J(qx,qy,qz,1,2,e) = Jtr(1,2);

               J(qx,qy,qz,2,0,e) = Jtr(2,0);
               J(qx,qy,qz,2,1,e) = Jtr(2,1);
               J(qx,qy,qz,2,2,e) = Jtr(2,2);
            }
         }
      }
   });

   switch (id)
   {
      case 0x21: return AddMultPA_Kernel_3D<2,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x22: return AddMultPA_Kernel_3D<2,2>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x23: return AddMultPA_Kernel_3D<2,3>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x24: return AddMultPA_Kernel_3D<2,4>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x25: return AddMultPA_Kernel_3D<2,5>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x26: return AddMultPA_Kernel_3D<2,6>(ne,W,B1d,G1d,Dpa,X,Y);

      case 0x31: return AddMultPA_Kernel_3D<3,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x32: return AddMultPA_Kernel_3D<3,2>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x33: return AddMultPA_Kernel_3D<3,3>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x34: return AddMultPA_Kernel_3D<3,4>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x35: return AddMultPA_Kernel_3D<3,5>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x36: return AddMultPA_Kernel_3D<3,6>(ne,W,B1d,G1d,Dpa,X,Y);

      case 0x41: return AddMultPA_Kernel_3D<4,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x42: return AddMultPA_Kernel_3D<4,2>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x43: return AddMultPA_Kernel_3D<4,3>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x44: return AddMultPA_Kernel_3D<4,4>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x45: return AddMultPA_Kernel_3D<4,5>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x46: return AddMultPA_Kernel_3D<4,6>(ne,W,B1d,G1d,Dpa,X,Y);

      case 0x51: return AddMultPA_Kernel_3D<5,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x52: return AddMultPA_Kernel_3D<5,2>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x53: return AddMultPA_Kernel_3D<5,3>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x54: return AddMultPA_Kernel_3D<5,4>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x55: return AddMultPA_Kernel_3D<5,5>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x56: return AddMultPA_Kernel_3D<5,6>(ne,W,B1d,G1d,Dpa,X,Y);
      default:  break;
   }
   dbg("id: %x",id);
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem
