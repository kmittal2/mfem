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
#include "tmop_pa.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AddMultGradPA_Kernel_2D,
                           const int NE,
                           const Array<double> &b_,
                           const Array<double> &g_,
                           const DenseTensor &j_,
                           const Vector &h_,
                           const Vector &x_,
                           Vector &y_,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto g = Reshape(g_.Read(), Q1D, D1D);
   const auto J = Reshape(j_.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto X = Reshape(x_.Read(), D1D, D1D, DIM, NE);
   const auto H = Reshape(h_.Read(), DIM, DIM, DIM, DIM, Q1D, Q1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, DIM, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double BG[2][MQ1*MD1];
      MFEM_SHARED double XY[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[4][NBZ][MQ1*MQ1];

      kernels::LoadX<MD1,NBZ>(e,D1D,X,XY);
      kernels::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);

      kernels::GradX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
      kernels::GradY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double *Jtr = &J(0,0,qx,qy,e);

            // Jrt = Jtr^{-1}
            double Jrt[4];
            kernels::CalcInverse<2>(Jtr, Jrt);

            // Jpr = X^T.DSh
            double Jpr[4];
            kernels::PullGradXY<MQ1,NBZ>(qx,qy,QQ,Jpr);

            // Jpt = Jpr . Jrt
            double Jpt[4];
            kernels::Mult(2,2,2, Jpr, Jrt, Jpt);

            // B = Jpt : H
            double B[4];
            DeviceMatrix M(B,2,2);
            ConstDeviceMatrix J(Jpt,2,2);
            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  M(i,j) = 0.0;
                  for (int r = 0; r < DIM; r++)
                  {
                     for (int c = 0; c < DIM; c++)
                     {
                        M(i,j) += H(r,c,i,j,qx,qy,e) * J(r,c);
                     }
                  }
               }
            }

            // C = Jrt . B
            double C[4];
            kernels::MultABt(2,2,2, Jrt, B, C);

            // Overwrite QQ = Jrt . (Jpt : H)^t
            kernels::PushGradXY<MQ1,NBZ>(qx,qy, C, QQ);
         }
      }
      MFEM_SYNC_THREAD;
      kernels::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,BG);
      kernels::GradYt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,DQ);
      kernels::GradXt<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,Y,e);
   });
}

MFEM_REGISTER_TMOP_KERNELS(void, AssembleDiagonalPA_Kernel_2D,
                           const int NE,
                           const Array<double> &b,
                           const Array<double> &g,
                           const DenseTensor &j,
                           const Vector &h,
                           Vector &diagonal,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto J = Reshape(j.Read(), DIM, DIM, Q1D, Q1D, NE);
   const auto H = Reshape(h.Read(), DIM, DIM, DIM, DIM, Q1D, Q1D, NE);

   auto D = Reshape(diagonal.ReadWrite(), D1D, D1D, DIM, NE);

   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;

      double qd[DIM*DIM*MQ1*MD1];
      DeviceTensor<4,double> QD(qd, DIM, DIM, MQ1, MD1);

      for (int r = 0; r < DIM; r++)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               QD(0,0,qx,dy) = 0.0;
               QD(0,1,qx,dy) = 0.0;
               QD(1,0,qx,dy) = 0.0;
               QD(1,1,qx,dy) = 0.0;

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double *Jtr = &J(0,0,qx,qy,e);

                  // Jrt = Jtr^{-1}
                  double j[4];
                  ConstDeviceMatrix Jrt(j,2,2);
                  kernels::CalcInverse<2>(Jtr, j);

                  const double gg = G(qy,dy) * G(qy,dy);
                  const double gb = G(qy,dy) * B(qy,dy);
                  const double bb = B(qy,dy) * B(qy,dy);
                  const double bgb[4] = { bb, gb, gb, gg };
                  ConstDeviceMatrix BG(bgb,2,2);

                  for (int i = 0; i < DIM; i++)
                  {
                     for (int j = 0; j < DIM; j++)
                     {
                        const double JJt = Jrt(i,j) * Jrt(j,i);
                        QD(i,j,qx,dy) += JJt * BG(i,j) * H(r,i,r,j,qx,qy,e);
                     }
                  }
               }
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               double d = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double gg = G(qx,dx) * G(qx,dx);
                  const double gb = G(qx,dx) * B(qx,dx);
                  const double bb = B(qx,dx) * B(qx,dx);
                  d += gg * QD(0,0,qx,dy);
                  d += gb * QD(0,1,qx,dy);
                  d += gb * QD(1,0,qx,dy);
                  d += bb * QD(1,1,qx,dy);
               }
               D(dx,dy,r,e) += d;
            }
         }
      }
   });
}

void TMOP_Integrator::AddMultGradPA_2D(const Vector &R, Vector &C) const
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &H = PA.H;

   MFEM_LAUNCH_TMOP_KERNEL(AddMultGradPA_Kernel_2D,id,N,B,G,J,H,R,C);
}

void TMOP_Integrator::AssembleDiagonalPA_2D(Vector &D)
{
   const int N = PA.ne;
   const int D1D = PA.maps->ndof;
   const int Q1D = PA.maps->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const DenseTensor &J = PA.Jtr;
   const Array<double> &B = PA.maps->B;
   const Array<double> &G = PA.maps->G;
   const Vector &H = PA.H;

   MFEM_LAUNCH_TMOP_KERNEL(AssembleDiagonalPA_Kernel_2D,id,N,B,G,J,H,D);
}

} // namespace mfem
