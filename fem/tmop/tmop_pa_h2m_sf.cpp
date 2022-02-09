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

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../linearform.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AddMultGradPA_Kernel_SF_2D,
                           const int NE,
                           const Array<double> &b_,
                           const Vector &h0_,
                           const Vector &r_,
                           Vector &c_,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 2;
   constexpr int NBZ = 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto H0 = Reshape(h0_.Read(), Q1D, Q1D, DIM, DIM, NE);
   const auto R = Reshape(r_.Read(), D1D, D1D, DIM, NE);

   auto Y = Reshape(c_.ReadWrite(), D1D, D1D, DIM, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      constexpr int DIM = 2;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double Xh[2];
            DeviceMatrix X(Xh,2,1);
            for (int i = 0; i < DIM; i++)
            {
               X(i,0) = R(qx,qy,i,e);
            }

            double H_data[4];
            DeviceMatrix H(H_data,2,2);
            for (int i = 0; i < DIM; i++)
            {
               for (int j = 0; j < DIM; j++)
               {
                  H(i,j) = H0(qx,qy,i,j,e);
               }
            }

            // p2 = H . Xh
            double p2[2];
            kernels::Mult(2,2,H_data,Xh,p2);
            for (int d = 0; d < DIM; d++)
            {
               Y(qx, qy, d, e) += p2[d];
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

void TMOP_Integrator::AddMultGradPA_SF_2D(const Vector &R,Vector &C) const
{
   const int N = PA.ne;
   const int D1D = PA.maps_surf->ndof;
   const int Q1D = PA.maps_surf->nqpt;
   const int id = (D1D << 4 ) | Q1D;
   const Array<double> &B = PA.maps_surf->B;
   const Vector &H0 = PA.Hsf;

   MFEM_LAUNCH_TMOP_KERNEL(AddMultGradPA_Kernel_SF_2D,id,N,B,H0,R,C);
}

} // namespace mfem
