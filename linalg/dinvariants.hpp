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

#ifndef MFEM_DINVARIANTS_HPP
#define MFEM_DINVARIANTS_HPP

#include "../config/config.hpp"
#include "../general/cuda.hpp"
#include "dtensor.hpp"
#include <cmath>

namespace mfem
{

namespace kernels
{

class InvariantsEvaluator3D
{
private:
   const double *J;
   double *B;
   double *dI1, *dI1b, *ddI1b;
   double *dI2, *dI2b, *ddI2, *ddI2b;
   double *dI3b;

public:

   MFEM_HOST_DEVICE
   InvariantsEvaluator3D(const double *J, double *B,
                         double *dI1, double *dI1b, double *ddI1b,
                         double *dI2, double *dI2b, double *ddI2, double *ddI2b,
                         double *dI3b):
      J(J), B(B),
      dI1(dI1), dI1b(dI1b), ddI1b(ddI1b),
      dI2(dI2), dI2b(dI2b), ddI2(ddI2), ddI2b(ddI2b), dI3b(dI3b) { }

   MFEM_HOST_DEVICE inline double Get_I3b(double &sign_detJ) // det(J) + sign
   {
      const double I3b = + J[0]*(J[4]*J[8] - J[7]*J[5])
                         - J[1]*(J[3]*J[8] - J[5]*J[6])
                         + J[2]*(J[3]*J[7] - J[4]*J[6]);
      sign_detJ = I3b >= 0.0 ? 1.0 : -1.0;
      return sign_detJ * I3b;
   }

   MFEM_HOST_DEVICE inline double Get_I3b() // det(J)
   {
      const double I3b = + J[0]*(J[4]*J[8] - J[7]*J[5])
                         - J[1]*(J[3]*J[8] - J[5]*J[6])
                         + J[2]*(J[3]*J[7] - J[4]*J[6]);
      return I3b;
   }

   MFEM_HOST_DEVICE inline double Get_I3() // det(J)^{2}
   {
      const double I3b = Get_I3b();
      return I3b * I3b;
   }

   MFEM_HOST_DEVICE inline double Get_I3b_p()  // I3b^{-2/3}
   {
      double sign_detJ;
      const double i3b = Get_I3b(sign_detJ);
      return sign_detJ * std::pow(i3b, -2./3.);
   }

   MFEM_HOST_DEVICE inline double Get_I3b_p(double &sign_detJ)  // I3b^{-2/3}
   {
      const double i3b = Get_I3b(sign_detJ);
      return sign_detJ * std::pow(i3b, -2./3.);
   }

   MFEM_HOST_DEVICE inline double Get_I1(double *B)
   {
      B[0] = J[0]*J[0] + J[3]*J[3] + J[6]*J[6];
      B[1] = J[1]*J[1] + J[4]*J[4] + J[7]*J[7];
      B[2] = J[2]*J[2] + J[5]*J[5] + J[8]*J[8];
      const double I1 = B[0] + B[1] + B[2];
      return I1;
   }

   MFEM_HOST_DEVICE inline
   double Get_I1b(double *B) // det(J)^{-2/3}*I_1 = I_1/I_3^{1/3}
   {
      const double I1b = Get_I1(B) * Get_I3b_p();
      return I1b;
   }

   MFEM_HOST_DEVICE inline void Get_B_offd(double *B)
   {
      // B = J J^t
      // B[3]=B(0,1), B[4]=B(0,2), B[5]=B(1,2)
      B[3] = J[0]*J[1] + J[3]*J[4] + J[6]*J[7]; // B(0,1)
      B[4] = J[0]*J[2] + J[3]*J[5] + J[6]*J[8]; // B(0,2)
      B[5] = J[1]*J[2] + J[4]*J[5] + J[7]*J[8]; // B(1,2)
   }

   MFEM_HOST_DEVICE inline double Get_I2(double *B)
   {
      Get_B_offd(B);
      const double I1 = Get_I1(B);
      const double BF2 = B[0]*B[0] + B[1]*B[1] + B[2]*B[2] +
                         2*(B[3]*B[3] + B[4]*B[4] + B[5]*B[5]);
      const double I2 = (I1*I1 - BF2)/2;
      return I2;
   }

   MFEM_HOST_DEVICE inline double Get_I2b(double *B) // I2b = I2*I3b^{-4/3}
   {
      const double I3b_p = Get_I3b_p();
      return Get_I2(B) * I3b_p * I3b_p;
   }

   MFEM_HOST_DEVICE inline double *Get_dI1(double *dI1)
   {
      for (int i = 0; i < 9; i++)
      {
         dI1[i] = 2*J[i];
      }
      return dI1;
   }

   MFEM_HOST_DEVICE inline double *Get_dI1b(double *B, double *dI3b, double *dI1b)
   {
      // I1b = I3b^{-2/3}*I1
      // dI1b = 2*I3b^{-2/3}*(J - (1/3)*I1/I3b*dI3b)
      double sign_detJ;
      const double I3b = Get_I3b(sign_detJ);
      const double I3b_p = Get_I3b_p();
      const double c1 = 2.0 * I3b_p;
      const double c2 = Get_I1(B)/(3.0 * I3b);
      Get_dI3b(sign_detJ, dI3b);
      for (int i = 0; i < 9; i++)
      {
         dI1b[i] = c1*(J[i] - c2*dI3b[i]);
      }
      return dI1b;
   }

   MFEM_HOST_DEVICE inline double *Get_dI2(double *B, double *dI2)
   {
      // dI2 = 2 I_1 J - 2 J J^t J = 2 (I_1 I - B) J
      const double I1 = Get_I1(B);
      Get_B_offd(B);
      // B[0]=B(0,0), B[1]=B(1,1), B[2]=B(2,2)
      // B[3]=B(0,1), B[4]=B(0,2), B[5]=B(1,2)
      const double C[6] =
      {
         2*(I1 - B[0]), 2*(I1 - B[1]), 2*(I1 - B[2]),
         -2*B[3], -2*B[4], -2*B[5]
      };
      //       | C[0]  C[3]  C[4] |  | J[0]  J[3]  J[6] |
      // dI2 = | C[3]  C[1]  C[5] |  | J[1]  J[4]  J[7] |
      //       | C[4]  C[5]  C[2] |  | J[2]  J[5]  J[8] |
      dI2[0] = C[0]*J[0] + C[3]*J[1] + C[4]*J[2];
      dI2[1] = C[3]*J[0] + C[1]*J[1] + C[5]*J[2];
      dI2[2] = C[4]*J[0] + C[5]*J[1] + C[2]*J[2];

      dI2[3] = C[0]*J[3] + C[3]*J[4] + C[4]*J[5];
      dI2[4] = C[3]*J[3] + C[1]*J[4] + C[5]*J[5];
      dI2[5] = C[4]*J[3] + C[5]*J[4] + C[2]*J[5];

      dI2[6] = C[0]*J[6] + C[3]*J[7] + C[4]*J[8];
      dI2[7] = C[3]*J[6] + C[1]*J[7] + C[5]*J[8];
      dI2[8] = C[4]*J[6] + C[5]*J[7] + C[2]*J[8];
      return dI2;
   }

   MFEM_HOST_DEVICE inline double *Get_dI2b(double *B,
                                            double *dI2,
                                            double *dI3b,
                                            double *dI2b)
   {
      // I2b = det(J)^{-4/3}*I2 = I3b^{-4/3}*I2
      // dI2b = (-4/3)*I3b^{-7/3}*I2*dI3b + I3b^{-4/3}*dI2
      //      = I3b^{-4/3} * [ dI2 - (4/3)*I2/I3b*dI3b ]
      double sign_detJ;
      const double I2 = Get_I2(B);
      const double I3b_p = Get_I3b_p();
      const double I3b = Get_I3b(sign_detJ);
      const double c1 = I3b_p*I3b_p;
      const double c2 = (4*I2/I3b)/3;
      Get_dI2(B, dI2);
      Get_dI3b(sign_detJ, dI3b);
      for (int i = 0; i < 9; i++)
      {
         dI2b[i] = c1*(dI2[i] - c2*dI3b[i]);
      }
      return dI2b;
   }

   MFEM_HOST_DEVICE inline double *Get_dI3(double *dI3b, double *dI3)
   {
      // I3 = I3b^2
      // dI3 = 2*I3b*dI3b = 2*det(J)*adj(J)^T
      double sign_detJ;
      const double c1 = 2*Get_I3b(sign_detJ);
      Get_dI3b(sign_detJ, dI3b);
      for (int i = 0; i < 9; i++)
      {
         dI3[i] = c1*dI3b[i];
      }
      return dI3;
   }

   MFEM_HOST_DEVICE inline double *Get_dI3b(const double sign_detJ, double *dI3b)
   {
      // I3b = det(J)
      // dI3b = adj(J)^T
      dI3b[0] = sign_detJ*(J[4]*J[8] - J[5]*J[7]);  // 0  3  6
      dI3b[1] = sign_detJ*(J[5]*J[6] - J[3]*J[8]);  // 1  4  7
      dI3b[2] = sign_detJ*(J[3]*J[7] - J[4]*J[6]);  // 2  5  8
      dI3b[3] = sign_detJ*(J[2]*J[7] - J[1]*J[8]);
      dI3b[4] = sign_detJ*(J[0]*J[8] - J[2]*J[6]);
      dI3b[5] = sign_detJ*(J[1]*J[6] - J[0]*J[7]);
      dI3b[6] = sign_detJ*(J[1]*J[5] - J[2]*J[4]);
      dI3b[7] = sign_detJ*(J[2]*J[3] - J[0]*J[5]);
      dI3b[8] = sign_detJ*(J[0]*J[4] - J[1]*J[3]);
      return dI3b;
   }

   // *****************************************************************************
   // ddI1b = X1 + X2 + X3, where
   // X1_ijkl = (2/3*I1b/I3) [ 2/3 dI3b_ij dI3b_kl + dI3b_kj dI3b_il ]
   // X2_ijkl = (I3b^{-2/3}) ddI1_ijkl
   // X3_ijkl = -(4/3*I3b^{-5/3}) (J_ij dI3b_kl + dI3b_ij J_kl)
   MFEM_HOST_DEVICE inline double *Get_ddI1b_ij(int i, int j,
                                                double *B, double *dI3b,
                                                double *ddI1b_ij)
   {


      // X1_ijkl = (2/3*I1b/I3) [ 2/3 dI3b_ij dI3b_kl + dI3b_kj dI3b_il ]
      double sign_detJ;
      Get_I3b(sign_detJ);
      double X1_p[9], X2_p[9], X3_p[9];
      DeviceMatrix X1(X1_p,3,3);
      const double I3 = Get_I3();
      const double I1b = Get_I1b(B);
      const double alpha = (2./3.)*I1b/I3;

      ConstDeviceMatrix di3b(Get_dI3b(sign_detJ,dI3b),3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            X1(k,l) = alpha * ((2./3.)*di3b(i,j) * di3b(k,l) +
                               di3b(k,j)*di3b(i,l));
         }
      }
      // ddI1_ijkl = 2 δ_ik δ_jl
      // X2_ijkl = (I3b^{-2/3}) ddI1_ijkl
      DeviceMatrix X2(X2_p,3,3);
      const double beta = Get_I3b_p();
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const double ddI1_ijkl = (i==k && j==l) ? 2.0 : 0.0;
            X2(k,l) = beta * ddI1_ijkl;
         }
      }
      // X3_ijkl = -(4/3*I3b^{-5/3}) (J_ij dI3b_kl + dI3b_ij J_kl)
      DeviceMatrix X3(X3_p,3,3);
      const double I3b = Get_I3b();
      const double gamma = -(4./3.)*Get_I3b_p()/I3b;
      ConstDeviceMatrix Jpt(J,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            X3(k,l) = gamma * (Jpt(i,j) * di3b(k,l) + di3b(i,j) * Jpt(k,l));
         }
      }
      DeviceMatrix ddI1b(ddI1b_ij,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            ddI1b(k,l) = X1(k,l) + X2(k,l) + X3(k,l);
         }
      }
      return ddI1b_ij;
   }

   // *****************************************************************************
   // ddI2 = x1 + x2 + x3
   //    x1_ijkl = (2 I1) δ_ik δ_jl
   //    x2_ijkl = 2 ( 2 δ_ku δ_iv - δ_ik δ_uv - δ_kv δ_iu ) J_vj J_ul
   //    x3_ijkl = -2 (J J^t)_ik δ_jl = -2 B_ik δ_jl
   MFEM_HOST_DEVICE inline double *Get_ddI2_ij(double *B,
                                               int i, int j,
                                               double *ddI2_ij)
   {
      double x1_p[9], x2_p[9], x3_p[9];
      DeviceMatrix x1(x1_p,3,3), x2(x2_p,3,3), x3(x3_p,3,3);
      // x1_ijkl = (2 I1) δ_ik δ_jl
      const double I1 = Get_I1(B);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const double ik_jl = (i==k && j==l) ? 1.0 : 0.0;
            x1(k,l) = 2.0 * I1 * ik_jl;
         }
      }
      // x2_ijkl = 2 ( 2 δ_ku δ_iv - δ_ik δ_uv - δ_kv δ_iu ) J_vj J_ul
      ConstDeviceMatrix Jpt(J,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            x2(k,l) = 0.0;
            for (int u=0; u<3; u++)
            {
               for (int v=0; v<3; v++)
               {
                  const double ku_iv = k==u && i==v ? 1.0 : 0.0;
                  const double ik_uv = i==k && u==v ? 1.0 : 0.0;
                  const double kv_iu = k==v && i==u ? 1.0 : 0.0;
                  x2(k,l) += 2.0*(2.*ku_iv-ik_uv-kv_iu)*Jpt(v,j)*Jpt(u,l);
               }
            }
         }
      }

      // x3_ijkl = -2 B_ik δ_jl
      B[0] = J[0]*J[0] + J[3]*J[3] + J[6]*J[6];
      B[1] = J[1]*J[1] + J[4]*J[4] + J[7]*J[7];
      B[2] = J[2]*J[2] + J[5]*J[5] + J[8]*J[8];
      B[3] = J[0]*J[1] + J[3]*J[4] + J[6]*J[7]; // B(0,1)
      B[4] = J[0]*J[2] + J[3]*J[5] + J[6]*J[8]; // B(0,2)
      B[5] = J[1]*J[2] + J[4]*J[5] + J[7]*J[8]; // B(1,2)
      const double b_p[9] =
      {
         B[0], B[3], B[4],
         B[3], B[1], B[5],
         B[4], B[5], B[2]
      };
      ConstDeviceMatrix b(b_p,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const double jl = j==l ? 1.0 : 0.0;
            x3(k,l) = -2.0 * b(i,k) * jl;
         }
      }
      // ddI2 = x1 + x2 + x3
      DeviceMatrix ddI2(ddI2_ij,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            ddI2(k,l) = x1(k,l) + x2(k,l) + x3(k,l);
         }
      }
      return ddI2_ij;
   }

   // *****************************************************************************
   // ddI2b = X1 + X2 + X3
   //    X1_ijkl = 16/9 det(J)^{-10/3} I2 dI3b_ij dI3b_kl +
   //               4/3 det(J)^{-10/3} I2 dI3b_il dI3b_kj
   //    X2_ijkl = -4/3 det(J)^{-7/3} (dI2_ij dI3b_kl + dI2_kl dI3b_ij)
   //    X3_ijkl =      det(J)^{-4/3} ddI2_ijkl
   MFEM_HOST_DEVICE inline double *Get_ddI2b_ij(int i, int j,
                                                double *B,
                                                double *dI2,
                                                double *dI3b,
                                                double *ddI2_ij,
                                                double *ddI2b_ij)
   {
      double X1_p[9], X2_p[9], X3_p[9];
      // X1_ijkl = 16/9 det(J)^{-10/3} I2 dI3b_ij dI3b_kl +
      //               4/3 det(J)^{-10/3} I2 dI3b_il dI3b_kj
      double sign_detJ;
      DeviceMatrix X1(X1_p,3,3);
      const double I3b_p = Get_I3b_p(); // I3b^{-2/3}
      const double I3b = Get_I3b(sign_detJ); // det(J)
      const double I2 = Get_I2(B);
      const double I3b_p43 = I3b_p*I3b_p;
      const double I3b_p73 = I3b_p*I3b_p/I3b;
      const double I3b_p103 = I3b_p*I3b_p/(I3b*I3b);
      ConstDeviceMatrix di3b(Get_dI3b(sign_detJ,dI3b),3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            const double up = (16./9.)*I3b_p103*I2*di3b(i,j)*di3b(k,l);
            const double down = (4./3.)*I3b_p103*I2*di3b(i,l)*di3b(k,j);
            X1(k,l) = up + down;
         }
      }
      // X2_ijkl = -4/3 det(J)^{-7/3} (dI2_ij dI3b_kl + dI2_kl dI3b_ij)
      DeviceMatrix X2(X2_p,3,3);
      ConstDeviceMatrix di2(Get_dI2(B,dI2),3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            X2(k,l) = -(4./3.)*I3b_p73*(di2(i,j)*di3b(k,l) + di2(k,l)*di3b(i,j));
         }
      }
      // X3_ijkl =  det(J)^{-4/3} ddI2_ijkl
      DeviceMatrix X3(X3_p,3,3);
      ConstDeviceMatrix ddI2(Get_ddI2_ij(B,i,j,ddI2_ij),3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            X3(k,l) = I3b_p43 * ddI2(k,l);
         }
      }
      // ddI2b = X1 + X2 + X3
      DeviceMatrix ddI2b(ddI2b_ij,3,3);
      for (int k=0; k<3; k++)
      {
         for (int l=0; l<3; l++)
         {
            ddI2b(k,l) = X1(k,l) + X2(k,l) + X3(k,l);
         }
      }
      return ddI2b_ij;
   }
};

} // namespace kernels

} // namespace mfem

#endif // MFEM_DINVARIANTS_HPP
