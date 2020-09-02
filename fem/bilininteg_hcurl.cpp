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

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "libceed/mass.hpp"

using namespace std;

namespace mfem
{

// Local maximum size of dofs and quads in 1D
constexpr int HCURL_MAX_D1D = 5;
constexpr int HCURL_MAX_Q1D = 6;

// PA H(curl) Mass Assemble 2D kernel
void PAHcurlSetup2D(const int Q1D,
                    const int coeffDim,
                    const int NE,
                    const Array<double> &w,
                    const Vector &j,
                    Vector &_coeff,
                    Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();

   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto coeff = Reshape(_coeff.Read(), coeffDim, NQ, NE);
   auto y = Reshape(op.Write(), NQ, 3, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double c_detJ1 = W[q] * coeff(0, q, e) / ((J11*J22)-(J21*J12));
         const double c_detJ2 = coeffDim == 2 ? W[q] * coeff(1, q, e)
         / ((J11*J22)-(J21*J12)) : c_detJ1;
         y(q,0,e) =  (c_detJ2*J12*J12 + c_detJ1*J22*J22); // 1,1
         y(q,1,e) = -(c_detJ2*J12*J11 + c_detJ1*J22*J21); // 1,2
         y(q,2,e) =  (c_detJ2*J11*J11 + c_detJ1*J21*J21); // 2,2
      }
   });
}

// PA H(curl) Mass Assemble 3D kernel
void PAHcurlSetup3D(const int Q1D,
                    const int coeffDim,
                    const int NE,
                    const Array<double> &w,
                    const Vector &j,
                    Vector &_coeff,
                    Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto coeff = Reshape(_coeff.Read(), coeffDim, NQ, NE);
   auto y = Reshape(op.Write(), NQ, 6, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J31 = J(q,2,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double J32 = J(q,2,1,e);
         const double J13 = J(q,0,2,e);
         const double J23 = J(q,1,2,e);
         const double J33 = J(q,2,2,e);
         const double detJ = J11 * (J22 * J33 - J32 * J23) -
         /* */               J21 * (J12 * J33 - J32 * J13) +
         /* */               J31 * (J12 * J23 - J22 * J13);
         const double w_detJ = W[q] / detJ;
         const double D1 = coeff(0, q, e);
         const double D2 = coeffDim == 3 ? coeff(1, q, e) : D1;
         const double D3 = coeffDim == 3 ? coeff(2, q, e) : D1;
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J32 * J13) - (J12 * J33);
         const double A13 = (J12 * J23) - (J22 * J13);
         const double A21 = (J31 * J23) - (J21 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J21 * J13) - (J11 * J23);
         const double A31 = (J21 * J32) - (J31 * J22);
         const double A32 = (J31 * J12) - (J11 * J32);
         const double A33 = (J11 * J22) - (J12 * J21);
         // detJ J^{-1} J^{-T} = (1/detJ) adj(J) D adj(J)^T
         y(q,0,e) = w_detJ * (D1*A11*A11 + D2*A12*A12 + D3*A13*A13); // 1,1
         y(q,1,e) = w_detJ * (D1*A11*A21 + D2*A12*A22 + D3*A13*A23); // 2,1
         y(q,2,e) = w_detJ * (D1*A11*A31 + D2*A12*A32 + D3*A13*A33); // 3,1
         y(q,3,e) = w_detJ * (D1*A21*A21 + D2*A22*A22 + D3*A23*A23); // 2,2
         y(q,4,e) = w_detJ * (D1*A21*A31 + D2*A22*A32 + D3*A23*A33); // 3,2
         y(q,5,e) = w_detJ * (D1*A31*A31 + D2*A32*A32 + D3*A33*A33); // 3,3
      }
   });
}

void PAHcurlMassApply2D(const int D1D,
                        const int Q1D,
                        const int NE,
                        const Array<double> &_Bo,
                        const Array<double> &_Bc,
                        const Array<double> &_Bot,
                        const Array<double> &_Bct,
                        const Vector &_op,
                        const Vector &_x,
                        Vector &_y)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 3, NE);
   auto x = Reshape(_x.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][VDIM];

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               mass[qy][qx][c] = 0.0;
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            double massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               massX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double t = x(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qy][qx][c] += massX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double O11 = op(qx,qy,0,e);
            const double O12 = op(qx,qy,1,e);
            const double O22 = op(qx,qy,2,e);
            const double massX = mass[qy][qx][0];
            const double massY = mass[qy][qx][1];
            mass[qy][qx][0] = (O11*massX)+(O12*massY);
            mass[qy][qx][1] = (O12*massX)+(O22*massY);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            double massX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               massX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] += mass[qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

void PAHcurlMassAssembleDiagonal2D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &_Bo,
                                   const Array<double> &_Bc,
                                   const Vector &_op,
                                   Vector &_diag)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 3, NE);
   auto diag = Reshape(_diag.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         double mass[MAX_Q1D];

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               mass[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);

                  mass[qx] += wy * wy * ((c == 0) ? op(qx,qy,0,e) : op(qx,qy,2,e));
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                  diag(dx + (dy * D1Dx) + osc, e) += mass[qx] * wx * wx;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop c
   }); // end of element loop
}

void PAHcurlMassAssembleDiagonal3D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &_Bo,
                                   const Array<double> &_Bc,
                                   const Vector &_op,
                                   Vector &_diag)
{
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto diag = Reshape(_diag.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         const int opc = (c == 0) ? 0 : ((c == 1) ? 3 : 5);

         double mass[MAX_Q1D];

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qx] = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);

                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const double wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);

                        mass[qx] += wy * wy * wz * wz * op(qx,qy,qz,opc,e);
                     }
                  }
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                     diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += mass[qx] * wx * wx;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop c
   }); // end of element loop
}

void PAHcurlMassApply3D(const int D1D,
                        const int Q1D,
                        const int NE,
                        const Array<double> &_Bo,
                        const Array<double> &_Bc,
                        const Array<double> &_Bot,
                        const Array<double> &_Bct,
                        const Vector &_op,
                        const Vector &_x,
                        Vector &_y)
{
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto x = Reshape(_x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  mass[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double massXY[MAX_Q1D][MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     mass[qz][qy][qx][c] += massXY[qy][qx] * wz;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop (c) over components

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O22 = op(qx,qy,qz,3,e);
               const double O23 = op(qx,qy,qz,4,e);
               const double O33 = op(qx,qy,qz,5,e);
               const double massX = mass[qz][qy][qx][0];
               const double massY = mass[qz][qy][qx][1];
               const double massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[MAX_D1D][MAX_D1D];

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += mass[qz][qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bot(dz,qz) : Bct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

// PA H(curl) curl-curl assemble 2D kernel
static void PACurlCurlSetup2D(const int Q1D,
                              const int NE,
                              const Array<double> &w,
                              const Vector &j,
                              Vector &_coeff,
                              Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto coeff = Reshape(_coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double detJ = (J11*J22)-(J21*J12);
         y(q,e) = W[q] * coeff(q,e) / detJ;
      }
   });
}

// PA H(curl) curl-curl assemble 3D kernel
static void PACurlCurlSetup3D(const int Q1D,
                              const int coeffDim,
                              const int NE,
                              const Array<double> &w,
                              const Vector &j,
                              Vector &_coeff,
                              Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto coeff = Reshape(_coeff.Read(), coeffDim, NQ, NE);
   auto y = Reshape(op.Write(), NQ, 6, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J31 = J(q,2,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double J32 = J(q,2,1,e);
         const double J13 = J(q,0,2,e);
         const double J23 = J(q,1,2,e);
         const double J33 = J(q,2,2,e);
         const double detJ = J11 * (J22 * J33 - J32 * J23) -
         /* */               J21 * (J12 * J33 - J32 * J13) +
         /* */               J31 * (J12 * J23 - J22 * J13);

         const double D1 = coeff(0, q, e);
         const double D2 = coeffDim == 3 ? coeff(1, q, e) : D1;
         const double D3 = coeffDim == 3 ? coeff(2, q, e) : D1;

         // set y to the 6 entries of J^T D J / det^2
         const double c_detJ = W[q] / detJ;

         y(q,0,e) = c_detJ * (D1*J11*J11 + D2*J21*J21 + D3*J31*J31); // 1,1
         y(q,1,e) = c_detJ * (D1*J11*J12 + D2*J21*J22 + D3*J31*J32); // 1,2
         y(q,2,e) = c_detJ * (D1*J11*J13 + D2*J21*J23 + D3*J31*J33); // 1,3
         y(q,3,e) = c_detJ * (D1*J12*J12 + D2*J22*J22 + D3*J32*J32); // 2,2
         y(q,4,e) = c_detJ * (D1*J12*J13 + D2*J22*J23 + D3*J32*J33); // 2,3
         y(q,5,e) = c_detJ * (D1*J13*J13 + D2*J23*J23 + D3*J33*J33); // 3,3
      }
   });
}

void CurlCurlIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fel = fes.GetFE(0);

   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*el, *el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   const int ndata = (dim == 2) ? 1 : 6;
   pa_data.SetSize(ndata * nq * ne, Device::GetMemoryType());

   Vector coeff(ne * nq);
   coeff = 1.0;
   if (Q)
   {
      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
            coeff[p + (e * nq)] = Q->Eval(*tr, ir->IntPoint(p));
         }
      }
   }

   if (el->GetDerivType() == mfem::FiniteElement::CURL && dim == 3)
   {
      PACurlCurlSetup3D(quad1D, 1, ne, ir->GetWeights(), geom->J,
                        coeff, pa_data);
   }
   else if (el->GetDerivType() == mfem::FiniteElement::CURL && dim == 2)
   {
      PACurlCurlSetup2D(quad1D, ne, ir->GetWeights(), geom->J,
                        coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

static void PACurlCurlApply2D(const int D1D,
                              const int Q1D,
                              const int NE,
                              const Array<double> &_Bo,
                              const Array<double> &_Bot,
                              const Array<double> &_Gc,
                              const Array<double> &_Gct,
                              const Vector &_op,
                              const Vector &_x,
                              Vector &_y)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto Gct = Reshape(_Gct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double curl[MAX_Q1D][MAX_Q1D];

      // curl[qy][qx] will be computed as du_y/dx - du_x/dy

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            curl[qy][qx] = 0;
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            double gradX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx] = 0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double t = x(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx] += t * ((c == 0) ? Bo(qx,dx) : Gc(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = (c == 0) ? -Gc(qy,dy) : Bo(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  curl[qy][qx] += gradX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            curl[qy][qx] *= op(qx,qy,e);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            double gradX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               gradX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradX[dx] += curl[qy][qx] * ((c == 0) ? Bot(dx,qx) : Gct(dx,qx));
               }
            }
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const double wy = (c == 0) ? -Gct(dy,qy) : Bot(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  y(dx + (dy * D1Dx) + osc, e) += gradX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PACurlCurlApply3D(const int D1D,
                              const int Q1D,
                              const int NE,
                              const Array<double> &_Bo,
                              const Array<double> &_Bc,
                              const Array<double> &_Bot,
                              const Array<double> &_Bct,
                              const Array<double> &_Gc,
                              const Array<double> &_Gct,
                              const Vector &_op,
                              const Vector &_x,
                              Vector &_y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
   // (\nabla\times u) \cdot (\nabla\times v) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{v}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto Gct = Reshape(_Gct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto x = Reshape(_x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double curl[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];
      // curl[qz][qy][qx] will be computed as the vector curl at each quadrature point.

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  curl[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      // We treat x, y, z components separately for optimization specific to each.

      int osc = 0;

      {
         // x component
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     curl[qz][qy][qx][1] += gradXY[qy][qx][1] * wDz; // (u_0)_{x_2}
                     curl[qz][qy][qx][2] -= gradXY[qy][qx][0] * wz;  // -(u_0)_{x_1}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // y component
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               double massY[MAX_Q1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = Bc(qx,dx);
                  const double wDx = Gc(qx,dx);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     curl[qz][qy][qx][0] -= gradXY[qy][qx][1] * wDz; // -(u_1)_{x_2}
                     curl[qz][qy][qx][2] += gradXY[qy][qx][0] * wz;  // (u_1)_{x_0}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // z component
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int dx = 0; dx < D1Dx; ++dx)
         {
            double gradYZ[MAX_Q1D][MAX_Q1D][2];
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradYZ[qz][qy][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massZ[MAX_Q1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wx = Bc(qx,dx);
               const double wDx = Gc(qx,dx);

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     curl[qz][qy][qx][0] += gradYZ[qz][qy][1] * wx;  // (u_2)_{x_1}
                     curl[qz][qy][qx][1] -= gradYZ[qz][qy][0] * wDx; // -(u_2)_{x_0}
                  }
               }
            }
         }
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O22 = op(qx,qy,qz,3,e);
               const double O23 = op(qx,qy,qz,4,e);
               const double O33 = op(qx,qy,qz,5,e);

               const double c1 = (O11 * curl[qz][qy][qx][0]) + (O12 * curl[qz][qy][qx][1]) +
                                 (O13 * curl[qz][qy][qx][2]);
               const double c2 = (O12 * curl[qz][qy][qx][0]) + (O22 * curl[qz][qy][qx][1]) +
                                 (O23 * curl[qz][qy][qx][2]);
               const double c3 = (O13 * curl[qz][qy][qx][0]) + (O23 * curl[qz][qy][qx][1]) +
                                 (O33 * curl[qz][qy][qx][2]);

               curl[qz][qy][qx][0] = c1;
               curl[qz][qy][qx][1] = c2;
               curl[qz][qy][qx][2] = c3;
            }
         }
      }

      // x component
      osc = 0;
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            double gradXY12[MAX_D1D][MAX_D1D];
            double gradXY21[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY12[dy][dx] = 0.0;
                  gradXY21[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[MAX_D1D][2];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massX[dx][n] = 0.0;
                  }
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     const double wx = Bot(dx,qx);

                     massX[dx][0] += wx * curl[qz][qy][qx][1];
                     massX[dx][1] += wx * curl[qz][qy][qx][2];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     gradXY21[dy][dx] += massX[dx][0] * wy;
                     gradXY12[dy][dx] += massX[dx][1] * wDy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     // (u_0)_{x_2} * (op * curl)_1 - (u_0)_{x_1} * (op * curl)_2
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (gradXY21[dy][dx] * wDz) - (gradXY12[dy][dx] * wz);
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // y component
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            double gradXY02[MAX_D1D][MAX_D1D];
            double gradXY20[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY02[dy][dx] = 0.0;
                  gradXY20[dy][dx] = 0.0;
               }
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               double massY[MAX_D1D][2];
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  massY[dy][0] = 0.0;
                  massY[dy][1] = 0.0;
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     const double wy = Bot(dy,qy);

                     massY[dy][0] += wy * curl[qz][qy][qx][2];
                     massY[dy][1] += wy * curl[qz][qy][qx][0];
                  }
               }
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double wx = Bct(dx,qx);
                  const double wDx = Gct(dx,qx);

                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     gradXY02[dy][dx] += massY[dy][0] * wDx;
                     gradXY20[dy][dx] += massY[dy][1] * wx;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     // -(u_1)_{x_2} * (op * curl)_0 + (u_1)_{x_0} * (op * curl)_2
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (-gradXY20[dy][dx] * wDz) + (gradXY02[dy][dx] * wz);
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // z component
      {
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int qx = 0; qx < Q1D; ++qx)
         {
            double gradYZ01[MAX_D1D][MAX_D1D];
            double gradYZ10[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  gradYZ01[dz][dy] = 0.0;
                  gradYZ10[dz][dy] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massZ[MAX_D1D][2];
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massZ[dz][n] = 0.0;
                  }
               }
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     const double wz = Bot(dz,qz);

                     massZ[dz][0] += wz * curl[qz][qy][qx][0];
                     massZ[dz][1] += wz * curl[qz][qy][qx][1];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     gradYZ01[dz][dy] += wy * massZ[dz][1];
                     gradYZ10[dz][dy] += wDy * massZ[dz][0];
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double wx = Bct(dx,qx);
               const double wDx = Gct(dx,qx);

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     // (u_2)_{x_1} * (op * curl)_0 - (u_2)_{x_0} * (op * curl)_1
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (gradYZ10[dz][dy] * wx) - (gradYZ01[dz][dy] * wDx);
                  }
               }
            }
         }  // loop qx
      }

   }); // end of element loop
}

void CurlCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      PACurlCurlApply3D(dofs1D, quad1D, ne, mapsO->B, mapsC->B, mapsO->Bt,
                        mapsC->Bt, mapsC->G, mapsC->Gt, pa_data, x, y);
   }
   else if (dim == 2)
   {
      PACurlCurlApply2D(dofs1D, quad1D, ne, mapsO->B, mapsO->Bt,
                        mapsC->G, mapsC->Gt, pa_data, x, y);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

static void PACurlCurlAssembleDiagonal2D(const int D1D,
                                         const int Q1D,
                                         const int NE,
                                         const Array<double> &_Bo,
                                         const Array<double> &_Gc,
                                         const Vector &_op,
                                         Vector &_diag)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, NE);
   auto diag = Reshape(_diag.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         double t[MAX_Q1D];

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               t[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bo(qy,dy) : -Gc(qy,dy);
                  t[qx] += wy * wy * op(qx,qy,e);
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = ((c == 0) ? Bo(qx,dx) : Gc(qx,dx));
                  diag(dx + (dy * D1Dx) + osc, e) += t[qx] * wx * wx;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop c
   }); // end of element loop
}

template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PACurlCurlAssembleDiagonal3D(const int D1D,
                                         const int Q1D,
                                         const int NE,
                                         const Array<double> &_Bo,
                                         const Array<double> &_Bc,
                                         const Array<double> &_Go,
                                         const Array<double> &_Gc,
                                         const Vector &_op,
                                         Vector &_diag)
{
   constexpr static int VDIM = 3;
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Go = Reshape(_Go.Read(), Q1D, D1D-1);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto diag = Reshape(_diag.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
      // (\nabla\times u) \cdot (\nabla\times u) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{u}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

      // For each c, we will keep 6 arrays for derivatives multiplied by the 6 entries of the symmetric 3x3 matrix (dF^T dF).

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         double zt[MAX_Q1D][MAX_Q1D][MAX_D1D][6][3];

         // z contraction
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int i=0; i<6; ++i)
                  {
                     for (int d=0; d<3; ++d)
                     {
                        zt[qx][qy][dz][i][d] = 0.0;
                     }
                  }

                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double wz = ((c == 2) ? Bo(qz,dz) : Bc(qz,dz));
                     const double wDz = ((c == 2) ? Go(qz,dz) : Gc(qz,dz));

                     for (int i=0; i<6; ++i)
                     {
                        zt[qx][qy][dz][i][0] += wz * wz * op(qx,qy,qz,i,e);
                        zt[qx][qy][dz][i][1] += wDz * wz * op(qx,qy,qz,i,e);
                        zt[qx][qy][dz][i][2] += wDz * wDz * op(qx,qy,qz,i,e);
                     }
                  }
               }
            }
         }  // end of z contraction

         double yt[MAX_Q1D][MAX_D1D][MAX_D1D][6][3][3];

         // y contraction
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dz = 0; dz < D1Dz; ++dz)
            {
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int i=0; i<6; ++i)
                  {
                     for (int d=0; d<3; ++d)
                        for (int j=0; j<3; ++j)
                        {
                           yt[qx][dy][dz][i][d][j] = 0.0;
                        }
                  }

                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = ((c == 1) ? Bo(qy,dy) : Bc(qy,dy));
                     const double wDy = ((c == 1) ? Go(qy,dy) : Gc(qy,dy));

                     for (int i=0; i<6; ++i)
                     {
                        for (int d=0; d<3; ++d)
                        {
                           yt[qx][dy][dz][i][d][0] += wy * wy * zt[qx][qy][dz][i][d];
                           yt[qx][dy][dz][i][d][1] += wDy * wy * zt[qx][qy][dz][i][d];
                           yt[qx][dy][dz][i][d][2] += wDy * wDy * zt[qx][qy][dz][i][d];
                        }
                     }
                  }
               }
            }
         }  // end of y contraction

         // x contraction
         for (int dz = 0; dz < D1Dz; ++dz)
         {
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                     const double wDx = ((c == 0) ? Go(qx,dx) : Gc(qx,dx));

                     // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
                     // (\nabla\times u) \cdot (\nabla\times u) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{u}
                     // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

                     /*
                       const double O11 = op(q,0,e);
                       const double O12 = op(q,1,e);
                       const double O13 = op(q,2,e);
                       const double O22 = op(q,3,e);
                       const double O23 = op(q,4,e);
                       const double O33 = op(q,5,e);
                     */

                     if (c == 0)
                     {
                        // (u_0)_{x_2} (O22 (u_0)_{x_2} - O23 (u_0)_{x_1}) - (u_0)_{x_1} (O32 (u_0)_{x_2} - O33 (u_0)_{x_1})

                        // (u_0)_{x_2} O22 (u_0)_{x_2}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                             e) += yt[qx][dy][dz][3][2][0] * wx * wx;

                        // -(u_0)_{x_2} O23 (u_0)_{x_1} - (u_0)_{x_1} O32 (u_0)_{x_2}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                             e) += -2.0 * yt[qx][dy][dz][4][1][1] * wx * wx;

                        // (u_0)_{x_1} O33 (u_0)_{x_1}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                             e) += yt[qx][dy][dz][5][0][2] * wx * wx;
                     }
                     else if (c == 1)
                     {
                        // (u_1)_{x_2} (O11 (u_1)_{x_2} - O13 (u_1)_{x_0}) + (u_1)_{x_0} (-O31 (u_1)_{x_2} + O33 (u_1)_{x_0})

                        // (u_1)_{x_2} O11 (u_1)_{x_2}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                             e) += yt[qx][dy][dz][0][2][0] * wx * wx;

                        // -(u_1)_{x_2} O13 (u_1)_{x_0} - (u_1)_{x_0} O31 (u_1)_{x_2}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                             e) += -2.0 * yt[qx][dy][dz][2][1][0] * wDx * wx;

                        // (u_1)_{x_0} O33 (u_1)_{x_0})
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                             e) += yt[qx][dy][dz][5][0][0] * wDx * wDx;
                     }
                     else
                     {
                        // (u_2)_{x_1} (O11 (u_2)_{x_1} - O12 (u_2)_{x_0}) - (u_2)_{x_0} (O21 (u_2)_{x_1} - O22 (u_2)_{x_0})

                        // (u_2)_{x_1} O11 (u_2)_{x_1}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                             e) += yt[qx][dy][dz][0][0][2] * wx * wx;

                        // -(u_2)_{x_1} O12 (u_2)_{x_0} - (u_2)_{x_0} O21 (u_2)_{x_1}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                             e) += -2.0 * yt[qx][dy][dz][1][0][1] * wDx * wx;

                        // (u_2)_{x_0} O22 (u_2)_{x_0}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                             e) += yt[qx][dy][dz][3][0][0] * wDx * wDx;
                     }
                  }
               }
            }
         }  // end of x contraction

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop c
   }); // end of element loop
}

void CurlCurlIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if (dim == 3)
   {
      // Reduce HCURL_MAX_D1D/Q1D to avoid using too much memory
      constexpr int MAX_D1D = 4;
      constexpr int MAX_Q1D = 5;
      PACurlCurlAssembleDiagonal3D<MAX_D1D,MAX_Q1D>(dofs1D, quad1D, ne,
                                                    mapsO->B, mapsC->B,
                                                    mapsO->G, mapsC->G,
                                                    pa_data, diag);
   }
   else if (dim == 2)
   {
      PACurlCurlAssembleDiagonal2D(dofs1D, quad1D, ne,
                                   mapsO->B, mapsC->G, pa_data, diag);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

// Apply to x corresponding to DOF's in H^1 (trial), whose gradients are
// integrated against H(curl) test functions corresponding to y.
void PAHcurlH1Apply3D(const int D1D,
                      const int Q1D,
                      const int NE,
                      const Array<double> &_Bc,
                      const Array<double> &_Gc,
                      const Array<double> &_Bot,
                      const Array<double> &_Bct,
                      const Vector &_op,
                      const Vector &_x,
                      Vector &_y)
{
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");

   constexpr static int VDIM = 3;

   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  mass[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      for (int dz = 0; dz < D1D; ++dz)
      {
         double gradXY[MAX_Q1D][MAX_Q1D][3];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradXY[qy][qx][0] = 0.0;
               gradXY[qy][qx][1] = 0.0;
               gradXY[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            double gradX[MAX_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = x(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * Bc(qx,dx);
                  gradX[qx][1] += s * Gc(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy  = Bc(qy,dy);
               const double wDy = Gc(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx  = gradX[qx][0];
                  const double wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx * wDy;
                  gradXY[qy][qx][2] += wx * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz  = Bc(qz,dz);
            const double wDz = Gc(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  mass[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  mass[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
               }
            }
         }
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O22 = op(qx,qy,qz,3,e);
               const double O23 = op(qx,qy,qz,4,e);
               const double O33 = op(qx,qy,qz,5,e);
               const double massX = mass[qz][qy][qx][0];
               const double massY = mass[qz][qy][qx][1];
               const double massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[MAX_D1D][MAX_D1D];

         int osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += mass[qz][qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bot(dz,qz) : Bct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

// Apply to x corresponding to DOFs in H^1 (domain) the (topological) gradient
// to get a dof in H(curl) (range). You can think of the range as the "test" space
// and the domain as the "trial" space, but there's no integration.
static void PAHcurlApplyGradient2D(const int c_dofs1D,
                                   const int o_dofs1D,
                                   const int NE,
                                   const Array<double> &_B,
                                   const Array<double> &_G,
                                   const Vector &_x,
                                   Vector &_y)
{
   auto B = Reshape(_B.Read(), c_dofs1D, c_dofs1D);
   auto G = Reshape(_G.Read(), o_dofs1D, c_dofs1D);
   // const int ned_dofs_per_elem = (2 * o_dofs1D * c_dofs1D);

   auto x = Reshape(_x.Read(), c_dofs1D, c_dofs1D, NE);
   auto y = Reshape(_y.ReadWrite(), 2 * c_dofs1D * o_dofs1D, NE);

   Vector hwork(c_dofs1D * c_dofs1D);
   auto hw = Reshape(hwork.ReadWrite(), c_dofs1D, c_dofs1D);

   Vector vwork(c_dofs1D * o_dofs1D);
   auto vw = Reshape(vwork.ReadWrite(), c_dofs1D, o_dofs1D);
   MFEM_FORALL(e, NE,
   {
      // horizontal part
      for (int dx = 0; dx < c_dofs1D; ++dx)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            hw(dx, ey) = 0.0;
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               hw(dx, ey) += B(ey, dy) * x(dx, dy, e);
            }
         }
      }

      for (int ey = 0; ey < c_dofs1D; ++ey)
      {
         for (int ex = 0; ex < o_dofs1D; ++ex)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               const int local_index = ey*o_dofs1D + ex;
               y(local_index, e) += G(ex, dx) * hw(dx, ey);
            }
         }
      }

      // vertical part
      for (int dx = 0; dx < c_dofs1D; ++dx)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            vw(dx, ey) = 0.0;
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               vw(dx, ey) += G(ey, dy) * x(dx, dy, e);
            }
         }
      }

      for (int ey = 0; ey < o_dofs1D; ++ey)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               const int local_index = c_dofs1D * o_dofs1D + ey*c_dofs1D + ex;
               y(local_index, e) += B(ex, dx) * vw(dx, ey);
            }
         }
      }
   });

}

static void PAHcurlApplyGradientTranspose2D(
   const int c_dofs1D, const int o_dofs1D, const int NE,
   const Array<double> &_B, const Array<double> &_G,
   const Vector &_x, Vector &_y)
{
   auto B = Reshape(_B.Read(), c_dofs1D, c_dofs1D);
   auto G = Reshape(_G.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(_x.Read(), 2 * c_dofs1D * o_dofs1D, NE);
   auto y = Reshape(_y.ReadWrite(), c_dofs1D, c_dofs1D, NE);

   Vector hwork(c_dofs1D * o_dofs1D);
   auto hw = Reshape(hwork.ReadWrite(), c_dofs1D, o_dofs1D);

   Vector vwork(c_dofs1D * c_dofs1D);
   auto vw = Reshape(vwork.ReadWrite(), c_dofs1D, c_dofs1D);

   MFEM_FORALL(e, NE,
   {
      // horizontal part (open x, closed y)
      for (int dy = 0; dy < c_dofs1D; ++dy)
      {
         for (int ex = 0; ex < o_dofs1D; ++ex)
         {
            hw(dy, ex) = 0.0;
            for (int ey = 0; ey < c_dofs1D; ++ey)
            {
               const int local_index = ey*o_dofs1D + ex;
               hw(dy, ex) += B(ey, dy) * x(local_index, e);
            }
         }
      }

      for (int dy = 0; dy < c_dofs1D; ++dy)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int ex = 0; ex < o_dofs1D; ++ex)
            {
               y(dx, dy, e) += G(ex, dx) * hw(dy, ex);
            }
         }
      }

      // vertical part (open y, closed x)
      for (int dy = 0; dy < c_dofs1D; ++dy)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            vw(dy, ex) = 0.0;
            for (int ey = 0; ey < o_dofs1D; ++ey)
            {
               const int local_index = c_dofs1D * o_dofs1D + ey*c_dofs1D + ex;
               vw(dy, ex) += G(ey, dy) * x(local_index, e);
            }
         }
      }

      for (int dy = 0; dy < c_dofs1D; ++dy)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               y(dx, dy, e) += B(ex, dx) * vw(dy, ex);
            }
         }
      }
   });
}

static void PAHcurlApplyGradient3D(const int c_dofs1D,
                                   const int o_dofs1D,
                                   const int NE,
                                   const Array<double> &_B,
                                   const Array<double> &_G,
                                   const Vector &_x,
                                   Vector &_y)
{
   auto B = Reshape(_B.Read(), c_dofs1D, c_dofs1D);
   auto G = Reshape(_G.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(_x.Read(), c_dofs1D, c_dofs1D, c_dofs1D, NE);
   auto y = Reshape(_y.ReadWrite(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);

   Vector pxwork1(c_dofs1D * c_dofs1D * c_dofs1D);
   auto pxw1 = Reshape(pxwork1.ReadWrite(), c_dofs1D, c_dofs1D, c_dofs1D);
   Vector pxwork2(c_dofs1D * c_dofs1D * c_dofs1D);
   auto pxw2 = Reshape(pxwork2.ReadWrite(), c_dofs1D, c_dofs1D, c_dofs1D);

   Vector pywork1(c_dofs1D * c_dofs1D * c_dofs1D);
   auto pyw1 = Reshape(pywork1.ReadWrite(), c_dofs1D, c_dofs1D, c_dofs1D);
   Vector pywork2(c_dofs1D * o_dofs1D * c_dofs1D);
   auto pyw2 = Reshape(pywork2.ReadWrite(), c_dofs1D, o_dofs1D, c_dofs1D);

   Vector pzwork1(c_dofs1D * c_dofs1D * o_dofs1D);
   auto pzw1 = Reshape(pzwork1.ReadWrite(), c_dofs1D, c_dofs1D, o_dofs1D);
   Vector pzwork2(c_dofs1D * c_dofs1D * o_dofs1D);
   auto pzw2 = Reshape(pzwork2.ReadWrite(), c_dofs1D, c_dofs1D, o_dofs1D);

   MFEM_FORALL(e, NE,
   {
      // ---
      // dofs that point parallel to x-axis (open in x, closed in y, z)
      // ---

      // contract in z
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               pxw1(dx, dy, ez) = 0.0;
               for (int dz = 0; dz < c_dofs1D; ++dz)
               {
                  pxw1(dx, dy, ez) += B(ez, dz) * x(dx, dy, dz, e);
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               pxw2(dx, ey, ez) = 0.0;
               for (int dy = 0; dy < c_dofs1D; ++dy)
               {
                  pxw2(dx, ey, ez) += B(ey, dy) * pxw1(dx, dy, ez);
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int ex = 0; ex < o_dofs1D; ++ex)
            {
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  const int local_index = ez*c_dofs1D*o_dofs1D + ey*o_dofs1D + ex;
                  y(local_index, e) += G(ex, dx) * pxw2(dx, ey, ez);
               }
            }
         }
      }


      // ---
      // dofs that point parallel to y-axis (open in y, closed in x, z)
      // ---

      // contract in z
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               pyw1(dx, dy, ez) = 0.0;
               for (int dz = 0; dz < c_dofs1D; ++dz)
               {
                  pyw1(dx, dy, ez) += B(ez, dz) * x(dx, dy, dz, e);
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               pyw2(dx, ey, ez) = 0.0;
               for (int dy = 0; dy < c_dofs1D; ++dy)
               {
                  pyw2(dx, ey, ez) += G(ey, dy) * pyw1(dx, dy, ez);
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  const int local_index = c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*o_dofs1D + ey*c_dofs1D + ex;
                  y(local_index, e) += B(ex, dx) * pyw2(dx, ey, ez);
               }
            }
         }
      }

      // ---
      // dofs that point parallel to z-axis (open in z, closed in x, y)
      // ---

      // contract in z
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               pzw1(dx, dy, ez) = 0.0;
               for (int dz = 0; dz < c_dofs1D; ++dz)
               {
                  pzw1(dx, dy, ez) += G(ez, dz) * x(dx, dy, dz, e);
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               pzw2(dx, ey, ez) = 0.0;
               for (int dy = 0; dy < c_dofs1D; ++dy)
               {
                  pzw2(dx, ey, ez) += B(ey, dy) * pzw1(dx, dy, ez);
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  const int local_index = 2*c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*c_dofs1D + ey*c_dofs1D + ex;
                  y(local_index, e) += B(ex, dx) * pzw2(dx, ey, ez);
               }
            }
         }
      }

   });
}

static void PAHcurlApplyGradientTranspose3D(
   const int c_dofs1D, const int o_dofs1D, const int NE,
   const Array<double> &_B, const Array<double> &_G,
   const Vector &_x, Vector &_y)
{
   auto B = Reshape(_B.Read(), c_dofs1D, c_dofs1D);
   auto G = Reshape(_G.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(_x.Read(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);
   auto y = Reshape(_y.ReadWrite(), c_dofs1D, c_dofs1D, c_dofs1D, NE);

   Vector pxwork1(o_dofs1D * c_dofs1D * c_dofs1D);
   auto pxw1 = Reshape(pxwork1.ReadWrite(), o_dofs1D, c_dofs1D, c_dofs1D);
   Vector pxwork2(o_dofs1D * c_dofs1D * c_dofs1D);
   auto pxw2 = Reshape(pxwork2.ReadWrite(), o_dofs1D, c_dofs1D, c_dofs1D);

   Vector pywork1(c_dofs1D * o_dofs1D * c_dofs1D);
   auto pyw1 = Reshape(pywork1.ReadWrite(), c_dofs1D, o_dofs1D, c_dofs1D);
   Vector pywork2(c_dofs1D * c_dofs1D * c_dofs1D);
   auto pyw2 = Reshape(pywork2.ReadWrite(), c_dofs1D, c_dofs1D, c_dofs1D);

   Vector pzwork1(c_dofs1D * c_dofs1D * c_dofs1D);
   auto pzw1 = Reshape(pzwork1.ReadWrite(), c_dofs1D, c_dofs1D, c_dofs1D);
   Vector pzwork2(c_dofs1D * c_dofs1D * c_dofs1D);
   auto pzw2 = Reshape(pzwork2.ReadWrite(), c_dofs1D, c_dofs1D, c_dofs1D);

   MFEM_FORALL(e, NE,
   {
      // ---
      // dofs that point parallel to x-axis (open in x, closed in y, z)
      // ---

      // contract in z
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int ex = 0; ex < o_dofs1D; ++ex)
         {
            for (int ey = 0; ey < c_dofs1D; ++ey)
            {
               pxw1(ex, ey, dz) = 0.0;
               for (int ez = 0; ez < c_dofs1D; ++ez)
               {
                  const int local_index = ez*c_dofs1D*o_dofs1D + ey*o_dofs1D + ex;
                  pxw1(ex, ey, dz) += B(ez, dz) * x(local_index, e);
               }
            }
         }
      }

      // contract in y
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int ex = 0; ex < o_dofs1D; ++ex)
            {
               pxw2(ex, dy, dz) = 0.0;
               for (int ey = 0; ey < c_dofs1D; ++ey)
               {
                  pxw2(ex, dy, dz) += B(ey, dy) * pxw1(ex, ey, dz);
               }
            }
         }
      }

      // contract in x
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               for (int ex = 0; ex < o_dofs1D; ++ex)
               {
                  y(dx, dy, dz, e) += G(ex, dx) * pxw2(ex, dy, dz);
               }
            }
         }
      }

      // ---
      // dofs that point parallel to y-axis (open in y, closed in x, z)
      // ---

      // contract in z
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            for (int ey = 0; ey < o_dofs1D; ++ey)
            {
               pyw1(ex, ey, dz) = 0.0;
               for (int ez = 0; ez < c_dofs1D; ++ez)
               {
                  const int local_index = c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*o_dofs1D + ey*c_dofs1D + ex;
                  pyw1(ex, ey, dz) += B(ez, dz) * x(local_index, e);
               }
            }
         }
      }

      // contract in y
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               pyw2(ex, dy, dz) = 0.0;
               for (int ey = 0; ey < o_dofs1D; ++ey)
               {
                  pyw2(ex, dy, dz) += G(ey, dy) * pyw1(ex, ey, dz);
               }
            }
         }
      }

      // contract in x
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               for (int ex = 0; ex < c_dofs1D; ++ex)
               {
                  y(dx, dy, dz, e) += B(ex, dx) * pyw2(ex, dy, dz);
               }
            }
         }
      }

      // ---
      // dofs that point parallel to z-axis (open in z, closed in x, y)
      // ---

      // contract in z
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int ex = 0; ex < c_dofs1D; ++ex)
         {
            for (int ey = 0; ey < c_dofs1D; ++ey)
            {
               pzw1(ex, ey, dz) = 0.0;
               for (int ez = 0; ez < o_dofs1D; ++ez)
               {
                  const int local_index = 2*c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*c_dofs1D + ey*c_dofs1D + ex;
                  pzw1(ex, ey, dz) += G(ez, dz) * x(local_index, e);
               }
            }
         }
      }

      // contract in y
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               pzw2(ex, dy, dz) = 0.0;
               for (int ey = 0; ey < c_dofs1D; ++ey)
               {
                  pzw2(ex, dy, dz) += B(ey, dy) * pzw1(ex, ey, dz);
               }
            }
         }
      }

      // contract in x
      for (int dz = 0; dz < c_dofs1D; ++dz)
      {
         for (int dy = 0; dy < c_dofs1D; ++dy)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               for (int ex = 0; ex < c_dofs1D; ++ex)
               {
                  y(dx, dy, dz, e) += B(ex, dx) * pzw2(ex, dy, dz);
               }
            }
         }
      }
   });
}

// Apply to x corresponding to DOF's in H^1 (trial), whose gradients are
// integrated against H(curl) test functions corresponding to y.
void PAHcurlH1Apply2D(const int D1D,
                      const int Q1D,
                      const int NE,
                      const Array<double> &_Bc,
                      const Array<double> &_Gc,
                      const Array<double> &_Bot,
                      const Array<double> &_Bct,
                      const Vector &_op,
                      const Vector &_x,
                      Vector &_y)
{
   constexpr static int VDIM = 2;
   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 3, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][VDIM];

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               mass[qy][qx][c] = 0.0;
            }
         }
      }

      for (int dy = 0; dy < D1D; ++dy)
      {
         double gradX[MAX_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = x(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] += s * Bc(qx,dx);
               gradX[qx][1] += s * Gc(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy  = Bc(qy,dy);
            const double wDy = Gc(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wx  = gradX[qx][0];
               const double wDx = gradX[qx][1];
               mass[qy][qx][0] += wDx * wy;
               mass[qy][qx][1] += wx * wDy;
            }
         }
      }

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double O11 = op(qx,qy,0,e);
            const double O12 = op(qx,qy,1,e);
            const double O22 = op(qx,qy,2,e);
            const double massX = mass[qy][qx][0];
            const double massY = mass[qy][qx][1];
            mass[qy][qx][0] = (O11*massX)+(O12*massY);
            mass[qy][qx][1] = (O12*massX)+(O22*massY);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         int osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            double massX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               massX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] += mass[qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }
   }); // end of element loop
}

// PA H(curl) Mass Assemble 3D kernel
static void PAHcurlL2Setup3D(const int Q1D,
                             const int coeffDim,
                             const int NE,
                             const Array<double> &w,
                             Vector &_coeff,
                             Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto coeff = Reshape(_coeff.Read(), coeffDim, NQ, NE);
   auto y = Reshape(op.Write(), coeffDim, NQ, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         for (int c=0; c<coeffDim; ++c)
         {
            y(c,q,e) = W[q] * coeff(c,q,e);
         }
      }
   });
}

void MixedVectorCurlIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
                                           const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with vector test and trial spaces.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &trial_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &trial_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   mapsCtest = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsOtest = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;
   dofs1Dtest = mapsCtest->ndof;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   testType = test_el->GetDerivType();
   trialType = trial_el->GetDerivType();

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   coeffDim = (testType == mfem::FiniteElement::DIV) ? symmDims : (DQ ? 3 : 1);

   pa_data.SetSize(coeffDim * nq * ne, Device::GetMemoryType());

   Vector coeff(coeffDim * nq * ne);
   coeff = 1.0;
   auto coeffh = Reshape(coeff.HostWrite(), coeffDim, nq, ne);
   if (Q || DQ)
   {
      Vector V(coeffDim);
      if (DQ)
      {
         MFEM_VERIFY(DQ->GetVDim() == coeffDim, "");
      }

      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);

         for (int p=0; p<nq; ++p)
         {
            if (DQ)
            {
               DQ->Eval(V, *tr, ir->IntPoint(p));
               for (int i=0; i<coeffDim; ++i)
               {
                  coeffh(i, p, e) = V[i];
               }
            }
            else
            {
               coeffh(0, p, e) = Q->Eval(*tr, ir->IntPoint(p));
            }
         }
      }
   }

   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      PAHcurlL2Setup3D(quad1D, coeffDim, ne, ir->GetWeights(), coeff, pa_data);
   }
   else if (testType == mfem::FiniteElement::DIV &&
            trialType == mfem::FiniteElement::CURL && dim == 3 &&
            test_fel->GetOrder() == trial_fel->GetOrder())
   {
      PACurlCurlSetup3D(quad1D, coeffDim, ne, ir->GetWeights(), geom->J, coeff,
                        pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

// Apply to x corresponding to DOF's in H(curl) (trial), whose curl is
// integrated against H(curl) test functions corresponding to y.
template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PAHcurlL2Apply3D(const int D1D,
                             const int Q1D,
                             const int coeffDim,
                             const int NE,
                             const Array<double> &_Bo,
                             const Array<double> &_Bc,
                             const Array<double> &_Bot,
                             const Array<double> &_Bct,
                             const Array<double> &_Gc,
                             const Vector &_op,
                             const Vector &_x,
                             Vector &_y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   // Using u = dF^{-T} \hat{u} and (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
   // (\nabla\times u) \cdot v = 1/det(dF) \hat{\nabla}\times\hat{u}^T dF^T dF^{-T} \hat{v}
   // = 1/det(dF) \hat{\nabla}\times\hat{u}^T \hat{v}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double curl[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];
      // curl[qz][qy][qx] will be computed as the vector curl at each quadrature point.

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  curl[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      // We treat x, y, z components separately for optimization specific to each.

      int osc = 0;

      {
         // x component
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     curl[qz][qy][qx][1] += gradXY[qy][qx][1] * wDz; // (u_0)_{x_2}
                     curl[qz][qy][qx][2] -= gradXY[qy][qx][0] * wz;  // -(u_0)_{x_1}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // y component
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               double massY[MAX_Q1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = Bc(qx,dx);
                  const double wDx = Gc(qx,dx);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     curl[qz][qy][qx][0] -= gradXY[qy][qx][1] * wDz; // -(u_1)_{x_2}
                     curl[qz][qy][qx][2] += gradXY[qy][qx][0] * wz;  // (u_1)_{x_0}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // z component
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int dx = 0; dx < D1Dx; ++dx)
         {
            double gradYZ[MAX_Q1D][MAX_Q1D][2];
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradYZ[qz][qy][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massZ[MAX_Q1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wx = Bc(qx,dx);
               const double wDx = Gc(qx,dx);

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     curl[qz][qy][qx][0] += gradYZ[qz][qy][1] * wx;  // (u_2)_{x_1}
                     curl[qz][qy][qx][1] -= gradYZ[qz][qy][0] * wDx; // -(u_2)_{x_0}
                  }
               }
            }
         }
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  curl[qz][qy][qx][c] *= op(coeffDim == 3 ? c : 0, qx,qy,qz,e);
               }
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[MAX_D1D][MAX_D1D];

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += curl[qz][qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bot(dz,qz) : Bct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

// Apply to x corresponding to DOF's in H(curl) (trial), whose curl is
// integrated against H(div) test functions corresponding to y.
template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PAHcurlHdivApply3D(const int D1D,
                               const int D1Dtest,
                               const int Q1D,
                               const int NE,
                               const Array<double> &_Bo,
                               const Array<double> &_Bc,
                               const Array<double> &_Bot,
                               const Array<double> &_Bct,
                               const Array<double> &_Gc,
                               const Vector &_op,
                               const Vector &_x,
                               Vector &_y)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");
   // Using Piola transformations (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u}
   // for u in H(curl) and w = (1 / det (dF)) dF \hat{w} for w in H(div), we get
   // (\nabla\times u) \cdot w = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{w}
   // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
   // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
   // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1Dtest-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1Dtest, Q1D);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto x = Reshape(_x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1Dtest-1)*(D1Dtest-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double curl[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];
      // curl[qz][qy][qx] will be computed as the vector curl at each quadrature point.

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  curl[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      // We treat x, y, z components separately for optimization specific to each.

      int osc = 0;

      {
         // x component
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * Bo(qx,dx);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     gradXY[qy][qx][0] += wx * wDy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     curl[qz][qy][qx][1] += gradXY[qy][qx][1] * wDz; // (u_0)_{x_2}
                     curl[qz][qy][qx][2] -= gradXY[qy][qx][0] * wz;  // -(u_0)_{x_1}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // y component
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][2];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               double massY[MAX_Q1D];
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  massY[qy] = 0.0;
               }

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     massY[qy] += t * Bo(qy,dy);
                  }
               }

               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = Bc(qx,dx);
                  const double wDx = Gc(qx,dx);
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = massY[qy];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = Bc(qz,dz);
               const double wDz = Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     curl[qz][qy][qx][0] -= gradXY[qy][qx][1] * wDz; // -(u_1)_{x_2}
                     curl[qz][qy][qx][2] += gradXY[qy][qx][0] * wz;  // (u_1)_{x_0}
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }

      {
         // z component
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int dx = 0; dx < D1Dx; ++dx)
         {
            double gradYZ[MAX_Q1D][MAX_Q1D][2];
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradYZ[qz][qy][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massZ[MAX_Q1D];
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  massZ[qz] = 0.0;
               }

               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     massZ[qz] += t * Bo(qz,dz);
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = Bc(qy,dy);
                  const double wDy = Gc(qy,dy);
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double wz = massZ[qz];
                     gradYZ[qz][qy][0] += wz * wy;
                     gradYZ[qz][qy][1] += wz * wDy;
                  }
               }
            }

            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wx = Bc(qx,dx);
               const double wDx = Gc(qx,dx);

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     curl[qz][qy][qx][0] += gradYZ[qz][qy][1] * wx;  // (u_2)_{x_1}
                     curl[qz][qy][qx][1] -= gradYZ[qz][qy][0] * wDx; // -(u_2)_{x_0}
                  }
               }
            }
         }
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O22 = op(qx,qy,qz,3,e);
               const double O23 = op(qx,qy,qz,4,e);
               const double O33 = op(qx,qy,qz,5,e);

               const double c1 = (O11 * curl[qz][qy][qx][0]) + (O12 * curl[qz][qy][qx][1]) +
                                 (O13 * curl[qz][qy][qx][2]);
               const double c2 = (O12 * curl[qz][qy][qx][0]) + (O22 * curl[qz][qy][qx][1]) +
                                 (O23 * curl[qz][qy][qx][2]);
               const double c3 = (O13 * curl[qz][qy][qx][0]) + (O23 * curl[qz][qy][qx][1]) +
                                 (O33 * curl[qz][qy][qx][2]);

               curl[qz][qy][qx][0] = c1;
               curl[qz][qy][qx][1] = c2;
               curl[qz][qy][qx][2] = c3;
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[HCURL_MAX_D1D][HCURL_MAX_D1D];  // Assuming HDIV_MAX_D1D <= HCURL_MAX_D1D

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1Dtest : D1Dtest - 1;
            const int D1Dy = (c == 1) ? D1Dtest : D1Dtest - 1;
            const int D1Dx = (c == 0) ? D1Dtest : D1Dtest - 1;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[HCURL_MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += curl[qz][qy][qx][c] *
                                  ((c == 0) ? Bct(dx,qx) : Bot(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = (c == 1) ? Bct(dy,qy) : Bot(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bct(dz,qz) : Bot(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) +=
                        massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

void MixedVectorCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
      PAHcurlL2Apply3D(dofs1D, quad1D, coeffDim, ne, mapsO->B, mapsC->B,
                       mapsO->Bt, mapsC->Bt, mapsC->G, pa_data, x, y);
   else if (testType == mfem::FiniteElement::DIV &&
            trialType == mfem::FiniteElement::CURL && dim == 3)
      PAHcurlHdivApply3D(dofs1D, dofs1Dtest, quad1D, ne, mapsO->B,
                         mapsC->B, mapsOtest->Bt, mapsCtest->Bt, mapsC->G,
                         pa_data, x, y);
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

void MixedVectorWeakCurlIntegrator::AssemblePA(const FiniteElementSpace
                                               &trial_fes,
                                               const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with vector test and trial spaces.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

   const VectorTensorFiniteElement *trial_el =
      dynamic_cast<const VectorTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   coeffDim = DQ ? 3 : 1;

   pa_data.SetSize(coeffDim * nq * ne, Device::GetMemoryType());

   Vector coeff(coeffDim * nq * ne);
   coeff = 1.0;
   auto coeffh = Reshape(coeff.HostWrite(), coeffDim, nq, ne);
   if (Q || DQ)
   {
      Vector V(coeffDim);
      if (DQ)
      {
         MFEM_VERIFY(DQ->GetVDim() == coeffDim, "");
      }

      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);

         for (int p=0; p<nq; ++p)
         {
            if (DQ)
            {
               DQ->Eval(V, *tr, ir->IntPoint(p));
               for (int i=0; i<coeffDim; ++i)
               {
                  coeffh(i, p, e) = V[i];
               }
            }
            else
            {
               coeffh(0, p, e) = Q->Eval(*tr, ir->IntPoint(p));
            }
         }
      }
   }

   testType = test_el->GetDerivType();
   trialType = trial_el->GetDerivType();

   if (trialType == mfem::FiniteElement::CURL && dim == 3)
   {
      PAHcurlL2Setup3D(quad1D, coeffDim, ne, ir->GetWeights(), coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

// Apply to x corresponding to DOF's in H(curl) (trial), integrated against curl
// of H(curl) test functions corresponding to y.
template<int MAX_D1D = HCURL_MAX_D1D, int MAX_Q1D = HCURL_MAX_Q1D>
static void PAHcurlL2Apply3DTranspose(const int D1D,
                                      const int Q1D,
                                      const int coeffDim,
                                      const int NE,
                                      const Array<double> &_Bo,
                                      const Array<double> &_Bc,
                                      const Array<double> &_Bot,
                                      const Array<double> &_Bct,
                                      const Array<double> &_Gct,
                                      const Vector &_op,
                                      const Vector &_x,
                                      Vector &_y)
{
   // See PAHcurlL2Apply3D for comments.

   MFEM_VERIFY(D1D <= MAX_D1D, "Error: D1D > MAX_D1D");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "Error: Q1D > MAX_Q1D");

   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto Gct = Reshape(_Gct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), coeffDim, Q1D, Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  mass[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double massXY[MAX_Q1D][MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     mass[qz][qy][qx][c] += massXY[qy][qx] * wz;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop (c) over components

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c=0; c<VDIM; ++c)
               {
                  mass[qz][qy][qx][c] *= op(coeffDim == 3 ? c : 0, qx,qy,qz,e);
               }
            }
         }
      }

      // x component
      osc = 0;
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D;
         const int D1Dx = D1D - 1;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            double gradXY12[MAX_D1D][MAX_D1D];
            double gradXY21[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY12[dy][dx] = 0.0;
                  gradXY21[dy][dx] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[MAX_D1D][2];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massX[dx][n] = 0.0;
                  }
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     const double wx = Bot(dx,qx);

                     massX[dx][0] += wx * mass[qz][qy][qx][1];
                     massX[dx][1] += wx * mass[qz][qy][qx][2];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     gradXY21[dy][dx] += massX[dx][0] * wy;
                     gradXY12[dy][dx] += massX[dx][1] * wDy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     // (u_0)_{x_2} * (op * curl)_1 - (u_0)_{x_1} * (op * curl)_2
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (gradXY21[dy][dx] * wDz) - (gradXY12[dy][dx] * wz);
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // y component
      {
         const int D1Dz = D1D;
         const int D1Dy = D1D - 1;
         const int D1Dx = D1D;

         for (int qz = 0; qz < Q1D; ++qz)
         {
            double gradXY02[MAX_D1D][MAX_D1D];
            double gradXY20[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradXY02[dy][dx] = 0.0;
                  gradXY20[dy][dx] = 0.0;
               }
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               double massY[MAX_D1D][2];
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  massY[dy][0] = 0.0;
                  massY[dy][1] = 0.0;
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     const double wy = Bot(dy,qy);

                     massY[dy][0] += wy * mass[qz][qy][qx][2];
                     massY[dy][1] += wy * mass[qz][qy][qx][0];
                  }
               }
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double wx = Bct(dx,qx);
                  const double wDx = Gct(dx,qx);

                  for (int dy = 0; dy < D1Dy; ++dy)
                  {
                     gradXY02[dy][dx] += massY[dy][0] * wDx;
                     gradXY20[dy][dx] += massY[dy][1] * wx;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = Bct(dz,qz);
               const double wDz = Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // \hat{\nabla}\times\hat{u} is [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     // -(u_1)_{x_2} * (op * curl)_0 + (u_1)_{x_0} * (op * curl)_2
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (-gradXY20[dy][dx] * wDz) + (gradXY02[dy][dx] * wz);
                  }
               }
            }
         }  // loop qz

         osc += D1Dx * D1Dy * D1Dz;
      }

      // z component
      {
         const int D1Dz = D1D - 1;
         const int D1Dy = D1D;
         const int D1Dx = D1D;

         for (int qx = 0; qx < Q1D; ++qx)
         {
            double gradYZ01[MAX_D1D][MAX_D1D];
            double gradYZ10[MAX_D1D][MAX_D1D];

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  gradYZ01[dz][dy] = 0.0;
                  gradYZ10[dz][dy] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massZ[MAX_D1D][2];
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     massZ[dz][n] = 0.0;
                  }
               }
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     const double wz = Bot(dz,qz);

                     massZ[dz][0] += wz * mass[qz][qy][qx][0];
                     massZ[dz][1] += wz * mass[qz][qy][qx][1];
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = Bct(dy,qy);
                  const double wDy = Gct(dy,qy);

                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     gradYZ01[dz][dy] += wy * massZ[dz][1];
                     gradYZ10[dz][dy] += wDy * massZ[dz][0];
                  }
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double wx = Bct(dx,qx);
               const double wDx = Gct(dx,qx);

               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dz = 0; dz < D1Dz; ++dz)
                  {
                     // \hat{\nabla}\times\hat{u} is [(u_2)_{x_1}, -(u_2)_{x_0}, 0]
                     // (u_2)_{x_1} * (op * curl)_0 - (u_2)_{x_0} * (op * curl)_1
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc,
                       e) += (gradYZ10[dz][dy] * wx) - (gradYZ01[dz][dy] * wDx);
                  }
               }
            }
         }  // loop qx
      }
   });
}

void MixedVectorWeakCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (testType == mfem::FiniteElement::CURL &&
       trialType == mfem::FiniteElement::CURL && dim == 3)
      PAHcurlL2Apply3DTranspose(dofs1D, quad1D, coeffDim, ne, mapsO->B, mapsC->B,
                                mapsO->Bt, mapsC->Bt, mapsC->Gt, pa_data, x, y);
   else
   {
      MFEM_ABORT("Unsupported dimension or space!");
   }
}

void GradientInterpolator::AssemblePA(const FiniteElementSpace &trial_fes,
                                      const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with a vector test space and H^1 trial space.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

   const NodalTensorFiniteElement *trial_el =
      dynamic_cast<const NodalTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only NodalTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *old_ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*old_ir, GeometricFactors::JACOBIANS);

   const int order = trial_el->GetOrder();
   fake_fe = new H1_SegmentElement(order);
   mfem::QuadratureFunctions1D qf1d;
   mfem::IntegrationRule closed_ir;
   closed_ir.SetSize(order + 1);
   qf1d.GaussLobatto(order + 1, &closed_ir);
   mfem::IntegrationRule open_ir;
   open_ir.SetSize(order);
   qf1d.GaussLegendre(order, &open_ir);

   maps_C_C = &fake_fe->GetDofToQuad(closed_ir, DofToQuad::TENSOR);
   maps_O_C = &fake_fe->GetDofToQuad(open_ir, DofToQuad::TENSOR);

   o_dofs1D = maps_O_C->nqpt;
   c_dofs1D = maps_C_C->nqpt;
   MFEM_VERIFY(maps_O_C->ndof == c_dofs1D, "Bad programming!");
   MFEM_VERIFY(maps_C_C->ndof == c_dofs1D, "Bad programming!");
}

void GradientInterpolator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      PAHcurlApplyGradient3D(c_dofs1D, o_dofs1D, ne, maps_C_C->B, maps_O_C->G,
                             x, y);
   }
   else if (dim == 2)
   {
      PAHcurlApplyGradient2D(c_dofs1D, o_dofs1D, ne, maps_C_C->B, maps_O_C->G,
                             x, y);
   }
   else
   {
      mfem_error("Bad dimension!");
   }
}

void GradientInterpolator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      PAHcurlApplyGradientTranspose3D(c_dofs1D, o_dofs1D, ne, maps_C_C->B,
                                      maps_O_C->G, x, y);
   }
   else if (dim == 2)
   {
      PAHcurlApplyGradientTranspose2D(c_dofs1D, o_dofs1D, ne, maps_C_C->B,
                                      maps_O_C->G, x, y);
   }
   else
   {
      mfem_error("Bad dimension!");
   }
}

static void PAHcurlVecH1IdentityApply3D(const int c_dofs1D,
                                        const int o_dofs1D,
                                        const int NE,
                                        const Array<double> &Bclosed,
                                        const Array<double> &Bopen,
                                        const Vector &pa_data,
                                        const Vector &_x,
                                        Vector &_y)
{
   constexpr int VDIM = 3;

   auto Bc = Reshape(Bclosed.Read(), c_dofs1D, c_dofs1D);
   auto Bo = Reshape(Bopen.Read(), o_dofs1D, c_dofs1D);

   auto x = Reshape(_x.Read(), c_dofs1D, c_dofs1D, c_dofs1D, VDIM, NE);
   auto y = Reshape(_y.ReadWrite(), (3 * c_dofs1D * c_dofs1D * o_dofs1D), NE);

   auto vk = Reshape(pa_data.Read(), VDIM, (3 * c_dofs1D * c_dofs1D * o_dofs1D),
                     NE);

   constexpr static int MAX_D1D = HCURL_MAX_D1D;
   //constexpr static int MAX_Q1D = HCURL_MAX_Q1D;

   MFEM_VERIFY(c_dofs1D <= MAX_D1D, "");

   MFEM_FORALL(e, NE,
   {
      double pxw1[MAX_D1D][MAX_D1D][MAX_D1D];
      double pxw2[MAX_D1D][MAX_D1D][MAX_D1D];

      double pyw1[MAX_D1D][MAX_D1D][MAX_D1D];
      double pyw2[MAX_D1D][MAX_D1D][MAX_D1D];

      double pzw1[MAX_D1D][MAX_D1D][MAX_D1D];
      double pzw2[MAX_D1D][MAX_D1D][MAX_D1D];

      // ---
      // dofs that point parallel to x-axis (open in x, closed in y, z)
      // ---

      // contract in z
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               pxw1[dx][dy][ez] = 0.0;
               for (int dz = 0; dz < c_dofs1D; ++dz)
               {
                  pxw1[dx][dy][ez] += Bc(ez, dz) * x(dx, dy, dz, 0, e);
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               pxw2[dx][ey][ez] = 0.0;
               for (int dy = 0; dy < c_dofs1D; ++dy)
               {
                  pxw2[dx][ey][ez] += Bc(ey, dy) * pxw1[dx][dy][ez];
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int ex = 0; ex < o_dofs1D; ++ex)
            {
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  const int local_index = ez*c_dofs1D*o_dofs1D + ey*o_dofs1D + ex;
                  y(local_index, e) += vk(0, local_index, e) * Bo(ex, dx) * pxw2[dx][ey][ez];
               }
            }
         }
      }

      // ---
      // dofs that point parallel to y-axis (open in y, closed in x, z)
      // ---

      // contract in z
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               pyw1[dx][dy][ez] = 0.0;
               for (int dz = 0; dz < c_dofs1D; ++dz)
               {
                  pyw1[dx][dy][ez] += Bc(ez, dz) * x(dx, dy, dz, 1, e);
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               pyw2[dx][ey][ez] = 0.0;
               for (int dy = 0; dy < c_dofs1D; ++dy)
               {
                  pyw2[dx][ey][ez] += Bo(ey, dy) * pyw1[dx][dy][ez];
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < c_dofs1D; ++ez)
      {
         for (int ey = 0; ey < o_dofs1D; ++ey)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  const int local_index = c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*o_dofs1D + ey*c_dofs1D + ex;
                  y(local_index, e) += vk(1, local_index, e) * Bc(ex, dx) * pyw2[dx][ey][ez];
               }
            }
         }
      }

      // ---
      // dofs that point parallel to z-axis (open in z, closed in x, y)
      // ---

      // contract in z
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int dx = 0; dx < c_dofs1D; ++dx)
         {
            for (int dy = 0; dy < c_dofs1D; ++dy)
            {
               pzw1[dx][dy][ez] = 0.0;
               for (int dz = 0; dz < c_dofs1D; ++dz)
               {
                  pzw1[dx][dy][ez] += Bo(ez, dz) * x(dx, dy, dz, 2, e);
               }
            }
         }
      }

      // contract in y
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int dx = 0; dx < c_dofs1D; ++dx)
            {
               pzw2[dx][ey][ez] = 0.0;
               for (int dy = 0; dy < c_dofs1D; ++dy)
               {
                  pzw2[dx][ey][ez] += Bc(ey, dy) * pzw1[dx][dy][ez];
               }
            }
         }
      }

      // contract in x
      for (int ez = 0; ez < o_dofs1D; ++ez)
      {
         for (int ey = 0; ey < c_dofs1D; ++ey)
         {
            for (int ex = 0; ex < c_dofs1D; ++ex)
            {
               for (int dx = 0; dx < c_dofs1D; ++dx)
               {
                  const int local_index = 2*c_dofs1D*c_dofs1D*o_dofs1D +
                                          ez*c_dofs1D*c_dofs1D + ey*c_dofs1D + ex;
                  y(local_index, e) += vk(2, local_index, e) * Bc(ex, dx) * pzw2[dx][ey][ez];
               }
            }
         }
      }

   });
}

static void PAHcurlVecH1IdentityApply2D(const int c_dofs1D,
                                        const int o_dofs1D,
                                        const int NE,
                                        const Array<double> &_B,
                                        const Array<double> &_G,
                                        const Vector &_x,
                                        Vector &_y)
{
   mfem_error("TODO");
}

void IdentityInterpolator::AssemblePA(const FiniteElementSpace &trial_fes,
                                      const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with a vector test space and H^1 trial space.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

   const NodalTensorFiniteElement *trial_el =
      dynamic_cast<const NodalTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only NodalTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *old_ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   // TODO: GetGeom... seems to corrupt pa_data memory?

   ne = trial_fes.GetNE();
   //geom = mesh->GetGeometricFactors(*old_ir, GeometricFactors::JACOBIANS);

   const int order = trial_el->GetOrder();
   fake_fe = new H1_SegmentElement(order);
   mfem::QuadratureFunctions1D qf1d;
   mfem::IntegrationRule closed_ir;
   closed_ir.SetSize(order + 1);
   qf1d.GaussLobatto(order + 1, &closed_ir);
   mfem::IntegrationRule open_ir;
   open_ir.SetSize(order);
   qf1d.GaussLegendre(order, &open_ir);

   maps_C_C = &fake_fe->GetDofToQuad(closed_ir, DofToQuad::TENSOR);
   maps_O_C = &fake_fe->GetDofToQuad(open_ir, DofToQuad::TENSOR);

   o_dofs1D = maps_O_C->nqpt;
   c_dofs1D = maps_C_C->nqpt;
   MFEM_VERIFY(maps_O_C->ndof == c_dofs1D, "Bad programming!");
   MFEM_VERIFY(maps_C_C->ndof == c_dofs1D, "Bad programming!");


   // TODO: put this in a separate function?
   MFEM_VERIFY(dim == 3, "");

   const Array<int> & test_dofmap = test_el->GetDofMap();
   const double tk[18] =
   { 1.,0.,0.,  0.,1.,0.,  0.,0.,1., -1.,0.,0.,  0.,-1.,0.,  0.,0.,-1. }; // Copied from ND_HexahedronElement

   const int ndof_test = 3 * c_dofs1D * c_dofs1D * o_dofs1D;
   pa_data.SetSize(ndof_test * dim * ne, Device::GetMemoryType());
   //pa_data.SetSize(ndof_test * dim * ne);

   //const int NQ = ;

   const IntegrationRule & Nodes = test_el->GetNodes();

   auto op = Reshape(pa_data.HostWrite(), 3, ndof_test, ne);
   //auto J = Reshape(geom->J.Read(), NQ, 3, 3, NE);

   for (int c=0; c<3; ++c)
   {
      for (int i=0; i<ndof_test/3; ++i)
      {
         //double crd[3];
         //Nodes.IntPoint(i).Get(crd, dim);

         const int d = (c*ndof_test/3) + i;

         const int dofmap = test_dofmap[d];
         const int dof2tk = (dofmap < 0) ? 3+c : c;

         for (int e=0; e<ne; ++e)
         {
            double v[3];
            ElementTransformation *tr = mesh->GetElementTransformation(e);
            tr->SetIntPoint(&Nodes.IntPoint(d));
            tr->Jacobian().Mult(tk + dof2tk*dim, v);

            for (int j=0; j<3; ++j)
            {
               op(j,d,e) = v[j];
            }
            //pa_data[(3*ndof_test*e) + 3*i + j] = v[j];
         }
      }
   }

}

void IdentityInterpolator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      PAHcurlVecH1IdentityApply3D(c_dofs1D, o_dofs1D, ne, maps_C_C->B, maps_O_C->B,
                                  pa_data, x, y);
   }
   else if (dim == 2)
   {
      PAHcurlVecH1IdentityApply2D(c_dofs1D, o_dofs1D, ne, maps_C_C->B, maps_O_C->G,
                                  x, y);
   }
   else
   {
      mfem_error("Bad dimension!");
   }
}

} // namespace mfem
