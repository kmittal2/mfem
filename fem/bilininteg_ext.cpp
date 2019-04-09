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

#include "../config/config.hpp"
#include "../general/forall.hpp"
#include "../linalg/vector.hpp"

#include "fem.hpp"
#include <map>
#include <cmath>
#include <algorithm>
#include "bilininteg.hpp"
#include "bilininteg_ext.hpp"

using namespace std;

namespace mfem
{

#ifdef MFEM_USE_OCCA
typedef std::pair<int,int> id_t;
typedef std::map<id_t, occa::kernel> occa_kernel_t;
#endif // MFEM_USE_OCCA

static const IntegrationRule &DefaultGetRule(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe)
{
   int order;
   if (trial_fe.Space() == FunctionSpace::Pk)
   {
      order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
   }
   else
   {
      // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
      order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() - 1;
   }
   if (trial_fe.Space() == FunctionSpace::rQk)
   {
      return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   return IntRules.Get(trial_fe.GetGeomType(), order);
}

// PA Diffusion Integrator

// OCCA 2D Assemble kernel
#ifdef MFEM_USE_OCCA
static void OccaPADiffusionAssemble2D(const int D1D,
                                      const int Q1D,
                                      const int NE,
                                      const double *W,
                                      const double *J,
                                      const double COEFF,
                                      double *op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = mfem::OccaPtr(W);
   const occa::memory o_J = mfem::OccaPtr(J);
   occa::memory o_op = mfem::OccaPtr(op);
   const id_t id = std::make_pair(D1D,Q1D);
   static occa_kernel_t OccaDiffSetup2D_ker;
   if (OccaDiffSetup2D_ker.find(id) == OccaDiffSetup2D_ker.end())
   {
      const occa::kernel DiffusionSetup2D =
         mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                     "DiffusionSetup2D", props);
      OccaDiffSetup2D_ker.emplace(id, DiffusionSetup2D);
   }
   OccaDiffSetup2D_ker.at(id)(NE, o_W, o_J, COEFF, o_op);
}

static void OccaPADiffusionAssemble3D(const int D1D,
                                      const int Q1D,
                                      const int NE,
                                      const double *W,
                                      const double *J,
                                      const double COEFF,
                                      double *op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = mfem::OccaPtr(W);
   const occa::memory o_J = mfem::OccaPtr(J);
   occa::memory o_op = mfem::OccaPtr(op);
   const id_t id = std::make_pair(D1D,Q1D);
   static occa_kernel_t OccaDiffSetup3D_ker;
   if (OccaDiffSetup3D_ker.find(id) == OccaDiffSetup3D_ker.end())
   {
      const occa::kernel DiffusionSetup3D =
         mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                     "DiffusionSetup3D", props);
      OccaDiffSetup3D_ker.emplace(id, DiffusionSetup3D);
   }
   OccaDiffSetup3D_ker.at(id)(NE, o_W, o_J, COEFF, o_op);
}
#endif // MFEM_USE_OCCA

// PA Diffusion Assemble 2D kernel
static void PADiffusionAssemble2D(const int Q1D,
                                  const int NE,
                                  const double* w,
                                  const double* j,
                                  const double COEFF,
                                  double* op)
{
   const int NQ = Q1D*Q1D;
   const DeviceVector W(w, NQ);
   const DeviceTensor<4> J(j, 2, 2, NQ, NE);
   DeviceTensor<3> y(op, 3, NQ, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(0,0,q,e);
         const double J12 = J(1,0,q,e);
         const double J21 = J(0,1,q,e);
         const double J22 = J(1,1,q,e);
         const double c_detJ = W(q) * COEFF / ((J11*J22)-(J21*J12));
         y(0,q,e) =  c_detJ * (J21*J21 + J22*J22);
         y(1,q,e) = -c_detJ * (J21*J11 + J22*J12);
         y(2,q,e) =  c_detJ * (J11*J11 + J12*J12);
      }
   });
}

// PA Diffusion Assemble 3D kernel
static void PADiffusionAssemble3D(const int Q1D,
                                  const int NE,
                                  const double* w,
                                  const double* j,
                                  const double COEFF,
                                  double* op)
{
   const int NQ = Q1D*Q1D*Q1D;
   const DeviceVector W(w, NQ);
   const DeviceTensor<4> J(j, 3, 3, NQ, NE);
   DeviceTensor<3> y(op, 6, NQ, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(0,0,q,e);
         const double J12 = J(1,0,q,e);
         const double J13 = J(2,0,q,e);
         const double J21 = J(0,1,q,e);
         const double J22 = J(1,1,q,e);
         const double J23 = J(2,1,q,e);
         const double J31 = J(0,2,q,e);
         const double J32 = J(1,2,q,e);
         const double J33 = J(2,2,q,e);
         const double detJ =
         ((J11 * J22 * J33) + (J12 * J23 * J31) +
         (J13 * J21 * J32) - (J13 * J22 * J31) -
         (J12 * J21 * J33) - (J11 * J23 * J32));
         const double c_detJ = W(q) * COEFF / detJ;
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J23 * J31) - (J21 * J33);
         const double A13 = (J21 * J32) - (J22 * J31);
         const double A21 = (J13 * J32) - (J12 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J12 * J31) - (J11 * J32);
         const double A31 = (J12 * J23) - (J13 * J22);
         const double A32 = (J13 * J21) - (J11 * J23);
         const double A33 = (J11 * J22) - (J12 * J21);
         // adj(J)^Tadj(J)
         y(0,q,e) = c_detJ * (A11*A11 + A21*A21 + A31*A31);
         y(1,q,e) = c_detJ * (A11*A12 + A21*A22 + A31*A32);
         y(2,q,e) = c_detJ * (A11*A13 + A21*A23 + A31*A33);
         y(3,q,e) = c_detJ * (A12*A12 + A22*A22 + A32*A32);
         y(4,q,e) = c_detJ * (A12*A13 + A22*A23 + A32*A33);
         y(5,q,e) = c_detJ * (A13*A13 + A23*A23 + A33*A33);
      }
   });
}

namespace internal
{

#ifdef MFEM_USE_OCCA
// This function is currently used to determine if an OCCA kernel should be
// used.
static bool DeviceUseOcca()
{
   return Device::Allows(Backend::OCCA_CUDA) ||
          (Device::Allows(Backend::OCCA_OMP) &&
           !Device::Allows(Backend::DEVICE_MASK)) ||
          (Device::Allows(Backend::OCCA_CPU) &&
           !Device::Allows(Backend::DEVICE_MASK|Backend::OMP_MASK));
}
#endif

}

static void PADiffusionAssemble(const int dim,
                                const int D1D,
                                const int Q1D,
                                const int NE,
                                const double* W,
                                const double* J,
                                const double COEFF,
                                double* op)
{
   if (dim == 1) { mfem_error("dim==1 not supported in PADiffusionAssemble"); }
   if (dim == 2)
   {
#ifdef MFEM_USE_OCCA
      if (internal::DeviceUseOcca())
      {
         OccaPADiffusionAssemble2D(D1D, Q1D, NE, W, J, COEFF, op);
         return;
      }
#endif // MFEM_USE_OCCA
      PADiffusionAssemble2D(Q1D, NE, W, J, COEFF, op);
   }
   if (dim == 3)
   {
#ifdef MFEM_USE_OCCA
      if (internal::DeviceUseOcca())
      {
         OccaPADiffusionAssemble3D(D1D, Q1D, NE, W, J, COEFF, op);
         return;
      }
#endif // MFEM_USE_OCCA
      PADiffusionAssemble3D(Q1D, NE, W, J, COEFF, op);
   }
}

void DiffusionIntegrator::Assemble(const FiniteElementSpace &fes)
{
   const Mesh *mesh = fes.GetMesh();
   const IntegrationRule *rule = IntRule;
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = rule?rule:&DefaultGetRule(el,el);
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   dofs1D = el.GetOrder() + 1;
   quad1D = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
   geom = GeometryExtension::Get(fes,*ir);
   maps = DofToQuad::Get(fes, fes, *ir);
   vec.SetSize(symmDims * nq * ne);
   const double coeff = static_cast<ConstantCoefficient*>(Q)->constant;
   PADiffusionAssemble(dim, dofs1D, quad1D, ne, maps->W, geom->J, coeff, vec);
}

#ifdef MFEM_USE_OCCA
// OCCA PA Diffusion MultAdd 2D kernel
static void OccaPADiffusionMultAdd2D(const int D1D,
                                     const int Q1D,
                                     const int NE,
                                     const double* B,
                                     const double* G,
                                     const double* Bt,
                                     const double* Gt,
                                     const double* op,
                                     const double* x,
                                     double* y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = mfem::OccaPtr(B);
   const occa::memory o_G = mfem::OccaPtr(G);
   const occa::memory o_Bt = mfem::OccaPtr(Bt);
   const occa::memory o_Gt = mfem::OccaPtr(Gt);
   const occa::memory o_op = mfem::OccaPtr(op);
   const occa::memory o_x = mfem::OccaPtr(x);
   occa::memory o_y = mfem::OccaPtr(y);
   const id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaDiffApply2D_cpu;
      if (OccaDiffApply2D_cpu.find(id) == OccaDiffApply2D_cpu.end())
      {
         const occa::kernel DiffusionApply2D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply2D_CPU", props);
         OccaDiffApply2D_cpu.emplace(id, DiffusionApply2D_CPU);
      }
      OccaDiffApply2D_cpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_op, o_x, o_y);
   }
   else
   {
      static occa_kernel_t OccaDiffApply2D_gpu;
      if (OccaDiffApply2D_gpu.find(id) == OccaDiffApply2D_gpu.end())
      {
         const occa::kernel DiffusionApply2D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply2D_GPU", props);
         OccaDiffApply2D_gpu.emplace(id, DiffusionApply2D_GPU);
      }
      OccaDiffApply2D_gpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_op, o_x, o_y);
   }
}

// OCCA PA Diffusion MultAdd 3D kernel
static void OccaPADiffusionMultAdd3D(const int D1D,
                                     const int Q1D,
                                     const int NE,
                                     const double* B,
                                     const double* G,
                                     const double* Bt,
                                     const double* Gt,
                                     const double* op,
                                     const double* x,
                                     double* y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = mfem::OccaPtr(B);
   const occa::memory o_G = mfem::OccaPtr(G);
   const occa::memory o_Bt = mfem::OccaPtr(Bt);
   const occa::memory o_Gt = mfem::OccaPtr(Gt);
   const occa::memory o_op = mfem::OccaPtr(op);
   const occa::memory o_x = mfem::OccaPtr(x);
   occa::memory o_y = mfem::OccaPtr(y);
   const id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaDiffApply3D_cpu;
      if (OccaDiffApply3D_cpu.find(id) == OccaDiffApply3D_cpu.end())
      {
         const occa::kernel DiffusionApply3D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply3D_CPU", props);
         OccaDiffApply3D_cpu.emplace(id, DiffusionApply3D_CPU);
      }
      OccaDiffApply3D_cpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_op, o_x, o_y);
   }
   else
   {
      static occa_kernel_t OccaDiffApply3D_gpu;
      if (OccaDiffApply3D_gpu.find(id) == OccaDiffApply3D_gpu.end())
      {
         const occa::kernel DiffusionApply3D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply3D_GPU", props);
         OccaDiffApply3D_gpu.emplace(id, DiffusionApply3D_GPU);
      }
      OccaDiffApply3D_gpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_op, o_x, o_y);
   }
}
#endif // MFEM_USE_OCCA


#define QUAD_2D_ID(X, Y) (X + ((Y) * Q1D))
#define QUAD_3D_ID(X, Y, Z) (X + ((Y) * Q1D) + ((Z) * Q1D*Q1D))

// PA Diffusion MultAdd 2D kernel
template<const int D1D,
         const int Q1D> static
void PADiffusionMultAssembled2D(const int NE,
                                const double* b,
                                const double* g,
                                const double* bt,
                                const double* gt,
                                const double* _op,
                                const double* _x,
                                double* _y)
{
   const int NQ = Q1D*Q1D;
   const DeviceMatrix B(b,Q1D,D1D);
   const DeviceMatrix G(g,Q1D,D1D);
   const DeviceMatrix Bt(bt,D1D,Q1D);
   const DeviceMatrix Gt(gt,D1D,Q1D);
   const DeviceTensor<3> op(_op,3,NQ,NE);
   const DeviceTensor<3> x(_x,D1D,D1D,NE);
   DeviceTensor<3> y(_y,D1D,D1D,NE);
   MFEM_FORALL(e, NE,
   {
      double grad[Q1D][Q1D][2];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            grad[qy][qx][0] = 0.0;
            grad[qy][qx][1] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double gradX[Q1D][2];
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
               gradX[qx][0] += s * B(qx,dx);
               gradX[qx][1] += s * G(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy  = B(qy,dy);
            const double wDy = G(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qy][qx][0] += gradX[qx][1] * wy;
               grad[qy][qx][1] += gradX[qx][0] * wDy;
            }
         }
      }
      // Calculate Dxy, xDy in plane
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = QUAD_2D_ID(qx, qy);

            const double O11 = op(0,q,e);
            const double O12 = op(1,q,e);
            const double O22 = op(2,q,e);

            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
            grad[qy][qx][1] = (O12 * gradX) + (O22 * gradY);
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double gradX[D1D][2];
         for (int dx = 0; dx < D1D; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double gX = grad[qy][qx][0];
            const double gY = grad[qy][qx][1];
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double wx  = Bt(dx,qx);
               const double wDx = Gt(dx,qx);
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double wy  = Bt(dy,qy);
            const double wDy = Gt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               y(dx,dy,e) += ((gradX[dx][0] * wy) + (gradX[dx][1] * wDy));
            }
         }
      }
   });
}

// PA Diffusion MultAdd 3D kernel
template<const int D1D,
         const int Q1D> static
void PADiffusionMultAssembled3D(const int NE,
                                const double* b,
                                const double* g,
                                const double* bt,
                                const double* gt,
                                const double* _op,
                                const double* _x,
                                double* _y)
{
   const int NQ = Q1D*Q1D*Q1D;
   const DeviceMatrix B(b,Q1D,D1D);
   const DeviceMatrix G(g,Q1D,D1D);
   const DeviceMatrix Bt(bt,D1D,Q1D);
   const DeviceMatrix Gt(gt,D1D,Q1D);
   const DeviceTensor<3> op(_op,6,NQ,NE);
   const DeviceTensor<4> x(_x,D1D,D1D,D1D,NE);
   DeviceTensor<4> y(_y,D1D,D1D,D1D,NE);
   MFEM_FORALL(e, NE,
   {
      double grad[Q1D][Q1D][Q1D][4];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qz][qy][qx][0] = 0.0;
               grad[qz][qy][qx][1] = 0.0;
               grad[qz][qy][qx][2] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         double gradXY[Q1D][Q1D][4];
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
            double gradX[Q1D][2];
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
                  gradX[qx][0] += s * B(qx,dx);
                  gradX[qx][1] += s * G(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy  = B(qy,dy);
               const double wDy = G(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx  = gradX[qx][0];
                  const double wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx  * wDy;
                  gradXY[qy][qx][2] += wx  * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz  = B(qz,dz);
            const double wDz = G(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
               }
            }
         }
      }
      // Calculate Dxyz, xDyz, xyDz in plane
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = QUAD_3D_ID(qx, qy, qz);
               const double O11 = op(0,q,e);
               const double O12 = op(1,q,e);
               const double O13 = op(2,q,e);
               const double O22 = op(3,q,e);
               const double O23 = op(4,q,e);
               const double O33 = op(5,q,e);
               const double gradX = grad[qz][qy][qx][0];
               const double gradY = grad[qz][qy][qx][1];
               const double gradZ = grad[qz][qy][qx][2];
               grad[qz][qy][qx][0] = (O11*gradX)+(O12*gradY)+(O13*gradZ);
               grad[qz][qy][qx][1] = (O12*gradX)+(O22*gradY)+(O23*gradZ);
               grad[qz][qy][qx][2] = (O13*gradX)+(O23*gradY)+(O33*gradZ);
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double gradXY[D1D][D1D][4];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradXY[dy][dx][0] = 0;
               gradXY[dy][dx][1] = 0;
               gradXY[dy][dx][2] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double gradX[D1D][4];
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradX[dx][0] = 0;
               gradX[dx][1] = 0;
               gradX[dx][2] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double gX = grad[qz][qy][qx][0];
               const double gY = grad[qz][qy][qx][1];
               const double gZ = grad[qz][qy][qx][2];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double wx  = Bt(dx,qx);
                  const double wDx = Gt(dx,qx);
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
                  gradX[dx][2] += gZ * wx;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double wy  = Bt(dy,qy);
               const double wDy = Gt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradXY[dy][dx][0] += gradX[dx][0] * wy;
                  gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                  gradXY[dy][dx][2] += gradX[dx][2] * wy;
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const double wz  = Bt(dz,qz);
            const double wDz = Gt(dz,qz);
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  y(dx,dy,dz,e) +=
                     ((gradXY[dy][dx][0] * wz) +
                      (gradXY[dy][dx][1] * wz) +
                      (gradXY[dy][dx][2] * wDz));
               }
            }
         }
      }
   });
}

typedef void (*fDiffusionMultAdd)(const int NE,
                                  const double* B,
                                  const double* G,
                                  const double* Bt,
                                  const double* Gt,
                                  const double* op,
                                  const double* x,
                                  double* y);

static void PADiffusionMultAssembled(const int dim,
                                     const int D1D,
                                     const int Q1D,
                                     const int NE,
                                     const double* B,
                                     const double* G,
                                     const double* Bt,
                                     const double* Gt,
                                     const double* op,
                                     const double* x,
                                     double* y)
{
#ifdef MFEM_USE_OCCA
   if (internal::DeviceUseOcca())
   {
      if (dim == 2)
      {
         OccaPADiffusionMultAdd2D(D1D, Q1D, NE, B, G, Bt, Gt, op, x, y);
         return;
      }
      if (dim == 3)
      {
         OccaPADiffusionMultAdd3D(D1D, Q1D, NE, B, G, Bt, Gt, op, x, y);
         return;
      }
      mfem_error("OCCA PADiffusionMultAssembled unknown kernel!");
   }
#endif // MFEM_USE_OCCA
   const int id = (dim<<8)|(D1D<<4)|(Q1D);
   static std::unordered_map<int, fDiffusionMultAdd> call =
   {
      // 2D
      {0x222,&PADiffusionMultAssembled2D<2,2>},
      {0x233,&PADiffusionMultAssembled2D<3,3>},
      {0x244,&PADiffusionMultAssembled2D<4,4>},
      {0x255,&PADiffusionMultAssembled2D<5,5>},
      // 3D
      {0x323,&PADiffusionMultAssembled3D<2,3>},
      {0x334,&PADiffusionMultAssembled3D<3,4>},
      {0x345,&PADiffusionMultAssembled3D<4,5>},
      {0x356,&PADiffusionMultAssembled3D<5,6>},
   };
   if (!call[id])
   {
      mfem::err << "\nUnknown kernel for PADiffusionMultAssembled with "
                << "dim = " << dim << ", "
                << "D1D = " << D1D << ", "
                << "Q1D = " << Q1D << " (add in fem/bilininteg_ext.cpp).\n";
      mfem_error("PADiffusionMultAssembled kernel not instantiated");
   }
   call[id](NE, B, G, Bt, Gt, op, x, y);
}

// PA Diffusion MultAdd kernel
void DiffusionIntegrator::MultAssembled(Vector &x, Vector &y)
{
   PADiffusionMultAssembled(dim, dofs1D, quad1D, ne,
                            maps->B, maps->G, maps->Bt, maps->Gt,
                            vec, x, y);
}

DiffusionIntegrator::~DiffusionIntegrator()
{
   delete geom;
   delete maps;
}

// PA Mass Assemble kernel
void MassIntegrator::Assemble(const FiniteElementSpace &fes)
{
   const Mesh *mesh = fes.GetMesh();
   const IntegrationRule *rule = IntRule;
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = rule?rule:&DefaultGetRule(el,el);
   dim = mesh->Dimension();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   dofs1D = el.GetOrder() + 1;
   quad1D = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
   geom = GeometryExtension::Get(fes,*ir);
   maps = DofToQuad::Get(fes, fes, *ir);
   vec.SetSize(ne*nq);
   ConstantCoefficient *const_coeff = dynamic_cast<ConstantCoefficient*>(Q);
   FunctionCoefficient *function_coeff = dynamic_cast<FunctionCoefficient*>(Q);
   // TODO: other types of coefficients ...
   if (dim==1) { mfem_error("Not supported yet... stay tuned!"); }
   if (dim==2)
   {
      double constant = 0.0;
      double (*function)(const Vector3&) = NULL;
      if (const_coeff)
      {
         constant = const_coeff->constant;
      }
      else if (function_coeff)
      {
         function = function_coeff->GetDeviceFunction();
      }
      else
      {
         MFEM_ABORT("Coefficient type not supported");
      }
      const int NE = ne;
      const int NQ = nq;
      const DeviceVector w(maps->W.GetData(), NQ);
      const DeviceTensor<3> x(geom->X.GetData(), 2,NQ,NE);
      const DeviceTensor<4> J(geom->J.GetData(), 2,2,NQ,NE);
      DeviceMatrix v(vec.GetData(), NQ, NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J(0,0,q,e);
            const double J12 = J(1,0,q,e);
            const double J21 = J(0,1,q,e);
            const double J22 = J(1,1,q,e);
            const double detJ = (J11*J22)-(J21*J12);
            const Vector3 Xq(x(0,q,e), x(1,q,e));
            const double coeff =
            const_coeff ? constant
            : function_coeff ? function(Xq)
            : 0.0;
            v(q,e) =  w[q] * coeff * detJ;
         }
      });
   }
   if (dim==3)
   {
      double constant = 0.0;
      double (*function)(const Vector3&) = NULL;
      if (const_coeff)
      {
         constant = const_coeff->constant;
      }
      else if (function_coeff)
      {
         function = function_coeff->GetDeviceFunction();
      }
      else
      {
         MFEM_ABORT("Coefficient type not supported");
      }
      const int NE = ne;
      const int NQ = nq;
      const DeviceVector W(maps->W.GetData(), NQ);
      const DeviceTensor<3> x(geom->X.GetData(), 3,NQ,NE);
      const DeviceTensor<4> J(geom->J.GetData(), 3,3,NQ,NE);
      DeviceMatrix v(vec.GetData(), NQ,NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J(0,0,q,e),J12 = J(1,0,q,e),J13 = J(2,0,q,e);
            const double J21 = J(0,1,q,e),J22 = J(1,1,q,e),J23 = J(2,1,q,e);
            const double J31 = J(0,2,q,e),J32 = J(1,2,q,e),J33 = J(2,2,q,e);
            const double detJ =
            ((J11 * J22 * J33) + (J12 * J23 * J31) + (J13 * J21 * J32) -
            (J13 * J22 * J31) - (J12 * J21 * J33) - (J11 * J23 * J32));
            const Vector3 Xq(x(0,q,e), x(1,q,e), x(2,q,e));
            const double coeff =
            const_coeff ? constant
            : function_coeff ? function(Xq)
            : 0.0;
            v(q,e) = W(q) * coeff * detJ;
         }
      });
   }
}

#ifdef MFEM_USE_OCCA
// OCCA PA Mass MultAdd 2D kernel
static void OccaPAMassMultAdd2D(const int D1D,
                                const int Q1D,
                                const int NE,
                                const double* B,
                                const double* Bt,
                                const double* op,
                                const double* x,
                                double* y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = mfem::OccaPtr(B);
   const occa::memory o_Bt = mfem::OccaPtr(Bt);
   const occa::memory o_op = mfem::OccaPtr(op);
   const occa::memory o_x = mfem::OccaPtr(x);
   occa::memory o_y = mfem::OccaPtr(y);
   const id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaMassApply2D_cpu;
      if (OccaMassApply2D_cpu.find(id) == OccaMassApply2D_cpu.end())
      {
         const occa::kernel MassApply2D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply2D_CPU", props);
         OccaMassApply2D_cpu.emplace(id, MassApply2D_CPU);
      }
      OccaMassApply2D_cpu.at(id)(NE, o_B, o_Bt, o_op, o_x, o_y);
   }
   else
   {
      static occa_kernel_t OccaMassApply2D_gpu;
      if (OccaMassApply2D_gpu.find(id) == OccaMassApply2D_gpu.end())
      {
         const occa::kernel MassApply2D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply2D_GPU", props);
         OccaMassApply2D_gpu.emplace(id, MassApply2D_GPU);
      }
      OccaMassApply2D_gpu.at(id)(NE, o_B, o_Bt, o_op, o_x, o_y);
   }
}

// OCCA PA Mass MultAdd 3D kernel
static void OccaPAMassMultAdd3D(const int D1D,
                                const int Q1D,
                                const int NE,
                                const double* B,
                                const double* Bt,
                                const double* op,
                                const double* x,
                                double* y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = mfem::OccaPtr(B);
   const occa::memory o_Bt = mfem::OccaPtr(Bt);
   const occa::memory o_op = mfem::OccaPtr(op);
   const occa::memory o_x = mfem::OccaPtr(x);
   occa::memory o_y = mfem::OccaPtr(y);
   const id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaMassApply3D_cpu;
      if (OccaMassApply3D_cpu.find(id) == OccaMassApply3D_cpu.end())
      {
         const occa::kernel MassApply3D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply3D_CPU", props);
         OccaMassApply3D_cpu.emplace(id, MassApply3D_CPU);
      }
      OccaMassApply3D_cpu.at(id)(NE, o_B, o_Bt, o_op, o_x, o_y);
   }
   else
   {
      static occa_kernel_t OccaMassApply3D_gpu;
      if (OccaMassApply3D_gpu.find(id) == OccaMassApply3D_gpu.end())
      {
         const occa::kernel MassApply3D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply3D_GPU", props);
         OccaMassApply3D_gpu.emplace(id, MassApply3D_GPU);
      }
      OccaMassApply3D_gpu.at(id)(NE, o_B, o_Bt, o_op, o_x, o_y);
   }
}
#endif // MFEM_USE_OCCA

template<const int D1D,
         const int Q1D> static
void PAMassMultAdd2D(const int NE,
                     const double* _B,
                     const double* _Bt,
                     const double* _op,
                     const double* _x,
                     double* _y)
{
   const DeviceMatrix B(_B, Q1D,D1D);
   const DeviceMatrix Bt(_Bt, D1D,Q1D);
   const DeviceTensor<3> op(_op, Q1D,Q1D,NE);
   const DeviceTensor<3> x(_x, D1D,D1D,NE);
   DeviceTensor<3> y(_y, D1D,D1D,NE);
   MFEM_FORALL(e,NE,
   {
      double sol_xy[Q1D][Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double sol_x[Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            sol_x[qy] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = x(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] += B(qx,dx)* s;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double d2q = B(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] += d2q * sol_x[qx];
            }
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] *= op(qx,qy,e);
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double sol_x[D1D];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] = 0.0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double s = sol_xy[qy][qx];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] += Bt(dx,qx) * s;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double q2d = Bt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               y(dx,dy,e) += q2d * sol_x[dx];
            }
         }
      }
   });
}

template<const int D1D,
         const int Q1D> static
void PAMassMultAdd3D(const int NE,
                     const double* _B,
                     const double* _Bt,
                     const double* _op,
                     const double* _x,
                     double* _y)
{
   const DeviceMatrix B(_B, Q1D,D1D);
   const DeviceMatrix Bt(_Bt, D1D,Q1D);
   const DeviceTensor<4> op(_op, Q1D,Q1D,Q1D,NE);
   const DeviceTensor<4> x(_x, D1D,D1D,D1D,NE);
   DeviceTensor<4> y(_y, D1D,D1D,D1D,NE);

   MFEM_FORALL(e,NE,
   {
      double sol_xyz[Q1D][Q1D][Q1D];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xyz[qz][qy][qx] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         double sol_xy[Q1D][Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            double sol_x[Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] = 0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = x(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_x[qx] += B(qx,dx) * s;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = B(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xy[qy][qx] += wy * sol_x[qx];
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz = B(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
               }
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xyz[qz][qy][qx] *= op(qx,qy,qz,e);
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double sol_xy[D1D][D1D];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_xy[dy][dx] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double sol_x[D1D];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double s = sol_xyz[qz][qy][qx];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_x[dx] += Bt(dx,qx) * s;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double wy = Bt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_xy[dy][dx] += wy * sol_x[dx];
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const double wz = Bt(dz,qz);
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  y(dx,dy,dz,e) += wz * sol_xy[dy][dx];
               }
            }
         }
      }
   });
}

typedef void (*fMassMultAdd)(const int NE,
                             const double* B,
                             const double* Bt,
                             const double* oper,
                             const double* x,
                             double* y);

static void PAMassMultAssembled(const int dim,
                                const int D1D,
                                const int Q1D,
                                const int NE,
                                const double* B,
                                const double* Bt,
                                const double* op,
                                const double* x,
                                double* y)
{
#ifdef MFEM_USE_OCCA
   if (internal::DeviceUseOcca())
   {
      if (dim == 2)
      {
         OccaPAMassMultAdd2D(D1D, Q1D, NE, B, Bt, op, x, y);
         return;
      }
      if (dim == 3)
      {
         OccaPAMassMultAdd3D(D1D, Q1D, NE, B, Bt, op, x, y);
         return;
      }
      mfem_error("OCCA PA Mass MultAssembled unknown kernel!");
   }
#endif // MFEM_USE_OCCA
   const int id = (dim<<8)|((D1D)<<4)|(Q1D);
   static std::unordered_map<int, fMassMultAdd> call =
   {
      // 2D
      {0x222,&PAMassMultAdd2D<2,2>},
      {0x224,&PAMassMultAdd2D<2,4>},
      {0x233,&PAMassMultAdd2D<3,3>},
      {0x234,&PAMassMultAdd2D<3,4>},
      {0x235,&PAMassMultAdd2D<3,5>},
      {0x236,&PAMassMultAdd2D<3,6>},
      {0x244,&PAMassMultAdd2D<4,4>},
      {0x245,&PAMassMultAdd2D<4,5>},
      {0x246,&PAMassMultAdd2D<4,6>},
      {0x248,&PAMassMultAdd2D<4,8>},
      {0x255,&PAMassMultAdd2D<5,5>},
      {0x258,&PAMassMultAdd2D<5,8>},
      // 3D
      {0x323,&PAMassMultAdd3D<2,3>},
      {0x324,&PAMassMultAdd3D<2,4>},
      {0x334,&PAMassMultAdd3D<3,4>},
      {0x345,&PAMassMultAdd3D<4,5>},
      {0x356,&PAMassMultAdd3D<5,6>},
   };
   if (!call[id])
   {
      mfem::err << "\nUnknown kernel for PAMassMultAdd with "
                << "dim = " << dim << ", "
                << "D1D = " << D1D << ", "
                << "Q1D = " << Q1D << " (add in fem/bilininteg_ext.cpp).\n";
      mfem_error("PAMassMultAssembled kernel not instantiated");
   }
   call[id](NE, B, Bt, op, x, y);
}

void MassIntegrator::MultAssembled(Vector &x, Vector &y)
{
   PAMassMultAssembled(dim, dofs1D, quad1D, ne,
                       maps->B, maps->Bt,
                       vec, x, y);
}

MassIntegrator::~MassIntegrator()
{
   delete geom;
   delete maps;
}

// DofToQuad
static std::map<std::string, DofToQuad* > AllDofQuadMaps;

DofToQuad::~DofToQuad()
{
   MFEM_ASSERT(AllDofQuadMaps.at(hash),"");
   AllDofQuadMaps.erase(hash);
}

DofToQuad* DofToQuad::Get(const FiniteElementSpace& fes,
                          const IntegrationRule& ir,
                          const bool transpose)
{
   return Get(*fes.GetFE(0), *fes.GetFE(0), ir, transpose);
}

DofToQuad* DofToQuad::Get(const FiniteElementSpace& trialFES,
                          const FiniteElementSpace& testFES,
                          const IntegrationRule& ir,
                          const bool transpose)
{
   return Get(*trialFES.GetFE(0), *testFES.GetFE(0), ir, transpose);
}

DofToQuad* DofToQuad::Get(const FiniteElement& trialFE,
                          const FiniteElement& testFE,
                          const IntegrationRule& ir,
                          const bool transpose)
{
   return GetTensorMaps(trialFE, testFE, ir, transpose);
}

DofToQuad* DofToQuad::GetTensorMaps(const FiniteElement& trialFE,
                                    const FiniteElement& testFE,
                                    const IntegrationRule& ir,
                                    const bool transpose)
{
   const TensorBasisElement& trialTFE =
      dynamic_cast<const TensorBasisElement&>(trialFE);
   const TensorBasisElement& testTFE =
      dynamic_cast<const TensorBasisElement&>(testFE);
   std::stringstream ss;
   ss << "TensorMap:"
      << " O1:"  << trialFE.GetOrder()
      << " O2:"  << testFE.GetOrder()
      << " BT1:" << trialTFE.GetBasisType()
      << " BT2:" << testTFE.GetBasisType()
      << " Q:"   << ir.GetNPoints();
   std::string hash = ss.str();
   // If we've already made the dof-quad maps, reuse them
   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }
   // Otherwise, build them
   DofToQuad *maps = new DofToQuad();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   const DofToQuad* trialMaps = GetD2QTensorMaps(trialFE, ir);
   const DofToQuad* testMaps  = GetD2QTensorMaps(testFE, ir, true);
   maps->B = trialMaps->B;
   maps->G = trialMaps->G;
   maps->Bt = testMaps->B;
   maps->Gt = testMaps->G;
   maps->W = testMaps->W;
   delete trialMaps;
   delete testMaps;
   return maps;
}

DofToQuad* DofToQuad::GetD2QTensorMaps(const FiniteElement& fe,
                                       const IntegrationRule& ir,
                                       const bool transpose)
{
   const IntegrationRule& ir1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder());

   const int dims = fe.GetDim();
   const int order = fe.GetOrder();
   const int numDofs = order + 1;
   const int numQuad1D = ir1D.GetNPoints();
   const int numQuad2D = numQuad1D * numQuad1D;
   const int numQuad3D = numQuad2D * numQuad1D;
   const int numQuad =
      (dims == 1) ? numQuad1D :
      (dims == 2) ? numQuad2D :
      (dims == 3) ? numQuad3D : 0;
   std::stringstream ss;
   ss << "D2QTensorMap:"
      << " dims:" << dims
      << " order:" << order
      << " numDofs:" << numDofs
      << " numQuad1D:" << numQuad1D
      << " transpose:"  << (transpose?"true":"false");
   std::string hash = ss.str();
   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }
   DofToQuad *maps = new DofToQuad();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   maps->B.SetSize(numQuad1D*numDofs);
   maps->G.SetSize(numQuad1D*numDofs);
   const int dim0 = (!transpose)?1:numDofs;
   const int dim1 = (!transpose)?numQuad1D:1;
   if (transpose) // Initialize quad weights only for transpose
   {
      maps->W.SetSize(numQuad);
   }
   mfem::Vector d2q(numDofs);
   mfem::Vector d2qD(numDofs);
   mfem::Array<double> W1d(numQuad1D);
   mfem::Array<double> B1d(numQuad1D*numDofs);
   mfem::Array<double> G1d(numQuad1D*numDofs);
   const TensorBasisElement& tbe = dynamic_cast<const TensorBasisElement&>(fe);
   const Poly_1D::Basis& basis = tbe.GetBasis1D();
   for (int q = 0; q < numQuad1D; ++q)
   {
      const IntegrationPoint& ip = ir1D.IntPoint(q);
      if (transpose)
      {
         W1d[q] = ip.weight;
      }
      basis.Eval(ip.x, d2q, d2qD);
      for (int d = 0; d < numDofs; ++d)
      {
         const double w = d2q[d];
         const double wD = d2qD[d];
         const int idx = dim0*q + dim1*d;
         B1d[idx] = w;
         G1d[idx] = wD;
      }
   }
   if (transpose)
   {
      mfem::Array<double> W(numQuad);
      for (int q = 0; q < numQuad; ++q)
      {
         const int qx = q % numQuad1D;
         const int qz = q / numQuad2D;
         const int qy = (q - qz*numQuad2D) / numQuad1D;
         double w = W1d[qx];
         if (dims > 1) { w *= W1d[qy]; }
         if (dims > 2) { w *= W1d[qz]; }
         W[q] = w;
      }
      maps->W = W;
   }
   mm::memcpy(maps->B, B1d, numQuad1D*numDofs*sizeof(double));
   mm::memcpy(maps->G, G1d, numQuad1D*numDofs*sizeof(double));
   return maps;
}

DofToQuad* DofToQuad::GetSimplexMaps(const FiniteElement& fe,
                                     const IntegrationRule& ir,
                                     const bool transpose)
{
   return GetSimplexMaps(fe, fe, ir, transpose);
}

DofToQuad* DofToQuad::GetSimplexMaps(const FiniteElement& trialFE,
                                     const FiniteElement& testFE,
                                     const IntegrationRule& ir,
                                     const bool transpose)
{
   std::stringstream ss;
   ss << "SimplexMap:"
      << " O1:" << trialFE.GetOrder()
      << " O2:" << testFE.GetOrder()
      << " Q:"  << ir.GetNPoints();
   std::string hash = ss.str();
   // If we've already made the dof-quad maps, reuse them
   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }
   DofToQuad *maps = new DofToQuad();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   const DofToQuad* trialMaps = GetD2QSimplexMaps(trialFE, ir);
   const DofToQuad* testMaps  = GetD2QSimplexMaps(testFE, ir, true);
   maps->B = trialMaps->B;
   maps->G = trialMaps->G;
   maps->Bt = testMaps->B;
   maps->Gt = testMaps->G;
   maps->W = testMaps->W;
   delete trialMaps;
   delete testMaps;
   return maps;
}

DofToQuad* DofToQuad::GetD2QSimplexMaps(const FiniteElement& fe,
                                        const IntegrationRule& ir,
                                        const bool transpose)
{
   const int dims = fe.GetDim();
   const int numDofs = fe.GetDof();
   const int numQuad = ir.GetNPoints();
   std::stringstream ss ;
   ss << "D2QSimplexMap:"
      << " Dim:" << dims
      << " numDofs:" << numDofs
      << " numQuad:" << numQuad
      << " transpose:" << (transpose?"true":"false");
   std::string hash = ss.str();
   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }
   DofToQuad* maps = new DofToQuad();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   maps->B.SetSize(numQuad*numDofs);
   maps->G.SetSize(dims*numQuad*numDofs);
   const int dim0 = (!transpose)?1:numDofs;
   const int dim1 = (!transpose)?numQuad:1;
   const int dim0D = (!transpose)?1:numQuad;
   const int dim1D = (!transpose)?dims:1;
   const int dim2D = dims*numQuad;
   if (transpose) // Initialize quad weights only for transpose
   {
      maps->W.SetSize(numQuad);
   }
   mfem::Vector d2q(numDofs);
   mfem::DenseMatrix d2qD(numDofs, dims);
   mfem::Array<double> W(numQuad);
   mfem::Array<double> B(numQuad*numDofs);
   mfem::Array<double> G(dims*numQuad*numDofs);
   for (int q = 0; q < numQuad; ++q)
   {
      const IntegrationPoint& ip = ir.IntPoint(q);
      if (transpose)
      {
         W[q] = ip.weight;
      }
      fe.CalcShape(ip, d2q);
      fe.CalcDShape(ip, d2qD);
      for (int d = 0; d < numDofs; ++d)
      {
         const double w = d2q[d];
         const int idx = dim0*q + dim1*d;
         B[idx] = w;
         for (int dim = 0; dim < dims; ++dim)
         {
            const double wD = d2qD(d, dim);
            const int idxD = dim0D*dim + dim1D*q + dim2D*d;
            G[idxD] = wD;
         }
      }
   }
   if (transpose)
   {
      mm::memcpy(maps->W, W, numQuad*sizeof(double));
   }
   mm::memcpy(maps->B, B, numQuad*numDofs*sizeof(double));
   mm::memcpy(maps->G, G, dims*numQuad*numDofs*sizeof(double));
   return maps;
}


static long sequence = -1;
static GeometryExtension *geom = NULL;

static void GeomFill(const int vdim,
                     const int NE, const int ND, const int NX,
                     const int* elementMap, int* eMap,
                     const double *_X, double *meshNodes)
{
   const DeviceArray d_elementMap(elementMap, ND*NE);
   DeviceArray d_eMap(eMap, ND*NE);
   const DeviceVector X(_X, NX);
   DeviceVector d_meshNodes(meshNodes, vdim*ND*NE);
   MFEM_FORALL(e, NE,
   {
      for (int d = 0; d < ND; ++d)
      {
         const int lid = d+ND*e;
         const int gid = d_elementMap[lid];
         d_eMap[lid] = gid;
         for (int v = 0; v < vdim; ++v)
         {
            const int moffset = v+vdim*lid;
            const int xoffset = v+vdim*gid;
            d_meshNodes[moffset] = X[xoffset];
         }
      }
   });
}

static void NodeCopyByVDim(const int elements,
                           const int numDofs,
                           const int ndofs,
                           const int dims,
                           const int* eMap,
                           const double* Sx,
                           double* nodes)
{
   MFEM_FORALL(e,elements,
   {
      for (int dof = 0; dof < numDofs; ++dof)
      {
         const int lid = dof+numDofs*e;
         const int gid = eMap[lid];
         for (int v = 0; v < dims; ++v)
         {
            const int moffset = v+dims*lid;
            const int voffset = gid+v*ndofs;
            nodes[moffset] = Sx[voffset];
         }
      }
   });
}

template<const int D1D,
         const int Q1D> static
void PAGeom2D(const int NE,
              const double* _B,
              const double* _G,
              const double* _X,
              double* _Xq,
              double* _J,
              double* _invJ,
              double* _detJ)
{
   const int ND = D1D*D1D;
   const int NQ = Q1D*Q1D;
   const DeviceTensor<2> B(_B, NQ,ND);
   const DeviceTensor<3> G(_G, 2,NQ,ND);
   const DeviceTensor<3> X(_X, 2,ND,NE);
   DeviceTensor<3> Xq(_Xq, 2,NQ,NE);
   DeviceTensor<4> J(_J, 2,2,NQ,NE);
   DeviceTensor<4> invJ(_invJ, 2,2,NQ,NE);
   DeviceMatrix detJ(_detJ, NQ,NE);
   MFEM_FORALL(e, NE,
   {
      double s_X[2*D1D*D1D];
      for (int q = 0; q < NQ; ++q)
      {
         for (int d = q; d < ND; d +=NQ)
         {
            s_X[0+d*2] = X(0,d,e);
            s_X[1+d*2] = X(1,d,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         double X0  = 0; double X1  = 0;
         double J11 = 0; double J12 = 0;
         double J21 = 0; double J22 = 0;
         for (int d = 0; d < ND; ++d)
         {
            const double b = B(q,d);
            const double wx = G(0,q,d);
            const double wy = G(1,q,d);
            const double x = s_X[0+d*2];
            const double y = s_X[1+d*2];
            J11 += (wx * x); J12 += (wx * y);
            J21 += (wy * x); J22 += (wy * y);
            X0 += b*x; X1 += b*y;
         }
         Xq(0,q,e) = X0; Xq(1,q,e) = X1;
         const double r_detJ = (J11 * J22)-(J12 * J21);
         J(0,0,q,e) = J11;
         J(1,0,q,e) = J12;
         J(0,1,q,e) = J21;
         J(1,1,q,e) = J22;
         const double r_idetJ = 1.0 / r_detJ;
         invJ(0,0,q,e) =  J22 * r_idetJ;
         invJ(1,0,q,e) = -J12 * r_idetJ;
         invJ(0,1,q,e) = -J21 * r_idetJ;
         invJ(1,1,q,e) =  J11 * r_idetJ;
         detJ(q,e) = r_detJ;
      }
   });
}

template<const int D1D,
         const int Q1D> static
void PAGeom3D(const int NE,
              const double* _B,
              const double* _G,
              const double* _X,
              double* _Xq,
              double* _J,
              double* _invJ,
              double* _detJ)
{
   const int ND = D1D*D1D*D1D;
   const int NQ = Q1D*Q1D*Q1D;
   const DeviceTensor<2> B(_B, NQ,NE);
   const DeviceTensor<3> G(_G, 3,NQ,NE);
   const DeviceTensor<3> X(_X, 3,ND,NE);
   DeviceTensor<3> Xq(_Xq, 3,NQ,NE);
   DeviceTensor<4> J(_J, 3,3,NQ,NE);
   DeviceTensor<4> invJ(_invJ, 3,3,NQ,NE);
   DeviceMatrix detJ(_detJ, NQ,NE);
   MFEM_FORALL(e,NE,
   {
      double s_nodes[3*D1D*D1D*D1D];
      for (int q = 0; q < NQ; ++q)
      {
         for (int d = q; d < ND; d += NQ)
         {
            s_nodes[0+d*3] = X(0,d,e);
            s_nodes[1+d*3] = X(1,d,e);
            s_nodes[2+d*3] = X(2,d,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         double J11 = 0; double J12 = 0; double J13 = 0;
         double J21 = 0; double J22 = 0; double J23 = 0;
         double J31 = 0; double J32 = 0; double J33 = 0;
         for (int d = 0; d < ND; ++d)
         {
            const double b = B(q,d);
            const double wx = G(0,q,d);
            const double wy = G(1,q,d);
            const double wz = G(2,q,d);
            const double x = s_nodes[0+d*3];
            const double y = s_nodes[1+d*3];
            const double z = s_nodes[2+d*3];
            J11 += (wx * x); J12 += (wx * y); J13 += (wx * z);
            J21 += (wy * x); J22 += (wy * y); J23 += (wy * z);
            J31 += (wz * x); J32 += (wz * y); J33 += (wz * z);
            Xq(0,q,e) = b*x; Xq(1,q,e) = b*y; Xq(2,q,e) = b*z;
         }
         const double r_detJ = ((J11 * J22 * J33) + (J12 * J23 * J31) +
                                (J13 * J21 * J32) - (J13 * J22 * J31) -
                                (J12 * J21 * J33) - (J11 * J23 * J32));
         J(0,0,q,e) = J11;
         J(1,0,q,e) = J12;
         J(2,0,q,e) = J13;
         J(0,1,q,e) = J21;
         J(1,1,q,e) = J22;
         J(2,1,q,e) = J23;
         J(0,2,q,e) = J31;
         J(1,2,q,e) = J32;
         J(2,2,q,e) = J33;
         const double r_idetJ = 1.0 / r_detJ;
         invJ(0,0,q,e) = r_idetJ * ((J22 * J33)-(J23 * J32));
         invJ(1,0,q,e) = r_idetJ * ((J32 * J13)-(J33 * J12));
         invJ(2,0,q,e) = r_idetJ * ((J12 * J23)-(J13 * J22));
         invJ(0,1,q,e) = r_idetJ * ((J23 * J31)-(J21 * J33));
         invJ(1,1,q,e) = r_idetJ * ((J33 * J11)-(J31 * J13));
         invJ(2,1,q,e) = r_idetJ * ((J13 * J21)-(J11 * J23));
         invJ(0,2,q,e) = r_idetJ * ((J21 * J32)-(J22 * J31));
         invJ(1,2,q,e) = r_idetJ * ((J31 * J12)-(J32 * J11));
         invJ(2,2,q,e) = r_idetJ * ((J11 * J22)-(J12 * J21));
         detJ(q,e) = r_detJ;
      }
   });
}

typedef void (*fIniGeom)(const int ne,
                         const double *B, const double *G, const double *X,
                         double *x, double *J, double *invJ, double *detJ);

static void PAGeom(const int dim,
                   const int D1D,
                   const int Q1D,
                   const int NE,
                   const double* B,
                   const double* G,
                   const double* X,
                   double* Xq,
                   double* J,
                   double* invJ,
                   double* detJ)
{
   const int id = (dim<<8)|(D1D)<<4|(Q1D);
   static std::unordered_map<int, fIniGeom> call =
   {
      // 2D
      {0x222,&PAGeom2D<2,2>},
      {0x223,&PAGeom2D<2,3>},
      {0x224,&PAGeom2D<2,4>},
      {0x225,&PAGeom2D<2,5>},
      {0x232,&PAGeom2D<3,2>},
      {0x234,&PAGeom2D<3,4>},
      {0x242,&PAGeom2D<4,2>},
      {0x244,&PAGeom2D<4,4>},
      {0x245,&PAGeom2D<4,5>},
      {0x246,&PAGeom2D<4,6>},
      {0x258,&PAGeom2D<5,8>},
      // 3D
      {0x323,&PAGeom3D<2,3>},
      {0x324,&PAGeom3D<2,4>},
      {0x325,&PAGeom3D<2,5>},
      {0x326,&PAGeom3D<2,6>},
      {0x334,&PAGeom3D<3,4>},
   };
   if (!call[id])
   {
      mfem::err << "\nUnknown kernel for PAGeom with "
                << "dim = " << dim << ", "
                << "D1D = " << D1D << ", "
                << "Q1D = " << Q1D << " (add in fem/bilininteg_ext.cpp).\n";
      mfem_error("PAGeom kernel not instantiated");
   }
   call[id](NE, B, G, X, Xq, J, invJ, detJ);
}

GeometryExtension* GeometryExtension::Get(const FiniteElementSpace& fes,
                                          const IntegrationRule& ir,
                                          const Vector& Sx)
{
   const Mesh *mesh = fes.GetMesh();
   const GridFunction *nodes = mesh->GetNodes();
   const FiniteElementSpace *fespace = nodes->FESpace();
   const FiniteElement *fe = fespace->GetFE(0);
   const IntegrationRule& ir1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder());
   const int dims     = fe->GetDim();
   const int numDofs  = fe->GetDof();
   const int D1D      = fe->GetOrder() + 1;
   const int Q1D      = ir1D.GetNPoints();
   const int elements = fespace->GetNE();
   const int ndofs    = fespace->GetNDofs();
   const DofToQuad* maps = DofToQuad::GetSimplexMaps(*fe, ir);
   NodeCopyByVDim(elements,numDofs,ndofs,dims,geom->eMap,Sx,geom->nodes);
   PAGeom(dims, D1D, Q1D, elements,
          maps->B, maps->G, geom->nodes,
          geom->X, geom->J, geom->invJ, geom->detJ);
   return geom;
}

GeometryExtension* GeometryExtension::Get(const FiniteElementSpace& fes,
                                          const IntegrationRule& ir)
{
   Mesh *mesh = fes.GetMesh();
   const bool geom_to_allocate = sequence < fes.GetSequence();
   sequence = fes.GetSequence();
   if (geom_to_allocate)
   {
      if (geom) { delete geom; }
      geom = new GeometryExtension();
   }
   mesh->EnsureNodes();
   const GridFunction *nodes = mesh->GetNodes();
   const mfem::FiniteElementSpace *fespace = nodes->FESpace();
   const mfem::FiniteElement *fe = fespace->GetFE(0);
   const IntegrationRule& ir1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder());
   const int dims     = fe->GetDim();
   const int elements = fespace->GetNE();
   const int numDofs  = fe->GetDof();
   const int D1D      = fe->GetOrder() + 1;
   const int Q1D      = ir1D.GetNPoints();
   const int numQuad  = ir.GetNPoints();
   const bool orderedByNODES = (fespace->GetOrdering() == Ordering::byNODES);
   if (orderedByNODES) { ReorderByVDim(nodes); }
   const int asize = dims*numDofs*elements;
   mfem::Array<double> meshNodes(asize);
   const Table& e2dTable = fespace->GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   mfem::Array<int> eMap(numDofs*elements);
   GeomFill(dims,
            elements,
            numDofs,
            nodes->Size(),
            elementMap,
            eMap,
            nodes->GetData(),
            meshNodes);
   if (geom_to_allocate)
   {
      geom->nodes.SetSize(dims*numDofs*elements);
      geom->eMap.SetSize(numDofs*elements);
   }
   geom->nodes = meshNodes;
   geom->eMap = eMap;
   // Reorder the original gf back
   if (orderedByNODES) { ReorderByNodes(nodes); }
   if (geom_to_allocate)
   {
      geom->X.SetSize(dims*numQuad*elements);
      geom->J.SetSize(dims*dims*numQuad*elements);
      geom->invJ.SetSize(dims*dims*numQuad*elements);
      geom->detJ.SetSize(numQuad*elements);
   }
   const DofToQuad* maps = DofToQuad::GetSimplexMaps(*fe, ir);
   PAGeom(dims, D1D, Q1D, elements,
          maps->B, maps->G, geom->nodes,
          geom->X, geom->J, geom->invJ, geom->detJ);
   delete maps;
   return geom;
}

void GeometryExtension::ReorderByVDim(const GridFunction *nodes)
{
   const mfem::FiniteElementSpace *fes = nodes->FESpace();
   const int size = nodes->Size();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetNDofs();
   double *data = nodes->GetData();
   double *temp = new double[size];
   int k=0;
   for (int d = 0; d < ndofs; d++)
      for (int v = 0; v < vdim; v++)
      {
         temp[k++] = data[d+v*ndofs];
      }
   for (int i = 0; i < size; i++)
   {
      data[i] = temp[i];
   }
   delete [] temp;
}

void GeometryExtension::ReorderByNodes(const GridFunction *nodes)
{
   const mfem::FiniteElementSpace *fes = nodes->FESpace();
   const int size = nodes->Size();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetNDofs();
   double *data = nodes->GetData();
   double *temp = new double[size];
   int k = 0;
   for (int j = 0; j < ndofs; j++)
      for (int i = 0; i < vdim; i++)
      {
         temp[j+i*ndofs] = data[k++];
      }
   for (int i = 0; i < size; i++)
   {
      data[i] = temp[i];
   }
   delete [] temp;
}

} // namespace mfem
