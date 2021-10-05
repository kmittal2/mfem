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

#include "bench.hpp"

#define MFEM_DEBUG_COLOR 206
#include "../../general/debug.hpp"

#ifdef MFEM_USE_BENCHMARK

constexpr int seed = 0x100001b3;

////////////////////////////////////////////////////////////////////////////////
/// Exact solution parameters:
static constexpr double sol_s[3] = { -0.32, 0.15, 0.24 };
static constexpr double sol_k[3] = { 1.21, 1.45, 1.37 };
static void gradu_exact(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   double *g = grad.GetData();
   double val = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      const double y = M_PI*(sol_s[d]+sol_k[d]*x(d));
      const double f = sin(y);
      for (int j = 0; j < d; j++) { g[j] *= f; }
      g[d] = val*M_PI*sol_k[d]*cos(y);
      val *= f;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Base class for the LinearForm extension test and the bench
struct LinExt
{
   const int problem, N, p, q, dim;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace fes;
   FiniteElementSpace *fespace;
   GridFunction x;
   const Geometry::Type geom_type;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   const int dofs;
   double mdofs;

   Vector v;
   ConstantCoefficient constant_coeff;
   VectorConstantCoefficient vector_constant_coeff;
   VectorFunctionCoefficient vector_function_coeff;

   LinearForm *lf[2];

   LinExt(int problem, int order, int dim, int vdim):
      problem(problem),
      N(Device::IsEnabled()?32:8),
      p(order),
      q(2*p),
      dim(dim),
      mesh(dim==3 ?
           Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON):
           Mesh::MakeCartesian2D(N,N,Element::QUADRILATERAL)),
      fec(p, dim),
      fes(&mesh, &fec, vdim),
      fespace(new FiniteElementSpace(&mesh, &fec, dim)),
      x(fespace),
      geom_type(fes.GetFE(0)->GetGeomType()),
      ir(&IntRules.Get(geom_type, q)),
      one(1.0),
      dofs(fes.GetTrueVSize()),
      mdofs(0.0),
      v(vdim),
      constant_coeff(M_PI),
      vector_constant_coeff((v.Randomize(seed),v)),
      vector_function_coeff(dim, gradu_exact),
      lf{new LinearForm(&fes), new LinearForm(&fes)}
   {
      assert(dim==2 || dim==3);
      SetupRandomMesh();
      SetupLinearForms();
      lf[0]->SetAssemblyLevel(LinearAssemblyLevel::LEGACY);
      lf[1]->SetAssemblyLevel(LinearAssemblyLevel::FULL);
   }

   ~LinExt() { delete lf[0]; delete lf[1]; }

   void SetupRandomMesh() noexcept
   {
      mesh.SetNodalFESpace(fespace);
      mesh.SetNodalGridFunction(&x);

      Vector h0(fespace->GetNDofs());
      h0 = infinity();
      Array<int> dofs;
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         // Get the local scalar element degrees of freedom in dofs.
         fespace->GetElementDofs(i, dofs);
         // Adjust the value of h0 in dofs based on the local mesh size.
         const double hi = mesh.GetElementSize(i);
         for (int j = 0; j < dofs.Size(); j++)
         {
            h0(dofs[j]) = fmin(h0(dofs[j]), hi);
         }
      }

      GridFunction rdm(fespace);
      rdm.Randomize(seed);
      rdm -= 0.25; // Shift to random values in [-0.5,0.5].
      rdm *= 1.0/M_PI;
      rdm.HostReadWrite();
      // Scale the random values to be of order of the local mesh size.
      for (int i = 0; i < fespace->GetNDofs(); i++)
      {
         for (int d = 0; d < dim; d++)
         {
            rdm(fespace->DofToVDof(i,d)) *= h0(i);
         }
      }
      Array<int> vdofs;
      for (int i = 0; i < fespace->GetNBE(); i++)
      {
         // Get the vector degrees of freedom in the boundary element.
         fespace->GetBdrElementVDofs(i, vdofs);
         // Set the boundary values to zero.
         for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
      }
      x -= rdm;
      x.SetTrueVector();
      x.SetFromTrueVector();
   }

   virtual void benchmark() = 0;

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }

   void SetupLinearForms() noexcept
   {
      for (int i=0; i<2; i++)
      {
         if (problem==1) // DomainLFIntegrator
         {
            DomainLFIntegrator *dlfi;
            dlfi = new DomainLFIntegrator(constant_coeff);
            dlfi->SetIntRule(ir);
            lf[i]->AddDomainIntegrator(dlfi);
         }
         else if (problem==2) // VectorDomainLFIntegrator
         {
            VectorDomainLFIntegrator *vdlfi;
            vdlfi = new VectorDomainLFIntegrator(vector_constant_coeff);
            vdlfi->SetIntRule(ir);
            lf[i]->AddDomainIntegrator(vdlfi);
         }
         else if (problem==3) // DomainLFGradIntegrator
         {
            DomainLFGradIntegrator *dlfgi;
            dlfgi = new DomainLFGradIntegrator(vector_function_coeff);
            dlfgi->SetIntRule(ir);
            lf[i]->AddDomainIntegrator(dlfgi);
         }
         else if (problem==4) // VectorDomainLFGradIntegrator
         {
            VectorDomainLFGradIntegrator *vdlfgi;
            vdlfgi = new VectorDomainLFGradIntegrator(vector_constant_coeff);
            vdlfgi->SetIntRule(ir);
            lf[i]->AddDomainIntegrator(vdlfgi);
         }
         else { MFEM_ABORT("Problem not specified!"); }
      }
   }
};

////////////////////////////////////////////////////////////////////////////////
/// TEST for LinearFormExtension
template<int DIM, int VDIM>
struct Test: public LinExt
{
   Test(int problem, int order): LinExt(problem,order,DIM,VDIM)
   {
      MFEM_DEVICE_SYNC;
   }

   void benchmark() override
   {
      lf[0]->Assemble();
      lf[1]->Assemble();
      const double dtd = (*lf[1]) * (*lf[1]);
      const double rtr = (*lf[0]) * (*lf[0]);
      MFEM_VERIFY(almost_equal(dtd,rtr,10), "almost_equal test error!");
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }
};

/// Scalar Linear Form Extension Tests
#define LinExtTest(Problem,Kernel,DIM,VDIM)\
static void TEST_##Kernel##_##DIM##D(bm::State &state){\
   const int order = state.range(0);\
   Test<DIM,VDIM> ker(Problem,order);\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(TEST_##Kernel##_##DIM##D)->DenseRange(1,6)->Unit(bm::kMillisecond);

/// Scalar linear form tests
LinExtTest(1,DomainLF,2,1)
LinExtTest(1,DomainLF,3,1)

/// Vector linear form tests
LinExtTest(2,VectorDomainLF,2,2)
LinExtTest(2,VectorDomainLF,3,3)

/// Grad linear form tests
LinExtTest(3,DomainLFGrad,2,1)
LinExtTest(3,DomainLFGrad,3,1)

/// Vector Grad linear form tests
LinExtTest(4,VectorDomainLFGrad,2,2)
LinExtTest(4,VectorDomainLFGrad,3,3)

////////////////////////////////////////////////////////////////////////////////
/// BENCH for LinearFormExtension
template<int DIM, int VDIM, enum LinearAssemblyLevel LEVEL>
struct Bench: public LinExt
{
   const int i;
   Bench(int problem, int order): LinExt(problem, order, DIM, VDIM),
      i(LEVEL == LinearAssemblyLevel::LEGACY ? 0 : 1)
   {
      MFEM_DEVICE_SYNC;
   }

   void benchmark() override
   {
      lf[i]->Assemble();
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }
};

/// Linear Form Extension Scalar Benchs
#define LinExtBench(Problem,Kernel,LVL,DIM,VDIM)\
static void BENCH_##LVL##_##Kernel##_##DIM##D(bm::State &state){\
   const int order = state.range(0);\
   Bench<DIM,VDIM,LinearAssemblyLevel::LVL> ker(Problem, order);\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);}\
BENCHMARK(BENCH_##LVL##_##Kernel##_##DIM##D)->DenseRange(1,6)->Unit(bm::kMicrosecond);

/// 2D scalar linear form bench
LinExtBench(1,DomainLF,LEGACY,2,1)
LinExtBench(1,DomainLF,FULL,2,1)

/// 3D scalar linear form bench
LinExtBench(1,DomainLF,LEGACY,3,1)
LinExtBench(1,DomainLF,FULL,3,1)

/// 2D Vector linear form bench
LinExtBench(2,VectorDomainLF,LEGACY,2,2)
LinExtBench(2,VectorDomainLF,FULL,2,2)

/// 3D Vector linear form bench
LinExtBench(2,VectorDomainLF,LEGACY,3,3)
LinExtBench(2,VectorDomainLF,FULL,3,3)

/// 2D Grad Scalar linear form bench
LinExtBench(3,DomainLFGrad,LEGACY,2,1)
LinExtBench(3,DomainLFGrad,FULL,2,1)

/// 3D Grad Scalar linear form bench
LinExtBench(3,DomainLFGrad,LEGACY,3,1)
LinExtBench(3,DomainLFGrad,FULL,3,1)

/// 2D Grad Vector linear form bench
LinExtBench(4,VectorDomainLFGrad,LEGACY,2,2)
LinExtBench(4,VectorDomainLFGrad,FULL,2,2)

/// 3D Grad Vector linear form bench
LinExtBench(4,VectorDomainLFGrad,LEGACY,3,3)
LinExtBench(4,VectorDomainLFGrad,FULL,3,3)

/** ****************************************************************************
 * @brief main entry point
 * --benchmark_filter=TEST
 * --benchmark_filter=BENCH_FULL
 * --benchmark_context=device=cuda
 */
int main(int argc, char *argv[])
{
   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string device_config = "cpu";
   if (bmi::global_context != nullptr)
   {
      const auto device = bmi::global_context->find("device");
      if (device != bmi::global_context->end())
      {
         mfem::out << device->first << " : " << device->second << std::endl;
         device_config = device->second;
      }
   }
   Device device(device_config.c_str());
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }
   bm::RunSpecifiedBenchmarks(&CR);
   return 0;
}

#endif // MFEM_USE_BENCHMARK
