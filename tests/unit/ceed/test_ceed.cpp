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

#include "catch.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../../../fem/libceed/coefficient.hpp"

using namespace mfem;

namespace ceed_test
{

#ifdef MFEM_USE_CEED

enum class CeedCoeffType { Const, Grid, Quad, VecConst, VecGrid, VecQuad };

double coeff_function(const Vector &x)
{
   return 1.0 + x[0]*x[0];
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   const double w = 1.0 + x[0]*x[0];
   switch (dim)
   {
      case 1: v(0) = w; break;
      case 2: v(0) = w*sqrt(2./3.); v(1) = w*sqrt(1./3.); break;
      case 3: v(0) = w*sqrt(3./6.); v(1) = w*sqrt(2./6.); v(2) = w*sqrt(1./6.); break;
   }
}

std::string getString(AssemblyLevel assembly)
{
   switch (assembly)
   {
   case AssemblyLevel::NONE:
      return "NONE";
      break;
   case AssemblyLevel::PARTIAL:
      return "PARTIAL";
      break;
   case AssemblyLevel::ELEMENT:
      return "ELEMENT";
      break;
   case AssemblyLevel::FULL:
      return "FULL";
      break;
   case AssemblyLevel::LEGACYFULL:
      return "LEGACYFULL";
      break;
   }
   mfem_error("Unknown AssemblyLevel.");
   return "";
}

std::string getString(CeedCoeffType coeff_type)
{
   switch (coeff_type)
   {
   case CeedCoeffType::Const:
      return "Const";
      break;
   case CeedCoeffType::Grid:
      return "Grid";
      break;
   case CeedCoeffType::Quad:
      return "Quad";
      break;
   case CeedCoeffType::VecConst:
      return "VecConst";
      break;
   case CeedCoeffType::VecGrid:
      return "VecGrid";
      break;
   case CeedCoeffType::VecQuad:
      return "VecQuad";
      break;
   }
   mfem_error("Unknown CeedCoeffType.");
   return "";
}

enum class Problem {Mass, Convection, Diffusion, VectorMass, VectorDiffusion};

std::string getString(Problem pb)
{
   switch (pb)
   {
   case Problem::Mass:
      return "Mass";
      break;
   case Problem::Convection:
      return "Convection";
      break;
   case Problem::Diffusion:
      return "Diffusion";
      break;
   case Problem::VectorMass:
      return "VectorMass";
      break;
   case Problem::VectorDiffusion:
      return "VectorDiffusion";
      break;
   }
   mfem_error("Unknown Problem.");
   return "";
}

enum class NLProblem {Convection};

std::string getString(NLProblem pb)
{
   switch (pb)
   {
   case NLProblem::Convection:
      return "Convection";
      break;
   }
   mfem_error("Unknown Problem.");
   return "";
}

void InitCoeff(Mesh &mesh, FiniteElementCollection &fec, const int dim,
               const CeedCoeffType coeff_type, GridFunction *&gf,
               FiniteElementSpace *& coeff_fes,
               Coefficient *&coeff, VectorCoefficient *&vcoeff)
{
   switch (coeff_type)
   {
      case CeedCoeffType::Const:
         coeff = new ConstantCoefficient(1.0);
         break;
      case CeedCoeffType::Grid:
      {
         FunctionCoefficient f_coeff(coeff_function);
         coeff_fes = new FiniteElementSpace(&mesh, &fec);
         gf = new GridFunction(coeff_fes);
         gf->ProjectCoefficient(f_coeff);
         coeff = new GridFunctionCoefficient(gf);
         break;
      }
      case CeedCoeffType::Quad:
         coeff = new FunctionCoefficient(coeff_function);
         break;
      case CeedCoeffType::VecConst:
      {
         Vector val(dim);
         for (int i = 0; i < dim; i++)
         {
            val(i) = 1.0;
         }
         vcoeff = new VectorConstantCoefficient(val);
         break;
      }
      case CeedCoeffType::VecGrid:
      {
         VectorFunctionCoefficient f_vcoeff(dim, velocity_function);
         coeff_fes = new FiniteElementSpace(&mesh, &fec, dim);
         gf = new GridFunction(coeff_fes);
         gf->ProjectCoefficient(f_vcoeff);
         vcoeff = new VectorGridFunctionCoefficient(gf);
         break;
      }
      case CeedCoeffType::VecQuad:
         vcoeff = new VectorFunctionCoefficient(dim, velocity_function);
         break;
   }
}

void test_ceed_operator(const char* input, int order, const CeedCoeffType coeff_type,
                        const Problem pb, const AssemblyLevel assembly)
{
   std::string section = "assembly: " + getString(assembly) + "\n" +
                         "coeff_type: " + getString(coeff_type) + "\n" +
                         "pb: " + getString(pb) + "\n" +
                         "order: " + std::to_string(order) + "\n" +
                         "mesh: " + input;
   INFO(section);
   Mesh mesh(input, 1, 1);
   mesh.EnsureNodes();
   int dim = mesh.Dimension();
   H1_FECollection fec(order, dim);

   // Coefficient Initialization
   GridFunction *gf = nullptr;
   FiniteElementSpace *coeff_fes = nullptr;
   Coefficient *coeff = nullptr;
   VectorCoefficient *vcoeff = nullptr;
   InitCoeff(mesh, fec, dim, coeff_type, gf, coeff_fes, coeff, vcoeff);

   // Build the BilinearForm
   bool vecOp = pb == Problem::VectorMass || pb == Problem::VectorDiffusion;
   const int vdim = vecOp ? dim : 1;
   FiniteElementSpace fes(&mesh, &fec, vdim);

   BilinearForm k_test(&fes);
   BilinearForm k_ref(&fes);
   switch (pb)
   {
   case Problem::Mass:
      k_ref.AddDomainIntegrator(new MassIntegrator(*coeff));
      k_test.AddDomainIntegrator(new MassIntegrator(*coeff));
      break;
   case Problem::Convection:
      k_ref.AddDomainIntegrator(new ConvectionIntegrator(*vcoeff,-1));
      k_test.AddDomainIntegrator(new ConvectionIntegrator(*vcoeff,-1));
      break;
   case Problem::Diffusion:
      k_ref.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
      k_test.AddDomainIntegrator(new DiffusionIntegrator(*coeff));
      break;
   case Problem::VectorMass:
      k_ref.AddDomainIntegrator(new VectorMassIntegrator(*coeff));
      k_test.AddDomainIntegrator(new VectorMassIntegrator(*coeff));
      break;
   case Problem::VectorDiffusion:
      k_ref.AddDomainIntegrator(new VectorDiffusionIntegrator(*coeff));
      k_test.AddDomainIntegrator(new VectorDiffusionIntegrator(*coeff));
      break;
   }

   k_ref.Assemble();
   k_ref.Finalize();

   k_test.SetAssemblyLevel(assembly);
   k_test.Assemble();

   // Compare ceed with mfem.
   GridFunction x(&fes), y_ref(&fes), y_test(&fes);

   x.Randomize(1);

   k_ref.Mult(x,y_ref);
   k_test.Mult(x,y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12);
   delete gf;
   delete coeff_fes;
   delete coeff;
   delete vcoeff;
}

void test_ceed_nloperator(const char* input, int order, const CeedCoeffType coeff_type,
                          const NLProblem pb, const AssemblyLevel assembly)
{
   std::string section = "assembly: " + getString(assembly) + "\n" +
                         "coeff_type: " + getString(coeff_type) + "\n" +
                         "pb: " + getString(pb) + "\n" +
                         "order: " + std::to_string(order) + "\n" +
                         "mesh: " + input;
   INFO(section);
   Mesh mesh(input, 1, 1);
   mesh.EnsureNodes();
   int dim = mesh.Dimension();
   H1_FECollection fec(order, dim);

   // Coefficient Initialization
   GridFunction *gf = nullptr;
   FiniteElementSpace *coeff_fes = nullptr;
   Coefficient *coeff = nullptr;
   VectorCoefficient *vcoeff = nullptr;
   InitCoeff(mesh, fec, dim, coeff_type, gf, coeff_fes, coeff, vcoeff);

   // Build the NonlinearForm
   bool vecOp = pb == NLProblem::Convection;
   const int vdim = vecOp ? dim : 1;
   FiniteElementSpace fes(&mesh, &fec, vdim);

   NonlinearForm k_test(&fes);
   NonlinearForm k_ref(&fes);
   switch (pb)
   {
   case NLProblem::Convection:
      k_ref.AddDomainIntegrator(new VectorConvectionNLFIntegrator(*coeff));
      k_test.AddDomainIntegrator(new VectorConvectionNLFIntegrator(*coeff));
      break;
   }

   k_test.SetAssemblyLevel(assembly);
   k_test.Setup();
   k_ref.Setup();

   // Compare ceed with mfem.
   GridFunction x(&fes), y_ref(&fes), y_test(&fes);

   x.Randomize(1);

   k_ref.Mult(x,y_ref);
   k_test.Mult(x,y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12);
   delete gf;
   delete coeff_fes;
   delete coeff;
   delete vcoeff;
}

TEST_CASE("CEED mass & diffusion", "[CEED]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::Const,CeedCoeffType::Grid,CeedCoeffType::Quad);
   auto pb = GENERATE(Problem::Mass,Problem::Diffusion,
                      Problem::VectorMass,Problem::VectorDiffusion);
   auto order = GENERATE(1);
   auto mesh = GENERATE("../../data/inline-quad.mesh","../../data/inline-hex.mesh",
                        "../../data/periodic-square.mesh",
                        "../../data/star-q2.mesh","../../data/fichera-q2.mesh",
                        "../../data/amr-quad.mesh","../../data/fichera-amr.mesh");
   test_ceed_operator(mesh, order, coeff_type, pb, assembly);
} // test case

TEST_CASE("CEED convection", "[CEED],[Convection]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::VecConst,CeedCoeffType::VecGrid,
                              CeedCoeffType::VecQuad);
   auto pb = GENERATE(Problem::Convection);
   auto order = GENERATE(1);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera-q2.mesh",
                        "../../data/amr-quad.mesh",
                        "../../data/fichera-amr.mesh");
   test_ceed_operator(mesh, order, coeff_type, pb, assembly);
} // test case

TEST_CASE("CEED non-linear convection", "[CEED],[NLConvection]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::Const,CeedCoeffType::Grid,CeedCoeffType::Quad);
   auto pb = GENERATE(NLProblem::Convection);
   auto order = GENERATE(1);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/periodic-square.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera.mesh");
   test_ceed_nloperator(mesh, order, coeff_type, pb, assembly);
} // test case

#endif

} // namespace ceed_test
