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

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <memory>
#include <array>

using namespace mfem;

#if defined(MFEM_USE_MPI)

namespace testhelper
{
double SmoothSolutionX(const mfem::Vector& x)
{
   return x(0);
}

double SmoothSolutionY(const mfem::Vector& x)
{
   return x(1);
}

double SmoothSolutionZ(const mfem::Vector& x)
{
   return x(2);
}

double NonsmoothSolutionX(const mfem::Vector& x)
{
   return std::abs(x(0)-0.5);
}

double NonsmoothSolutionY(const mfem::Vector& x)
{
   return std::abs(x(1)-0.5);
}

double NonsmoothSolutionZ(const mfem::Vector& x)
{
   return std::abs(x(2)-0.5);
}
}

TEST_CASE("Kelly Error Estimator on 2D NCMesh",
          "[NCMesh], [Parallel]")
{
   // Setup
   const auto order = GENERATE(1, 3, 5);
   auto mesh = new Mesh(2, 2, Element::QUADRILATERAL);

   // Make the mesh NC
   mesh->EnsureNCMesh();
   {
      Array<int> elements_to_refine(1);
      elements_to_refine[0] = 1;
      mesh->GeneralRefinement(elements_to_refine, 1, 0);
   }

   auto pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   H1_FECollection fe_coll(order, pmesh->Dimension());
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   SECTION("Perfect Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionX);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Perfect Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionY);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Nonsmooth Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionX);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   SECTION("Nonsmooth Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionY);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   delete pmesh;
}

TEST_CASE("Kelly Error Estimator on 2D NCMesh embedded in 3D",
          "[NCMesh], [Parallel]")
{
   // Setup
   const auto order = GENERATE(1, 3, 5);

   // Manually construct embedded mesh
   std::array<double, 4*3> vertices =
   {
      0.0,0.0,0.0,
      0.0,1.0,0.0,
      1.0,1.0,0.0,
      1.0,0.0,0.0
   };

   std::array<int, 4> element_indices =
   {
      0,1,2,3
   };

   std::array<int, 1> element_attributes =
   {
      1
   };

   std::array<int, 8> boundary_indices =
   {
      0,1,
      1,2,
      2,3,
      3,0
   };

   std::array<int, 4> boundary_attributes =
   {
      1,
      1,
      1,
      1
   };

   auto mesh = new Mesh(
      vertices.data(), 4,
      element_indices.data(), Geometry::SQUARE,
      element_attributes.data(), 1,
      boundary_indices.data(), Geometry::SEGMENT,
      boundary_attributes.data(), 4,
      2, 3
   );
   mesh->UniformRefinement();
   mesh->Finalize();

   // Make the mesh NC
   mesh->EnsureNCMesh();
   {
      Array<int> elements_to_refine(1);
      elements_to_refine[0] = 1;
      mesh->GeneralRefinement(elements_to_refine, 1, 0);
   }

   auto pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   H1_FECollection fe_coll(order, pmesh->Dimension());
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   SECTION("Perfect Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionX);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Perfect Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionY);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Nonsmooth Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionX);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   SECTION("Nonsmooth Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionY);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   delete pmesh;
}

TEST_CASE("Kelly Error Estimator on 3D NCMesh",
          "[NCMesh], [Parallel]")
{
   // Setup
   const auto order = GENERATE(1, 3, 5);
   auto mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON);

   // Make the mesh NC
   mesh->EnsureNCMesh();
   {
      Array<int> elements_to_refine(1);
      elements_to_refine[0] = 1;
      mesh->GeneralRefinement(elements_to_refine, 1, 0);
   }

   auto pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   H1_FECollection fe_coll(order, pmesh->Dimension());
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   SECTION("Perfect Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionX);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Perfect Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionY);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Perfect Approximation Z")
   {
      FunctionCoefficient u_analytic(testhelper::SmoothSolutionZ);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) == MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() == MFEM_Approx(0.0));
   }

   SECTION("Nonsmooth Approximation X")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionX);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   SECTION("Nonsmooth Approximation Y")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionY);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= MFEM_Approx(0.0));
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   SECTION("Nonsmooth Approximation Z")
   {
      FunctionCoefficient u_analytic(testhelper::NonsmoothSolutionZ);
      ParGridFunction u_gf(&fespace);
      u_gf.ProjectCoefficient(u_analytic);

      L2_FECollection flux_fec(order, pmesh->Dimension());
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec, pmesh->SpaceDimension());
      DiffusionIntegrator di;
      KellyErrorEstimator estimator(di, u_gf, flux_fes);

      auto &local_errors = estimator.GetLocalErrors();
      for (int i=0; i<local_errors.Size(); i++)
      {
         REQUIRE(local_errors(i) >= 0.0);
      }
      REQUIRE(estimator.GetTotalError() > 0.0);
   }

   delete pmesh;
}

#endif
