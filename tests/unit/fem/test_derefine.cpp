// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

namespace derefine
{

int dimension;
double coeff(const Vector& x)
{
   if (dimension == 2)
   {
      return sin(10.0*(x[0]+x[1]));
   }
   else
   {
      return sin(10.0*(x[0]+x[1]+x[2]));
   }
}

TEST_CASE("Derefine")
{
   for (dimension = 2; dimension <= 3; ++dimension)
   {
      for (int order = 0; order <= 2; ++order)
      {
         for (int map_type = FiniteElement::VALUE; map_type <= FiniteElement::INTEGRAL;
              ++map_type)
         {
            const int ne = 8;
            Mesh mesh;
            if (dimension == 2)
            {
               mesh = Mesh::MakeCartesian2D(
                         ne, ne, Element::QUADRILATERAL, true, 1.0, 1.0);
            }
            else
            {
               mesh = Mesh::MakeCartesian3D(
                         ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
            }

            mesh.EnsureNCMesh();
            mesh.SetCurvature(std::max(order,1), false, dimension, Ordering::byNODES);

            L2_FECollection fec(order, dimension, BasisType::Positive, map_type);

            FiniteElementSpace fespace(&mesh, &fec);

            GridFunction x(&fespace);

            FunctionCoefficient c(coeff);
            x.ProjectCoefficient(c);

            fespace.Update();
            x.Update();

            Array<Refinement> refinements;
            refinements.Append(Refinement(1));
            refinements.Append(Refinement(2));

            int nonconformity_limit = 0; // 0 meaning allow unlimited ratio

            // First refine two elements.
            mesh.GeneralRefinement(refinements, 1, nonconformity_limit);

            fespace.Update();
            x.Update();

            // Now refine one more element and then derefine it, comparing x before and after.
            Vector diff(x);

            refinements.DeleteAll();
            refinements.Append(Refinement(2));
            mesh.GeneralRefinement(refinements, 1, nonconformity_limit);

            fespace.Update();
            x.Update();

            // Derefine by setting 0 error on the fine elements in coarse element 2.
            Table coarse_to_fine_;
            const CoarseFineTransformations &rtrans = mesh.GetRefinementTransforms();
            rtrans.MakeCoarseToFineTable(coarse_to_fine_);
            Array<int> tabrow;

            Vector local_err(mesh.GetNE());
            double threshold = 1.0;
            local_err = 2*threshold;
            coarse_to_fine_.GetRow(2, tabrow);
            for (int j = 0; j < tabrow.Size(); j++) { local_err(tabrow[j]) = 0.0; }
            mesh.DerefineByError(local_err, threshold, 0, 1);

            fespace.Update();
            x.Update();

            diff -= x;
            REQUIRE(diff.Norml2() / x.Norml2() < 1e-11);
         }
      }
   }
}

// project linear function on one element.
// refine, project again.
double linear_coeff(const Vector& x)
{
   if (dimension == 2)
   {
      return x[0]+x[1];
   }
   else
   {
      return x[0]+x[1]+x[2];
   }
}

// for order 0, linear, order 1, quadratic, etc.

struct PolyCoeff
{
   static int order_;

   static double poly_coeff(const Vector& x)
      {
         if (dimension == 2)
         {
            return
                pow(x[0],order_+1)
               +pow(x[1],order_+1);
         }
         else
         {
            return
               pow(x[0],order_+1)
               +pow(x[1],order_+1)
               +pow(x[2],order_+1);
         }
      }
};
int PolyCoeff::order_ = -1;

void test_derefine_L2_element(int order, int basis_type)
{
   Mesh mesh;
   if (dimension == 2) {
      mesh = Mesh::MakeCartesian2D(
         1, 1, Element::QUADRILATERAL, true, 1.0, 1.0);
   }
   if (dimension == 3) {
      mesh = Mesh::MakeCartesian3D(
         1, 1, 1, Element::HEXAHEDRON, true, 1.0, 1.0, 1.0);
   }
   mesh.EnsureNCMesh();

   L2_FECollection fec(order, dimension, basis_type);
   FiniteElementSpace fespace(&mesh, &fec);
   GridFunction x(&fespace);

   PolyCoeff pcoeff;
   pcoeff.order_ = order;
   FunctionCoefficient c(PolyCoeff::poly_coeff);
   x.ProjectCoefficient(c);

   Vector coarse_projection = x;

   Array<Refinement> refinements;
   refinements.Append(Refinement(0));
   mesh.GeneralRefinement(refinements);

   fespace.Update();
   x.Update();

   // re-project to get function that isn't exactly representable in
   // the coarse space.
   x.ProjectCoefficient(c);

   Vector local_err(mesh.GetNE());
   local_err = 0.;
   double threshold = 1.0;
   mesh.DerefineByError(local_err, threshold);

   fespace.Update();
   x.Update();

   Vector coarse_derefinement = x;

   REQUIRE(coarse_projection.Norml2() -coarse_derefinement.Norml2() < 1e-11);
}

void test_derefine_H1_element(int order, int basis_type)
{
   Mesh mesh = Mesh::MakeCartesian2D(
      1, 1, Element::QUADRILATERAL, true, 1.0, 1.0);
   mesh.EnsureNCMesh();

   H1_FECollection fec(order, dimension, basis_type);
   FiniteElementSpace fespace(&mesh, &fec);
   GridFunction x(&fespace);

   FunctionCoefficient c(linear_coeff);
   x.ProjectCoefficient(c);

   Vector coarse_projection = x;

   Array<Refinement> refinements;
   refinements.Append(Refinement(0));
   mesh.GeneralRefinement(refinements);

   fespace.Update();
   x.Update();

   // re-project to get function that isn't exactly representable in
   // the coarse space.
   x.ProjectCoefficient(c);

   Vector local_err(mesh.GetNE());
   local_err = 0.;
   double threshold = 1.0;
   mesh.DerefineByError(local_err, threshold);

   fespace.Update();
   x.Update();

   Vector coarse_derefinement = x;

   REQUIRE(coarse_projection.Norml2() -coarse_derefinement.Norml2() < 1e-11);
}

TEST_CASE("Coarsen L2 Element, Verify Projection")
{
   dimension = 2;
   test_derefine_L2_element(0, BasisType::Positive);
   test_derefine_L2_element(1, BasisType::Positive);
   // test_derefine_L2_element(2, BasisType::Positive);

   // test_derefine_L2_element(0, BasisType::GaussLegendre);
   // test_derefine_L2_element(1, BasisType::GaussLegendre);
   // test_derefine_L2_element(2, BasisType::GaussLegendre);

   // test_derefine_L2_element(0, BasisType::GaussLobatto);
   // test_derefine_L2_element(1, BasisType::GaussLobatto);
   // test_derefine_L2_element(2, BasisType::GaussLobatto);

   // dimension = 3;
   // test_derefine_L2_element(0, BasisType::Positive);
   // test_derefine_L2_element(1, BasisType::Positive);
   // test_derefine_L2_element(2, BasisType::Positive);

   // test_derefine_L2_element(0, BasisType::GaussLegendre);
   // test_derefine_L2_element(1, BasisType::GaussLegendre);
   // test_derefine_L2_element(2, BasisType::GaussLegendre);

   // test_derefine_L2_element(0, BasisType::GaussLobatto);
   // test_derefine_L2_element(1, BasisType::GaussLobatto);
   // test_derefine_L2_element(2, BasisType::GaussLobatto);
}

TEST_CASE("Coarsen H1 Element, Verify Projection")
{
   // dimension = 2;
   // test_derefine_H1_element(1, BasisType::GaussLobatto);

   // test_derefine_H1_element(1, BasisType::GaussLegendre);
   // test_derefine_H1_element(2, BasisType::GaussLegendre);
}

#ifdef MFEM_USE_MPI
TEST_CASE("ParDerefine", "[Parallel]")
{
   for (dimension = 2; dimension <= 3; ++dimension)
   {
      for (int order = 0; order <= 2; ++order)
      {
         for (int map_type = FiniteElement::VALUE; map_type <= FiniteElement::INTEGRAL;
              ++map_type)
         {
            const int ne = 8;
            Mesh mesh;
            if (dimension == 2)
            {
               mesh = Mesh::MakeCartesian2D(
                         ne, ne, Element::QUADRILATERAL, true, 1.0, 1.0);
            }
            else
            {
               mesh = Mesh::MakeCartesian3D(
                         ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
            }

            mesh.EnsureNCMesh();
            mesh.SetCurvature(std::max(order,1), false, dimension, Ordering::byNODES);

            ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

            L2_FECollection fec(order, dimension, BasisType::Positive, map_type);
            ParFiniteElementSpace fespace(pmesh, &fec);

            ParGridFunction x(&fespace);

            FunctionCoefficient c(coeff);
            x.ProjectCoefficient(c);

            fespace.Update();
            x.Update();

            // Refine two elements on each process and then derefine, comparing
            // x before and after.
            Vector diff(x);

            Array<Refinement> refinements;
            refinements.Append(Refinement(1));
            refinements.Append(Refinement(2));

            int nonconformity_limit = 0; // 0 meaning allow unlimited ratio

            pmesh->GeneralRefinement(refinements, 1, nonconformity_limit);

            fespace.Update();
            x.Update();

            // Derefine by setting 0 error on all fine elements.
            Vector local_err(pmesh->GetNE());
            double threshold = 1.0;
            local_err = 0.0;
            pmesh->DerefineByError(local_err, threshold, 0, 1);

            fespace.Update();
            x.Update();

            diff -= x;
            REQUIRE(diff.Norml2() / x.Norml2() < 1e-11);

            delete pmesh;
         }
      }
   }
}
#endif
}
