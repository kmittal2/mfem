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
//
//                      -------------------------------
//                      Convergence Rates Test (Serial)
//                      -------------------------------
//
// Compile with: make rates
//
// Sample runs:  rates -m ../../data/inline-segment.mesh -sr 4 -prob 0 -o 1
//               rates -m ../../data/inline-quad.mesh -sr 3 -prob 0 -o 2
//               rates -m ../../data/inline-quad.mesh -sr 3 -prob 1 -o 2
//               rates -m ../../data/inline-quad.mesh -sr 3 -prob 2 -o 2
//               rates -m ../../data/inline-tri.mesh -sr 2 -prob 2 -o 3
//               rates -m ../../data/star.mesh -sr 2 -prob 1 -o 4
//               rates -m ../../data/fichera.mesh -sr 3 -prob 2 -o 1
//               rates -m ../../data/inline-wedge.mesh -sr 1 -prob 0 -o 2
//               rates -m ../../data/inline-hex.mesh -sr 1 -prob 1 -o 2
//               rates -m ../../data/square-disc.mesh -sr 2 -prob 1 -o 1
//               rates -m ../../data/star.mesh -sr 2 -prob 3 -o 2
//               rates -m ../../data/star.mesh -sr 2 -prob 3 -o 2 -j 0
//               rates -m ../../data/inline-hex.mesh -sr 1 -prob 3 -o 1
//
// Description:  This example code demonstrates the use of MFEM to define and
//               solve finite element problem for various discretizations and
//               provide convergence rates in serial.
//
//               prob 0: H1 projection:
//                       (grad u, grad v) + (u,v) = (grad u_exact, grad v) + (u_exact, v)
//               prob 1: H(curl) projection
//                       (curl u, curl v) + (u,v) = (curl u_exact, curl v) + (u_exact, v)
//               prob 2: H(div) projection
//                       (div  u, div  v) + (u,v) = (div  u_exact, div  v) + (u_exact, v)
//               prob 3: DG discretization for the Poisson problem
//                       -Delta u = f

#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;

// Exact solution parameters:
double sol_s[3] = { -0.32, 0.15, 0.24 };
double sol_k[3] = { 1.21, 1.45, 1.37 };

// H1
double scalar_u_exact(const Vector &x);
double rhs_func(const Vector &x);
void gradu_exact(const Vector &x, Vector &gradu);

// Vector FE
void vector_u_exact(const Vector &x, Vector & vector_u);
// H(curl)
void curlu_exact(const Vector &x, Vector &curlu);
// H(div)
double divu_exact(const Vector &x);

int dim;
int prob=0;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   bool visualization = 1;
   int sr = 1;
   int jump_scaling_type = 1;
   double sigma = -1.0;
   double kappa = -1.0;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&prob, "-prob", "--problem",
                  "Problem kind: 0: H1, 1: H(curl), 2: H(div), 3: DG ");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&jump_scaling_type, "-j", "--jump-scaling",
                  "Scaling of the jump error for DG methods: "
                  "0: no scaling, 1: 1/h, 2: p^2/h");
   args.AddOption(&sr, "-sr", "--serial_ref",
                  "Number of serial refinements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (prob >3 || prob <0) { prob = 0; } // default problem = H1
   if (prob == 3)
   {
      if (kappa < 0)
      {
         kappa = (order+1)*(order+1);
      }
   }
   args.PrintOptions(cout);

   // 2. Read the (serial) mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // 3. Refine the serial mesh on all processors to increase the resolution.
   mesh->UniformRefinement();

   // 4. Define a finite element space on the parallel mesh.
   FiniteElementCollection *fec=nullptr;
   switch (prob)
   {
      case 0: fec = new H1_FECollection(order,dim);   break;
      case 1: fec = new ND_FECollection(order,dim);   break;
      case 2: fec = new RT_FECollection(order-1,dim); break;
      case 3: fec = new DG_FECollection(order,dim); break;
      default: break;
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   // 5. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace.
   GridFunction x(fespace);
   x = 0.0;

   // 6. Set up the linear form b(.) and the bilinear form a(.,.).
   FunctionCoefficient *f=nullptr;
   FunctionCoefficient *scalar_u=nullptr;
   FunctionCoefficient *divu=nullptr;
   VectorFunctionCoefficient *vector_u=nullptr;
   VectorFunctionCoefficient *gradu=nullptr;
   VectorFunctionCoefficient *curlu=nullptr;

   ConstantCoefficient one(1.0);
   LinearForm b(fespace);
   BilinearForm a(fespace);

   switch (prob)
   {
      case 0:
         //(grad u_ex, grad v) + (u_ex,v)
         scalar_u = new FunctionCoefficient(scalar_u_exact);
         gradu = new VectorFunctionCoefficient(dim,gradu_exact);
         b.AddDomainIntegrator(new DomainLFGradIntegrator(*gradu));
         b.AddDomainIntegrator(new DomainLFIntegrator(*scalar_u));

         // (grad u, grad v) + (u,v)
         a.AddDomainIntegrator(new DiffusionIntegrator(one));
         a.AddDomainIntegrator(new MassIntegrator(one));

         break;
      case 1:
         //(curl u_ex, curl v) + (u_ex,v)
         vector_u = new VectorFunctionCoefficient(dim,vector_u_exact);
         curlu = new VectorFunctionCoefficient((dim==3)?dim:1,curlu_exact);
         b.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(*curlu));
         b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*vector_u));

         // (curl u, curl v) + (u,v)
         a.AddDomainIntegrator(new CurlCurlIntegrator(one));
         a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
         break;

      case 2:
         //(div u_ex, div v) + (u_ex,v)
         vector_u = new VectorFunctionCoefficient(dim,vector_u_exact);
         divu = new FunctionCoefficient(divu_exact);
         b.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(*divu));
         b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*vector_u));

         // (div u, div v) + (u,v)
         a.AddDomainIntegrator(new DivDivIntegrator(one));
         a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
         break;

      case 3:
         scalar_u = new FunctionCoefficient(scalar_u_exact);
         f = new FunctionCoefficient(rhs_func);
         gradu = new VectorFunctionCoefficient(dim,gradu_exact);
         b.AddDomainIntegrator(new DomainLFIntegrator(*f));
         b.AddBdrFaceIntegrator(
            new DGDirichletLFIntegrator(*scalar_u, one, sigma, kappa));
         a.AddDomainIntegrator(new DiffusionIntegrator(one));
         a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
         a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
         break;

      default:
         break;
   }

   // 7. Perform successive refinements, compute the errors and the
   //    corresponding rates of convergence.
   ConvergenceStudy rates;
   for (int l = 0; l <= sr; l++)
   {
      b.Assemble();
      a.Assemble();
      a.Finalize();
      const SparseMatrix &A = a.SpMat();
      GSSmoother M(A);
      if (prob == 3 && sigma != -1.0)
      {
         GMRES(A, M, b, x, 0, 500, 10, 1e-12, 0.0);
      }
      else
      {
         PCG(A, M, b, x, 0, 500, 1e-12, 0.0);
      }

      JumpScaling js(1.0, jump_scaling_type == 2 ? JumpScaling::P_SQUARED_OVER_H
                     : jump_scaling_type == 1 ? JumpScaling::ONE_OVER_H
                     : JumpScaling::CONSTANT);

      switch (prob)
      {
         case 0: rates.AddH1GridFunction(&x,scalar_u,gradu); break;
         case 1: rates.AddHcurlGridFunction(&x,vector_u,curlu); break;
         case 2: rates.AddHdivGridFunction(&x,vector_u,divu);  break;
         case 3: rates.AddL2GridFunction(&x,scalar_u,gradu,&one,js); break;
      }

      if (l==sr) { break; }

      mesh->UniformRefinement();
      fespace->Update();
      a.Update();
      b.Update();
      x.Update();
   }
   rates.Print();

   // 8. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x <<
               "window_title 'Numerical Solution' "
               << flush;
   }

   // 9. Free the used memory.
   delete f;
   delete scalar_u;
   delete divu;
   delete vector_u;
   delete gradu;
   delete curlu;
   delete fespace;
   delete fec;
   delete mesh;
   return 0;
}

double rhs_func(const Vector &x)
{
   double val = 1.0, lap = 0.0;
   for (int d = 0; d < x.Size(); d++)
   {
      const double f = sin(M_PI*(sol_s[d]+sol_k[d]*x(d)));
      val *= f;
      lap = lap*f + val*M_PI*M_PI*sol_k[d]*sol_k[d];
   }
   return lap;
}

double scalar_u_exact(const Vector &x)
{
   double val = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      val *= sin(M_PI*(sol_s[d]+sol_k[d]*x(d)));
   }
   return val;
}

void gradu_exact(const Vector &x, Vector &grad)
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

void vector_u_exact(const Vector &x, Vector & vector_u)
{
   vector_u.SetSize(x.Size());
   vector_u=0.0;
   vector_u[0] = scalar_u_exact(x);
}

// H(curl)
void curlu_exact(const Vector &x, Vector &curlu)
{
   Vector grad;
   gradu_exact(x,grad);
   int n = (x.Size()==3)?3:1;
   curlu.SetSize(n);
   if (x.Size()==3)
   {
      curlu[0] = 0.0;
      curlu[1] = grad[2];
      curlu[2] = -grad[1];
   }
   else if (x.Size()==2)
   {
      curlu[0] = -grad[1];
   }
}

// H(div)
double divu_exact(const Vector &x)
{
   Vector grad;
   gradu_exact(x,grad);

   return grad[0];
}
