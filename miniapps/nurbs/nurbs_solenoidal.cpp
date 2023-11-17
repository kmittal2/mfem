// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
//        ------------------------------------------------------------
//        NURBS Solenoidal Miniapp: Project solenoidal velocity
//        ------------------------------------------------------------
//
//
// Compile with: make nurbs_solenoidal
//
// Sample runs:  nurbs_solenoidal -m ../../data/square-nurbs.mesh -o 2
//               nurbs_solenoidal -m ../../data/cube-nurbs.mesh -o 2
//               nurbs_solenoidal -m ../../data/pipe-nurbs-2d.mesh -o 2
//               nurbs_solenoidal -m ../../data/square-disc-nurbs.mesh -o -1
//               nurbs_solenoidal -m ../../data/disc-nurbs.mesh -o -1
//               nurbs_solenoidal -m ../../data/pipe-nurbs.mesh -o -1
//               nurbs_solenoidal -m ../../data/beam-hex-nurbs.mesh -pm 1 -ps 2
//
//
// Description:  This code projects a velocity field, and forces this field
//               to be solenoidal, viz. the divergence is zero. If the correct
//               discrete spaces are chosen the divergence is pointwise zero.
//
//               This achived by solving a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system   (similar to ex5)
//
//                                 u + grad p = u_ex
//                                    - div u = 0
//
//               Here, u_ex is the specified velocity field. If u_ex is
//               divergence free, we expect the pressure to converge to zero.
//               We discretize with H(div) and L2/H1 conforming elements.


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void u_ex(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));

   int p2 = 1;
   int p3 = 1;
   double c1 = 2.0;

   int p1 = p3 + 1;
   int p4 = p2 + 1;
   double c2 = -p1*(c1/p4);

   u(0) = c1*pow(xi,p1)*pow(yi,p2);
   u(1) = c2*pow(xi,p3)*pow(yi,p4);

   //u(0) = xi;
   //u(1) = 0.0;
}

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/square-nurbs.mesh";
   int ref_levels = -1;
   int order = 1;
   const char *device_config = "cpu";
   bool visualization = 1;
   bool NURBS = true;
   bool div_free = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&div_free, "-df", "--dif-free", "-p","--proj",
                  "Div-free or standard projection.");
   args.AddOption(&NURBS, "-n", "--nurbs", "-nn","--no-nurbs",
                  "NURBS.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   {
      if (ref_levels < 0)
      {
         ref_levels =
            (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *hdiv_coll = nullptr;
   FiniteElementCollection *l2_coll = nullptr;
   NURBSExtension *NURBSext = nullptr;

   if (mesh->NURBSext&& NURBS)//(false) //(mesh->NURBSext)
   {
      hdiv_coll = new NURBS_HDivFECollection(order);
      l2_coll   = new NURBSFECollection(order);
      NURBSext  = new NURBSExtension(mesh->NURBSext, order);
      mfem::out<<"Create NURBS fec and ext"<<std::endl;
   }
   else
   {
      hdiv_coll = new RT_FECollection(order, dim);
      l2_coll   = new L2_FECollection(order, dim);
      mfem::out<<"Create Normal fec"<<std::endl;
   }

   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, NURBSext, l2_coll);
   FiniteElementSpace *R_space = new FiniteElementSpace(mesh,
                                                        W_space->StealNURBSext(),
                                                        hdiv_coll);

   // 6. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = R_space->GetVSize();
   block_offsets[2] = W_space->GetVSize();
   block_offsets.PartialSum();

   std::cout << "***********************************************************\n";
   std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   VectorFunctionCoefficient ucoeff(dim, u_ex);

   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);
   rhs = 0.0;

   LinearForm *fform(new LinearForm);
   fform->Update(R_space, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(ucoeff));
   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   // 9. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   BilinearForm *mVarf(new BilinearForm(R_space));
   MixedBilinearForm *bVarf(new MixedBilinearForm(R_space, W_space));

   mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mVarf->Assemble();
   mVarf->Finalize();

   bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf->Assemble();
   bVarf->Finalize();

   SparseMatrix &M(mVarf->SpMat());
   SparseMatrix &B(bVarf->SpMat());
   B *= -1.;
   TransposeOperator *Bt = new TransposeOperator(&B);

   BlockOperator darcyOp(block_offsets);
   darcyOp.SetBlock(0,0, &M);
   if (div_free) { darcyOp.SetBlock(0,1, Bt); }
   if (div_free) { darcyOp.SetBlock(1,0, &B); }

   // 10. Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     pressure Schur Complement
   Vector Md(mVarf->Height());
   M.GetDiag(Md);
   Md.HostReadWrite();
   SparseMatrix *MinvBt = Transpose(B);
   for (int i = 0; i < Md.Size(); i++)
   {
      MinvBt->ScaleRow(i, 1./Md(i));
   }
   SparseMatrix *S = Mult(B, *MinvBt);
   Solver *invS;
#ifndef MFEM_USE_SUITESPARSE
   invS = new GSSmoother(*S);
#else
   invS = new UMFPackSolver(*S);
#endif
   invS->iterative_mode = false;

   Solver *invM = new GSSmoother(M);
   invM->iterative_mode = false;

   BlockDiagonalPreconditioner darcyPrec(block_offsets);
   darcyPrec.SetDiagonalBlock(0, invM);
   darcyPrec.SetDiagonalBlock(1, invS);

   // 11. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.
   int maxIter(10000);
   double rtol(1.e-16);
   double atol(1.e-16);

   chrono.Clear();
   chrono.Start();
   MINRESSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(darcyOp);
   solver.SetPreconditioner(darcyPrec);
   solver.SetPrintLevel(2);
   x = 0.0;
   solver.Mult(rhs, x);
   if (device.IsEnabled()) { x.HostRead(); }
   chrono.Stop();

   if (solver.GetConverged())
   {
      std::cout << "MINRES converged in " << solver.GetNumIterations()
                << " iterations with a residual norm of "
                << solver.GetFinalNorm() << ".\n";
   }
   else
   {
      std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                << " iterations. Residual norm is " << solver.GetFinalNorm()
                << ".\n";
   }
   std::cout << "MINRES solver took " << chrono.RealTime() << "s.\n";

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u, p, uu, vv;
   u.MakeRef(R_space, x.GetBlock(0), 0);
   p.MakeRef(W_space, x.GetBlock(1), 0);
   if (NURBS)
   {
      uu.MakeRef(R_space->fes1, x.GetBlock(0), 0);
      vv.MakeRef(R_space->fes2, x.GetBlock(0),R_space->GetVSize()/2);
   }

   //int order_quad = max(2, 2*order+1);
   int order_quad = 2*order+1+5;
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_u  = u.ComputeL2Error(ucoeff, irs);
   double err_p  = p.ComputeL2Error(zero, irs);
   double err_div  = u.ComputeDivError(&zero, irs);

   std::cout << "|| u_h - u_ex ||  = " << err_u  << "\n";
   std::cout << "|| div u_h - div u_ex ||  = " << err_div  << "\n";
   std::cout << "|| p_h - p_ex ||  = " << err_p  << "\n";

   // 13. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ex5.mesh -g sol_u.gf" or "glvis -m ex5.mesh -g
   //     sol_p.gf".
   {
      ofstream mesh_ofs("ex5.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream u_ofs("sol_u.gf");
      u_ofs.precision(8);
      u.Save(u_ofs);

      ofstream p_ofs("sol_p.gf");
      p_ofs.precision(8);
      p.Save(p_ofs);
   }

   // 14. Save data in the VisIt format
   VisItDataCollection visit_dc("Example5", mesh);
   visit_dc.RegisterField("velocity", &u);
   visit_dc.RegisterField("pressure", &p);
   if (NURBS) { visit_dc.RegisterField("uu", &uu); }
   if (NURBS) { visit_dc.RegisterField("vv", &vv); }
   visit_dc.Save();

   // 15. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("Example5", mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("velocity",&u);
   paraview_dc.RegisterField("pressure",&p);
   paraview_dc.Save();

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "solution\n" << *mesh << u << "window_title 'Velocity'" << endl;
      socketstream p_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "solution\n" << *mesh << p << "window_title 'Pressure'" << endl;
   }

   // 17. Free the used memory.
   delete fform;
   delete invM;
   delete invS;
   delete S;
   delete Bt;
   delete MinvBt;
   delete mVarf;
   delete bVarf;
   delete W_space;
   delete R_space->fes1;
   delete R_space->fes2;
   delete R_space;
   delete l2_coll;
   delete hdiv_coll;
   delete mesh;

   return 0;
}






