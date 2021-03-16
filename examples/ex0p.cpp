//                       MFEM Example 0 - Parallel Version
//
// Compile with: make ex0p
//
// Sample runs:  mpirun -np 4 ex0p
//               mpirun -np 4 ex0p -m ../data/fichera.mesh
//               mpirun -np 4 ex0p -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic parallel usage of
//              MFEM to define a simple finite element discretization of the
//              Laplace problem -Delta u = 1 with homogeneous Dirichlet boundary
//              conditions. The mesh file and finite element polynomial degree
//              are given by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI
   MPI_Session mpi(argc, argv);

   // 2. Parse command line options
   const char *mesh_file = "../data/star.mesh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (mpi.Root()) { args.PrintOptions(cout); }

   // 3. Read the serial mesh from the given mesh file.
   Mesh serial_mesh(mesh_file);

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh once in parallel to increase the resolution.
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear(); // Delete the serial mesh --- we don't need it anymore
   mesh.UniformRefinement();

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order.
   H1_FECollection fec(order, mesh.Dimension());
   ParFiniteElementSpace fespace(&mesh, &fec);
   HYPRE_Int total_num_dofs = fespace.GlobalTrueVSize();
   if (mpi.Root()) { cout << "Number of unknowns: " << total_num_dofs << endl; }

   // 6. Get a list of all the boundary DOFs. These will be marked as essential
   //    in order to enforce Dirichlet boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which also determines the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 8. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 9. Set up the bilinear form a(.,.) corresponding to the Laplacian -Delta.
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     etc.
   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 11. Solve using preconditioned CG with hypre's BoomerAMG preconditioner.
   HypreBoomerAMG M(A);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(M);
   cg.SetOperator(A);
   cg.Mult(B, X);

   // 12. Recover the solution as a grid function and save to files. The output
   //     can be viewed using GLVis with the command:
   //     glvis -np <np> -m mesh -g sol
   a.RecoverFEMSolution(X, b, x);

   ostringstream mesh_name, sol_name;
   mesh_name << "mesh." << setfill('0') << setw(6) << mpi.WorldRank();
   sol_name << "sol." << setfill('0') << setw(6) << mpi.WorldRank();

   ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);

   ofstream sol_ofs(sol_name.str().c_str());
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   return 0;
}
