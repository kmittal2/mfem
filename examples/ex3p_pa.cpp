//                       MFEM Example 3 - Parallel Version
//
// Compile with: make ex3p_pa
//
// Sample runs:  mpirun -np 4 ex3p_pa -m ../data/star.mesh
//               mpirun -np 4 ex3p_pa -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex3p_pa -m ../data/beam-tet.mesh
//               mpirun -np 4 ex3p_pa -m ../data/beam-hex.mesh
//               mpirun -np 4 ex3p_pa -m ../data/escher.mesh
//               mpirun -np 4 ex3p_pa -m ../data/escher.mesh -o 2
//               mpirun -np 4 ex3p_pa -m ../data/fichera.mesh
//               mpirun -np 4 ex3p_pa -m ../data/fichera-q2.vtk
//               mpirun -np 4 ex3p_pa -m ../data/fichera-q3.mesh
//               mpirun -np 4 ex3p_pa -m ../data/square-disc-nurbs.mesh
//               mpirun -np 4 ex3p_pa -m ../data/beam-hex-nurbs.mesh
//               mpirun -np 4 ex3p_pa -m ../data/amr-quad.mesh -o 2
//               mpirun -np 4 ex3p_pa -m ../data/amr-hex.mesh
//               mpirun -np 4 ex3p_pa -m ../data/star-surf.mesh -o 2
//               mpirun -np 4 ex3p_pa -m ../data/mobius-strip.mesh -o 2 -f 0.1
//               mpirun -np 4 ex3p_pa -m ../data/klein-bottle.mesh -o 2 -f 0.1
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Static condensation is
//               also illustrated.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;

double alphaCoefficient(const Vector & x)
{
  if (x.Size() == 3)
    {
      return (10.0 * x(0)) + (5.0 * x(1)) + x(2);
    }
  else
    return (10.0 * x(0)) + (5.0 * x(1));
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/inline-hex.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }
   kappa = freq * M_PI;

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels = (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels-1; l++)
      {
         mesh->UniformRefinement();
      }
   }

   if (myid == 0)
     cout << "Serial mesh number of elements " << mesh->GetNE() << endl;

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   pmesh->ReorientTetMesh();

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient f(sdim, f_exact);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   VectorFunctionCoefficient E(sdim, E_exact);
   x.ProjectCoefficient(E);

   // 10. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
   //     mass domain integrators.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));

   // 11. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   if (myid == 0)
   {
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   // 11.5 Partial assembly. 
   {
     ParGridFunction x_test(fespace);
     ParGridFunction y_fa(fespace);
     ParGridFunction y_pa(fespace);

     FunctionCoefficient alpha(alphaCoefficient);
     //ConstantCoefficient alpha(10.0);

     x_test.Randomize(1);
     //x_test = 0.0;
     //x_test[0] = 1.0;

     ParBilinearForm *a_fa = new ParBilinearForm(fespace);
     a_fa->AddDomainIntegrator(new VectorFEMassIntegrator(alpha));
     a_fa->AddDomainIntegrator(new CurlCurlIntegrator(alpha));

     a_fa->Assemble();
     a_fa->Finalize();

     ParBilinearForm *a_pa = new ParBilinearForm(fespace);
     a_pa->SetAssemblyLevel(AssemblyLevel::PARTIAL);
     a_pa->AddDomainIntegrator(new VectorFEMassIntegrator(alpha));
     a_pa->AddDomainIntegrator(new CurlCurlIntegrator(alpha));
     a_pa->Assemble();

     a_fa->Mult(x_test, y_fa);
     a_pa->Mult(x_test, y_pa);

     cout << myid << ": Norm of y_fa " << y_fa.Norml2() << endl;
     cout << myid << ": Norm of y_pa " << y_pa.Norml2() << endl;

     y_fa -= y_pa;
     
     cout << myid << ": Norm of diff " << y_fa.Norml2() << endl;

     const bool testDiag = true;
     if (testDiag)
       {
	 ParGridFunction diag_pa(fespace);
	 diag_pa = 0.0;
	 a_pa->AssembleDiagonal(diag_pa);

	 Vector tdiag_pa(fespace->GetTrueVSize());
	 diag_pa.GetTrueDofs(tdiag_pa);

	 Vector ej(fespace->GetTrueVSize());
	 HYPRE_Int *tdos = fespace->GetTrueDofOffsets();

	 double maxRelErr = 0.0;

	 for (int i=0; i<fespace->GlobalTrueVSize(); ++i)
	   {
	     ej = 0.0;

	     if (tdos[0] <= i && i < tdos[1])
	       ej[i - tdos[0]] = 1.0;

	     x_test.SetFromTrueDofs(ej);
	     a_pa->Mult(x_test, y_pa);

	     y_pa.GetTrueDofs(ej);

	     if (tdos[0] <= i && i < tdos[1])
	       {
		 const double d_i = ej[i - tdos[0]];
		 const double relErr = fabs(tdiag_pa[i - tdos[0]] - d_i) / std::max(fabs(tdiag_pa[i - tdos[0]]), fabs(d_i));
		 if (relErr > 1.0e-1)
		   cout << "diag " << i << ": " << d_i << " = " << tdiag_pa[i - tdos[0]] << ", error " << fabs(tdiag_pa[i - tdos[0]] - d_i) << endl;

		 if (relErr > maxRelErr)
		   maxRelErr = relErr;
	       }
	   }

	 cout << myid << ": maximum relative error in diagonal assembly: " << maxRelErr << endl;

	 MPI_Barrier(MPI_COMM_WORLD);
       }

     delete a_fa;
     delete a_pa;

     MPI_Barrier(MPI_COMM_WORLD);
   }

   return 0;

   // 12. Define and apply a parallel PCG solver for AX=B with the AMS
   //     preconditioner from hypre.
   //ParFiniteElementSpace *prec_fespace =
   //   (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
   //HypreSolver *ams = new HypreAMS(A, prec_fespace);
   HyprePCG *pcg = new HyprePCG(A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   //pcg->SetPreconditioner(*ams);
   pcg->Mult(B, X);

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 14. Compute and print the L^2 norm of the error.
   {
      double err = x.ComputeL2Error(E);
      if (myid == 0)
      {
         cout << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
      }
   }

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 17. Free the used memory.
   delete pcg;
   //delete ams;
   delete a;
   delete sigma;
   delete muinv;
   delete b;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   if (dim == 3)
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(2));
      E(2) = sin(kappa * x(0));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}
