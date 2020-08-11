//                       MFEM Example 4 - Parallel Version
//
// Compile with: make ex4p
//
// Sample runs:  mpirun -np 4 ex4p -m ../data/square-disc.mesh
//               mpirun -np 4 ex4p -m ../data/star.mesh
//               mpirun -np 4 ex4p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex4p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex4p -m ../data/beam-hex.mesh -o 2 -pa
//               mpirun -np 4 ex4p -m ../data/escher.mesh -o 2 -sc
//               mpirun -np 4 ex4p -m ../data/fichera.mesh -o 2 -hb
//               mpirun -np 4 ex4p -m ../data/fichera-q2.vtk
//               mpirun -np 4 ex4p -m ../data/fichera-q3.mesh -o 2 -sc
//               mpirun -np 4 ex4p -m ../data/square-disc-nurbs.mesh -o 3
//               mpirun -np 4 ex4p -m ../data/beam-hex-nurbs.mesh -o 3
//               mpirun -np 4 ex4p -m ../data/periodic-square.mesh -no-bc
//               mpirun -np 4 ex4p -m ../data/periodic-cube.mesh -no-bc
//               mpirun -np 4 ex4p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex4p -m ../data/amr-hex.mesh -o 2 -sc
//               mpirun -np 4 ex4p -m ../data/amr-hex.mesh -o 2 -hb
//               mpirun -np 4 ex4p -m ../data/star-surf.mesh -o 3 -hb
//
// Device sample runs:
//               mpirun -np 4 ex4p -m ../data/star.mesh -pa -d cuda
//               mpirun -np 4 ex4p -m ../data/star.mesh -pa -d raja-cuda
//               mpirun -np 4 ex4p -m ../data/star.mesh -pa -d raja-omp
//               mpirun -np 4 ex4p -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple 2D/3D H(div) diffusion
//               problem corresponding to the second order definite equation
//               -grad(alpha div F) + beta F = f with boundary condition F dot n
//               = <given normal field>. Here, we use a given exact solution F
//               and compute the corresponding r.h.s. f.  We discretize with
//               Raviart-Thomas finite elements.
//
//               The example demonstrates the use of H(div) finite element
//               spaces with the grad-div and H(div) vector finite element mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Bilinear form
//               hybridization and static condensation are also illustrated.
//
//               We recommend viewing examples 1-3 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <chrono>

using namespace std;
using namespace mfem;

// Exact solution, F, and r.h.s., f. See below for implementation.
void F_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-hex.mesh";
   int order = 1;
   bool set_bc = true;
   bool static_cond = false;
   bool hybridization = true;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;
   bool amgx = true;
   const char *amgx_cfg = 0;
   int ser_ref_levels = 2;
   int par_ref_levels = 2;
   int ndevices = 4;
   int nsolves = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&set_bc, "-bc", "--impose-bc", "-no-bc", "--dont-impose-bc",
                  "Impose or not essential boundary conditions.");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&amgx_cfg, "-c","--c","AMGX solver file");
   args.AddOption(&amgx, "-amgx","--amgx","-no-amgx",
                  "--no-amgx","Use AMGX");
   args.AddOption(&ser_ref_levels, "-nr","--nr","Number of Serial Refinements");
   args.AddOption(&par_ref_levels, "-np","--np","Number of Parallel Refinements");
   args.AddOption(&ndevices, "-nd","--nd","Number of Devices");
   args.AddOption(&nsolves, "-nsolves","--nsolves","Number of Solves");
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

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume, as well as periodic meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      for (int l = 0; l < ser_ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them (this is needed in the ADS solver below).
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   pmesh->ReorientTetMesh();

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *fec = new RT_FECollection(order-1, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = set_bc ? 1 : 0;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient f(sdim, f_exact);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary faces will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   VectorFunctionCoefficient F(sdim, F_exact);
   x.ProjectCoefficient(F);

   // 11. Set up the parallel bilinear form corresponding to the H(div)
   //     diffusion operator grad alpha div + beta I, by adding the div-div and
   //     the mass domain integrators.
   Coefficient *alpha = new ConstantCoefficient(1.0);
   Coefficient *beta  = new ConstantCoefficient(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new DivDivIntegrator(*alpha));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*beta));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation,
   //     hybridization, etc.
   FiniteElementCollection *hfec = NULL;
   ParFiniteElementSpace *hfes = NULL;
   if (static_cond)
   {
      a->EnableStaticCondensation();
   }
   else if (hybridization)
   {
      hfec = new DG_Interface_FECollection(order-1, dim);
      hfes = new ParFiniteElementSpace(pmesh, hfec);
      a->EnableHybridization(hfes, new NormalTraceJumpIntegrator(),
                             ess_tdof_list);
   }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // 13. Define and apply a parallel PCG solver for A X = B with the 2D AMS or
   //     the 3D ADS preconditioners from hypre. If using hybridization, the
   //     system is preconditioned with hypre's BoomerAMG. In the partial
   //     assembly case, use Jacobi preconditioning.
   if (amgx)
   {

#ifdef MFEM_USE_AMGX
      std::string amgx_str;
      amgx_str = amgx_cfg;
      AmgXSolver amgx;
      auto start = std::chrono::steady_clock::now();
      amgx.initialize(MPI_COMM_WORLD, "dDDI", amgx_str, ndevices);
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_seconds = end-start;
      std::cout << "Initialize: " << elapsed_seconds.count() << "s\n";

      auto start2 = std::chrono::steady_clock::now();
      amgx.setA(*A.As<HypreParMatrix>());
      auto end2 = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_seconds2 = end2-start2;
      std::cout << "Setup: " << elapsed_seconds2.count() << "s\n";

      auto start3 = std::chrono::steady_clock::now();
      for (int i = 0; i < nsolves; i++)
      {
         X = 0.0; //set to zero
         amgx.solve(X, B);

      }
      cudaDeviceSynchronize();
      auto end3 = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_seconds3 = end3-start3;
      std::cout << "Solve: " << elapsed_seconds3.count() << "s\n";
#endif
   }
   else
   {
      auto start4 = std::chrono::steady_clock::now();
      Solver *prec = NULL;
      CGSolver *pcg = new CGSolver(MPI_COMM_WORLD);
      pcg->SetOperator(*A);
      pcg->SetRelTol(1e-12);
      pcg->SetMaxIter(2000);
      pcg->SetPrintLevel(1);
      if (hybridization) { prec = new HypreBoomerAMG(*A.As<HypreParMatrix>()); }
      else if (pa) { prec = new OperatorJacobiSmoother(*a, ess_tdof_list); }
      else
      {
         ParFiniteElementSpace *prec_fespace =
            (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
         if (dim == 2)   { prec = new HypreAMS(*A.As<HypreParMatrix>(), prec_fespace); }
         else            { prec = new HypreADS(*A.As<HypreParMatrix>(), prec_fespace); }
      }
      pcg->SetPreconditioner(*prec);
      auto end4 = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_seconds4 = end4-start4;
      std::cout << "Set up: " << elapsed_seconds4.count() << "s\n";
      auto start5 = std::chrono::steady_clock::now();
      for (int i = 0; i < nsolves; i++)
      {
         X = 0.0;
         pcg->Mult(B, X);
      }
      auto end5 = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_seconds5 = end5-start5;
      std::cout << "Solve: " << elapsed_seconds5.count() << "s\n";
      delete pcg;
      delete prec;
   }

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 15. Compute and print the L^2 norm of the error.
   {
      double err = x.ComputeL2Error(F);
      if (myid == 0)
      {
         cout << "\n|| F_h - F ||_{L^2} = " << err << '\n' << endl;
      }
   }

   // 16. Save the refined mesh and the solution in parallel. This output can
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

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 18. Free the used memory.
   delete hfes;
   delete hfec;
   delete a;
   delete alpha;
   delete beta;
   delete b;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}


// The exact solution (for non-surface meshes)
void F_exact(const Vector &p, Vector &F)
{
   int dim = p.Size();

   double x = p(0);
   double y = p(1);
   // double z = (dim == 3) ? p(2) : 0.0; // Uncomment if F is changed to depend on z

   F(0) = cos(kappa*x)*sin(kappa*y);
   F(1) = cos(kappa*y)*sin(kappa*x);
   if (dim == 3)
   {
      F(2) = 0.0;
   }
}

// The right hand side
void f_exact(const Vector &p, Vector &f)
{
   int dim = p.Size();

   double x = p(0);
   double y = p(1);
   // double z = (dim == 3) ? p(2) : 0.0; // Uncomment if f is changed to depend on z

   double temp = 1 + 2*kappa*kappa;

   f(0) = temp*cos(kappa*x)*sin(kappa*y);
   f(1) = temp*cos(kappa*y)*sin(kappa*x);
   if (dim == 3)
   {
      f(2) = 0;
   }
}
