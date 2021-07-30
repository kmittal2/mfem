//                                MFEM Example 6
//
// Compile with: make ex6
//
// Sample runs:  ex6 -m ../data/square-disc.mesh -o 1
//               ex6 -m ../data/square-disc.mesh -o 2
//               ex6 -m ../data/square-disc-nurbs.mesh -o 2
//               ex6 -m ../data/star.mesh -o 3
//               ex6 -m ../data/escher.mesh -o 2
//               ex6 -m ../data/fichera.mesh -o 2
//               ex6 -m ../data/disc-nurbs.mesh -o 2
//               ex6 -m ../data/ball-nurbs.mesh
//               ex6 -m ../data/pipe-nurbs.mesh
//               ex6 -m ../data/star-surf.mesh -o 2
//               ex6 -m ../data/square-disc-surf.mesh -o 2
//               ex6 -m ../data/amr-quad.mesh
//               ex6 -m ../data/inline-segment.mesh -o 1 -md 100
//
// Device sample runs:
//               ex6 -pa -d cuda
//               ex6 -pa -d occa-cuda
//               ex6 -pa -d raja-omp
//               ex6 -pa -d ceed-cpu
//             * ex6 -pa -d ceed-cuda
//               ex6 -pa -d ceed-cuda:/gpu/cuda/shared
//
// Description:  This is a version of Example 1 with a simple adaptive mesh
//               refinement loop. The problem being solved is again the Laplace
//               equation -Delta u = 1 with homogeneous Dirichlet boundary
//               conditions. The problem is solved on a sequence of meshes which
//               are locally refined in a conforming (triangles, tetrahedrons)
//               or non-conforming (quadrilaterals, hexahedra) manner according
//               to a simple ZZ error estimator.
//
//               The example demonstrates MFEM's capability to work with both
//               conforming and nonconforming refinements, in 2D and 3D, on
//               linear, curved and surface meshes. Interpolation of functions
//               from coarse to fine meshes, as well as persistent GLVis
//               visualization are also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double wavefront_exsol(const Vector &p)
{
   double x = p(0), y = p(1);
   double alpha = 1000.0;
   // double xc = -0.05, yc = -0.05;
   double xc = 0.0, yc = 0.0;
   double r0 = 0.7;
   double r = sqrt(pow(x - xc,2.0) + pow(y - yc,2.0));
   return atan(alpha * (r - r0));
}

void wavefront_exgrad(const Vector &p, Vector &grad)
{
   double x = p(0), y = p(1);
   double alpha = 1000.0;
   double xc = -0.05, yc = -0.05;
   double r0 = 0.7;
   grad(0) = 0.0;
   grad(1) = 0.0;
}

double wavefront_laplace(const Vector &p)
{
   double x = p(0), y = p(1);
   double alpha = 1000.0;
   // double xc = -0.05, yc = -0.05;
   double xc = 0.0, yc = 0.0;
   double r0 = 0.7;
   double r = sqrt(pow(x - xc,2.0) + pow(y - yc,2.0));
   double num = - ( alpha - pow(alpha,3) * (pow(r,2) - pow(r0,2)) );
   double denom = pow(r * ( pow(alpha,2) * pow(r0,2) + pow(alpha,2) * pow(r,2) \
                            - 2 * pow(alpha,2) * r0 * r + 1.0 ),2);
   denom = max(denom,1e-8);
   // return num / denom;
   if (p.Normlp(2.0) > 0.4 && p.Normlp(2.0) < 0.6) { return 1; }
   if (p.Normlp(2.0) < 0.4 || p.Normlp(2.0) > 0.6) { return 2; }
   return 0;
}

double wavefront_laplace_alt(const Vector &p)
{
   double x = p(0), y = p(1);
   double alpha = 1000.0;
   double xc = -0.5, yc = -0.5;
   double r0 = 0.7;
   double r = sqrt(pow(x - xc,2.0) + pow(y - yc,2.0));
   double num = - ( alpha - pow(alpha,3) * (pow(r,2) - pow(r0,2)) );
   double denom = pow(r * ( pow(alpha,2) * pow(r0,2) + pow(alpha,2) * pow(r,2) \
                            - 2 * pow(alpha,2) * r0 * r + 1.0 ),2);
   denom = max(denom,1e-8);
   return num / denom;
}

double load3(const Vector &p)
{
   double x = p(0), y = p(1);
   double r = sqrt(x*x + y*y) + 1.0e-8;
   return r * sin(1.0/r);
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool pa = false;
   const char *device_config = "cpu";
   int max_dofs = 50000;
   bool visualization = true;
   int nc_limit = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&max_dofs, "-md", "--max-dofs",
                  "Stop after reaching this many degrees of freedom.");
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
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();

   // 4. Since a NURBS mesh can currently only be refined uniformly, we need to
   //    convert it to a piecewise-polynomial curved mesh. First we refine the
   //    NURBS mesh a bit more and then project the curvature to quadratic Nodes.
   if (mesh.NURBSext)
   {
      for (int i = 0; i < 2; i++)
      {
         mesh.UniformRefinement();
      }
      mesh.SetCurvature(2);
   }

   // 5. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // 6. As in Example 1, we set up bilinear and linear forms corresponding to
   //    the Laplace problem -\Delta u = 1. We don't assemble the discrete
   //    problem yet, this will be done in the main loop.
   BilinearForm a(&fespace);
   if (pa)
   {
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.SetDiagonalPolicy(Operator::DIAG_ONE);
   }
   LinearForm b(&fespace);

   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   Coefficient * exsol = nullptr;
   Coefficient * rhs = nullptr;
   exsol = new FunctionCoefficient(wavefront_exsol);
   rhs = new FunctionCoefficient(wavefront_laplace);

   BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
   a.AddDomainIntegrator(integ);
   // b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.AddDomainIntegrator(new DomainLFIntegrator(*rhs));

   // 7. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   GridFunction x(&fespace);
   x = 0.0;

   // 8. All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 9. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost, visport);
   }

   // 10. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     that uses the ComputeElementFlux method of the DiffusionIntegrator to
   //     recover a smoothed flux (gradient) that is subtracted from the element
   //     flux to get an error indicator. We need to supply the space for the
   //     smoothed flux: an (H1)^sdim (i.e., vector-valued) space is used here.
   FiniteElementSpace flux_fespace(&mesh, &fec, sdim);
   ZienkiewiczZhuEstimator estimator(*integ, x, flux_fespace);
   estimator.SetAnisotropic();

   // 11. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);

   const IntegrationRule *irs[Geometry::NumGeom];
   int order_quad = 2*order + 5;
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   // 11.5. Preprocess mesh to control osc
   double osc;
   double osc_tol = 1e-3;
   CoefficientRefiner coeffrefiner(order);
   coeffrefiner.SetCoefficient(*rhs);
   // coeffrefiner.SetIntRule(irs);
   coeffrefiner.SetThreshold(osc_tol);
   coeffrefiner.SetNCLimit(nc_limit);
   coeffrefiner.PreprocessMesh(mesh);

   osc = coeffrefiner.GetOsc();
   std::cout << " osc(f1) = " << osc << std::endl;

   Coefficient * rhs2 = nullptr;
   rhs2 = new FunctionCoefficient(wavefront_laplace_alt);
   coeffrefiner.SetCoefficient(*rhs2);
   coeffrefiner.PreprocessMesh(mesh);

   osc = coeffrefiner.GetOsc();
   std::cout << " osc(f2) = " << osc << std::endl;

   // Coefficient * rhs3 = nullptr;
   // rhs3 = new FunctionCoefficient(load3);
   // coeffrefiner.SetCoefficient(*rhs3);
   // coeffrefiner.PreprocessMesh(mesh);



   sol_sock.precision(8);
   sol_sock << "mesh\n" << mesh << flush;

   cout << "press any key" << endl;
   cin.get();
   fespace.Update(false);
   b.Update();
   a.Update();
   x.Update();

   // 12. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   for (int it = 0; ; it++)
   {
      int cdofs = fespace.GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of unknowns: " << cdofs << endl;

      // 13. Assemble the right-hand side.
      b.Assemble();

      // 14. Set Dirichlet boundary values in the GridFunction x.
      //     Determine the list of Dirichlet true DOFs in the linear system.
      Array<int> ess_tdof_list;
      x.ProjectBdrCoefficient(*exsol, ess_bdr);
      // x.ProjectBdrCoefficient(zero, ess_bdr);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // 15. Assemble the stiffness matrix.
      a.Assemble();

      // 16. Create the linear system: eliminate boundary conditions, constrain
      //     hanging nodes and possibly apply other transformations. The system
      //     will be solved for true (unconstrained) DOFs only.
      OperatorPtr A;
      Vector B, X;

      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);

      // 17. Solve the linear system A X = B.
      if (!pa)
      {
#ifndef MFEM_USE_SUITESPARSE
         // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
         GSSmoother M((SparseMatrix&)(*A));
         PCG(*A, M, B, X, 3, 200, 1e-12, 0.0);
#else
         // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(*A);
         umf_solver.Mult(B, X);
#endif
      }
      else // Diagonal preconditioning in partial assembly mode.
      {
         OperatorJacobiSmoother M(a, ess_tdof_list);
         PCG(*A, M, B, X, 3, 2000, 1e-12, 0.0);
      }

      // 18. After solving the linear system, reconstruct the solution as a
      //     finite element GridFunction. Constrained nodes are interpolated
      //     from true DOFs (it may therefore happen that x.Size() >= X.Size()).
      a.RecoverFEMSolution(X, b, x);

      // 19. Send solution by socket to the GLVis server.
      if (visualization && sol_sock.good())
      {
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << x << flush;
      }

      if (cdofs > max_dofs)
      {
         cout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // 20. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(mesh);
      if (refiner.Stop())
      {
         cout << "Stopping criterion satisfied. Stop." << endl;
         break;
      }

      // 21. Update the space to reflect the new state of the mesh. Also,
      //     interpolate the solution x so that it lies in the new space but
      //     represents the same function. This saves solver iterations later
      //     since we'll have a good initial guess of x in the next step.
      //     Internally, FiniteElementSpace::Update() calculates an
      //     interpolation matrix which is then used by GridFunction::Update().
      fespace.Update();
      x.Update();

      // 22. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();
   }

   return 0;
}
