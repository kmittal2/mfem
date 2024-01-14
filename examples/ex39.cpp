//                                MFEM Example 39
//
// Compile with: make ex39
//
// Sample runs:  ex39
//               ex39 -ess "Southern Boundary"
//               ex39 -src Base
//
// Device sample runs:
//               ex39 -fa -d cuda
//
// Description:  This example code demonstrates the use of named attribute
//               sets in MFEM to specify material regions, boundary regions,
//               or source regions by name rather than attribute numbers. It
//               also demonstrates how new named attribute sets may be created
//               from arbitrary groupings of attribute number and used as a
//               convenient shorthand to refer to those groupings in other
//               portions of the application or through the command line.
//
//               The particular problem being solved here is nearly the same
//               as that in example 1 i.e. a simple finite element
//               discretization of the Laplace problem -Delta u = 1 with
//               homogeneous Dirichlet boundary conditions and, in this case,
//               an inhomogeneous diffusion coefficient. The diffusion
//               coefficient is given a small default value throughout the
//               domain which is increased by two separate amounts in two named
//               regions.
//
//               The example highlights the use of named attribute sets for
//               both subdomains and boundaries in different contexts as well
//               as basic methods to create named sets from existing attributes.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/compass.msh";
   int order = 1;
   string source_name = "Rose Even";
   string ess_name = "Boundary";
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&source_name,"-src","--source-attr-name",
                  "Name of attribute set containing source.");
   args.AddOption(&ess_name,"-ess","--ess-attr-name",
                  "Name of attribute set containing essential BC.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic",
                  "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
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

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   AttributeSets &attr_sets = mesh.attribute_sets;

   // Display attribute set names contained in the initial mesh
   {
      std::set<string> names = attr_sets.GetAttributeSetNames();
      cout << "Element Attribute Set Names: ";
      for (auto const &set_name : names)
      {
         cout << " \"" << set_name << "\"";
      }
      cout << endl;

      std::set<string> bdr_names = attr_sets.GetBdrAttributeSetNames();
      cout << "Boundary Attribute Set Names: ";
      for (auto const &bdr_set_name : bdr_names)
      {
         cout << " \"" << bdr_set_name << "\"";
      }
      cout << endl;
   }

   // Define new regions based on existing attribute sets
   {
      Array<int> & Na = attr_sets.GetAttributeSet("N Even");
      Array<int> & Nb = attr_sets.GetAttributeSet("N Odd");
      Array<int> & Sa = attr_sets.GetAttributeSet("S Even");
      Array<int> & Sb = attr_sets.GetAttributeSet("S Odd");
      Array<int> & Ea = attr_sets.GetAttributeSet("E Even");
      Array<int> & Eb = attr_sets.GetAttributeSet("E Odd");
      Array<int> & Wa = attr_sets.GetAttributeSet("W Even");
      Array<int> & Wb = attr_sets.GetAttributeSet("W Odd");

      // Create a new set spanning the North point
      attr_sets.SetAttributeSet("North", Na);
      attr_sets.AddToAttributeSet("North", Nb);

      // Create a new set spanning the South point
      attr_sets.SetAttributeSet("South", Sa);
      attr_sets.AddToAttributeSet("South", Sb);

      // Create a new set spanning the East point
      attr_sets.SetAttributeSet("East", Ea);
      attr_sets.AddToAttributeSet("East", Eb);

      // Create a new set spanning the West point
      attr_sets.SetAttributeSet("West", Wa);
      attr_sets.AddToAttributeSet("West", Wb);

      // Create a new set consisting of the "a" sides of the compass rose
      attr_sets.SetAttributeSet("Rose Even", Na);
      attr_sets.AddToAttributeSet("Rose Even", Sa);
      attr_sets.AddToAttributeSet("Rose Even", Ea);
      attr_sets.AddToAttributeSet("Rose Even", Wa);

      // Create a new set consisting of the "b" sides of the compass rose
      attr_sets.SetAttributeSet("Rose Odd", Nb);
      attr_sets.AddToAttributeSet("Rose Odd", Sb);
      attr_sets.AddToAttributeSet("Rose Odd", Eb);
      attr_sets.AddToAttributeSet("Rose Odd", Wb);


      // Create a new set consisting of the full compass rose
      Array<int> & Ra = attr_sets.GetAttributeSet("Rose Even");
      Array<int> & Rb = attr_sets.GetAttributeSet("Rose Odd");
      attr_sets.SetAttributeSet("Rose", Ra);
      attr_sets.AddToAttributeSet("Rose", Rb);
   }
   // Define new boundary regions based on existing boundary attribute sets
   {
      Array<int> & NNE = attr_sets.GetBdrAttributeSet("NNE");
      Array<int> & NNW = attr_sets.GetBdrAttributeSet("NNW");
      Array<int> & ENE = attr_sets.GetBdrAttributeSet("ENE");
      Array<int> & ESE = attr_sets.GetBdrAttributeSet("ESE");
      Array<int> & SSE = attr_sets.GetBdrAttributeSet("SSE");
      Array<int> & SSW = attr_sets.GetBdrAttributeSet("SSW");
      Array<int> & WNW = attr_sets.GetBdrAttributeSet("WNW");
      Array<int> & WSW = attr_sets.GetBdrAttributeSet("WSW");

      attr_sets.SetBdrAttributeSet("Northern Boundary", NNE);
      attr_sets.AddToBdrAttributeSet("Northern Boundary", NNW);

      attr_sets.SetBdrAttributeSet("Southern Boundary", SSE);
      attr_sets.AddToBdrAttributeSet("Southern Boundary", SSW);

      attr_sets.SetBdrAttributeSet("Eastern Boundary", ENE);
      attr_sets.AddToBdrAttributeSet("Eastern Boundary", ESE);

      attr_sets.SetBdrAttributeSet("Western Boundary", WNW);
      attr_sets.AddToBdrAttributeSet("Western Boundary", WSW);

      attr_sets.SetBdrAttributeSet("Boundary",
                                   attr_sets.GetBdrAttributeSet
                                   ("Northern Boundary"));
      attr_sets.AddToBdrAttributeSet("Boundary",
                                     attr_sets.GetBdrAttributeSet
                                     ("Southern Boundary"));
      attr_sets.AddToBdrAttributeSet("Boundary",
                                     attr_sets.GetBdrAttributeSet
                                     ("Eastern Boundary"));
      attr_sets.AddToBdrAttributeSet("Boundary",
                                     attr_sets.GetBdrAttributeSet
                                     ("Western Boundary"));
   }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary regions corresponding to the boundary attributes
   //    contained in the set named "ess_name" as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr_marker;
      mesh.BdrAttrToMarker(attr_sets.GetBdrAttributeSet(ess_name),
                           ess_bdr_marker);
      fespace.GetEssentialTrueDofs(ess_bdr_marker, ess_tdof_list);
   }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1_s,phi_i) where phi_i
   //    are the basis functions in fespace and 1_s is an indicator function
   //    equal to 1 on the region defined by the named set "source_name" and
   //    zero elsewhere.
   Array<int> source_marker;
   mesh.AttrToMarker(attr_sets.GetAttributeSet(source_name), source_marker);

   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one), source_marker);
   b.Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the
   //    Diffusion domain integrator.
   BilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (fa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      // Sort the matrix column indices when running on GPU or with OpenMP (i.e.
      // when Device::IsEnabled() returns true). This makes the results
      // bit-for-bit deterministic at the cost of somewhat longer run time.
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }

   ConstantCoefficient defaultCoef(1.0e-6);
   ConstantCoefficient baseCoef(1.0);
   ConstantCoefficient roseCoef(2.0);

   Array<int> base_marker;
   Array<int> rose_marker;
   mesh.AttrToMarker(attr_sets.GetAttributeSet("Base"), base_marker);
   mesh.AttrToMarker(attr_sets.GetAttributeSet("Rose Even"), rose_marker);

   a.AddDomainIntegrator(new DiffusionIntegrator(defaultCoef));
   a.AddDomainIntegrator(new DiffusionIntegrator(baseCoef), base_marker);
   a.AddDomainIntegrator(new DiffusionIntegrator(roseCoef), rose_marker);

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 800, 1e-12, 0.0);
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
   }
   else
   {
      if (UsesTensorBasis(fespace))
      {
         if (algebraic_ceed)
         {
            ceed::AlgebraicSolver M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 800, 1e-12, 0.0);
         }
         else
         {
            OperatorJacobiSmoother M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 800, 1e-12, 0.0);
         }
      }
      else
      {
         CG(*A, B, X, 1, 800, 1e-12, 0.0);
      }
   }

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << "keys Rjmm" << flush;
   }

   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
