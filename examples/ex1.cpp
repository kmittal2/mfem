//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               ex1 -pa -d cuda
//               ex1 -pa -d raja-cuda
//               ex1 -pa -d occa-cuda
//               ex1 -pa -d raja-omp
//               ex1 -pa -d occa-omp
//               ex1 -pa -d ceed-cpu
//               ex1 -pa -d ceed-cuda
//               ex1 -m ../data/beam-hex.mesh -pa -d cuda
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cpu
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cuda:/gpu/cuda/ref
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

GridFunction* ProlongToMaxOrder(const GridFunction *x);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int ref_levels = 1;
   int seed = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool relaxed_hp = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--ref-levels",
                  "Number of mesh refinement levels.");
   args.AddOption(&seed, "-s", "--seed", "Random seed");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&relaxed_hp, "-x", "--relaxed-hp", "-no-x",
                  "--no-relaxed-hp", "Set relaxed hp conformity.");
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
   mesh->EnsureNCMesh();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   srand(1);
   {
      //mesh->UniformRefinement();
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->RandomRefinement(0.5, true);
      }
   }
   srand(seed);

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   fespace->SetRelaxedHpConformity(relaxed_hp);

   // 6. At this point all elements have the default order (specified when
   //    construction the FECollection). Now we can p-refine some of them to
   //    obtain a variable-order space...
   /*{
      Array<Refinement> refs;
      refs.Append(Refinement(1, 4));
      mesh->GeneralRefinement(refs);
      refs[0].ref_type = 2;
      mesh->GeneralRefinement(refs);
   }
   fespace->Update(false);*/
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      fespace->SetElementOrder(i, (rand()%5)+order);
      //fespace->SetElementOrder(i, order);
      //fespace->SetElementOrder(i, i ? 3 : 2);
   }
   fespace->Update(false);

   /*fespace->SetElementOrder(0, order+1);
   fespace->Update(false);*/

   /*Array<int> dofs;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      fespace->GetElementDofs(i, dofs);
      mfem::out << "Element " << i << " DOFs:";
      for (int j = 0; j < dofs.Size(); j++) {
         mfem::out << " " << dofs[j];
      }
      mfem::out << std::endl;
   }*/
   cout << "Space size (all DOFs): " << fespace->GetNDofs() << endl;
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   /*cout << "P matrix:\n";
   fespace->GetConformingProlongation()->Print();*/

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   cout << "Essential DOFs: " << ess_tdof_list.Size() << endl;

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
   }
   else // Jacobi preconditioning in partial assembly mode
   {
      if (UsesTensorBasis(*fespace))
      {
         OperatorJacobiSmoother M(*a, ess_tdof_list);
         PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
      }
      else
      {
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
   }

   // 12. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      // Prolong the solution vector onto L2 space of max order (for GLVis)
      GridFunction *vis_x = ProlongToMaxOrder(&x);

#if 1
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << *vis_x
               //<< "keys Rjlm\n"
               << flush;
#endif
      delete vis_x;

#if 1
      L2_FECollection l2fec(0, dim);
      FiniteElementSpace l2fes(mesh, &l2fec);
      GridFunction orders(&l2fes);

      for (int i = 0; i < orders.Size(); i++)
      {
         orders(i) = fespace->GetElementOrder(i);
      }

      socketstream ord_sock(vishost, visport);
      ord_sock.precision(8);
      ord_sock << "solution\n" << *mesh << orders
               //<< "keys Rjlmc\n"
               << flush;
#endif

      // visualize the basis functions
      if (0)
      {
         socketstream b_sock(vishost, visport);
         b_sock.precision(8);

#if 1
         int first = 0;
#else
         int first = fespace->GetNV() - 10;
#endif
         cout << "first = " << first << endl;

         for (int i = first; i < X.Size(); i++)
         {
            X = 0.0;
            X(i) = 1.0;
            a->RecoverFEMSolution(X, *b, x);
            vis_x = ProlongToMaxOrder(&x);

            b_sock << "solution\n" << *mesh << *vis_x << flush;
            if (i == first) { b_sock << "keys miIMA\n"; }
            b_sock << "pause\n" << flush;
            // delete vis_x;
         }
      }

      /*std::ofstream f("mesh.dump");
      mesh->ncmesh->DebugDump(f);*/
   }

   // 15. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}

GridFunction* ProlongToMaxOrder(const GridFunction *x)
{
   const FiniteElementSpace *fespace = x->FESpace();
   Mesh *mesh = fespace->GetMesh();
   const FiniteElementCollection *fec = fespace->FEColl();

   // Find the max order in the space
   int max_order = 1;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      max_order = std::max(fespace->GetElementOrder(i), max_order);
   }

   // Create a visualization space of max order for all elements
   FiniteElementCollection *visualization_fec =
      new L2_FECollection(max_order, mesh->Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace *visualization_space =
      new FiniteElementSpace(mesh, visualization_fec);

   IsoparametricTransformation T;
   DenseMatrix I;

   GridFunction *visualization_x = new GridFunction(visualization_space);

   // Interpolate solution vector in the visualization space
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Geometry::Type geometry = mesh->GetElementGeometry(i);
      T.SetIdentityTransformation(geometry);

      Array<int> dofs;
      fespace->GetElementDofs(i, dofs);
      Vector elemvect, visualization_vect;
      x->GetSubVector(dofs, elemvect);

      const auto *fe = fec->GetFE(geometry, fespace->GetElementOrder(i));
      const auto *visualization_fe = visualization_fec->GetFE(geometry, max_order);

      visualization_fe->GetTransferMatrix(*fe, T, I);
      visualization_space->GetElementDofs(i, dofs);
      visualization_vect.SetSize(dofs.Size());

      I.Mult(elemvect, visualization_vect);
      visualization_x->SetSubVector(dofs, visualization_vect);
   }

   visualization_x->MakeOwner(visualization_fec);
   return visualization_x;
}
