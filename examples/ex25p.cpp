//                       MFEM Example 25 - Parallel Version
//
// Compile with: make ex25p
//
// Sample runs:  mpirun -np 4 ex25p -m ../data/star.mesh
//               mpirun -np 4 ex25p -m ../data/fichera.mesh
//               mpirun -np 4 ex25p -m ../data/beam-hex.mesh
//
// Device sample runs:
//               mpirun -np 4 ex25p -d cuda
//               mpirun -np 4 ex25p -d occa-cuda
//               mpirun -np 4 ex25p -d raja-omp
//               mpirun -np 4 ex25p -d ceed-cpu
//               mpirun -np 4 ex25p -d ceed-cuda
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions
//               as in example 1. It highlights on the creation of a hierarchy
//               of discretization spaces with partial assembly and the
//               construction of an efficient p-multigrid preconditioner for the
//               iterative solver.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class MultigridDiffusionOperator : public MultigridOperator
{
private:
   ParMesh* pmesh_lor;
   H1_FECollection* fec_lor;
   ParFiniteElementSpace* fespace_lor;
   Array<BilinearForm*> bfs;
   Array<Array<int>*> essentialTrueDofs;
   ConstantCoefficient one;

public:
   /// Constructor for a multigrid diffusion operator for a given SpaceHierarchy. Uses Chebyshev accelerated smoothing.
   MultigridDiffusionOperator(ParSpaceHierarchy& spaceHierarchy,
                              Array<int>& ess_bdr, int chebyshevOrder = 2)
      : one(1.0)
   {
      ParMesh* pmesh = spaceHierarchy.GetFESpaceAtLevel(0).GetParMesh();
      pmesh_lor = new ParMesh(pmesh, 1, BasisType::GaussLobatto);
      fec_lor =
         new H1_FECollection(1, pmesh->Dimension(), BasisType::GaussLobatto);
      fespace_lor = new ParFiniteElementSpace(pmesh_lor, fec_lor);
      ParBilinearForm* a_lor = new ParBilinearForm(fespace_lor);
      AddIntegrators(a_lor);
      a_lor->UsePrecomputedSparsity();
      a_lor->Assemble();
      bfs.Append(a_lor);

      essentialTrueDofs.Append(new Array<int>());
      spaceHierarchy.GetFESpaceAtLevel(0).GetEssentialTrueDofs(
         ess_bdr, *essentialTrueDofs.Last());

      HypreParMatrix* hypreCoarseMat = new HypreParMatrix();
      a_lor->FormSystemMatrix(*essentialTrueDofs.Last(), *hypreCoarseMat);

      HypreBoomerAMG* amg = new HypreBoomerAMG(*hypreCoarseMat);
      amg->SetPrintLevel(-1);

      AddCoarsestLevel(hypreCoarseMat, amg, true, true);

      for (int level = 1; level < spaceHierarchy.GetNumLevels(); ++level)
      {
         ParBilinearForm* form = new ParBilinearForm(&spaceHierarchy.GetFESpaceAtLevel(
                                                        level));
         form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         AddIntegrators(form);
         form->Assemble();
         bfs.Append(form);

         essentialTrueDofs.Append(new Array<int>());
         spaceHierarchy.GetFESpaceAtLevel(level).GetEssentialTrueDofs(
            ess_bdr, *essentialTrueDofs.Last());

         OperatorPtr opr;
         opr.SetType(Operator::ANY_TYPE);
         form->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
         opr.SetOperatorOwner(false);

         Vector diag(spaceHierarchy.GetFESpaceAtLevel(level).GetTrueVSize());
         form->AssembleDiagonal(diag);

         Solver* smoother = new OperatorChebyshevSmoother(
            opr.Ptr(), diag, *essentialTrueDofs.Last(), chebyshevOrder, pmesh->GetComm());

         Operator* P =
            new TrueTransferOperator(spaceHierarchy.GetFESpaceAtLevel(level - 1),
                                     spaceHierarchy.GetFESpaceAtLevel(level));

         AddLevel(opr.Ptr(), smoother, P, true, true, true);
      }
   }

   virtual ~MultigridDiffusionOperator()
   {
      for (int i = 0; i < bfs.Size(); ++i)
      {
         delete bfs[i];
      }

      for (int i = 0; i < essentialTrueDofs.Size(); ++i)
      {
         delete essentialTrueDofs[i];
      }

      delete fespace_lor;
      delete fec_lor;
      delete pmesh_lor;
   }

   void EliminateBCs(Vector& x, Vector& b, Vector& X, Vector& B)
   {
      Operator *oper;
      GetOperatorAtFinestLevel()->FormLinearSystem(*essentialTrueDofs.Last(), x, b,
                                                   oper, X, B);
      delete oper;
   }

   void RecoverFEMSolution(const Vector& X, const Vector& b, Vector& x)
   {
      bfs.Last()->RecoverFEMSolution(X, b, x);
   }

private:
   void AddIntegrators(BilinearForm* form)
   {
      form->AddDomainIntegrator(new DiffusionIntegrator(one));
   }
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int orderrefinements = 2;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&orderrefinements, "-or", "--orderrefinements",
                  "Number of order refinements. Finest level in the hierarchy has order 2^{or}");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec = new H1_FECollection(1, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   Array<FiniteElementCollection*> collections;
   collections.Append(fec);
   ParSpaceHierarchy* spaceHierarchy = new ParSpaceHierarchy(pmesh, fespace, true,
                                                             true);
   for (int level = 0; level < orderrefinements; ++level)
   {
      collections.Append(new H1_FECollection(std::pow(2, level+1), dim));
      spaceHierarchy->AddOrderRefinedLevel(collections.Last());
   }

   HYPRE_Int size = spaceHierarchy->GetFinestFESpace().GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr = 1;
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(&spaceHierarchy->GetFinestFESpace());
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   ParGridFunction x(&spaceHierarchy->GetFinestFESpace());
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.


   // 13. Solve the linear system A X = B.
   //     * With p-multigrid preconditioner

   MultigridDiffusionOperator* mgOperator = new MultigridDiffusionOperator(
      *spaceHierarchy, ess_bdr);
   MultigridSolver* prec = new MultigridSolver(mgOperator,
                                               MultigridSolver::CycleType::VCYCLE,
                                               1, 1);

   Vector X, B;
   mgOperator->EliminateBCs(x, *b, X, B);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(*mgOperator);
   cg.SetPreconditioner(*prec);
   cg.Mult(B, X);

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   mgOperator->RecoverFEMSolution(X, *b, x);

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
   delete prec;
   delete mgOperator;
   delete b;
   delete spaceHierarchy;
   for (int level = 0; level < collections.Size(); ++level)
   {
      delete collections[level];
   }

   MPI_Finalize();

   return 0;
}
