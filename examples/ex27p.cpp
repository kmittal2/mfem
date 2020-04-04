//                       MFEM Example 27 - Parallel Version
//
// Compile with: make ex27p
//
// Sample runs:  mpirun -np 4 ex27p
//               mpirun -np 4 ex27p -dg
//               mpirun -np 4 ex27p -dg -dbc 8 -nbc -2
//               mpirun -np 4 ex27p -rbc-a 1 -rbc-b 8
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 0 with a variety of boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order using a continuous or discontinuous space.  We then
//               apply Dirichlet, Neumann (both homogeneous and inhomogeneous),
//               Robin, and Periodic boundary conditions on different portions
//               of a predefined mesh.
//
//               The predefined mesh consists of a rectangle with two
//               holes removed (see below).  The narrow ends of the
//               mesh are connected to form a Periodic boundary
//               condition.  The lower edge (tagged with attribute 1)
//               receives an inhomogeneous Neumann boundary condition.
//               A Robin boundary condition is applied to upper edge
//               (attribute 2).  The circular hole on the left
//               (attribute 3) enforces a Dirichlet boundary
//               condition.  Finally, a natural boundary condition, or
//               homogeneous Neumann BC, is applied to the circular
//               hole on the right (attribute 4).
//
//                  Attribute 3    ^ y  Attribute 2
//                        \        |      /
//                     +-----------+-----------+
//                     |    \_     |     _     |
//                     |    / \    |    / \    |
//                  <--+---+---+---+---+---+---+--> x
//                     |    \_/    |    \_/    |
//                     |           |      \    |
//                     +-----------+-----------+       (hole radii are
//                          /      |        \            adjustable)
//                  Attribute 1    v    Attribute 4
//
//               The boundary conditions are defined as (where u is
//               the solution field):
//                  Dirichlet: u = d
//                  Neumann:   n.Grad(u) = g
//                  Robin:     n.Grad(u) + a u = b
//
//               The user can adjust the values of 'd', 'g', 'a', and
//               'b' with command line options.
//
//               This example highlights the differing implementations of
//               boundary conditions with continuous and discontinuous Galerkin
//               formulations of the Laplace problem.
//
//               We recommend viewing examples 1 and 14 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static double a_ = 0.2;

// Normal to hole with boundary attribute 4
void n4Vec(const Vector &x, Vector &n) { n = x; n[0] -= 0.5; n /= -n.Norml2(); }

Mesh * GenerateSerialMesh(int ref);

double IntegrateBC(ParGridFunction &x, Array<int> &bdr, bool h1);
double IntegrateNGradBC(ParGridFunction &u, VectorCoefficient &nCoef,
                        Array<int> &bdr, bool h1,
                        OperatorHandle &M, double a = 0.0);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi;
   if (!mpi.Root()) { mfem::out.Disable(); mfem::err.Disable(); }

   // 2. Parse command-line options.
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 1;
   double sigma = -1.0;
   double kappa = -1.0;
   bool h1 = true;
   bool visualization = true;

   double mat_val = 1.0;
   double dbc_val = 0.0;
   double nbc_val = 1.0;
   double rbc_a_val = 1.0; // du/dn + a * u = b
   double rbc_b_val = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&h1, "-h1", "--continuous", "-dg", "--discontinuous",
                  "Select continuous \"H1\" or discontinuous \"DG\" basis.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&mat_val, "-mat", "--material-value",
                  "Constant value for material coefficient "
                  "in the Laplace operator.");
   args.AddOption(&dbc_val, "-dbc", "--dirichlet-value",
                  "Constant value for Dirichlet Boundary Condition.");
   args.AddOption(&nbc_val, "-nbc", "--neumann-value",
                  "Constant value for Neumann Boundary Condition.");
   args.AddOption(&rbc_a_val, "-rbc-a", "--robin-a-value",
                  "Constant 'a' value for Robin Boundary Condition: "
                  "du/dn + a * u = b.");
   args.AddOption(&rbc_b_val, "-rbc-b", "--robin-b-value",
                  "Constant 'b' value for Robin Boundary Condition: "
                  "du/dn + a * u = b.");
   args.AddOption(&a_, "-a", "--radius",
                  "Radius of holes in the mesh.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   if (kappa < 0 && !h1)
   {
      kappa = (order+1)*(order+1);
   }
   args.PrintOptions(mfem::out);

   if (a_ < 0.01)
   {
      mfem::out << "Hole radius too small, resetting to 0.01.\n";
      a_ = 0.01;
   }
   if (a_ > 0.49)
   {
      mfem::out << "Hole radius too large, resetting to 0.49.\n";
      a_ = 0.49;
   }

   // 3. Construct the (serial) mesh and refine it if requested.
   Mesh *mesh = GenerateSerialMesh(ser_ref_levels);
   int dim = mesh->Dimension();

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   // 5. Define a parallel finite element space on the parallel mesh. Here we
   //    use either continuous Lagrange finite elements or discontinuous
   //    Galerkin finite elements of the specified order.
   FiniteElementCollection *fec =
      h1 ? (FiniteElementCollection*)new H1_FECollection(order, dim) :
      (FiniteElementCollection*)new DG_FECollection(order, dim);
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_Int size = fespace.GlobalTrueVSize();
   mfem::out << "Number of finite element unknowns: " << size << endl;

   // 6. Create "marker arrays" to define the portions of the boundary
   //    associated with each type of boundary condition.  These arrays
   //    have an entry corresponding to each boundary attribute.
   //    Placing a '1' in entry i marks attribute i+1 as being
   //    active, '0' is inactive.
   Array<int> nbc_bdr(pmesh.bdr_attributes.Max());
   Array<int> rbc_bdr(pmesh.bdr_attributes.Max());
   Array<int> dbc_bdr(pmesh.bdr_attributes.Max());

   nbc_bdr = 0; nbc_bdr[0] = 1;
   rbc_bdr = 0; rbc_bdr[1] = 1;
   dbc_bdr = 0; dbc_bdr[2] = 1;

   Array<int> ess_tdof_list(0);
   if (h1 && pmesh.bdr_attributes.Size())
   {
      // For a continuous basis the linear system must be modifed to enforce
      // an essential (Dirichlet) boundary condition.  In the DG case this is
      // not necessary as the boundary condition will only be enforced weakly.
      fespace.GetEssentialTrueDofs(dbc_bdr, ess_tdof_list);
   }

   // 7. Setup the various coefficients needed for the Laplace operator and
   //    the various boundary conditions.  In general these coefficients could
   //    be functions of position but here we use only constants.
   ConstantCoefficient matCoef(mat_val);
   ConstantCoefficient dbcCoef(dbc_val);
   ConstantCoefficient nbcCoef(nbc_val);
   ConstantCoefficient rbcACoef(rbc_a_val);
   ConstantCoefficient rbcBCoef(rbc_b_val);

   // Since the n.Grad(u) terms arise by integrating -Div(m Grad(u)) by parts
   // we must introduce the coefficient 'm' into the boundary conditions.
   // Therefore, in the case of the Neumann BC, we actually enforce
   // m n.Grad(u) = m g rather than simply n.Grad(u) = g.
   ProductCoefficient m_nbcCoef(matCoef, nbcCoef);
   ProductCoefficient m_rbcACoef(matCoef, rbcACoef);
   ProductCoefficient m_rbcBCoef(matCoef, rbcBCoef);

   // 8. Define the solution vector u as a parallel finite element grid function
   //    corresponding to fespace. Initialize u with initial guess of zero.
   ParGridFunction u(&fespace);
   u = 0.0;

   // 9. Set up the parallel bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(matCoef));
   if (h1)
   {
      // Add a Mass integrator on the Robin boundary
      a.AddBoundaryIntegrator(new MassIntegrator(m_rbcACoef), rbc_bdr);
   }
   else
   {
      // Add the interfacial portion of the Lapalce operator
      a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(matCoef,
                                                            sigma, kappa));

      // Counteract the n.Grad(u) term on the Dirichlet portion of the boundary
      a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(matCoef, sigma, kappa),
                             dbc_bdr);

      // Augment the n.Grad(u) term with a*u on the Robin portion of boundary
      a.AddBdrFaceIntegrator(new BoundaryMassIntegrator(m_rbcACoef),
                             rbc_bdr);
   }
   a.Assemble();

   // 10. Assemble the parallel linear form for the right hand side vector.
   ParLinearForm b(&fespace);

   if (h1)
   {
      // Set the Dirchlet values in the solution vector
      u.ProjectBdrCoefficient(dbcCoef, dbc_bdr);

      // Add the desired value for n.Grad(u) on the Neumann boundary
      b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbcCoef), nbc_bdr);

      // Add the desired value for n.Grad(u) + a*u on the Robin boundary
      b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_rbcBCoef), rbc_bdr);
   }
   else
   {
      // Add the desired value for the Dirchlet boundary
      b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(dbcCoef, matCoef,
                                                         sigma, kappa),
                             dbc_bdr);

      // Add the desired value for n.Grad(u) on the Neumann boundary
      b.AddBdrFaceIntegrator(new BoundaryLFIntegrator(m_nbcCoef),
                             nbc_bdr);

      // Add the desired value for n.Grad(u) + a*u on the Robin boundary
      b.AddBdrFaceIntegrator(new BoundaryLFIntegrator(m_rbcBCoef),
                             rbc_bdr);
   }
   b.Assemble();

   // 11. Construct the linear system.
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

   // 12. Solve the linear system A X = B.
   HypreSolver *amg = new HypreBoomerAMG;
   if (h1 || sigma == -1.0)
   {
      HyprePCG pcg(MPI_COMM_WORLD);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(*amg);
      pcg.SetOperator(*A);
      pcg.Mult(B, X);
   }
   else
   {
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetAbsTol(0.0);
      gmres.SetRelTol(1e-12);
      gmres.SetMaxIter(200);
      gmres.SetKDim(10);
      gmres.SetPrintLevel(1);
      gmres.SetPreconditioner(*amg);
      gmres.SetOperator(*A);
      gmres.Mult(B, X);
   }
   delete amg;

   // 13. Recover the parallel grid function corresponding to U. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, u);

   // 14. Build a mass matrix to help solve for n.Grad(u) where 'n' is
   //     a surface normal.
   ParBilinearForm m(&fespace);
   m.AddDomainIntegrator(new MassIntegrator);
   m.Assemble();

   ess_tdof_list.SetSize(0);
   OperatorPtr M;
   m.FormSystemMatrix(ess_tdof_list, M);

   // 15. Compute the various boundary integrals.
   mfem::out << endl << "Verifying boundary conditions:" << endl;
   {
      // Integrate the solution on the Dirichlet boundary and compare
      // to the expected value. The factor of (2 pi a) divides by the
      // circumference of the hole.
      double dbc_int = IntegrateBC(u, dbc_bdr, h1) / (2.0 * M_PI * a_);
      double dbc_err = fabs(dbc_int - dbc_val);

      bool hom_dbc = (dbc_val == 0.0);
      dbc_err /=  hom_dbc ? 1.0 : fabs(dbc_val);
      mfem::out << "dbc " << dbc_int << ", "
                << (hom_dbc ? "absolute" : "relative")
                << " error " << dbc_err << endl;
   }
   {
      // Integrate n.Grad(u) with n = (0, -1) on the inhomogeneous
      // Neumann boundary and compare to the expected value.
      Vector nVec(2); nVec[0] = 0.0; nVec[1] = -1.0;
      VectorConstantCoefficient nCoef(nVec);

      // The factor of 0.5 divides by the length of the boundary.
      double nbc_int = IntegrateNGradBC(u, nCoef, nbc_bdr, h1, M) * 0.5;
      double nbc_err = fabs(nbc_int - nbc_val);

      bool hom_nbc = (nbc_val == 0.0);
      nbc_err /=  hom_nbc ? 1.0 : fabs(nbc_val);
      mfem::out << "nbc " << nbc_int << ", "
                << (hom_nbc ? "absolute" : "relative")
                << " error " << nbc_err << endl;
   }
   {
      // Integrate n.Grad(u) with n given by the function n4Vec on the
      // homogeneous Neumann boundary and compare to the expected
      // value of zero.
      Array<int> nbc0_bdr(pmesh.bdr_attributes.Max());
      nbc0_bdr = 0;
      nbc0_bdr[3] = 1;

      // The factor of (2 pi a) divides by the circumference of the hole.
      VectorFunctionCoefficient n0Coef(2, n4Vec);
      double nbc_int = IntegrateNGradBC(u, n0Coef, nbc0_bdr, h1, M) /
                       (2.0 * M_PI * a_);
      double nbc_err = fabs(nbc_int);

      bool hom_nbc = true;
      mfem::out << "nbc0 " << nbc_int << ", "
                << (hom_nbc ? "absolute" : "relative")
                << " error " << nbc_err << endl;
   }
   {
      // Integrate n.Grad(u) + a * u with n = (0, 1) on the Robin
      // boundary and compare to the expected value.
      Vector nVec(2); nVec[0] = 0.0; nVec[1] = 1.0;
      VectorConstantCoefficient nCoef(nVec);

      // The factor of 0.5 divides by the length of the boundary.
      double rbc_int = IntegrateNGradBC(u, nCoef, rbc_bdr, h1, M, rbc_a_val)
                       * 0.5;
      double rbc_err = fabs(rbc_int - rbc_b_val);

      bool hom_rbc = (rbc_b_val == 0.0);
      rbc_err /=  hom_rbc ? 1.0 : fabs(rbc_b_val);
      mfem::out << "rbc " << rbc_int << ", "
                << (hom_rbc ? "absolute" : "relative")
                << " error " << rbc_err << endl;
   }

   // 16. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << mpi.WorldRank();
      sol_name << "sol." << setfill('0') << setw(6) << mpi.WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      u.Save(sol_ofs);
   }

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      string title_str = h1 ? "H1" : "DG";
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << mpi.WorldSize()
               << " " << mpi.WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << u
               << "window_title '" << title_str << " Solution'"
               << " keys 'mmc'" << flush;
   }

   // 18. Free the used memory.
   delete fec;

   return 0;
}

void quad_trans(double u, double v, double &x, double &y, bool log = false)
{
   double a = a_; // Radius of disc

   double d = 4.0 * a * (M_SQRT2 - 2.0 * a) * (1.0 - 2.0 * v);

   double v0 = (1.0 + M_SQRT2) * (M_SQRT2 * a - 2.0 * v) *
               ((4.0 - 3 * M_SQRT2) * a +
                (8.0 * (M_SQRT2 - 1.0) * a - 2.0) * v) / d;

   double r = 2.0 * ((M_SQRT2 - 1.0) * a * a * (1.0 - 4.0 *v) +
                     2.0 * (1.0 + M_SQRT2 *
                            (1.0 + 2.0 * (2.0 * a - M_SQRT2 - 1.0) * a)) * v * v
                    ) / d;

   double t = asin(v / r) * u / v;
   if (log)
   {
      mfem::out << "u, v, r, v0, t "
                << u << " " << v << " " << r << " " << v0 << " " << t
                << endl;
   }
   x = r * sin(t);
   y = r * cos(t) - v0;
}

void trans(const Vector &u, Vector &x)
{
   double tol = 1e-4;

   if (u[1] > 0.5 - tol || u[1] < -0.5 + tol)
   {
      x = u;
      return;
   }
   if (u[0] > 1.0 - tol || u[0] < -1.0 + tol || fabs(u[0]) < tol)
   {
      x = u;
      return;
   }

   if (u[0] > 0.0)
   {
      if (u[1] > fabs(u[0] - 0.5))
      {
         quad_trans(u[0] - 0.5, u[1], x[0], x[1]);
         x[0] += 0.5;
         return;
      }
      if (u[1] < -fabs(u[0] - 0.5))
      {
         quad_trans(u[0] - 0.5, -u[1], x[0], x[1]);
         x[0] += 0.5;
         x[1] *= -1.0;
         return;
      }
      if (u[0] - 0.5 > fabs(u[1]))
      {
         quad_trans(u[1], u[0] - 0.5, x[1], x[0]);
         x[0] += 0.5;
         return;
      }
      if (u[0] - 0.5 < -fabs(u[1]))
      {
         quad_trans(u[1], 0.5 - u[0], x[1], x[0]);
         x[0] *= -1.0;
         x[0] += 0.5;
         return;
      }
   }
   else
   {
      if (u[1] > fabs(u[0] + 0.5))
      {
         quad_trans(u[0] + 0.5, u[1], x[0], x[1]);
         x[0] -= 0.5;
         return;
      }
      if (u[1] < -fabs(u[0] + 0.5))
      {
         quad_trans(u[0] + 0.5, -u[1], x[0], x[1]);
         x[0] -= 0.5;
         x[1] *= -1.0;
         return;
      }
      if (u[0] + 0.5 > fabs(u[1]))
      {
         quad_trans(u[1], u[0] + 0.5, x[1], x[0]);
         x[0] -= 0.5;
         return;
      }
      if (u[0] + 0.5 < -fabs(u[1]))
      {
         quad_trans(u[1], -0.5 - u[0], x[1], x[0]);
         x[0] *= -1.0;
         x[0] -= 0.5;
         return;
      }
   }
   x = u;
}

Mesh * GenerateSerialMesh(int ref)
{
   Mesh * mesh = new Mesh(2, 29, 16, 24, 2);

   int vi[4];

   for (int i=0; i<2; i++)
   {
      int o = 13 * i;
      vi[0] = o + 0; vi[1] = o + 3; vi[2] = o + 4; vi[3] = o + 1;
      mesh->AddQuad(vi);

      vi[0] = o + 1; vi[1] = o + 4; vi[2] = o + 5; vi[3] = o + 2;
      mesh->AddQuad(vi);

      vi[0] = o + 5; vi[1] = o + 8; vi[2] = o + 9; vi[3] = o + 2;
      mesh->AddQuad(vi);

      vi[0] = o + 8; vi[1] = o + 12; vi[2] = o + 15; vi[3] = o + 9;
      mesh->AddQuad(vi);

      vi[0] = o + 11; vi[1] = o + 14; vi[2] = o + 15; vi[3] = o + 12;
      mesh->AddQuad(vi);

      vi[0] = o + 10; vi[1] = o + 13; vi[2] = o + 14; vi[3] = o + 11;
      mesh->AddQuad(vi);

      vi[0] = o + 6; vi[1] = o + 13; vi[2] = o + 10; vi[3] = o + 7;
      mesh->AddQuad(vi);

      vi[0] = o + 0; vi[1] = o + 6; vi[2] = o + 7; vi[3] = o + 3;
      mesh->AddQuad(vi);
   }

   vi[0] =  0; vi[1] =  6; mesh->AddBdrSegment(vi, 1);
   vi[0] =  6; vi[1] = 13; mesh->AddBdrSegment(vi, 1);
   vi[0] = 13; vi[1] = 19; mesh->AddBdrSegment(vi, 1);
   vi[0] = 19; vi[1] = 26; mesh->AddBdrSegment(vi, 1);

   vi[0] = 28; vi[1] = 22; mesh->AddBdrSegment(vi, 2);
   vi[0] = 22; vi[1] = 15; mesh->AddBdrSegment(vi, 2);
   vi[0] = 15; vi[1] =  9; mesh->AddBdrSegment(vi, 2);
   vi[0] =  9; vi[1] =  2; mesh->AddBdrSegment(vi, 2);

   for (int i=0; i<2; i++)
   {
      int o = 13 * i;
      vi[0] = o +  3; vi[1] = o +  7; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  7; vi[1] = o + 10; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o + 10; vi[1] = o + 11; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o + 11; vi[1] = o + 12; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o + 12; vi[1] = o +  8; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  8; vi[1] = o +  5; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  5; vi[1] = o +  4; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  4; vi[1] = o +  3; mesh->AddBdrSegment(vi, 3 + i);
   }

   double d[2];
   double a = a_ / M_SQRT2;

   d[0] = -1.0; d[1] = -0.5; mesh->AddVertex(d);
   d[0] = -1.0; d[1] =  0.0; mesh->AddVertex(d);
   d[0] = -1.0; d[1] =  0.5; mesh->AddVertex(d);

   d[0] = -0.5 - a; d[1] =   -a; mesh->AddVertex(d);
   d[0] = -0.5 - a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] = -0.5 - a; d[1] =    a; mesh->AddVertex(d);

   d[0] = -0.5; d[1] = -0.5; mesh->AddVertex(d);
   d[0] = -0.5; d[1] =   -a; mesh->AddVertex(d);
   d[0] = -0.5; d[1] =    a; mesh->AddVertex(d);
   d[0] = -0.5; d[1] =  0.5; mesh->AddVertex(d);

   d[0] = -0.5 + a; d[1] =   -a; mesh->AddVertex(d);
   d[0] = -0.5 + a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] = -0.5 + a; d[1] =    a; mesh->AddVertex(d);

   d[0] =  0.0; d[1] = -0.5; mesh->AddVertex(d);
   d[0] =  0.0; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  0.0; d[1] =  0.5; mesh->AddVertex(d);

   d[0] =  0.5 - a; d[1] =   -a; mesh->AddVertex(d);
   d[0] =  0.5 - a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  0.5 - a; d[1] =    a; mesh->AddVertex(d);

   d[0] =  0.5; d[1] = -0.5; mesh->AddVertex(d);
   d[0] =  0.5; d[1] =   -a; mesh->AddVertex(d);
   d[0] =  0.5; d[1] =    a; mesh->AddVertex(d);
   d[0] =  0.5; d[1] =  0.5; mesh->AddVertex(d);

   d[0] =  0.5 + a; d[1] =   -a; mesh->AddVertex(d);
   d[0] =  0.5 + a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  0.5 + a; d[1] =    a; mesh->AddVertex(d);

   d[0] =  1.0; d[1] = -0.5; mesh->AddVertex(d);
   d[0] =  1.0; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  1.0; d[1] =  0.5; mesh->AddVertex(d);

   mesh->FinalizeTopology();

   mesh->SetCurvature(1, true);

   // Stitch the ends of the stack together
   {
      Array<int> v2v(mesh->GetNV());
      for (int i = 0; i < v2v.Size() - 3; i++)
      {
         v2v[i] = i;
      }
      // identify vertices on the narrow ends of the rectangle
      v2v[v2v.Size() - 3] = 0;
      v2v[v2v.Size() - 2] = 1;
      v2v[v2v.Size() - 1] = 2;

      // renumber elements
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         Element *el = mesh->GetElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
      }
      // renumber boundary elements
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         Element *el = mesh->GetBdrElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
      }
      mesh->RemoveUnusedVertices();
      mesh->RemoveInternalBoundaries();
   }
   mesh->SetCurvature(3, true);

   for (int l = 0; l < ref; l++)
   {
      mesh->UniformRefinement();
   }

   mesh->Transform(trans);

   return mesh;
}

double IntegrateBC(ParGridFunction &x, Array<int> &bdr, bool h1)
{
   ParFiniteElementSpace * fespace = x.ParFESpace();
   ConstantCoefficient one(1.0);

   // Integrate the basis functions along the given boundary
   ParLinearForm lf(fespace);
   if (h1)
   {
      lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), bdr);
   }
   else
   {
      lf.AddBdrFaceIntegrator(new BoundaryLFIntegrator(one), bdr);
   }
   lf.Assemble();

   // Compute the integral of x along the given boundary
   return lf(x);
}

double IntegrateNGradBC(ParGridFunction &u, VectorCoefficient &nCoef,
                        Array<int> &bdr, bool h1,
                        OperatorHandle &M, double a)
{
   ParFiniteElementSpace * fespace = u.ParFESpace();

   // Compute the operator n.Grad
   ParBilinearForm nd(fespace);
   nd.AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(nCoef));
   nd.Assemble();

   // Apply the operator to the solution vector
   ParLinearForm b(fespace);
   nd.Mult(u, b);

   // Solve for n.Grad(u)
   Vector B, X;
   X.SetSize(fespace->TrueVSize());
   X = 0.0;

   B.SetSize(X.Size());
   b.ParallelAssemble(B);

   CGSolver mcg(MPI_COMM_WORLD);
   mcg.SetRelTol(1e-12);
   mcg.SetMaxIter(2000);
   mcg.SetPrintLevel(0);
   mcg.SetOperator(*M);
   mcg.Mult(B, X);

   ParGridFunction ndu(fespace);
   ndu.Distribute(X);

   // For Robin BCs we add a * u
   if (a != 0.0)
   {
      ndu.Add(a, u);
   }

   // Integrate n.Grad(u) + a * u along the given boundary
   return IntegrateBC(ndu, bdr, h1);
}
