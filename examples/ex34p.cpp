//                         MFEM Example 34 - Parallel Version
//
//
// Compile with: make ex34p
//
// Sample runs: mpirun -np 2 ex34p -o 2
//              mpirun -np 2 ex34p -o 2 -r 4
//
//
// Description: This example code demonstrates the use of MFEM to solve the
//              bound-constrained energy minimization problem
//
//                      minimize ||∇u||² subject to u ≥ ϕ in H¹₀.
//
//              This is known as the obstacle problem, and it is a simple
//              mathematical model for contact mechanics.
//
//              In this example, the obstacle ϕ is a half-sphere centered
//              at the origin of a circular domain Ω. After solving to a
//              specified tolerance, the numerical solution is compared to
//              a closed-form exact solution to assess accuracy.
//
//              The problem is discretized and solved using the entropic
//              finite element method (EFEM) introduced by Keith and
//              Surowiec [1].
//
//              This example highlights the ability of MFEM to deliver high-
//              order solutions to variation inequality problems and
//              showcases how to set up and solve nonlinear mixed methods.
//
//
// [1] Keith, B. and Surowiec, T. (2023) The entropic finite element method
//     (in preparation).


#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double spherical_obstacle(const Vector &pt);
void spherical_obstacle_gradient(const Vector &pt, Vector &grad);
double exact_solution_obstacle(const Vector &pt);
void exact_solution_gradient_obstacle(const Vector &pt, Vector &grad);

class LogarithmGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   Coefficient *obstacle;
   double min_val;

public:
   LogarithmGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_,
                                    double min_val_=-36)
      : u(&u_), obstacle(&obst_), min_val(min_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ExponentialGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   Coefficient *obstacle;
   double min_val;
   double max_val;

public:
   ExponentialGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_,
                                      double min_val_=0.0, double max_val_=1e6)
      : u(&u_), obstacle(&obst_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/disk.mesh";
   int order = 1;
   bool visualization = true;
   int max_it = 10;
   double tol = 1e-5;
   int ref_levels = 3;
   double alpha0 = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  "isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&tol, "-tol", "--tol",
                  "Stopping criteria based on the difference between"
                  "successive solution updates");
   args.AddOption(&alpha0, "-step", "--step",
                  "Initial step size alpha");
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
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 3. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   int curvature_order = max(order,2);
   mesh.SetCurvature(curvature_order);
   mesh.EnsureNCMesh();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection H1fec(order, dim);
   ParFiniteElementSpace H1fes(&pmesh, &H1fec);

   L2_FECollection L2fec(order-1, dim);
   ParFiniteElementSpace L2fes(&pmesh, &L2fec);

   if (myid == 0)
   {
      cout << "Number of finite element unknowns: "
           << H1fes.GetTrueVSize()
           << " "
           << L2fes.GetTrueVSize() << endl;
   }

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = H1fes.GetVSize();
   offsets[2] = L2fes.GetVSize();
   offsets.PartialSum();

   Array<int> toffsets(3);
   toffsets[0] = 0;
   toffsets[1] = H1fes.GetTrueVSize();
   toffsets[2] = L2fes.GetTrueVSize();
   toffsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   BlockVector tx(toffsets), trhs(toffsets);
   tx = 0.0; trhs = 0.0;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> empty;
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Define an initial guess for the solution.
   auto IC_func = [](const Vector &x)
   {
      double r0 = 1.0;
      double rr = 0.0;
      for (int i=0; i<x.Size(); i++)
      {
         rr += x(i)*x(i);
      }
      return r0*r0 - rr;
   };
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);

   // 7. Define the solution vectors as a finite element grid functions
   //    corresponding to the fespaces.
   ParGridFunction u_gf, delta_psi_gf;
   u_gf.MakeRef(&H1fes,x.GetBlock(0).GetData());
   delta_psi_gf.MakeRef(&L2fes,x.GetBlock(1).GetData());
   delta_psi_gf = 0.0;

   ParGridFunction u_old_gf(&H1fes);
   ParGridFunction psi_old_gf(&L2fes);
   ParGridFunction psi_gf(&L2fes);
   u_old_gf = 0.0;
   psi_old_gf = 0.0;

   // 8. Define the function coefficients for the solution and use them to
   //    initialize the initial guess
   FunctionCoefficient exact_coef(exact_solution_obstacle);
   VectorFunctionCoefficient exact_grad_coef(dim,exact_solution_gradient_obstacle);
   FunctionCoefficient IC_coef(IC_func);
   ConstantCoefficient f(0.0);
   FunctionCoefficient obstacle(spherical_obstacle);
   u_gf.ProjectCoefficient(IC_coef);
   u_old_gf = u_gf;

   // 9. Initialize the slack variable ψₕ = exp(uₕ)
   LogarithmGridFunctionCoefficient ln_u(u_gf, obstacle);
   psi_gf.ProjectCoefficient(ln_u);
   psi_old_gf = psi_gf;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   ParGridFunction u_alt_gf(&L2fes);
   ParGridFunction error_gf(&L2fes);

   ExponentialGridFunctionCoefficient exp_psi(psi_gf,obstacle);
   u_alt_gf.ProjectCoefficient(exp_psi);

   if (visualization)
   {
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock << "solution\n" << pmesh << u_alt_gf <<
               "window_title 'Discrete solution'" << flush;
   }
   else
   {
      sol_sock.close();
   }

   // 10. Iterate
   int k;
   int total_iterations = 0;
   double increment_u = 0.1;
   for (k = 0; k < max_it; k++)
   {
      double alpha = alpha0 * (k+1);

      ParGridFunction u_tmp(&H1fes);
      u_tmp = u_old_gf;

      if (myid == 0)
      {
         mfem::out << "\nOUTER ITERATION " << k+1 << endl;
      }

      int j;
      for ( j = 0; j < 15; j++)
      {
         total_iterations++;

         ConstantCoefficient alpha_cf(alpha);

         ParLinearForm b0,b1;
         b0.Update(&H1fes,rhs.GetBlock(0),0);
         b1.Update(&L2fes,rhs.GetBlock(1),0);

         ExponentialGridFunctionCoefficient exp_psi(psi_gf, zero);
         ProductCoefficient neg_exp_psi(-1.0,exp_psi);
         GradientGridFunctionCoefficient grad_u_old(&u_old_gf);
         ProductCoefficient alpha_f(alpha, f);
         GridFunctionCoefficient psi_cf(&psi_gf);
         GridFunctionCoefficient psi_old_cf(&psi_old_gf);
         SumCoefficient psi_old_minus_psi(psi_old_cf, psi_cf, 1.0, -1.0);

         b0.AddDomainIntegrator(new DomainLFIntegrator(alpha_f));
         b0.AddDomainIntegrator(new DomainLFIntegrator(psi_old_minus_psi));
         b0.Assemble();

         b1.AddDomainIntegrator(new DomainLFIntegrator(exp_psi));
         b1.AddDomainIntegrator(new DomainLFIntegrator(obstacle));
         b1.Assemble();

         ParBilinearForm a00(&H1fes);
         a00.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
         a00.AddDomainIntegrator(new DiffusionIntegrator(alpha_cf));
         a00.Assemble();
         HypreParMatrix A00;
         a00.FormLinearSystem(ess_tdof_list, x.GetBlock(0), rhs.GetBlock(0),
                              A00, tx.GetBlock(0), trhs.GetBlock(0));


         ParMixedBilinearForm a10(&H1fes,&L2fes);
         a10.AddDomainIntegrator(new MixedScalarMassIntegrator());
         a10.Assemble();
         HypreParMatrix A10;
         a10.FormRectangularLinearSystem(ess_tdof_list, empty, x.GetBlock(0),
                                         rhs.GetBlock(1),
                                         A10, tx.GetBlock(0), trhs.GetBlock(1));

         HypreParMatrix &A01 = *A10.Transpose();

         ParBilinearForm a11(&L2fes);
         a11.AddDomainIntegrator(new MassIntegrator(neg_exp_psi));
         ConstantCoefficient eps_cf(-1e-6);
         a11.AddDomainIntegrator(new DiffusionIntegrator(eps_cf));
         a11.Assemble();
         a11.Finalize();
         HypreParMatrix A11;
         a11.FormSystemMatrix(empty, A11);

         BlockOperator A(toffsets);
         A.SetBlock(0,0,&A00);
         A.SetBlock(1,0,&A10);
         A.SetBlock(0,1,&A01);
         A.SetBlock(1,1,&A11);

         BlockDiagonalPreconditioner prec(toffsets);
         HypreBoomerAMG P00(A00);
         P00.SetPrintLevel(0);
         HypreSmoother P11(A11);
         prec.SetDiagonalBlock(0,&P00);
         prec.SetDiagonalBlock(1,new HypreSmoother(A11));

         GMRESSolver gmres(MPI_COMM_WORLD);
         gmres.SetPrintLevel(-1);
         gmres.SetRelTol(1e-12);
         gmres.SetMaxIter(20000);
         gmres.SetKDim(500);
         gmres.SetOperator(A);
         gmres.SetPreconditioner(prec);
         gmres.Mult(trhs,tx);

         u_gf.SetFromTrueDofs(tx.GetBlock(0));
         delta_psi_gf.SetFromTrueDofs(tx.GetBlock(1));

         u_tmp -= u_gf;
         double Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         double gamma = 1.0;
         delta_psi_gf *= gamma;
         psi_gf += delta_psi_gf;

         if (visualization)
         {
            sol_sock << "parallel " << num_procs << " " << myid << "\n";
            sol_sock << "solution\n" << pmesh << u_gf << "window_title 'Discrete solution'"
                     << flush;
         }

         if (myid == 0)
         {
            mfem::out << "Newton_update_size = " << Newton_update_size << endl;
         }

         if (Newton_update_size < increment_u)
         {
            break;
         }
      }

      u_tmp = u_gf;
      u_tmp -= u_old_gf;
      increment_u = u_tmp.ComputeL2Error(zero);

      if (myid == 0)
      {
         mfem::out << "Number of Newton iterations = " << j+1 << endl;
         mfem::out << "Increment (|| uₕ - uₕ_prvs||) = " << increment_u << endl;
      }

      u_old_gf = u_gf;
      psi_old_gf = psi_gf;

      if (increment_u < tol || k == max_it-1)
      {
         break;
      }

      double L2_error = u_gf.ComputeL2Error(exact_coef);
      if (myid == 0)
      {
         mfem::out << "L2-error  (|| u - uₕ||)       = " << L2_error << endl;
      }

   }

   if (myid == 0)
   {
      mfem::out << "\n Outer iterations: " << k+1
                << "\n Total iterations: " << total_iterations
                << "\n dofs:             " << H1fes.GetTrueVSize() + L2fes.GetTrueVSize()
                << endl;
   }

   // 11. Exact solution.
   if (visualization)
   {
      socketstream err_sock(vishost, visport);
      err_sock.precision(8);

      ParGridFunction error(&H1fes);
      error = 0.0;
      error.ProjectCoefficient(exact_coef);
      error -= u_gf;

      err_sock << "parallel " << num_procs << " " << myid << "\n";
      err_sock << "solution\n" << pmesh << error << "window_title 'Error'"  << flush;
   }

   {
      ExponentialGridFunctionCoefficient exp_psi(psi_gf,obstacle);
      u_alt_gf.ProjectCoefficient(exp_psi);
      error_gf = 0.0;
      error_gf.ProjectCoefficient(exact_coef);
      error_gf -= u_alt_gf;
      error_gf *= -1.0;

      double L2_error = u_gf.ComputeL2Error(exact_coef);
      double H1_error = u_gf.ComputeH1Error(&exact_coef,&exact_grad_coef);
      double L2_error_alt = u_alt_gf.ComputeL2Error(exact_coef);

      if (myid == 0)
      {
         mfem::out << "\n Final L2-error (|| u - uₕ||)          = " << L2_error <<
                   endl;
         mfem::out << " Final H1-error (|| u - uₕ||)          = " << H1_error << endl;
         mfem::out << " Final L2-error (|| u - ϕ - exp(ψₕ)||) = " << L2_error_alt <<
                   endl;
      }
   }

   return 0;
}

double LogarithmGridFunctionCoefficient::Eval(ElementTransformation &T,
                                              const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip) - obstacle->Eval(T, ip);
   return max(min_val, log(val));
}

double ExponentialGridFunctionCoefficient::Eval(ElementTransformation &T,
                                                const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, exp(val) + obstacle->Eval(T, ip)));
}

double spherical_obstacle(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double beta = 0.9;

   double b = r0*beta;
   double tmp = sqrt(r0*r0 - b*b);
   double B = tmp + b*b/tmp;
   double C = -b/tmp;

   if (r > b)
   {
      return B + r * C;
   }
   else
   {
      return sqrt(r0*r0 - r*r);
   }
}

void spherical_obstacle_gradient(const Vector &pt, Vector &grad)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double beta = 0.9;

   double b = r0*beta;
   double tmp = sqrt(r0*r0-b*b);
   double C = -b/tmp;

   if (r > b)
   {
      grad(0) = C * x / r;
      grad(1) = C * y / r;
   }
   else
   {
      grad(0) = - x / sqrt( r0*r0 - r*r );
      grad(1) = - y / sqrt( r0*r0 - r*r );
   }
}

double exact_solution_obstacle(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double a =  0.348982574111686;
   double A = -0.340129705945858;

   if (r > a)
   {
      return A * log(r);
   }
   else
   {
      return sqrt(r0*r0-r*r);
   }
}

void exact_solution_gradient_obstacle(const Vector &pt, Vector &grad)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double a =  0.348982574111686;
   double A = -0.340129705945858;

   if (r > a)
   {
      grad(0) =  A * x / (r*r);
      grad(1) =  A * y / (r*r);
   }
   else
   {
      grad(0) = - x / sqrt( r0*r0 - r*r );
      grad(1) = - y / sqrt( r0*r0 - r*r );
   }
}