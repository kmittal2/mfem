//                 MFEM Ultraweal DPG parallel example for diffusion
//
// Compile with: make pdiffusion
//
// Sample runs
// mpirun -np 4 pdiffusion -m ../../../data/inline-quad.mesh -o 3 -sref 1 -pref 3 -theta 0.0 -prob 0
// mpirun -np 4 pdiffusion -m ../../../data/inline-hex.mesh -o 2 -sref 0 -pref 2 -theta 0.0 -prob 0 -sc
// mpirun -np 4 pdiffusion -m ../../../data/beam-tet.mesh -o 3 -sref 0 -pref 2 -theta 0.0 -prob 0 -sc

// lshape runs
// Note: uniform ref are expected to give sub-optimal rate for the l-shape problem (rate = 2/3)
// mpirun -np 4 pdiffusion -o 2 -sref 1 -pref 5 -theta 0.0 -prob 1 

// L-shape AMR runs
// mpirun -np 4 pdiffusion -o 1 -sref 1 -pref 20 -theta 0.75 -prob 1
// mpirun -np 4 pdiffusion -o 2 -sref 1 -pref 20 -theta 0.75 -prob 1 -sc
// mpirun -np 4 pdiffusion -o 3 -sref 1 -pref 20 -theta 0.75 -prob 1 -sc -do 2

// Description:  
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Poisson problem in parallel

//       - Δ u = f,   in Ω
//         u = u_0, on ∂Ω
//
// It solves two kinds of problems 
// a) A manufactured solution problem where u_exact = sin(π * (x + y + z)). 
//    This example computes and prints out convergence rates for the L2 error.
// b) The l-shape benchmark problem with AMR. The AMR process is driven by the 
//    DPG built-in residual indicator. 

// The DPG UW deals with the First Order System
//   ∇ u - σ = 0, in Ω
// - ∇⋅σ     = f, in Ω
//        u  = u_0, in ∂Ω

// Ultraweak-DPG is obtained by integration by parts of both equations and the 
// introduction of trace unknowns on the mesh skeleton

// u ∈ L^2(Ω), σ ∈ (L^2(Ω))^dim 
// û ∈ H^1/2, σ̂ ∈ H^-1/2  
// -(u , ∇⋅τ) + < û, τ⋅n> - (σ , τ) = 0,      ∀ τ ∈ H(div,Ω)      
//  (σ , ∇ v) - < σ̂, v  >           = (f,v)   ∀ v ∈ H^1(Ω)
//                                û = u_0        on ∂Ω 

// Note: 
// û := u and σ̂ := -σ

// -------------------------------------------------------------
// |   |     u     |     σ     |    û      |    σ̂    |  RHS    |
// -------------------------------------------------------------
// | τ | -(u,∇⋅τ)  |  -(σ,τ)   | < û, τ⋅n> |         |    0    |
// |   |           |           |           |         |         |
// | v |           |  (σ,∇ v)  |           | -<σ̂,v>  |  (f,v)  |  


// where (τ,v) ∈  H(div,Ω) × H^1(Ω) 

#include "mfem.hpp"
#include "util/pweakform.hpp"
#include "../../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

enum prob_type
{
   manufactured,
   lshape
};

prob_type prob;

void solution(const Vector & X, double & u, Vector & du, double & d2u);
double exact_u(const Vector & X);
void exact_sigma(const Vector & X, Vector & sigma);
double exact_hatu(const Vector & X);
void exact_hatsigma(const Vector & X, Vector & hatsigma);
double f_exact(const Vector & X);

int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int myid = mpi.WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   int sref = 0; // initial uniform mesh refinements
   int pref = 0; // parallel mesh refinements for AMR 
   bool visualization = true;
   int iprob = 0;
   bool static_cond = false;
   double theta = 0.7; 

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&sref, "-sref", "--num_serial_refinements",
                  "Number of initial serial uniform refinements");    
   args.AddOption(&pref, "-pref", "--num_parallel_refinements",
                  "Number of AMR refinements");                     
   args.AddOption(&theta, "-theta", "--theta_factor",
                  "Refinement factor (0 indicates uniform refinements) ");                              
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: manufactured, 1: l-shape");       
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
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   if (iprob > 1) { iprob = 1; }
   prob = (prob_type)iprob;

   if (prob == prob_type::lshape)
   {
      mesh_file = "lshape2.mesh"; // this might change 
   }

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   for (int i = 0; i<sref; i++)
   {
      mesh.UniformRefinement();
   }

   mesh.EnsureNCMesh();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define spaces
   // L2 space for u
   FiniteElementCollection *u_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *u_fes = new ParFiniteElementSpace(&pmesh,u_fec);

   // Vector L2 space for σ 
   FiniteElementCollection *sigma_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *sigma_fes = new ParFiniteElementSpace(&pmesh,sigma_fec, dim); 

   // H^1/2 space for û 
   FiniteElementCollection * hatu_fec = new H1_Trace_FECollection(order,dim);
   ParFiniteElementSpace *hatu_fes = new ParFiniteElementSpace(&pmesh,hatu_fec);

   // H^-1/2 space for σ̂ 
   FiniteElementCollection * hatsigma_fec = new RT_Trace_FECollection(order-1,dim);   
   ParFiniteElementSpace *hatsigma_fes = new ParFiniteElementSpace(&pmesh,hatsigma_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * tau_fec = new RT_FECollection(test_order-1, dim);
   FiniteElementCollection * v_fec = new H1_FECollection(test_order, dim);

   Array<ParFiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(u_fes);
   trial_fes.Append(sigma_fes);
   trial_fes.Append(hatu_fes);
   trial_fes.Append(hatsigma_fes);
   test_fec.Append(tau_fec);
   test_fec.Append(v_fec);

   // Required coefficients for the weak formulation
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   FunctionCoefficient f(f_exact); // rhs for the manufactured solution problem

   // Required coefficients for the exact solutions
   FunctionCoefficient uex(exact_u);
   VectorFunctionCoefficient sigmaex(dim,exact_sigma);
   FunctionCoefficient hatuex(exact_hatu);

   ParDPGWeakForm * a = new ParDPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices(true); // this is needed for estimation of residual

   //  -(u,∇⋅τ)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),0,0);

   // -(σ,τ) 
   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(negone)),1,0);

   // (σ,∇ v)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),1,1);

   //  <û,τ⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,2,0);

   // -<σ̂,v> (sign is included in σ̂)
   a->AddTrialIntegrator(new TraceIntegrator,3,1);

   // test integrators (space-induced norm for H(div) × H1)
   // (∇⋅τ,∇⋅δτ)
   a->AddTestIntegrator(new DivDivIntegrator(one),0,0);
   // (τ,δτ)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),0,0);
   // (∇v,∇δv)
   a->AddTestIntegrator(new DiffusionIntegrator(one),1,1);
   // (v,δv)
   a->AddTestIntegrator(new MassIntegrator(one),1,1);

   // RHS
   if (prob == prob_type::manufactured)
   {
      a->AddDomainLFIntegrator(new DomainLFIntegrator(f),1);
   }

   // GridFunction for Dirichlet bdr data
   ParGridFunction hatu_gf;

   // Visualization streams
   socketstream u_out;
   socketstream sigma_out;

   if (myid == 0)
   {
      mfem::out << "\n Refinement |" 
                << "    Dofs    |" 
                << "  L2 Error  |" 
                << "  Rate  |" 
                << "  Residual  |" 
                << "  Rate  |" 
                << " CG it  |" << endl;
      mfem::out << " --------------------"      
                <<  "-------------------"    
                <<  "-------------------"    
                <<  "-------------------" << endl;   
   }

   Array<int> elements_to_refine; // for AMR
   double err0 = 0.;
   int dof0=0.;
   double res0=0.0;
   if (static_cond) { a->EnableStaticCondensation(); }
   for (int iref = 0; iref<=pref; iref++)
   {
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         hatu_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // shift the ess_tdofs
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         ess_tdof_list[i] += u_fes->GetTrueVSize() + sigma_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = u_fes->GetVSize();
      offsets[2] = sigma_fes->GetVSize();
      offsets[3] = hatu_fes->GetVSize();
      offsets[4] = hatsigma_fes->GetVSize();
      offsets.PartialSum();
      BlockVector x(offsets);
      x = 0.0;
      hatu_gf.MakeRef(hatu_fes,x.GetBlock(2));
      hatu_gf.ProjectBdrCoefficient(uex,ess_bdr);

      Vector X,B;
      OperatorPtr Ah;
      a->FormLinearSystem(ess_tdof_list,x,Ah,X,B);

      BlockOperator * A = Ah.As<BlockOperator>();

      BlockDiagonalPreconditioner M(A->RowOffsets());
      M.owns_blocks = 1;
      int skip = 0;
      if (!static_cond)
      {
         HypreBoomerAMG * amg0 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(0,0));
         HypreBoomerAMG * amg1 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(1,1));
         amg0->SetPrintLevel(0);
         amg1->SetPrintLevel(0);
         M.SetDiagonalBlock(0,amg0);
         M.SetDiagonalBlock(1,amg1);
         skip=2;
      }
      HypreBoomerAMG * amg2 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(skip,skip));
      amg2->SetPrintLevel(0);
      M.SetDiagonalBlock(skip,amg2);
      HypreSolver * prec;
      if (dim == 2)
      {
         prec = new HypreAMS((HypreParMatrix &)A->GetBlock(skip+1,skip+1), hatsigma_fes);
      }
      else
      {
         prec = new HypreADS((HypreParMatrix &)A->GetBlock(skip+1,skip+1), hatsigma_fes);
      }
      M.SetDiagonalBlock(skip+1,prec);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(*M);
      cg.SetOperator(*A);
      cg.Mult(B, X);

      a->RecoverFEMSolution(X,x);

      Vector & residuals = a->ComputeResidual(x);

      double residual = residuals.Norml2();

      double maxresidual = residuals.Max(); 
      double globalresidual = residual * residual; 

      MPI_Allreduce(MPI_IN_PLACE,&maxresidual,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&globalresidual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      globalresidual = sqrt(globalresidual);

      ParGridFunction u_gf;
      u_gf.MakeRef(u_fes,x.GetBlock(0));

      ParGridFunction sigma_gf;
      sigma_gf.MakeRef(sigma_fes,x.GetBlock(1));

      int dofs = u_fes->GlobalTrueVSize() + sigma_fes->GlobalTrueVSize()
                 + hatu_fes->GlobalTrueVSize() + hatsigma_fes->GlobalTrueVSize();

      double u_err = u_gf.ComputeL2Error(uex);
      double sigma_err = sigma_gf.ComputeL2Error(sigmaex);
      double L2Error = sqrt(u_err*u_err + sigma_err*sigma_err);
      double rate_err = (iref) ? dim*log(err0/L2Error)/log((double)dof0/dofs) : 0.0;
      double rate_res = (iref) ? dim*log(res0/globalresidual)/log((double)dof0/dofs) : 0.0;
      err0 = L2Error;
      res0 = globalresidual;
      dof0 = dofs;

      if (myid == 0)
      {
         mfem::out << std::right << std::setw(11) << iref << " | " 
               << std::setw(10) <<  dof0 << " | " 
               << std::setprecision(3) 
               << std::setw(10) << std::scientific <<  err0 << " | " 
               << std::setprecision(2) 
               << std::setw(6) << std::fixed << rate_err << " | " 
               << std::setw(10) << std::scientific <<  res0 << " | " 
               << std::setprecision(2) 
               << std::setw(6) << std::fixed << rate_res << " | " 
               << std::setw(6) << std::fixed << cg.GetNumIterations() << " | " 
               << std::setprecision(3) 
               << std::resetiosflags(std::ios::showbase)
               << std::endl;
      }

      if (visualization)
      {
         const char * keys = (iref == 0) ? "jRcm\n" : "";
         char vishost[] = "localhost";
         int  visport   = 19916;

         common::VisualizeField(u_out,vishost,visport,u_gf,
                               "Numerical u", 0,0,500,500,keys);
         common::VisualizeField(sigma_out,vishost,visport,sigma_gf,
                               "Numerical flux", 500,0,500,500,keys);
      }


      if (iref == pref) { break; }

      elements_to_refine.SetSize(0);
      for (int iel = 0; iel<pmesh.GetNE(); iel++)
      {
         if (residuals[iel] >= theta * maxresidual)
         {
            elements_to_refine.Append(iel);
         }
      }

      pmesh.GeneralRefinement(elements_to_refine);

      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

   delete a;
   delete tau_fec;
   delete v_fec;
   delete hatsigma_fes;
   delete hatsigma_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete sigma_fec;
   delete sigma_fes;
   delete u_fec;
   delete u_fes;

   return 0;
}

void solution(const Vector & X, double & u, Vector & du, double & d2u)
{
   du.SetSize(X.Size());
   switch (prob)
   {
      case prob_type::lshape:
      {
         double x = X[0];
         double y = X[1];
         double r = sqrt(x*x + y*y);
         double alpha = 2./3.;
         double phi = atan2(y,x);
         if (phi < 0) phi += 2*M_PI;

         u = pow(r,alpha) * sin(alpha * phi);

         double r_x = x/r;
         double r_y = y/r;
         double phi_x = - y / (r*r);
         double phi_y = x / (r*r);
         double beta = alpha * pow(r,alpha - 1.);
         du[0] = beta*(r_x * sin(alpha*phi) + r * phi_x * cos(alpha*phi));
         du[1] = beta*(r_y * sin(alpha*phi) + r * phi_y * cos(alpha*phi));

         d2u = 0.0; // Not computed since it's not needed for rhs (f = 0)
      }   
      break;
   
      default:
      {
         double alpha = M_PI * (X.Sum());
         u = sin(alpha);
         du.SetSize(X.Size());
         for (int i = 0; i<du.Size(); i++)
         {
            du[i] = M_PI * cos(alpha);
         }
         d2u = - M_PI*M_PI * u * du.Size();
      }
      break;
   }
}

double exact_u(const Vector & X)
{
   double u, d2u;
   Vector du;
   solution(X,u,du,d2u);
   return u;
}

void exact_sigma(const Vector & X, Vector & sigma)
{
   double u, d2u;
   Vector du;
   solution(X,u,du,d2u);
   // σ = ∇ u
   sigma = du;
}

double exact_hatu(const Vector & X)
{
   return exact_u(X);
}

void exact_hatsigma(const Vector & X, Vector & hatsigma)
{
   exact_sigma(X,hatsigma);
   hatsigma *= -1.;
}

double f_exact(const Vector & X)
{
   MFEM_VERIFY(prob!=prob_type::lshape, 
         "f_exact should not be called for l-shape benchmark problem, i.e., f = 0")
   double u, d2u;
   Vector du;
   solution(X,u,du,d2u);
   return -d2u;
}
