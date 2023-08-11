//                              MFEM Example 35
//
//
// Compile with: make ex35
//
// Sample runs:
//     ex35 -alpha 10
//     ex35 -lambda 0.1 -mu 0.1
//     ex35 -r 5 -o 2 -alpha 5.0 -epsilon 0.01 -mi 50 -mf 0.5 -tol 1e-5
//     ex35 -r 6 -o 1 -alpha 10.0 -epsilon 0.01 -mi 50 -mf 0.5 -tol 1e-5
//
//
// Description: This example code demonstrates the use of MFEM to solve a
//              density-filtered [3] topology optimization problem. The
//              objective is to minimize the compliance
//
//                  minimize ∫_Ω f⋅u dx over u ∈ [H¹(Ω)]² and ρ ∈ L²(Ω)
//
//                  subject to
//
//                    -Div(r(ρ̃)Cε(u)) = f       in Ω + BCs
//                    -ϵ²Δρ̃ + ρ̃ = ρ             in Ω + Neumann BCs
//                    0 ≤ ρ ≤ 1                 in Ω
//                    ∫_Ω ρ dx = θ vol(Ω)
//
//              Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
//              penalization (SIMP) law, C is the elasticity tensor for an
//              isotropic linearly elastic material, ϵ > 0 is the design
//              length scale, and 0 < θ < 1 is the volume fraction.
//
//              The problem is discretized and gradients are computing using
//              finite elements [1]. The design is optimized using an entropic
//              mirror descent algorithm introduced by Keith and Surowiec [2]
//              that is tailored to the bound constraint 0 ≤ ρ ≤ 1.
//
//              This example highlights the ability of MFEM to deliver high-
//              order solutions to inverse design problems and showcases how
//              to set up and solve PDE-constrained optimization problems
//              using the so-called reduced space approach.
//
//
// [1] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O.
//    (2011). Efficient topology optimization in MATLAB using 88 lines of
//    code. Structural and Multidisciplinary Optimization, 43(1), 1-16.
// [2] Keith, B. and Surowiec, T. (2023) The entropic finite element method
//     (in preparation).
// [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
//     based on Helmholtz‐type differential equations. International Journal
//     for Numerical Methods in Engineering, 86(6), 765-781.

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include "ex35.hpp"

using namespace std;
using namespace mfem;

/**
 * @brief Nonlinear projection of ψ onto the subspace
 *        ∫_Ω sigmoid(ψ) dx = θ vol(Ω) as follows.
 *
 *        1. Compute the root of the R → R function
 *            f(c) = ∫_Ω sigmoid(ψ + c) dx - θ vol(Ω)
 *        2. Set ψ ← ψ + c.
 *
 * @param psi a GridFunction to be updated
 * @param target_volume θ vol(Ω)
 * @param tol Newton iteration tolerance
 * @param max_its Newton maximum iteration number
 * @return double Final volume, ∫_Ω sigmoid(ψ)
 */
double projit(GridFunction &psi, double target_volume, double tol=1e-12,
              int max_its=10)
{
   MappedGridFunctionCoefficient sigmoid_psi(&psi, sigmoid);
   MappedGridFunctionCoefficient der_sigmoid_psi(&psi, der_sigmoid);

   LinearForm int_sigmoid_psi(psi.FESpace());
   int_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi));
   LinearForm int_der_sigmoid_psi(psi.FESpace());
   int_der_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(
                                              der_sigmoid_psi));
   bool done = false;
   for (int k=0; k<max_its; k++) // Newton iteration
   {
      int_sigmoid_psi.Assemble(); // Recompute f(c) with updated ψ
      const double f = int_sigmoid_psi.Sum() - target_volume;

      int_der_sigmoid_psi.Assemble(); // Recompute df(c) with updated ψ
      const double df = int_der_sigmoid_psi.Sum();

      const double dc = -f/df;
      psi += dc;
      if (abs(dc) < tol) { done = true; break; }
   }
   if (!done)
   {
      mfem_warning("Projection reached maximum iteration without converging. Result may not be accurate.");
   }
   int_sigmoid_psi.Assemble();
   return int_sigmoid_psi.Sum();
}


/**
 * @brief Enforce boundedness, -max_val ≤ psi ≤ max_val
 *
 * @param psi a GridFunction to be bounded (in place)
 * @param max_val upper and lower bound
 */
inline void clip(GridFunction &psi, const double max_val)
{
   for (auto &val : psi) { val = min(max_val, max(-max_val, val)); }
}
/**
 * ---------------------------------------------------------------
 *                      ALGORITHM PREAMBLE
 * ---------------------------------------------------------------
 *
 *  The Lagrangian for this problem is
 *
 *          L(u,ρ,ρ̃,w,w̃) = (f,u) - (r(ρ̃) C ε(u),ε(w)) + (f,w)
 *                       - (ϵ² ∇ρ̃,∇w̃) - (ρ̃,w̃) + (ρ,w̃)
 *
 *  where
 *
 *    r(ρ̃) = ρ₀ + ρ̃³ (1 - ρ₀)       (SIMP rule)
 *
 *    ε(u) = (∇u + ∇uᵀ)/2           (symmetric gradient)
 *
 *    C e = λtr(e)I + 2μe           (isotropic material)
 *
 *  NOTE: The Lame parameters can be computed from Young's modulus E
 *        and Poisson's ratio ν as follows:
 *
 *             λ = E ν/((1+ν)(1-2ν)),      μ = E/(2(1+ν))
 *
 * ---------------------------------------------------------------
 *
 *  Discretization choices:
 *
 *     u ∈ V ⊂ (H¹)ᵈ (order p)
 *     ψ ∈ L² (order p - 1), ρ = sigmoid(ψ)
 *     ρ̃ ∈ H¹ (order p - 1)
 *     w ∈ V  (order p)
 *     w̃ ∈ H¹ (order p - 1)
 *
 * ---------------------------------------------------------------
 *                          ALGORITHM
 * ---------------------------------------------------------------
 *
 *  Update ρ with projected mirror descent via the following algorithm.
 *
 *  1. Initialize ψ = inv_sigmoid(mass_fraction) so that ∫ sigmoid(ψ) = θ vol(Ω)
 *
 *  While not converged:
 *
 *     2. Solve filter equation ∂_w̃ L = 0; i.e.,
 *
 *           (ϵ² ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)   ∀ v ∈ H¹.
 *
 *     3. Solve primal problem ∂_w L = 0; i.e.,
 *
 *      (λ(ρ̃) ∇⋅u, ∇⋅v) + (2 μ(ρ̃) ε(u), ε(v)) = (f,v)   ∀ v ∈ V,
 *
 *     where λ(ρ̃) := λ r(ρ̃) and  μ(ρ̃) := μ r(ρ̃).
 *
 *     NB. The dual problem ∂_u L = 0 is the same as the primal problem due to symmetry.
 *
 *     4. Solve for filtered gradient ∂_ρ̃ L = 0; i.e.,
 *
 *      (ϵ² ∇ w̃ , ∇ v ) + (w̃ ,v) = (-r'(ρ̃) ( λ(ρ̃) |∇⋅u|² + 2 μ(ρ̃) |ε(u)|²),v)   ∀ v ∈ H¹.
 *
 *     5. Construct gradient G ∈ L²; i.e.,
 *
 *                         (G,v) = (w̃,v)   ∀ v ∈ L².
 *
 *     6. Mirror descent update until convergence; i.e.,
 *
 *                      ψ ← clip(projit(ψ - αG)),
 *
 *     where
 *
 *          α > 0                                    (step size parameter)
 *
 *          clip(y) = min(max_val, max(min_val, y))  (boundedness enforcement)
 *
 *     and projit is a (compatible) projection operator enforcing ∫_Ω ρ(=sigmoid(ψ)) dx = θ vol(Ω).
 *
 *  end
 *
 */

int main(int argc, char *argv[])
{

   // 1. Parse command-line options.
   int ref_levels = 4;
   int order = 2;
   bool visualization = true;
   double alpha = 1.0;
   double epsilon = 0.01;
   double mass_fraction = 0.5;
   int max_it = 1e3;
   double tol = 1e-4;
   double rho_min = 1e-6;
   double lambda = 1.0;
   double mu = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&alpha, "-alpha", "--alpha-step-length",
                  "Step length for gradient descent.");
   args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                  "epsilon phase field thickness");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&tol, "-tol", "--tol",
                  "Exit tolerance for ρ ");
   args.AddOption(&mass_fraction, "-mf", "--mass-fraction",
                  "Mass fraction for diffusion coefficient.");
   args.AddOption(&lambda, "-lambda", "--lambda",
                  "Lame constant λ");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lame constant μ");
   args.AddOption(&rho_min, "-rmin", "--psi-min",
                  "Minimum of density coefficient.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   Mesh mesh = Mesh::MakeCartesian2D(3,1,mfem::Element::Type::QUADRILATERAL,true,
                                     3.0,1.0);

   int dim = mesh.Dimension();

   // 2. Set BCs.
   for (int i = 0; i<mesh.GetNBE(); i++)
   {
      Element * be = mesh.GetBdrElement(i);
      Array<int> vertices;
      be->GetVertices(vertices);

      double * coords1 = mesh.GetVertex(vertices[0]);
      double * coords2 = mesh.GetVertex(vertices[1]);

      Vector center(2);
      center(0) = 0.5*(coords1[0] + coords2[0]);
      center(1) = 0.5*(coords1[1] + coords2[1]);

      if (abs(center(0) - 0.0) < 1e-10)
      {
         // the left edge
         be->SetAttribute(1);
      }
      else
      {
         // all other boundaries
         be->SetAttribute(2);
      }
   }
   mesh.SetAttributes();

   // 3. Refine the mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for ψ
   FiniteElementSpace state_fes(&mesh, &state_fec,dim);
   FiniteElementSpace filter_fes(&mesh, &filter_fec);
   FiniteElementSpace control_fes(&mesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   int filter_size = filter_fes.GetTrueVSize();
   mfem::out << "Number of state unknowns: " << state_size << std::endl;
   mfem::out << "Number of filter unknowns: " << filter_size << std::endl;
   mfem::out << "Number of control unknowns: " << control_size << std::endl;

   // 5. Set the initial guess for ρ.
   GridFunction u(&state_fes);
   GridFunction psi(&control_fes);
   GridFunction psi_old(&control_fes);
   GridFunction rho_filter(&filter_fes);
   u = 0.0;
   rho_filter = mass_fraction;
   psi = inv_sigmoid(mass_fraction);
   psi_old = inv_sigmoid(mass_fraction);

   const double sigmoid_bound = -inv_sigmoid(rho_min);

   // ρ = sigmoid(ψ)
   MappedGridFunctionCoefficient rho(&psi, sigmoid);
   // Interpolation of ρ = sigmoid(ψ) in control fes
   GridFunction rho_gf(&control_fes);
   // ρ - ρ_old = sigmoid(ψ) - sigmoid(ψ_old)
   DiffMappedGridFunctionCoefficient succ_diff_rho(&psi, &psi_old, sigmoid);

   // 6. Set-up the physics solver.
   int maxat = mesh.bdr_attributes.Max();
   Array<int> ess_bdr(maxat);
   ess_bdr = 0;
   ess_bdr[0] = 1;
   ConstantCoefficient one(1.0);
   ConstantCoefficient lambda_cf(lambda);
   ConstantCoefficient mu_cf(mu);
   LinearElasticitySolver * ElasticitySolver = new LinearElasticitySolver();
   ElasticitySolver->SetMesh(&mesh);
   ElasticitySolver->SetOrder(state_fec.GetOrder());
   ElasticitySolver->SetupFEM();
   Vector center(2); center(0) = 2.9; center(1) = 0.5;
   Vector force(2); force(0) = 0.0; force(1) = -1.0;
   double r = 0.05;
   VolumeForceCoefficient vforce_cf(r,center,force);
   ElasticitySolver->SetRHSCoefficient(&vforce_cf);
   ElasticitySolver->SetEssentialBoundary(ess_bdr);

   // 7. Set-up the filter solver.
   ConstantCoefficient eps2_cf(epsilon*epsilon);
   DiffusionSolver * FilterSolver = new DiffusionSolver();
   FilterSolver->SetMesh(&mesh);
   FilterSolver->SetOrder(filter_fec.GetOrder());
   FilterSolver->SetDiffusionCoefficient(&eps2_cf);
   FilterSolver->SetMassCoefficient(&one);
   Array<int> ess_bdr_filter;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr_filter.SetSize(mesh.bdr_attributes.Max());
      ess_bdr_filter = 0;
   }
   FilterSolver->SetEssentialBoundary(ess_bdr_filter);
   FilterSolver->SetupFEM();

   BilinearForm mass(&control_fes);
   mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(one)));
   mass.Assemble();
   SparseMatrix M;
   Array<int> empty;
   mass.FormSystemMatrix(empty,M);

   // 8. Define the Lagrange multiplier and gradient functions
   GridFunction grad(&control_fes);
   GridFunction w_filter(&filter_fes);

   // 9. Define some tools for later
   ConstantCoefficient zero(0.0);
   GridFunction onegf(&control_fes);
   onegf = 1.0;
   GridFunction zerogf(&control_fes);
   zerogf = 0.0;
   LinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   double domain_volume = vol_form(onegf);
   const double target_volume = domain_volume * mass_fraction;

   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_u,sout_r,sout_rho;
   if (visualization)
   {
      sout_u.open(vishost, visport);
      sout_rho.open(vishost, visport);
      sout_r.open(vishost, visport);
      sout_u.precision(8);
      sout_rho.precision(8);
      sout_r.precision(8);
   }

   rho_gf.ProjectCoefficient(rho);
   mfem::ParaViewDataCollection paraview_dc("Elastic_compliance", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("displacement",&u);
   paraview_dc.RegisterField("density",&rho_gf);
   paraview_dc.RegisterField("filtered_density",&rho_filter);

   // 11. Iterate
   for (int k = 1; k < max_it; k++)
   {
      if (k > 1) { alpha *= ((double) k) / ((double) k-1); }

      mfem::out << "\nStep = " << k << std::endl;

      // Step 1 - Filter solve
      // Solve (ϵ^2 ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)
      FilterSolver->SetRHSCoefficient(&rho);
      FilterSolver->Solve();
      rho_filter = *FilterSolver->GetFEMSolution();

      // Step 2 - State solve
      // Solve (λ(ρ̃) ∇⋅u, ∇⋅v) + (2 μ(ρ̃) ε(u), ε(v)) = (f,v)
      SIMPInterpolationCoefficient SIMP_cf(&rho_filter,rho_min, 1.0);
      ProductCoefficient lambda_SIMP_cf(lambda_cf,SIMP_cf);
      ProductCoefficient mu_SIMP_cf(mu_cf,SIMP_cf);
      ElasticitySolver->SetLameCoefficients(&lambda_SIMP_cf,&mu_SIMP_cf);
      ElasticitySolver->Solve();
      u = *ElasticitySolver->GetFEMSolution();

      // Step 3 - Adjoint filter solve
      // Solve (ϵ² ∇ w̃, ∇ v) + (w̃ ,v) = (-r'(ρ̃) ( λ(ρ̃) |∇⋅u|² + 2 μ(ρ̃) |ε(u)|²),v)
      StrainEnergyDensityCoefficient rhs_cf(&lambda_cf,&mu_cf,&u, &rho_filter,
                                            rho_min);
      FilterSolver->SetRHSCoefficient(&rhs_cf);
      FilterSolver->Solve();
      w_filter = *FilterSolver->GetFEMSolution();

      // Step 4 - Compute gradient
      // Solve G = M⁻¹w̃
      GridFunctionCoefficient w_cf(&w_filter);
      LinearForm w_rhs(&control_fes);
      w_rhs.AddDomainIntegrator(new DomainLFIntegrator(w_cf));
      w_rhs.Assemble();
      M.Mult(w_rhs,grad);

      // Step 5 - Update design variable ψ ← clip(projit(ψ - αG))
      psi.Add(-alpha, grad);
      const double material_volume = projit(psi, target_volume);
      clip(psi, sigmoid_bound);

      // Compute ||ρ - ρ_old|| in control fes.
      double norm_reduced_gradient = zerogf.ComputeL2Error(succ_diff_rho)/alpha;
      psi_old = psi;

      double compliance = (*(ElasticitySolver->GetLinearForm()))(u);
      mfem::out << "norm of reduced gradient = " << norm_reduced_gradient <<
                std::endl;
      mfem::out << "compliance = " << compliance << std::endl;
      mfem::out << "mass_fraction = " << material_volume / domain_volume << std::endl;

      if (visualization)
      {
         sout_u << "solution\n" << mesh << u
                << "window_title 'Displacement u'" << flush;

         rho_gf.ProjectCoefficient(rho);
         sout_rho << "solution\n" << mesh << rho_gf
                  << "window_title 'Control variable ρ'" << flush;

         GridFunction r_gf(&filter_fes);
         r_gf.ProjectCoefficient(SIMP_cf);
         sout_r << "solution\n" << mesh << r_gf
                << "window_title 'Design density r(ρ̃)'" << flush;

         paraview_dc.SetCycle(k);
         paraview_dc.SetTime((double)k);
         paraview_dc.Save();
      }

      if (norm_reduced_gradient < tol)
      {
         break;
      }
   }

   delete ElasticitySolver;
   delete FilterSolver;

   return 0;
}
