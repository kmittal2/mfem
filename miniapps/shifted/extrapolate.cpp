// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//            ------------------------------------------------
//            Extrapolation Miniapp: PDE-based extrapolation
//            ------------------------------------------------
//
// Compile with: make extrapolate
//
// Sample runs:
//     mpirun -np 4 extrapolate -o 3 -et 0 -eo 2
//     mpirun -np 4 extrapolate -rs 3 -o 2 -p 1 -et 0 -eo 1

#include <fstream>
#include <iostream>
#include "../common/mfem-common.hpp"
#include "marking.hpp"

using namespace std;
using namespace mfem;

int problem = 0;

const char vishost[] = "localhost";
const int  visport   = 19916;
const int wsize = 450; // glvis window size.

double domainLS(const Vector &coord)
{
   // Map from [0,1] to [-1,1].
   const int dim = coord.Size();
   const double x = coord(0)*2.0 - 1.0,
                y = coord(1)*2.0 - 1.0,
                z = (dim > 2) ? coord(2)*2.0 - 1.0 : 0.0;
   switch(problem)
   {
      case 0:
      {
         // 2d circle.
         return 0.75 - sqrt(x*x + y*y + 1e-12);
      }
      case 1:
      {
         // 2d star.
         return 0.60 - sqrt(x*x + y*y + 1e-12) +
                0.25 * (y*y*y*y*y + 5.0*x*x*x*x*y - 10.0*x*x*y*y*y) /
                       pow(x*x + y*y + 1e-12, 2.5);
      }
      default: MFEM_ABORT("Bad option for --problem!"); return 0.0;
   }
}

double solution0(const Vector &coord)
{
   // Map from [0,1] to [-1,1].
   const int dim = coord.Size();
   const double x = coord(0)*2.0 - 1.0,
                y = coord(1)*2.0 - 1.0,
                z = (dim > 2) ? coord(2)*2.0 - 1.0 : 0.0;

   return std::cos(M_PI * x) * std::sin(M_PI * y);
}

class LevelSetNormalGradCoeff : public VectorCoefficient
{
private:
   const ParGridFunction &ls_gf;

public:
   LevelSetNormalGradCoeff(const ParGridFunction &ls) :
      VectorCoefficient(ls.ParFESpace()->GetMesh()->Dimension()), ls_gf(ls) { }

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector grad_ls(vdim), n(vdim);
      ls_gf.GetGradient(T, grad_ls);
      const double norm_grad = grad_ls.Norml2();
      V = grad_ls;
      if (norm_grad > 0.0) { V /= norm_grad; }

      // Since positive level set values correspond to the known region, we
      // transport into the opposite direction of the gradient.
      V *= -1;
   }
};

void PrintNorm(int myid, Vector &v, std::string text)
{
   double norm = v.Norml1();
   MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   if (myid == 0)
   {
      std::cout << std::setprecision(12) << std::fixed
                << text << norm << std::endl;
   }
}

void PrintIntegral(int myid, ParGridFunction &g, std::string text)
{
   ConstantCoefficient zero(0.0);
   double norm = g.ComputeL1Error(zero);
   if (myid == 0)
   {
      std::cout << std::setprecision(12) << std::fixed
                << text << norm << std::endl;
   }
}

class GradComponentCoeff : public Coefficient
{
private:
   const ParGridFunction &u_gf;
   int comp;

public:
   GradComponentCoeff(const ParGridFunction &u, int c) : u_gf(u), comp(c) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector grad_u(T.GetDimension());
      u_gf.GetGradient(T, grad_u);
      return grad_u(comp);
   }
};

class NormalGradCoeff : public Coefficient
{
private:
   const ParGridFunction &u_gf;
   VectorCoefficient &n_coeff;

public:
   NormalGradCoeff(const ParGridFunction &u, VectorCoefficient &n)
      : u_gf(u), n_coeff(n) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      const int dim = T.GetDimension();
      Vector n(dim), grad_u(dim);
      n_coeff.Eval(n, T, ip);
      u_gf.GetGradient(T, grad_u);
      return n * grad_u;
   }
};

class NormalGradComponentCoeff : public Coefficient
{
private:
   const ParGridFunction &du_dx, &du_dy;
   VectorCoefficient &n_coeff;

public:
   NormalGradComponentCoeff(const ParGridFunction &dx,
                            const ParGridFunction &dy, VectorCoefficient &n)
      : du_dx(dx), du_dy(dy), n_coeff(n) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      const int dim = T.GetDimension();
      Vector n(dim), grad_u(dim);
      n_coeff.Eval(n, T, ip);
      grad_u(0) = du_dx.GetValue(T, ip);
      grad_u(1) = du_dy.GetValue(T, ip);
      return n * grad_u;
   }
};

class DiscreteUpwindLOSolver
{
protected:
   ParFiniteElementSpace &pfes;
   const SparseMatrix &K;
   mutable SparseMatrix D;

   Array<int> K_smap;
   const Vector &M_lumped;

   void ComputeDiscreteUpwindMatrix() const;
   void ApplyDiscreteUpwindMatrix(ParGridFunction &u, Vector &du) const;

public:
   DiscreteUpwindLOSolver(ParFiniteElementSpace &space, const SparseMatrix &adv,
                          const Vector &Mlump);

   virtual void CalcLOSolution(const Vector &u, const Vector &rhs,
                               Vector &du) const;

   Array<int> &GetKmap() { return K_smap; }
};

class FluxBasedFCT
{
protected:
   ParFiniteElementSpace &pfes;
   double dt;

   const SparseMatrix &K, &M;
   const Array<int> &K_smap;

   // Temporary computation objects.
   mutable SparseMatrix flux_ij;
   mutable ParGridFunction gp, gm;

   void ComputeFluxMatrix(const ParGridFunction &u, const Vector &du_ho,
                          SparseMatrix &flux_mat) const;
   void AddFluxesAtDofs(const SparseMatrix &flux_mat,
                        Vector &flux_pos, Vector &flux_neg) const;
   void ComputeFluxCoefficients(const Vector &u, const Vector &du_lo,
      const Vector &m, const Vector &u_min, const Vector &u_max,
      Vector &coeff_pos, Vector &coeff_neg) const;
   void UpdateSolutionAndFlux(const Vector &du_lo, const Vector &m,
      ParGridFunction &coeff_pos, ParGridFunction &coeff_neg,
      SparseMatrix &flux_mat, Vector &du) const;

public:
   FluxBasedFCT(ParFiniteElementSpace &space, double delta_t,
                const SparseMatrix &adv_mat, const Array<int> &adv_smap,
                const SparseMatrix &mass_mat)
      : pfes(space), dt(delta_t),
        K(adv_mat), M(mass_mat), K_smap(adv_smap), flux_ij(adv_mat),
        gp(&pfes), gm(&pfes) { }

   void CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                        const Vector &du_ho, const Vector &du_lo,
                        const Vector &u_min, const Vector &u_max,
                        Vector &du) const;
};

class AdvectionOper : public TimeDependentOperator
{
private:
   Array<bool> &active_zones;
   ParBilinearForm &M, &K;
   const Vector &b;
   double dt = 0.0;

   mutable ParBilinearForm M_Lump;

   void ComputeElementsMinMax(const ParGridFunction &gf,
                              Vector &el_min, Vector &el_max) const;
   void ComputeBounds(const ParFiniteElementSpace &pfes,
                      const Vector &el_min, const Vector &el_max,
                      Vector &dof_min, Vector &dof_max) const;

public:
   // 0 is stanadard HO; 1 is upwind diffusion; 2 is FCT.
   enum AdvectionMode {HO, LO, FCT} mode = HO;

   AdvectionOper(Array<bool> &zones,
                 ParBilinearForm &Mbf, ParBilinearForm &Kbf, const Vector &rhs)
      : TimeDependentOperator(Mbf.Size()),
        active_zones(zones),
        M(Mbf), K(Kbf), b(rhs),
        M_Lump(M.ParFESpace())
   {
      M_Lump.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
      M_Lump.Assemble();
      M_Lump.Finalize();
   }

   void SetDt(double delta_t) { dt = delta_t; }

   virtual void Mult(const Vector &x, Vector &dx) const
   {
      ParFiniteElementSpace &pfes = *M.ParFESpace();
      const int NE = pfes.GetNE();
      const int nd = pfes.GetFE(0)->GetDof(), size = NE * nd;
      Array<int> dofs(nd);

      if (mode == LO)
      {
         Vector lumpedM; M_Lump.SpMat().GetDiag(lumpedM);
         DiscreteUpwindLOSolver lo_solver(pfes, K.SpMat(), lumpedM);
         lo_solver.CalcLOSolution(x, b, dx);
         for (int k = 0; k < NE; k++)
         {
            pfes.GetElementDofs(k, dofs);

            if (active_zones[k] == false)
            {
               dx.SetSubVector(dofs, 0.0);
               continue;
            }
         }
         return;
      }

      Vector rhs(x.Size());
      HypreParMatrix *K_mat = K.ParallelAssemble(&K.SpMat());
      K_mat->Mult(x, rhs);
      rhs += b;

      DenseMatrix M_loc(nd);
      DenseMatrixInverse M_loc_inv(&M_loc);
      Vector rhs_loc(nd), dx_loc(nd);
      for (int k = 0; k < NE; k++)
      {
         pfes.GetElementDofs(k, dofs);

         if (active_zones[k] == false)
         {
            dx.SetSubVector(dofs, 0.0);
            continue;
         }

         rhs.GetSubVector(dofs, rhs_loc);
         M.SpMat().GetSubMatrix(dofs, dofs, M_loc);
         M_loc_inv.Factor();
         M_loc_inv.Mult(rhs_loc, dx_loc);
         dx.SetSubVector(dofs, dx_loc);
      }

      if (mode == FCT)
      {
         Vector dx_LO(size);
         Vector lumpedM; M_Lump.SpMat().GetDiag(lumpedM);
         DiscreteUpwindLOSolver lo_solver(pfes, K.SpMat(), lumpedM);
         lo_solver.CalcLOSolution(x, b, dx_LO);

         Vector dx_HO(dx);

         Vector el_min(NE), el_max(NE);
         Vector x_min(size), x_max(size);
         ParGridFunction x_gf(&pfes);
         x_gf = x;
         x_gf.ExchangeFaceNbrData();
         ComputeElementsMinMax(x_gf, el_min, el_max);
         ComputeBounds(pfes, el_min, el_max, x_min, x_max);
         FluxBasedFCT fct_solver(pfes, dt, K.SpMat(),
                                 lo_solver.GetKmap(), M.SpMat());
         fct_solver.CalcFCTSolution(x_gf, lumpedM, dx_HO, dx_LO,
                                    x_min, x_max, dx);

         for (int k = 0; k < NE; k++)
         {
            pfes.GetElementDofs(k, dofs);

            if (active_zones[k] == false)
            {
               dx.SetSubVector(dofs, 0.0);
               continue;
            }
         }
      }
   }
};

class Extrapolator
{
private:
   void TimeLoop(ParGridFunction &sltn, ODESolver &ode_solver,
                 double dt, int vis_x_pos, std::string vis_name)
   {
      socketstream sock;

      const int myid  = sltn.ParFESpace()->GetMyRank();
      const double t_final = 0.4;
      bool done = false;
      double t = 0.0;
      for (int ti = 0; !done;)
      {
         double dt_real = min(dt, t_final - t);
         ode_solver.Step(sltn, t, dt_real);
         ti++;

         done = (t >= t_final - 1e-8*dt);
         if (done || ti % vis_steps == 0)
         {
            if (myid == 0)
            {
               cout << "time step: " << ti << ", time: " << t << endl;
            }
            if (visualization)
            {
               common::VisualizeField(sock, vishost, visport, sltn,
                                      vis_name.c_str(), vis_x_pos, wsize+60,
                                      wsize, wsize, "rRjmm********A");
               MPI_Barrier(sltn.ParFESpace()->GetComm());
            }
         }
      }
   }

public:
   enum XtrapType {ASLAM = 0, BOCHKOV = 1} xtrap_type = ASLAM;
   int xtrap_order    = 1;
   bool visualization = false;
   int vis_steps      = 5;

   Extrapolator() { }

   // The known values taken from elements where level_set > 0, and extrapolated
   // to all other elements. The known values are not changed.
   void Extrapolate(Coefficient &level_set, const ParGridFunction &input,
                    ParGridFunction &xtrap)
   {
      ParMesh &pmesh = *input.ParFESpace()->GetParMesh();
      const int order = input.ParFESpace()->GetOrder(0),
                dim   = pmesh.Dimension(), NE = pmesh.GetNE();

      // Get a ParGridFunction and mark elements.
      H1_FECollection fec(order, dim);
      ParFiniteElementSpace pfes_H1(&pmesh, &fec);
      ParGridFunction ls_gf(&pfes_H1);
      ls_gf.ProjectCoefficient(level_set);
      if (visualization)
      {
         socketstream sock1, sock2;
         common::VisualizeField(sock1, vishost, visport, ls_gf,
                                "Domain level set", 0, 0, wsize, wsize,
                                "rRjmm********A");
         common::VisualizeField(sock2, vishost, visport, input,
                                "Input u", 0, wsize+60, wsize, wsize,
                                "rRjmm********A");
         MPI_Barrier(pmesh.GetComm());
      }
      // Mark elements.
      Array<int> elem_marker;
      ShiftedFaceMarker marker(pmesh, pfes_H1, false);
      ls_gf.ExchangeFaceNbrData();
      marker.MarkElements(ls_gf, elem_marker);

      // The active zones are where we extrapolate (where the PDE is solved).
      Array<bool> active_zones(NE);
      for (int k = 0; k < NE; k++)
      {
         // Extrapolation is done in zones that are CUT or OUTSIDE.
         active_zones[k] =
            (elem_marker[k] == ShiftedFaceMarker::INSIDE) ? false : true;
      }

      // Setup a VectorCoefficient for n = - grad_ls / |grad_ls|.
      // The sign makes it point out of the known region.
      // The coefficient must be continuous to have well-defined transport.
      LevelSetNormalGradCoeff ls_n_coeff_L2(ls_gf);
      ParFiniteElementSpace pfes_H1_vec(&pmesh, &fec, dim);
      ParGridFunction lsn_gf(&pfes_H1_vec);
      ls_gf.ExchangeFaceNbrData();
      lsn_gf.ProjectDiscCoefficient(ls_n_coeff_L2, GridFunction::ARITHMETIC);
      VectorGridFunctionCoefficient ls_n_coeff(&lsn_gf);

      // Initial solution.
      // Trim to the known values (only elements inside the known region).
      Array<int> dofs;
      L2_FECollection fec_L2(order, dim);
      ParFiniteElementSpace pfes_L2(&pmesh, &fec_L2);
      ParGridFunction u(&pfes_L2);
      u.ProjectGridFunction(input);
      for (int k = 0; k < NE; k++)
      {
         pfes_L2.GetElementDofs(k, dofs);
         if (elem_marker[k] != ShiftedFaceMarker::INSIDE)
         { u.SetSubVector(dofs, 0.0); }
      }
      if (visualization)
      {
         socketstream sock;
         common::VisualizeField(sock, vishost, visport, u,
                                "Fixed (known) u values", wsize, 0,
                                wsize, wsize, "rRjmm********A");
      }

      // Normal derivative function.
      ParGridFunction n_grad_u(&pfes_L2);
      NormalGradCoeff n_grad_u_coeff(u, ls_n_coeff);
      n_grad_u.ProjectCoefficient(n_grad_u_coeff);
      if (visualization && xtrap_order >= 1)
      {
         socketstream sock;
         common::VisualizeField(sock, vishost, visport, n_grad_u,
                                "n.grad(u)", 2*wsize, 0, wsize, wsize,
                                "rRjmm********A");
      }

      // 2nd normal derivative function.
      ParGridFunction n_grad_n_grad_u(&pfes_L2);
      NormalGradCoeff n_grad_n_grad_u_coeff(n_grad_u, ls_n_coeff);
      n_grad_n_grad_u.ProjectCoefficient(n_grad_n_grad_u_coeff);
      if (visualization && xtrap_order == 2)
      {
         socketstream sock;
         common::VisualizeField(sock, vishost, visport, n_grad_n_grad_u,
                                "n.grad(n.grad(u))", 3*wsize, 0, wsize, wsize,
                                "rRjmm********A");
      }

      ParBilinearForm lhs_bf(&pfes_L2), rhs_bf(&pfes_L2);
      lhs_bf.AddDomainIntegrator(new MassIntegrator);
      const double alpha = -1.0;
      rhs_bf.AddDomainIntegrator(new ConvectionIntegrator(ls_n_coeff, alpha));
      auto trace_i = new NonconservativeDGTraceIntegrator(ls_n_coeff, alpha);
      rhs_bf.AddInteriorFaceIntegrator(trace_i);
      rhs_bf.KeepNbrBlock(true);

      ls_gf.ExchangeFaceNbrData();
      lhs_bf.Assemble();
      lhs_bf.Finalize();
      rhs_bf.Assemble(0);
      rhs_bf.Finalize(0);

      // Compute a CFL time step.
      double h_min = std::numeric_limits<double>::infinity();
      for (int k = 0; k < NE; k++)
      {
         h_min = std::min(h_min, pmesh.GetElementSize(k));
      }
      MPI_Allreduce(MPI_IN_PLACE, &h_min, 1, MPI_DOUBLE, MPI_MIN,
                    pfes_L2.GetComm());
      h_min /= order;
      // The propagation speed is 1.
      double dt = 0.05 * h_min / 1.0;

      // Time loops.
      Vector rhs(pfes_L2.GetVSize());
      AdvectionOper adv_oper(active_zones, lhs_bf, rhs_bf, rhs);
      RK2Solver ode_solver(1.0);
      ode_solver.Init(adv_oper);
      adv_oper.SetDt(dt);

      if (xtrap_type == ASLAM)
      {
         switch (xtrap_order)
         {
            case 0:
            {
               // Constant extrapolation of u.
               rhs = 0.0;
               adv_oper.mode = AdvectionOper::FCT;
               TimeLoop(u, ode_solver, dt, wsize, "u - constant extrap");
               break;
            }
            case 1:
            {
               // Constant extrapolation of [n.grad_u].
               rhs = 0.0;
               adv_oper.mode = AdvectionOper::LO;
               TimeLoop(n_grad_u, ode_solver, dt, 2*wsize, "n.grad(u)");

               // Linear sextrapolation of u.
               lhs_bf.Mult(n_grad_u, rhs);
               adv_oper.mode = AdvectionOper::LO;
               TimeLoop(u, ode_solver, dt, wsize, "u - linear Aslam extrap");
               break;
            }
            case 2:
            {
               // Constant extrapolation of [n.grad(n.grad(u))].
               rhs = 0.0;
               adv_oper.mode = AdvectionOper::LO;
               TimeLoop(n_grad_n_grad_u, ode_solver, dt, 3*wsize, "n.grad(n.grad(u))");

               // Linear extrapolation of [n.grad_u].
               lhs_bf.Mult(n_grad_n_grad_u, rhs);
               adv_oper.mode = AdvectionOper::LO;
               TimeLoop(n_grad_u, ode_solver, dt, 2*wsize, "n.grad(u)");

               // Quadratic extrapolation of u.
               lhs_bf.Mult(n_grad_u, rhs);
               adv_oper.mode = AdvectionOper::LO;
               TimeLoop(u, ode_solver, dt, wsize, "u - quadratic Aslam extrap");
               break;
            }
            default: MFEM_ABORT("Wrong extrapolation order.");
         }
      }
      else if (xtrap_type == BOCHKOV)
      {
         switch (xtrap_order)
         {
            case 0:
            {
               // Constant extrapolation of u.
               rhs = 0.0;
               TimeLoop(u, ode_solver, dt, wsize, "u - constant extrap");
               break;
            }
            case 1:
            {
               // Constant extrapolation of all grad(u) components.
               rhs = 0.0;
               ParGridFunction grad_u_0(&pfes_L2), grad_u_1(&pfes_L2);
               GradComponentCoeff grad_u_0_coeff(u, 0), grad_u_1_coeff(u, 1);
               grad_u_0.ProjectCoefficient(grad_u_0_coeff);
               grad_u_1.ProjectCoefficient(grad_u_1_coeff);
               adv_oper.mode = AdvectionOper::HO;
               TimeLoop(grad_u_0, ode_solver, dt, 2*wsize, "grad_u_0");
               TimeLoop(grad_u_1, ode_solver, dt, 3*wsize, "grad_u_1");

               // Linear extrapolation of u.
               ParLinearForm rhs_lf(&pfes_L2);
               NormalGradComponentCoeff grad_u_n(grad_u_0, grad_u_1, ls_n_coeff);
               rhs_lf.AddDomainIntegrator(new DomainLFIntegrator(grad_u_n));
               rhs_lf.Assemble();
               rhs = rhs_lf;
               adv_oper.mode = AdvectionOper::LO;
               TimeLoop(u, ode_solver, dt, wsize, "u - linear Bochkov extrap");
               break;
            }
            case 2:  MFEM_ABORT("Quadratic Bochkov method is not implemented.");
            default: MFEM_ABORT("Wrong extrapolation order.");
         }
      }
      else { MFEM_ABORT("Wrong extrapolation type option"); }

      xtrap.ProjectGridFunction(u);
   }
};

int main(int argc, char *argv[])
{
   // Initialize MPI.
   MPI_Session mpi;
   int myid = mpi.WorldRank();

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int rs_levels = 2;
   Extrapolator::XtrapType xtrap_type = Extrapolator::ASLAM;
   int xtrap_order = 1;
   int order = 2;
   bool visualization = true;
   int vis_steps = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption((int*)&xtrap_type, "-et", "--extrap-type",
                  "Extrapolation type: Aslam (0) or Bochkov (1).");
   args.AddOption(&xtrap_order, "-eo", "--extrap-order",
                  "Extrapolation order: 0/1/2 for constant/linear/quadratic.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&problem, "-p", "--problem",
                  "0 - 2D circle,\n\t"
                  "1 - 2D star");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }


   // Refine the mesh and distribute.
   Mesh mesh(mesh_file, 1, 1);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   const int dim = pmesh.Dimension();

   // Input function.
   L2_FECollection fec_L2(order, dim);
   ParFiniteElementSpace pfes_L2(&pmesh, &fec_L2);
   ParGridFunction u(&pfes_L2);
   FunctionCoefficient u0_coeff(solution0);
   u.ProjectCoefficient(u0_coeff);

   // Extrapolate.
   Extrapolator xtrap;
   xtrap.xtrap_type  = xtrap_type;
   xtrap.xtrap_order = xtrap_order;
   xtrap.visualization = true;
   xtrap.vis_steps = 5;
   FunctionCoefficient ls_coeff(domainLS);
   ParGridFunction ux(&pfes_L2);
   xtrap.Extrapolate(ls_coeff, u, ux);

   GridFunctionCoefficient u_exact_coeff(&u);
   double err_L1 = ux.ComputeL1Error(u_exact_coeff),
          err_L2 = ux.ComputeL2Error(u_exact_coeff);
   if (myid == 0)
   {
      std::cout << "L1 error: " << err_L1 << std::endl
                << "L2 error: " << err_L2 << std::endl;
   }

   // ParaView output.
   ParaViewDataCollection dacol("ParaViewExtrapolate", &pmesh);
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("extrapolated sltn", &ux);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();

   return 0;
}

void AdvectionOper::ComputeElementsMinMax(const ParGridFunction &gf,
                                          Vector &el_min, Vector &el_max) const
{
   ParFiniteElementSpace &pfes = *gf.ParFESpace();
   const int NE = pfes.GetNE(), ndof = pfes.GetFE(0)->GetDof();
   for (int k = 0; k < NE; k++)
   {
      el_min(k) = numeric_limits<double>::infinity();
      el_max(k) = -numeric_limits<double>::infinity();

      for (int i = 0; i < ndof; i++)
      {
         el_min(k) = min(el_min(k), gf(k*ndof + i));
         el_max(k) = max(el_max(k), gf(k*ndof + i));
      }
   }
}

void AdvectionOper::ComputeBounds(const ParFiniteElementSpace &pfes,
                                  const Vector &el_min, const Vector &el_max,
                                  Vector &dof_min, Vector &dof_max) const
{
   ParMesh *pmesh = pfes.GetParMesh();
   L2_FECollection fec_bounds(0, pmesh->Dimension());
   ParFiniteElementSpace pfes_bounds(pmesh, &fec_bounds);
   ParGridFunction el_min_gf(&pfes_bounds), el_max_gf(&pfes_bounds);
   const int NE = pmesh->GetNE(), ndofs = dof_min.Size() / NE;

   el_min_gf = el_min;
   el_max_gf = el_max;

   el_min_gf.ExchangeFaceNbrData(); el_max_gf.ExchangeFaceNbrData();
   const Vector &min_nbr = el_min_gf.FaceNbrData(),
                &max_nbr = el_max_gf.FaceNbrData();
   const Table &el_to_el = pmesh->ElementToElementTable();
   Array<int> face_nbr_el;
   for (int k = 0; k < NE; k++)
   {
      double k_min = el_min_gf(k), k_max = el_max_gf(k);

      el_to_el.GetRow(k, face_nbr_el);
      for (int n = 0; n < face_nbr_el.Size(); n++)
      {
         if (face_nbr_el[n] < NE)
         {
            // Local neighbor.
            k_min = std::min(k_min, el_min_gf(face_nbr_el[n]));
            k_max = std::max(k_max, el_max_gf(face_nbr_el[n]));
         }
         else
         {
            // MPI face neighbor.
            k_min = std::min(k_min, min_nbr(face_nbr_el[n] - NE));
            k_max = std::max(k_max, max_nbr(face_nbr_el[n] - NE));
         }
      }

      for (int j = 0; j < ndofs; j++)
      {
         dof_min(k*ndofs + j) = k_min;
         dof_max(k*ndofs + j) = k_max;
      }
   }
}

DiscreteUpwindLOSolver::DiscreteUpwindLOSolver(ParFiniteElementSpace &space,
                                               const SparseMatrix &adv,
                                               const Vector &Mlump)
   : pfes(space), K(adv), D(adv), K_smap(), M_lumped(Mlump)
{
   // Assuming it is finalized.
   const int *I = K.GetI(), *J = K.GetJ(), n = K.Size();
   K_smap.SetSize(I[n]);
   for (int row = 0, j = 0; row < n; row++)
   {
      for (int end = I[row+1]; j < end; j++)
      {
         int col = J[j];
         // Find the offset, _j, of the (col,row) entry and store it in smap[j].
         for (int _j = I[col], _end = I[col+1]; true; _j++)
         {
            MFEM_VERIFY(_j != _end, "Can't find the symmetric entry!");

            if (J[_j] == row) { K_smap[j] = _j; break; }
         }
      }
   }
}

void DiscreteUpwindLOSolver::CalcLOSolution(const Vector &u, const Vector &rhs,
                                            Vector &du) const
{
   ComputeDiscreteUpwindMatrix();
   ParGridFunction u_gf(&pfes);
   u_gf = u;

   ApplyDiscreteUpwindMatrix(u_gf, du);

   const int s = du.Size();
   for (int i = 0; i < s; i++)
   {
      du(i) = (du(i) + rhs(i)) / M_lumped(i);
   }
}

void DiscreteUpwindLOSolver::ComputeDiscreteUpwindMatrix() const
{
   const int *I = K.HostReadI(), *J = K.HostReadJ(), n = K.Size();

   const double *K_data = K.HostReadData();

   double *D_data = D.HostReadWriteData();
   D.HostReadWriteI(); D.HostReadWriteJ();

   for (int i = 0, k = 0; i < n; i++)
   {
      double rowsum = 0.;
      for (int end = I[i+1]; k < end; k++)
      {
         int j = J[k];
         double kij = K_data[k];
         double kji = K_data[K_smap[k]];
         double dij = fmax(fmax(0.0,-kij),-kji);
         D_data[k] = kij + dij;
         D_data[K_smap[k]] = kji + dij;
         if (i != j) { rowsum += dij; }
      }
      D(i,i) = K(i,i) - rowsum;
   }
}

void DiscreteUpwindLOSolver::ApplyDiscreteUpwindMatrix(ParGridFunction &u,
                                                       Vector &du) const
{
   const int s = u.Size();
   const int *I = D.HostReadI(), *J = D.HostReadJ();
   const double *D_data = D.HostReadData();

   u.ExchangeFaceNbrData();
   const Vector &u_np = u.FaceNbrData();

   for (int i = 0; i < s; i++)
   {
      du(i) = 0.0;
      for (int k = I[i]; k < I[i + 1]; k++)
      {
         int j = J[k];
         double u_j  = (j < s) ? u(j) : u_np[j - s];
         double d_ij = D_data[k];
         du(i) += d_ij * u_j;
      }
   }
}

void FluxBasedFCT::CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                   const Vector &du_ho, const Vector &du_lo,
                                   const Vector &u_min, const Vector &u_max,
                                   Vector &du) const
{
   // Construct the flux matrix (it gets recomputed every time).
   ComputeFluxMatrix(u, du_ho, flux_ij);

   // Iterated FCT correction.
   Vector du_lo_fct(du_lo);
   for (int fct_iter = 0; fct_iter < 1; fct_iter++)
   {
      // Compute sums of incoming/outgoing fluxes at each DOF.
      AddFluxesAtDofs(flux_ij, gp, gm);

      // Compute the flux coefficients (aka alphas) into gp and gm.
      ComputeFluxCoefficients(u, du_lo_fct, m, u_min, u_max, gp, gm);

      // Apply the alpha coefficients to get the final solution.
      // Update the flux matrix for iterative FCT (when iter_cnt > 1).
      UpdateSolutionAndFlux(du_lo_fct, m, gp, gm, flux_ij, du);

      du_lo_fct = du;
   }
}

void FluxBasedFCT::ComputeFluxMatrix(const ParGridFunction &u,
                                     const Vector &du_ho,
                                     SparseMatrix &flux_mat) const
{
   const int s = u.Size();
   double *flux_data = flux_mat.HostReadWriteData();
   flux_mat.HostReadI(); flux_mat.HostReadJ();
   const int *K_I = K.HostReadI(), *K_J = K.HostReadJ();
   const double *K_data = K.HostReadData();
   const double *u_np = u.FaceNbrData().HostRead();
   u.HostRead();
   du_ho.HostRead();
   for (int i = 0; i < s; i++)
   {
      for (int k = K_I[i]; k < K_I[i + 1]; k++)
      {
         int j = K_J[k];
         if (j <= i) { continue; }

         double kij  = K_data[k], kji = K_data[K_smap[k]];
         double dij  = max(max(0.0, -kij), -kji);
         double u_ij = (j < s) ? u(i) - u(j)
                       : u(i) - u_np[j - s];

         flux_data[k] = dt * dij * u_ij;
      }
   }

   const int NE = pfes.GetMesh()->GetNE();
   const int ndof = s / NE;
   Array<int> dofs;
   DenseMatrix Mz(ndof);
   Vector du_z(ndof);
   for (int k = 0; k < NE; k++)
   {
      pfes.GetElementDofs(k, dofs);
      M.GetSubMatrix(dofs, dofs, Mz);
      du_ho.GetSubVector(dofs, du_z);
      for (int i = 0; i < ndof; i++)
      {
         int j = 0;
         for (; j <= i; j++) { Mz(i, j) = 0.0; }
         for (; j < ndof; j++) { Mz(i, j) *= dt * (du_z(i) - du_z(j)); }
      }
      flux_mat.AddSubMatrix(dofs, dofs, Mz, 0);
   }
}

// Compute sums of incoming fluxes for every DOF.
void FluxBasedFCT::AddFluxesAtDofs(const SparseMatrix &flux_mat,
                                   Vector &flux_pos, Vector &flux_neg) const
{
   const int s = flux_pos.Size();
   const double *flux_data = flux_mat.GetData();
   const int *flux_I = flux_mat.GetI(), *flux_J = flux_mat.GetJ();
   flux_pos = 0.0;
   flux_neg = 0.0;
   flux_pos.HostReadWrite();
   flux_neg.HostReadWrite();
   for (int i = 0; i < s; i++)
   {
      for (int k = flux_I[i]; k < flux_I[i + 1]; k++)
      {
         int j = flux_J[k];

         // The skipped fluxes will be added when the outer loop is at j as
         // the flux matrix is always symmetric.
         if (j <= i) { continue; }

         const double f_ij = flux_data[k];

         if (f_ij >= 0.0)
         {
            flux_pos(i) += f_ij;
            // Modify j if it's on the same MPI task (prevents x2 counting).
            if (j < s) { flux_neg(j) -= f_ij; }
         }
         else
         {
            flux_neg(i) += f_ij;
            // Modify j if it's on the same MPI task (prevents x2 counting).
            if (j < s) { flux_pos(j) -= f_ij; }
         }
      }
   }
}

// Compute the so-called alpha coefficients that scale the fluxes into gp, gm.
void FluxBasedFCT::
ComputeFluxCoefficients(const Vector &u, const Vector &du_lo, const Vector &m,
                        const Vector &u_min, const Vector &u_max,
                        Vector &coeff_pos, Vector &coeff_neg) const
{
   const int s = u.Size();
   for (int i = 0; i < s; i++)
   {
      const double u_lo = u(i) + dt * du_lo(i);
      const double max_pos_diff = max((u_max(i) - u_lo) * m(i), 0.0),
                   min_neg_diff = min((u_min(i) - u_lo) * m(i), 0.0);
      const double sum_pos = coeff_pos(i), sum_neg = coeff_neg(i);

      coeff_pos(i) = (sum_pos > max_pos_diff) ? max_pos_diff / sum_pos : 1.0;
      coeff_neg(i) = (sum_neg < min_neg_diff) ? min_neg_diff / sum_neg : 1.0;
   }
}

void FluxBasedFCT::
UpdateSolutionAndFlux(const Vector &du_lo, const Vector &m,
                      ParGridFunction &coeff_pos, ParGridFunction &coeff_neg,
                      SparseMatrix &flux_mat, Vector &du) const
{
   Vector &a_pos_n = coeff_pos.FaceNbrData(),
          &a_neg_n = coeff_neg.FaceNbrData();
   coeff_pos.ExchangeFaceNbrData();
   coeff_neg.ExchangeFaceNbrData();

   du = du_lo;

   coeff_pos.HostReadWrite();
   coeff_neg.HostReadWrite();
   du.HostReadWrite();

   double *flux_data = flux_mat.HostReadWriteData();
   const int *flux_I = flux_mat.HostReadI(), *flux_J = flux_mat.HostReadJ();
   const int s = du.Size();
   for (int i = 0; i < s; i++)
   {
      for (int k = flux_I[i]; k < flux_I[i + 1]; k++)
      {
         int j = flux_J[k];
         if (j <= i) { continue; }

         double fij = flux_data[k], a_ij;
         if (fij >= 0.0)
         {
            a_ij = (j < s) ? min(coeff_pos(i), coeff_neg(j))
                   : min(coeff_pos(i), a_neg_n(j - s));
         }
         else
         {
            a_ij = (j < s) ? min(coeff_neg(i), coeff_pos(j))
                   : min(coeff_neg(i), a_pos_n(j - s));
         }
         fij *= a_ij;

         du(i) += fij / m(i) / dt;
         if (j < s) { du(j) -= fij / m(j) / dt; }

         flux_data[k] -= fij;
      }
   }
}
