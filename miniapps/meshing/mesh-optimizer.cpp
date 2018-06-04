// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//            --------------------------------------------------
//            Mesh Optimizer Miniapp: Optimize high-order meshes
//            --------------------------------------------------
//
// This miniapp performs mesh optimization using the Target-Matrix Optimization
// Paradigm (TMOP) by P.Knupp et al., and a global variational minimization
// approach. It minimizes the quantity sum_T int_T mu(J(x)), where T are the
// target (ideal) elements, J is the Jacobian of the transformation from the
// target to the physical element, and mu is the mesh quality metric. This
// metric can measure shape, size or alignment of the region around each
// quadrature point. The combination of targets & quality metrics is used to
// optimize the physical node positions, i.e., they must be as close as possible
// to the shape / size / alignment of their targets. This code also demonstrates
// a possible use of nonlinear operators (the class TMOP_QualityMetric, defining
// mu(J), and the class TMOP_Integrator, defining int mu(J)), as well as their
// coupling to Newton methods for solving minimization problems. Note that the
// utilized Newton methods are oriented towards avoiding invalid meshes with
// negative Jacobian determinants. Each Newton step requires the inversion of a
// Jacobian matrix, which is done through an inner linear solver.
//
// Compile with: make mesh-optimizer
//
//  r-adapt WIP:
// ./mesh-optimizer -m TG_Q3_3R.mesh -rs 0 -o 3 -mid 9 -tid 5 -ls 2 -bnd -vl 2 -ni 200 -li 100 -qo 4 -qt 2
// r-adapt size:
// ./mesh-optimizer -m square.mesh -rs 2 -o 3 -mid 9 -tid 5 -ls 2 -bnd -vl 2 -ni 200 -li 100 -qo 4 -qt 2
// r-adapt shape:
// ./mesh-optimizer -m square.mesh -rs 2 -o 3 -mid 2 -tid 6 -ls 2 -bnd -vl 2 -ni 200 -li 100 -qo 4 -qt 2
// r-adapt shape+size:
// ./mesh-optimizer -m square.mesh -rs 2 -o 3 -mid 9 -tid 7 -ls 2 -bnd -vl 2 -ni 200 -li 100 -qo 4 -qt 2
//
//
// Sample runs:
//   Blade shape:
//     mesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   Blade limited shape:
//     mesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8 -lc 5000
//   ICF shape and equal size:
//     mesh-optimizer -o 3 -rs 0 -mid 9 -tid 2 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF shape and initial size:
//     mesh-optimizer -o 3 -rs 0 -mid 9 -tid 3 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF shape:
//     mesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF limited shape:
//     mesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8 -lc 10
//   ICF combo shape + size (rings, slow convergence):
//     mesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 1000 -ls 2 -li 100 -bnd -qt 1 -qo 8 -cmb
//   3D pinched sphere shape (the mesh is in the mfem/data GitHub repository):
//   * mesh-optimizer -m ../../../mfem_data/ball-pert.mesh -o 4 -rs 0 -mid 303 -tid 1 -ni 20 -ls 2 -li 500 -fix-bnd

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <ctime>

using namespace mfem;
using namespace std;

double weight_fun(const Vector &x);

// Metric values are visualized by creating an L2 finite element functions and
// computing the metric values at the nodes.
void vis_metric(int order, TMOP_QualityMetric &qm, const TargetConstructor &tc,
                Mesh &mesh, char *title, int position)
{
   L2_FECollection fec(order, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 1);
   GridFunction metric(&fes);
   InterpolateTMOP_QualityMetric(qm, tc, mesh, metric);
   osockstream sock(19916, "localhost");
   sock << "solution\n";
   mesh.Print(sock);
   metric.Save(sock);
   sock.send();
   sock << "window_title '"<< title << "'\n"
        << "window_geometry "
        << position << " " << 0 << " " << 600 << " " << 600 << "\n"
        << "keys jRmclA" << endl;
}

//-------------- Begin CGO Solver
class CGOSolver : public IterativeSolver
{

private:
   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;
   FiniteElementSpace *fes;
   mutable GridFunction x_gf;

protected:
   mutable Vector r, c, s;

public:
   CGOSolver(const IntegrationRule &irule, FiniteElementSpace *f)
      : ir(irule), fes(f) { }

   virtual void SetOperator(const Operator &op);

   /** This method is equivalent to calling SetPreconditioner(). */
   virtual void SetSolver(Solver &solver) { prec = &solver; }

   /// Solve the nonlinear system with right-hand side @a b.
   /** If `b.Size() != Height()`, then @a b is assumed to be zero. */
   virtual void Mult(const Vector &b, Vector &x) const;

   /** @brief This method can be overloaded in derived classes to implement line
       search algorithms. */
   /** The base class implementation (NewtonSolver) simply returns 1. A return
       value of 0 indicates a failure, interrupting the Newton iteration. */
   virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const;

   virtual void ProcessNewState(const Vector &x) const { }
};

void CGOSolver::SetOperator(const Operator &op)
{
   oper = &op;
   height = op.Height();
   width = op.Width();
   MFEM_ASSERT(height == width, "square Operator is required.");

   r.SetSize(width);
   c.SetSize(width);
}


void CGOSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

   int it;
   double norm0, norm, norm_goal, beta, num,den;
   const bool have_b = (b.Size() == Height());

   if (!iterative_mode)
   {
      x = 0.0;
   }

   oper->Mult(x, r); // r = b-Ax
   if (have_b)
   {
      r -= b;
   }

   c = r;

   norm0 = norm = Norm(r);
   norm_goal = std::max(rel_tol*norm, abs_tol);

   prec->iterative_mode = false;

   for (it = 0; true; it++)
   {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_level >= 0)
      {
         mfem::out << "Newton iteration " << setw(2) << it
                   << " : ||r|| = " << norm;
         if (it > 0)
         {
            mfem::out << ", ||r||/||r_0|| = " << norm/norm0;
         }
         mfem::out << '\n';
      }



      if (norm <= norm_goal)
      {
         converged = 1;
         break;
      }

      if (it >= max_iter)
      {
         converged = 0;
         break;
      }
/*
      if (it % 50 == 0)
      { 
        oper->Mult(x, r); // r = b-Ax
        if (have_b)
        {  
         r -= b;
        }
        c = r;
      }
*/
      const double c_scale = ComputeScalingFactor(x, b);
      if (c_scale == 0.0)
      {
         converged = 0;
         break;
      }
      add(x, -c_scale, c, x);

      oper->Mult(x, r);
      if (have_b)
      {
         r -= b;
      }

      num = Dot(r,r); // g_{k+1}^T g_{k+1}
      den = Dot(c,c); // g_k ^T g_k 
      beta = num/den; //
      add(r,beta,c,c); // c = r - beta(c)

      norm = Norm(r);
   }

   final_iter = it;
   final_norm = norm;
}

double CGOSolver::ComputeScalingFactor(const Vector &x,
                                                 const Vector &b) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   MFEM_VERIFY(nlf != NULL, "invalid Operator subclass");
   const bool have_b = (b.Size() == Height());

   const int NE = fes->GetMesh()->GetNE(), dim = fes->GetFE(0)->GetDim(),
             dof = fes->GetFE(0)->GetDof(), nsp = ir.GetNPoints();
   Array<int> xdofs(dof * dim);
   DenseMatrix Jpr(dim), dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);

   Vector x_out(x.Size());
   bool x_out_ok = false;
   const double energy_in = nlf->GetEnergy(x);
   double scale = 1.0, energy_out;
   double norm0 = Norm(r);
   x_gf.MakeTRef(fes, x_out, 0);

   // Decreases the scaling of the update until the new mesh is valid.
   for (int i = 0; i < 20; i++)
   {
      add(x, -scale, c, x_out);
      x_gf.SetFromTrueVector();

      energy_out = nlf->GetGridFunctionEnergy(x_gf);
      if (energy_out > 1.2*energy_in || isnan(energy_out) != 0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Increasing energy." << endl; }
         scale *= 0.1; continue;
      }

      int jac_ok = 1;
      for (int i = 0; i < NE; i++)
      {
         fes->GetElementVDofs(i, xdofs);
         x_gf.GetSubVector(xdofs, posV);
         for (int j = 0; j < nsp; j++)
         {
            fes->GetFE(i)->CalcDShape(ir.IntPoint(j), dshape);
            MultAtB(pos, dshape, Jpr);
            if (Jpr.Det() <= 0.0) { jac_ok = 0; goto break2; }
         }
      }
   break2:
      if (jac_ok == 0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Neg det(J) found." << endl; }
         scale *= 0.1; continue;
      }

      oper->Mult(x_out, r);
      if (have_b) { r -= b; }
      double norm = Norm(r);

      if (norm > 1.2*norm0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Norm increased." << endl; }
         scale *= 0.1; continue;
      }
      else { x_out_ok = true; break; }
   }

   if (print_level >= 0)
   {
      cout << "Energy decrease: "
           << (energy_in - energy_out) / energy_in * 100.0
           << "% with " << scale << " scaling." << endl;
   }

   if (x_out_ok == false) { scale = 0.0; }

   return scale;
}
//--------------------------- END CGO SOLVER

double ind_values(const Vector &x)
{
   // Sub-square.
   //if (x(0) > 0.3 && x(0) < 0.5 && x(1) > 0.5 && x(1) < 0.7) { return 1.0; }

   // Circle from origin.
   //const double r = sqrt(x(0)*x(0) + x(1)*x(1));
   //if (r > 0.5 && r < 0.6) { return 1.0; }

   // 3point.
   //if (x(0) >= 0.1 && x(0) <= 0.2) { return 1.0; }
   //if (x(1) >= 0.45 && x(1) <= 0.55 && x(0) >= 0.1 ) { return 1.0; }

   // Sine wave.
   const double X = x(0), Y = x(1);
   return std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) + 1) -
          std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) - 1);


   // Circle in the middle.
   //const double xc = x(0) - 0.5, yc = x(1) - 0.5;
   //const double r = sqrt(xc*xc + yc*yc);
   //if (r > 0.2 && r < 0.3) { return 1.0; }

   return 0.0;
}

void normalize(Vector &v)
{
   const double max = v.Max();
   v /= max;
}

// Performs an advection step.
class AdvectorCGOperator : public TimeDependentOperator
{
private:
   GridFunction &x0, &u, &x_now;

   VectorGridFunctionCoefficient u_coeff;
   mutable BilinearForm M, K;

public:
   // Note: fes must be the FiniteElementSpace of the mesh that will be moved.
   //       xn must be the Nodes GridFunction of the mesh that will be moved.
   AdvectorCGOperator(GridFunction &x_start, GridFunction &vel,
                      GridFunction &xn, FiniteElementSpace &fes)
      : TimeDependentOperator(fes.GetVSize()),
        x0(x_start), u(vel), x_now(xn), u_coeff(&u), M(&fes), K(&fes)
   {
      ConvectionIntegrator *Kinteg = new ConvectionIntegrator(u_coeff);
      K.AddDomainIntegrator(Kinteg);
      K.Assemble(0);
      K.Finalize(0);

      MassIntegrator *Minteg = new MassIntegrator;
      M.AddDomainIntegrator(Minteg);
      M.Assemble();
      M.Finalize();
   }

   virtual void Mult(const Vector &ind, Vector &di_dt) const
   {
      const double t = GetTime();

      // Move the mesh.
      add(x0, t, u, x_now);

      // Assemble on the new mesh.
      K.BilinearForm::operator=(0.0);
      K.Assemble();
      Vector rhs(ind);
      K.Mult(ind, rhs);
      M.BilinearForm::operator=(0.0);
      M.Assemble();

      // Solve.
      GSSmoother M_prec(M.SpMat());
      PCG(M, M_prec, rhs, di_dt);
   }
};

// Performs the whole advection loop.
class AdvectorCG
{
private:
   Mesh mesh;
   FiniteElementSpace fes;
   RK4Solver ode_solver;

public:
   AdvectorCG(Mesh &m, const FiniteElementCollection &field_fec)
      : mesh(m, true), fes(&mesh, &field_fec), ode_solver() { }

   // Advects ind from x_start to x_end.
   void Advect(GridFunction &x_start, GridFunction &x_end,
               GridFunction &ind)
   {
      GridFunction *mesh_nodes = mesh.GetNodes();
      *mesh_nodes = x_start;

      GridFunction u(x_start.FESpace());
      subtract(x_end, x_start, u);

      // This must be the fes of the ind, associated with the object's mesh.
      AdvectorCGOperator oper(x_start, u, *mesh_nodes, fes);
      ode_solver.Init(oper);

      // Compute some time step [mesh_size / speed].
      double min_h = numeric_limits<double>::infinity();
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         min_h = std::min(min_h, mesh.GetElementSize(1));
      }
      double v_max = 0.0;
      int s = u.FESpace()->GetVSize() / 2;
      for (int i = 0; i < s; i++)
      {
         double vel = std::sqrt( u(i) * u(i) + u(i+s) * u(i+s) + 1e-14);
         v_max = std::max(v_max, vel);
      }
      double dt = 0.5 * min_h / v_max;

      double t = 0.0;
      bool last_step = false;
      for (int ti = 1; !last_step; ti++)
      {
         if (t + dt >= 1.0)
         {
            std::cout << "Remap with dt = " << dt
                      << " took " << ti << " steps." << std::endl;
            dt = 1.0 - t;
            last_step = true;
         }
         ode_solver.Step(ind, t, dt);
      }

      // Trim to put it in [0, 1].
      for (int i = 0; i < ind.Size(); i++)
      {
         if (ind(i) < 0.0) { ind(i) = 0.0; }
         if (ind(i) > 1.0) { ind(i) = 1.0; }
      }
   }
};

class RelaxedNewtonSolver : public NewtonSolver
{
private:
   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;
   FiniteElementSpace *fes;
   mutable GridFunction x_gf;

   // Advection related.
   GridFunction *x0, *ind0, *ind;
   AdvectorCG *advector;

public:
   RelaxedNewtonSolver(const IntegrationRule &irule, FiniteElementSpace *f,
                       GridFunction *x0_, GridFunction *ind0_,
                       GridFunction *ind_, AdvectorCG *adv)
      : ir(irule), fes(f), x_gf(),
        x0(x0_), ind0(ind0_), ind(ind_), advector(adv) { }

   virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const;
   virtual void ProcessNewState(const Vector &x) const
   {
      if (x0 && ind0 && ind && advector)
      {
         // GridFunction with the current positions.
         Vector x_copy(x);
         x_gf.MakeTRef(fes, x_copy, 0);

         // Reset the indicator to its values on the initial positions.
         *ind = *ind0;

         // Advect the indicator from the original to the new posiions.
         advector->Advect(*x0, x_gf, *ind);
      }
   }
};

double RelaxedNewtonSolver::ComputeScalingFactor(const Vector &x,
                                                 const Vector &b) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   MFEM_VERIFY(nlf != NULL, "invalid Operator subclass");
   const bool have_b = (b.Size() == Height());

   const int NE = fes->GetMesh()->GetNE(), dim = fes->GetFE(0)->GetDim(),
             dof = fes->GetFE(0)->GetDof(), nsp = ir.GetNPoints();
   Array<int> xdofs(dof * dim);
   DenseMatrix Jpr(dim), dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);

   Vector x_out(x.Size());
   bool x_out_ok = false;
   const double energy_in = nlf->GetEnergy(x);
   double scale = 1.0, energy_out;
   double norm0 = Norm(r);
   x_gf.MakeTRef(fes, x_out, 0);

   // Decreases the scaling of the update until the new mesh is valid.
   for (int i = 0; i < 12; i++)
   {
      add(x, -scale, c, x_out);
      x_gf.SetFromTrueVector();

      energy_out = nlf->GetGridFunctionEnergy(x_gf);
      if (energy_out > 1.2*energy_in || isnan(energy_out) != 0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Increasing energy." << endl; }
         scale *= 0.5; continue;
      }

      int jac_ok = 1;
      for (int i = 0; i < NE; i++)
      {
         fes->GetElementVDofs(i, xdofs);
         x_gf.GetSubVector(xdofs, posV);
         for (int j = 0; j < nsp; j++)
         {
            fes->GetFE(i)->CalcDShape(ir.IntPoint(j), dshape);
            MultAtB(pos, dshape, Jpr);
            if (Jpr.Det() <= 0.0) { jac_ok = 0; goto break2; }
         }
      }
   break2:
      if (jac_ok == 0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Neg det(J) found." << endl; }
         scale *= 0.5; continue;
      }

      oper->Mult(x_out, r);
      if (have_b) { r -= b; }
      double norm = Norm(r);

      if (norm > 1.2*norm0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Norm increased." << endl; }
         scale *= 0.5; continue;
      }
      else { x_out_ok = true; break; }
   }

   if (print_level >= 0)
   {
      cout << "Energy decrease: "
           << (energy_in - energy_out) / energy_in * 100.0
           << "% with " << scale << " scaling." << endl;
   }

   if (x_out_ok == false) { scale = 0.0; }

   return scale;
}



// Allows negative Jacobians. Used in untangling metrics.
class DescentNewtonSolver : public NewtonSolver
{
private:
   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;
   FiniteElementSpace *fes;
   mutable GridFunction x_gf;

public:
   DescentNewtonSolver(const IntegrationRule &irule, FiniteElementSpace *f)
      : ir(irule), fes(f) { }

   virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const;
};

double DescentNewtonSolver::ComputeScalingFactor(const Vector &x,
                                                 const Vector &b) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   MFEM_VERIFY(nlf != NULL, "invalid Operator subclass");

   const int NE = fes->GetMesh()->GetNE(), dim = fes->GetFE(0)->GetDim(),
             dof = fes->GetFE(0)->GetDof(), nsp = ir.GetNPoints();
   Array<int> xdofs(dof * dim);
   DenseMatrix Jpr(dim), dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);

   x_gf.MakeTRef(fes, x.GetData());
   x_gf.SetFromTrueVector();

   double min_detJ = infinity();
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, xdofs);
      x_gf.GetSubVector(xdofs, posV);
      for (int j = 0; j < nsp; j++)
      {
         fes->GetFE(i)->CalcDShape(ir.IntPoint(j), dshape);
         MultAtB(pos, dshape, Jpr);
         min_detJ = min(min_detJ, Jpr.Det());
      }
   }
   cout << "Minimum det(J) = " << min_detJ << endl;

   Vector x_out(x.Size());
   bool x_out_ok = false;
   const double energy_in = nlf->GetGridFunctionEnergy(x_gf);
   double scale = 1.0, energy_out;

   for (int i = 0; i < 7; i++)
   {
      add(x, -scale, c, x_out);

      energy_out = nlf->GetEnergy(x_out);
      if (energy_out > energy_in || isnan(energy_out) != 0)
      {
         scale *= 0.5;
      }
      else { x_out_ok = true; break; }
   }

   cout << "Energy decrease: " << (energy_in - energy_out) / energy_in * 100.0
        << "% with " << scale << " scaling." << endl;

   if (x_out_ok == false) { return 0.0;}

   return scale;
}

// Additional IntegrationRules that can be used with the --quad-type option.
IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);


int main (int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   double jitter         = 0.0;
   int metric_id         = 1;
   int target_id         = 1;
   double lim_const      = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int newton_iter       = 10;
   double newton_rtol    = 1e-12;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = true;
   bool combomet         = 0;
   bool visualization    = true;
   int verbosity_level   = 0;
   int solver_type       = 0;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric:\n\t"
                  "1  : |T|^2                          -- 2D shape\n\t"
                  "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                  "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                  "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
                  "22 : 0.5(|T|^2-2*tau)/(tau-tau_0)   -- 2D untangling\n\t"
                  "50 : 0.5|T^tT|^2/tau^2-1            -- 2D shape\n\t"
                  "55 : (tau-1)^2                      -- 2D size\n\t"
                  "56 : 0.5(sqrt(tau)-1/sqrt(tau))^2   -- 2D size\n\t"
                  "58 : |T^tT|^2/(tau^2)-2*|T|^2/tau+2 -- 2D shape\n\t"
                  "77 : 0.5(tau-1/tau)^2               -- 2D size\n\t"
                  "211: (tau-1)^2-tau+sqrt(tau^2)      -- 2D untangling\n\t"
                  "252: 0.5(tau-1)^2/(tau-tau_0)       -- 2D untangling\n\t"
                  "301: (|T||T^-1|)/3-1              -- 3D shape\n\t"
                  "302: (|T|^2|T^-1|^2)/9-1          -- 3D shape\n\t"
                  "303: (|T|^2)/3*tau^(2/3)-1        -- 3D shape\n\t"
                  "315: (tau-1)^2                    -- 3D size\n\t"
                  "316: 0.5(sqrt(tau)-1/sqrt(tau))^2 -- 3D size\n\t"
                  "321: |T-T^-t|^2                   -- 3D shape+size\n\t"
                  "352: 0.5(tau-1)^2/(tau-tau_0)     -- 3D untangling");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target (ideal element) type:\n\t"
                  "1: Ideal shape, unit size\n\t"
                  "2: Ideal shape, equal size\n\t"
                  "3: Ideal shape, initial size");
   args.AddOption(&lim_const, "-lc", "--limit-const", "Limiting constant.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&solver_type, "-st", "--solver_type",
                  "Set the non-linear solver - 0 or 1");
   args.AddOption(&newton_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver: 0 - l1-Jacobi, 1 - CG, 2 - MINRES.");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&combomet, "-cmb", "--combo-met", "-no-cmb", "--no-combo-met",
                  "Combination of metrics.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   cout << "Mesh curvature: ";
   if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
   else { cout << "(NONE)"; }
   cout << endl;

   // 3. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   FiniteElementCollection *fec;
   if (mesh_poly_deg <= 0)
   {
      fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else { fec = new H1_FECollection(mesh_poly_deg, dim); }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);

   // 4. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   mesh->SetNodalFESpace(fespace);

   // 5. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 6. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   GridFunction *x = mesh->GetNodes();

   // 7. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in fespace.
   Vector h0(fespace->GetNDofs());
   h0 = infinity();
   Array<int> dofs;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      fespace->GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), mesh->GetElementSize(i));
      }
   }

   // 8. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   GridFunction rdm(fespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < fespace->GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(fespace->DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < fespace->GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      fespace->GetBdrElementVDofs(i, vdofs);
      // Set the boundary values to zero.
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   *x -= rdm;

   // 9. Save the starting (prior to the optimization) mesh to a file. This
   //    output can be viewed later using GLVis: "glvis -m perturbed.mesh".
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }

   // 10. Store the starting (prior to the optimization) positions.
   GridFunction x0(fespace);
   x0 = *x;

   // 11. Form the integrator that uses the chosen metric and target.
   double tauval = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 22: metric = new TMOP_Metric_022(tauval); break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 56: metric = new TMOP_Metric_056; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 77: metric = new TMOP_Metric_077; break;
      case 211: metric = new TMOP_Metric_211; break;
      case 252: metric = new TMOP_Metric_252(tauval); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 352: metric = new TMOP_Metric_352(tauval); break;
      default: cout << "Unknown metric_id: " << metric_id << endl; return 3;
   }
   TargetConstructor::TargetType target_t;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4: target_t = TargetConstructor::IDEAL_SHAPE_ADAPTIVE_SIZE; break;
      case 5: target_t = TargetConstructor::IDEAL_SHAPE_ADAPTIVE_SIZE_7; break;
      case 6: target_t = TargetConstructor::ADAPTIVE_SHAPE; break;
      case 7: target_t = TargetConstructor::ADAPTIVE_SHAPE_AND_SIZE; break;
      default: cout << "Unknown target_id: " << target_id << endl;
         delete metric; return 3;
   }
   TargetConstructor *target_c = new TargetConstructor(target_t);
   target_c->SetNodes(x0);
   TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, target_c);
   // Indicator function.
   // Copy of the initial mesh.
   Mesh mesh0(*mesh);
   FunctionCoefficient ind_coeff(ind_values);
   L2_FECollection ind_fec(2, dim);
   FiniteElementSpace ind_fes(&mesh0, &ind_fec);
   GridFunction ind_gf(&ind_fes);
   ind_gf.ProjectCoefficient(ind_coeff);
   normalize(ind_gf);

   H1_FECollection remap_fec(2, dim);
   FiniteElementSpace remap_fes(mesh, &remap_fec);
   GridFunction remap_gf(&remap_fes);
   remap_gf.ProjectCoefficient(ind_coeff);
   normalize(remap_gf);
   GridFunction remap_gf_init(remap_gf);

   // Adaptivity tests.
   if (target_t == TargetConstructor::IDEAL_SHAPE_ADAPTIVE_SIZE)
   {
      target_c->SetMeshAndIndicator(mesh0, ind_gf, 20.0);
      target_c->SetMeshNodes(*x);
   }
   if (target_t == TargetConstructor::IDEAL_SHAPE_ADAPTIVE_SIZE_7)
   {
      target_c->SetIndicator(remap_gf, 10.0);
   }
   if (target_t == TargetConstructor::ADAPTIVE_SHAPE)
   {
      target_c->SetIndicator(remap_gf, 3.0);
   }
   if (target_t == TargetConstructor::ADAPTIVE_SHAPE_AND_SIZE)
   {
      target_c->SetIndicator(remap_gf, 3.0);
   }

   AdvectorCG *advector = NULL;
   if (target_t > 4)
   {
      advector = new AdvectorCG(mesh0, *remap_gf.FESpace()->FEColl());
   }
//   AdvectorCG advector(mesh0, *remap_gf.FESpace()->FEColl());

   if (visualization &&
       (target_t == TargetConstructor::IDEAL_SHAPE_ADAPTIVE_SIZE_7 ||
        target_t == TargetConstructor::ADAPTIVE_SHAPE ||
        target_t == TargetConstructor::ADAPTIVE_SHAPE_AND_SIZE) )
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh0.Print(sock);
      remap_gf.Save(sock);
      sock.send();
      sock << "window_title 'Adaptivity Indicator'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;
   }

   if (visualization &&
       target_t == TargetConstructor::IDEAL_SHAPE_ADAPTIVE_SIZE)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh0.Print(sock);
      ind_gf.Save(sock);
      sock.send();
      sock << "window_title 'Adaptivity Indicator'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;
   }

   // 12. Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = NULL;
   const int geom_type = fespace->GetFE(0)->GetGeomType();
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order); break;
      case 2: ir = &IntRules.Get(geom_type, quad_order); break;
      case 3: ir = &IntRulesCU.Get(geom_type, quad_order); break;
      default: cout << "Unknown quad_type: " << quad_type << endl;
         delete he_nlf_integ; return 3;
   }
   cout << "Quadrature points per cell: " << ir->GetNPoints() << endl;
   he_nlf_integ->SetIntegrationRule(*ir);

   // 13. Limit the node movement.
   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, lim_coeff); }

   // 14. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights. Note that there are no
   //     command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   NonlinearForm a(fespace);
   Coefficient *coeff1 = NULL;
   TMOP_QualityMetric *metric2 = NULL;
   TargetConstructor *target_c2 = NULL;
   FunctionCoefficient coeff2(weight_fun);
   if (combomet == 1)
   {
      // Weight of the original metric.
      coeff1 = new ConstantCoefficient(1.25);
      he_nlf_integ->SetCoefficient(*coeff1);
      a.AddDomainIntegrator(he_nlf_integ);

      metric2 = new TMOP_Metric_077;
      target_c2 = new TargetConstructor(
         TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE);
      target_c2->SetVolumeScale(0.01);
      target_c2->SetNodes(x0);
      TMOP_Integrator *he_nlf_integ2 = new TMOP_Integrator(metric2, target_c2);
      he_nlf_integ2->SetIntegrationRule(*ir);

      // Weight of metric2.
      he_nlf_integ2->SetCoefficient(coeff2);
      a.AddDomainIntegrator(he_nlf_integ2);
   }
   else { a.AddDomainIntegrator(he_nlf_integ); }
   const double init_en = a.GetGridFunctionEnergy(*x);
   cout << "Initial strain energy: " << init_en << endl;

   // 15. Visualize the starting mesh and metric values.
   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_metric(mesh_poly_deg, *metric, *target_c, *mesh, title, 0);
   }

   // 16. Fix all boundary nodes, or fix only a given component depending on the
   //     boundary attributes of the given mesh. Attributes 1/2/3 correspond to
   //     fixed x/y/z components of the node. Attribute 4 corresponds to an
   //     entirely fixed node. Other boundary attributes do not affect the node
   //     movement boundary conditions.
   if (move_bnd == false)
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      const int nd  = fespace->GetBE(0)->GetDof();
      int n = 0;
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int attr = mesh->GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n), vdofs;
      n = 0;
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int attr = mesh->GetBdrElement(i)->GetAttribute();
         fespace->GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         else if (attr == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+2*nd]; }
         }
         else if (attr == 4) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      a.SetEssentialVDofs(ess_vdofs);
   }

   // 17. As we use the Newton method to solve the resulting nonlinear system,
   //     here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL;
   DSmoother *prec = NULL;
   const double linsol_rtol = 1e-12;
   if (lin_solver == 0)
   {
      S = new DSmoother(1, 1.0, max_lin_iter);
   }
   else if (lin_solver == 1)
   {
      CGSolver *cg = new CGSolver;
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(linsol_rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver;
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      minres->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);

      prec = new DSmoother(1);
      minres->SetPreconditioner(*prec);

      S = minres;
   }

   // 18. Compute the minimum det(J) of the starting mesh.
   tauval = infinity();
   const int NE = mesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      ElementTransformation *transf = mesh->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir->IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   cout << "Minimum det(J) of the original mesh is " << tauval << endl;

   // 19. Finally, perform the nonlinear optimization.
   NewtonSolver *newton = NULL;
   CGOSolver    *cgo    = NULL;
   int start_s=clock();
   if (solver_type == 0) 
   {
   if (tauval > 0.0)
   {
      tauval = 0.0;
      newton = new RelaxedNewtonSolver(*ir, fespace,
                                       &x0, &remap_gf_init,
                                       &remap_gf, advector);
      cout << "The RelaxedNewtonSolver is used (as all det(J)>0)." << endl;
   }
   else
   {
      if ( (dim == 2 && metric_id != 22 && metric_id != 252) ||
           (dim == 3 && metric_id != 352) )
      {
         cout << "The mesh is inverted. Use an untangling metric." << endl;
         return 3;
      }
      tauval -= 0.01 * h0.Min(); // Slightly below minJ0 to avoid div by 0.
      newton = new DescentNewtonSolver(*ir, fespace);
      cout << "The DescentNewtonSolver is used (as some det(J)<0)." << endl;
   }

   newton->SetPreconditioner(*S);
   newton->SetMaxIter(newton_iter);
   newton->SetRelTol(newton_rtol);
   newton->SetAbsTol(0.0);
   newton->SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
   newton->SetOperator(a);
   newton->Mult(b, *x);
   if (newton->GetConverged() == false)
   {
      cout << "NewtonIteration: rtol = " << newton_rtol << " not achieved."
           << endl;
   }
   delete newton;
   }
   else
   {
     cgo = new CGOSolver(*ir, fespace);
     cout << "The CG Optimizer is used." << endl;
    int start_s=clock();
    cgo->SetPreconditioner(*S);
    cgo->SetMaxIter(newton_iter);
    cgo->SetRelTol(newton_rtol);
    cgo->SetAbsTol(0.0);
    cgo->SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
    cgo->SetOperator(a);
    cgo->Mult(b, *x);
    x->SetFromTrueVector();
    if (cgo->GetConverged() == false)
    {
       cout << "NewtonIteration: rtol = " << newton_rtol << " not achieved."
            << endl;
    }
   delete cgo;
   }
   // 20. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized.mesh".
   {
      ofstream mesh_ofs("optimized.mesh");
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }
   int stop_s=clock();
   cout << "time taken (sec): " << (stop_s-start_s)/1000000. << endl;

   // 21. Compute the amount of energy decrease.
   const double fin_en = a.GetGridFunctionEnergy(*x);
   cout << "Final strain energy : " << fin_en << endl;
   cout << "The strain energy decreased by: " << setprecision(12)
        << (init_en - fin_en) * 100.0 / init_en << " %." << endl;

   // 22. Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_metric(mesh_poly_deg, *metric, *target_c, *mesh, title, 600);
   }

   // Remap test.
   //AdvectorCG advector2(mesh0, *remap_gf.FESpace()->FEColl());
   //advector2.Advect(x0, *x, remap_gf);
   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      remap_gf.Save(sock);
      sock.send();
      sock << "window_title 'Remapped Final'\n"
           << "window_geometry "
           << 700 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;
   }

   // 23. Visualize the mesh displacement.
   if (visualization)
   {
      x0 -= *x;
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      x0.Save(sock);
      sock.send();
      sock << "window_title 'Displacements'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;
   }

   // 24. Free the used memory.
   delete prec;
   delete S;
   delete advector;
   delete target_c2;
   delete metric2;
   delete coeff1;
   delete target_c;
   delete metric;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}

// Defined with respect to the icf mesh.
double weight_fun(const Vector &x)
{
   const double r = sqrt(x(0)*x(0) + x(1)*x(1) + 1e-12);
   const double den = 0.002;
   double l2 = 0.2 + 0.5*std::tanh((r-0.16)/den) - 0.5*std::tanh((r-0.17)/den)
               + 0.5*std::tanh((r-0.23)/den) - 0.5*std::tanh((r-0.24)/den);
   return l2;
}
