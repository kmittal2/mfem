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
//    ---------------------------------------------------------------------
//    Mesh Optimizer Miniapp: Optimize high-order meshes - Parallel Version
//    ---------------------------------------------------------------------
//
// This miniapp performs mesh optimization using the Target-Matrix Optimization
// Paradigm (TMOP), and a global variational minimization approach. It minimizes
// the quantity sum_T int_T mu(Jtr(x)), where T are the target (ideal) elements,
// Jtr is the Jacobian of the transformation from the reference to the target
// element, and mu is the mesh quality metric. This metric can measure shape,
// size or alignment of the region around each quadrature point. The combination
// of targets & quality metrics is used to optimize the physical node positions,
// i.e., they must be as close as possible to the shape / size / alignment of
// their targets. This code also demonstrates a possible use of nonlinear
// operators (the class HyperelasticModel, defining mu(Jtr), and the class
// HyperelasticNLFIntegrator, defining int mu(Jtr)), as well as their coupling
// to Newton methods for solving minimization problems. Note that the utilized
// Newton methods are oriented towards avoiding invalid meshes with negative
// Jacobian determinants. Each Newton step requires the inversion of a Jacobian
// matrix, which is done through an inner linear solver.
//
// Compile with: make pmesh-optimizer
//
// Sample runs:
//   mpirun -np 4 pmesh-optimizer -m ../../data/blade.mesh -o 2 -rs 0 -mid 2 -tid 1 -ni 20 -ls 1 -bnd
//   mpirun -np 4 pmesh-optimizer i -o 2 -rs 0 -ji 0.0 -mid 2 -tid 1 -lim -lc 0.001 -ni 10 -ls 1 -bnd

#include "mfem.hpp"

using namespace mfem;
using namespace std;

double weight_fun(const Vector &x);

// Metric values are visualized by creating an L2 finite element functions and
// computing the metric values at the nodes.
void vis_metric(int order, HyperelasticModel &model, const TargetJacobian &tj,
                ParMesh &pmesh, char *title, int position)
{
   L2_FECollection fec(order, pmesh.Dimension(), BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&pmesh, &fec, 1);
   ParGridFunction metric(&fes);
   InterpolateHyperElasticModel(model, tj, pmesh, metric);
   osockstream sock(19916, "localhost");
   sock << "solution\n";
   pmesh.PrintAsOne(sock);
   metric.SaveAsOne(sock);
   sock.send();
   sock << "window_title '"<< title << "'\n"
        << "window_geometry "
        << position << " " << 0 << " " << 600 << " " << 600 << "\n"
        << "keys JRmcl" << endl;
}

class RelaxedNewtonSolver : public NewtonSolver
{
private:
   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;

public:
   RelaxedNewtonSolver(const IntegrationRule &irule) : ir(irule) { }

#ifdef MFEM_USE_MPI
   RelaxedNewtonSolver(const IntegrationRule &irule, MPI_Comm _comm)
      : NewtonSolver(_comm), ir(irule) { }
#endif

   virtual double ComputeScalingFactor(const Vector &x, const Vector &c) const;
};

double RelaxedNewtonSolver::ComputeScalingFactor(const Vector &x,
                                                 const Vector &c) const
{
   const ParNonlinearForm *nlf = dynamic_cast<const ParNonlinearForm *>(oper);
   ParFiniteElementSpace *pfes = nlf->ParFESpace();

   const int NE = pfes->GetParMesh()->GetNE(), dim = pfes->GetFE(0)->GetDim(),
             dof = pfes->GetFE(0)->GetDof(), nsp = ir.GetNPoints();
   Array<int> xdofs(dof * dim);
   DenseMatrix Jpr(dim), dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);

   Vector x_out(x.Size());
   ParGridFunction x_out_gf(nlf->ParFESpace());
   bool x_out_ok = false;
   const double energy_in = nlf->GetEnergy(x);
   double scale = 1.0, energy_out;

   // Decreases the scaling of the update until the new mesh is valid.
   for (int i = 0; i < 7; i++)
   {
      add(x, -scale, c, x_out);
      x_out_gf.Distribute(x_out);

      energy_out = nlf->GetEnergy(x_out_gf);
      if (energy_out > energy_in || isnan(energy_out) != 0)
      {
         scale *= 0.5; continue;
      }

      int jac_ok = 1;
      for (int i = 0; i < NE; i++)
      {
         pfes->GetElementVDofs(i, xdofs);
         x_out_gf.GetSubVector(xdofs, posV);
         for (int j = 0; j < nsp; j++)
         {
            pfes->GetFE(i)->CalcDShape(ir.IntPoint(j), dshape);
            MultAtB(pos, dshape, Jpr);
            if (Jpr.Det() <= 0.0) { jac_ok = 0; goto break2; }
         }
      }
   break2:
      int jac_ok_all;
      MPI_Allreduce(&jac_ok, &jac_ok_all, 1, MPI_INT, MPI_LAND,
                    pfes->GetComm());

      if (jac_ok_all == 0) { scale *= 0.5; }
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

public:
   DescentNewtonSolver(const IntegrationRule &irule) : ir(irule) { }

#ifdef MFEM_USE_MPI
   DescentNewtonSolver(const IntegrationRule &irule, MPI_Comm _comm)
      : NewtonSolver(_comm), ir(irule) { }
#endif

   virtual double ComputeScalingFactor(const Vector &x, const Vector &c) const;
};

double DescentNewtonSolver::ComputeScalingFactor(const Vector &x,
                                                 const Vector &c) const
{
   const ParNonlinearForm *nlf = dynamic_cast<const ParNonlinearForm *>(oper);
   ParFiniteElementSpace *pfes = nlf->ParFESpace();

   const int NE = pfes->GetParMesh()->GetNE(), dim = pfes->GetFE(0)->GetDim(),
             dof = pfes->GetFE(0)->GetDof(), nsp = ir.GetNPoints();
   Array<int> xdofs(dof * dim);
   DenseMatrix Jpr(dim), dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);

   ParGridFunction x_gf(nlf->ParFESpace());
   x_gf.Distribute(x);

   double min_detJ = numeric_limits<double>::infinity();
   for (int i = 0; i < NE; i++)
   {
      pfes->GetElementVDofs(i, xdofs);
      x_gf.GetSubVector(xdofs, posV);
      for (int j = 0; j < nsp; j++)
      {
         pfes->GetFE(i)->CalcDShape(ir.IntPoint(j), dshape);
         MultAtB(pos, dshape, Jpr);
         min_detJ = min(min_detJ, Jpr.Det());
      }
   }
   double min_detJ_all;
   MPI_Allreduce(&min_detJ, &min_detJ_all, 1, MPI_DOUBLE, MPI_MIN,
                 pfes->GetComm());
   if (print_level >= 0)
   { cout << "Minimum det(J) = " << min_detJ_all << endl; }

   Vector x_out(x.Size());
   bool x_out_ok = false;
   const double energy_in = nlf->GetEnergy(x_gf);
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

   if (print_level >= 0)
   {
      cout << "Energy decrease: "
           << (energy_in - energy_out) / energy_in * 100.0
           << "% with " << scale << " scaling." << endl;
   }

   if (x_out_ok == false) { return 0.0; }

   return scale;
}

int main (int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Set the method's default parameters.
   const char *mesh_file = "../../data/icf-pert.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   int rp_levels         = 0;
   double jitter         = 0.0;
   int metric_id         = 1;
   int target_id         = 1;
   bool limited          = false;
   double lim_eps        = 1.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int newton_iter       = 10;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = false;
   bool visualization    = true;
   int combomet          = 0;

   // 2. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
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
                  "52 : 0.5(tau-1)^2/(tau-tau_0)       -- 2D untangling\n\t"
                  "55 : (tau-1)^2                      -- 2D size\n\t"
                  "56 : 0.5(sqrt(tau)-1/sqrt(tau))^2   -- 2D size\n\t"
                  "58 : |T^tT|^2/(tau^2)-2*|T|^2/tau+2 -- 2D shape\n\t"
                  "77 : 0.5(tau-1/tau)^2               -- 2D size\n\t"
                  "211: (tau-1)^2-tau+sqrt(tau^2)      -- 2D untangling\n\t"
                  "301: (|T||T^-1|)/3-1              -- 3D shape\n\t"
                  "302: (|T|^2|T^-1|^2)/9-1          -- 3D shape\n\t"
                  "303: (|T|^2)/3*tau^(2/3)-1        -- 3D shape\n\t"
                  "315: (tau-1)^2                    -- 3D size\n\t"
                  "316: 0.5(sqrt(tau)-1/sqrt(tau))^2 -- 3D size\n\t"
                  "321: |T-T^-t|^2                   -- 3D shape+size\n\t"
                  "352: 0.5(tau-1)^2/(tau-tau_0)     -- 3D untangling");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target (ideal element) type:\n\t"
                  "1: IDEAL\n\t"
                  "2: IDEAL_EQ_SIZE\n\t"
                  "3: IDEAL_INIT_SIZE");
   args.AddOption(&limited, "-lim", "--limiting", "-no-lim", "--no-limiting",
                  "Enable limiting of the node movement.");
   args.AddOption(&lim_eps, "-lc", "--limit-const", "Limiting constant.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&newton_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "ODE solver: 0 - l1-Jacobi, 1 - CG, 2 - MINRES.");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&combomet, "-cmb", "--combination-of-metrics",
                  "Metric combination");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // 3. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1,false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   if (myid == 0)
   {
      cout << "Mesh curvature: ";
      if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
      else { cout << "(NONE)"; }
      cout << endl;
   }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   // 4. Define a finite element space on the mesh. Here we use vector finite
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
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim);

   // 5. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   pmesh->SetNodalFESpace(pfespace);

   // 6. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   ParGridFunction x(pfespace);
   pmesh->SetNodalGridFunction(&x);

   // 8. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in pfespace.
   Vector h0(pfespace->GetNDofs());
   h0 = numeric_limits<double>::infinity();
   Array<int> dofs;
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      pfespace->GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), pmesh->GetElementSize(i));
      }
   }

   // 9. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   ParGridFunction rdm(pfespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < pfespace->GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(pfespace->DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < pfespace->GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      pfespace->GetBdrElementVDofs(i, vdofs);
      // Set the boundary values to zero.
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   // Average the perturbation of the overlapping nodes.
   HypreParVector *trueF = rdm.ParallelAverage();
   rdm = *trueF;
   x -= rdm;
   delete trueF;

   // 10. Save the starting (prior to the optimization) mesh to a file. This
   //     output can be viewed later using GLVis: "glvis -m perturbed -np
   //     num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "perturbed." << setfill('0') << setw(6) << myid;
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);
   }

   // 11. Store the starting (prior to the optimization) positions.
   ParGridFunction x0(pfespace);
   x0 = x;

   // 12. Form the integrator that uses the chosen metric and target.
   double tauval = -0.1;
   HyperelasticModel *model = NULL;
   switch (metric_id)
   {
      case 1: model = new TMOPHyperelasticModel001; break;
      case 2: model = new TMOPHyperelasticModel002; break;
      case 7: model = new TMOPHyperelasticModel007; break;
      case 9: model = new TMOPHyperelasticModel009; break;
      case 22: model = new TMOPHyperelasticModel022(tauval); break;
      case 50: model = new TMOPHyperelasticModel050; break;
      case 52: model = new TMOPHyperelasticModel052(tauval); break;
      case 55: model = new TMOPHyperelasticModel055; break;
      case 56: model = new TMOPHyperelasticModel056; break;
      case 58: model = new TMOPHyperelasticModel058; break;
      case 77: model = new TMOPHyperelasticModel077; break;
      case 211: model = new TMOPHyperelasticModel211; break;
      case 301: model = new TMOPHyperelasticModel301; break;
      case 302: model = new TMOPHyperelasticModel302; break;
      case 303: model = new TMOPHyperelasticModel303; break;
      case 315: model = new TMOPHyperelasticModel315; break;
      case 316: model = new TMOPHyperelasticModel316; break;
      case 321: model = new TMOPHyperelasticModel321; break;
      case 352: model = new TMOPHyperelasticModel352(tauval); break;
      default:
         if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 3;
   }
   TargetJacobian *tj = NULL;
   switch (target_id)
   {
      case 1: tj = new TargetJacobian(TargetJacobian::IDEAL); break;
      case 2: tj = new TargetJacobian(TargetJacobian::IDEAL_EQ_SIZE); break;
      case 3: tj = new TargetJacobian(TargetJacobian::IDEAL_INIT_SIZE); break;
      default:
         if (myid == 0) { cout << "Unknown target_id: " << target_id << endl; }
         return 3;
   }
   tj->SetNodes(x);
   tj->SetInitialNodes(x0);
   HyperelasticNLFIntegrator *he_nlf_integ;
   he_nlf_integ = new HyperelasticNLFIntegrator(model, tj);

   // 13. Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = NULL;
   const int geom_type = pfespace->GetFE(0)->GetGeomType();
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order); break;
      case 2: ir = &IntRules.Get(geom_type, quad_order); break;
      case 3: ir = &IntRulesCU.Get(geom_type, quad_order); break;
      default:
         if (myid == 0) { cout << "Unknown quad_type: " << target_id << endl; }
         return 3;
   }
   if (myid == 0)
   { cout << "Quadrature point per cell: " << ir->GetNPoints() << endl; }
   he_nlf_integ->SetIntegrationRule(*ir);

   // 14. Limit the node movement.
   if (limited) { he_nlf_integ->SetLimited(lim_eps, x0); }

   // 15. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights.  Note that there are
   //     no command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   ParNonlinearForm a(pfespace);
   Coefficient *coeff1 = NULL;
   HyperelasticModel *model2 = NULL;
   TargetJacobian *tj2 = NULL;
   FunctionCoefficient coeff2(weight_fun);
   if (combomet == 1)
   {
      // Weight of the original metric.
      coeff1 = new ConstantCoefficient(1.25);
      he_nlf_integ->SetCoefficient(*coeff1);
      a.AddDomainIntegrator(he_nlf_integ);

      model2 = new TMOPHyperelasticModel077;
      tj2    = new TargetJacobian(TargetJacobian::IDEAL_EQ_SIZE);
      tj2->size_scale = 0.005;
      tj2->SetNodes(x);
      tj2->SetInitialNodes(x0);
      HyperelasticNLFIntegrator *he_nlf_integ2;
      he_nlf_integ2 = new HyperelasticNLFIntegrator(model2, tj2);
      he_nlf_integ2->SetIntegrationRule(*ir);

      // Weight of metric2.
      he_nlf_integ2->SetCoefficient(coeff2);
      a.AddDomainIntegrator(he_nlf_integ2);
   }
   else { a.AddDomainIntegrator(he_nlf_integ); }
   const double init_en = a.GetEnergy(x);
   if (myid == 0) { cout << "Initial strain energy: " << init_en << endl; }

   // 16. Visualize the starting mesh and metric values.
   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_metric(mesh_poly_deg, *model, *tj, *pmesh, title, 0);
   }

   // 17. Fix all boundary nodes, or fix only a given component depending on the
   //     boundary attributes of the given mesh.  Attributes 1/2/3 correspond to
   //     fixed x/y/z components of the node.  Attribute 4 corresponds to an
   //     entirely fixed node.  Other boundary attributes do not affect the node
   //     movement boundary conditions.
   if (move_bnd == false)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      const int nd  = pfespace->GetBE(0)->GetDof();
      int n = 0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n), vdofs;
      n = 0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         pfespace->GetBdrElementVDofs(i, vdofs);
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

   // 18. As we use the Newton method to solve the resulting nonlinear system,
   //     here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL;
   const double rtol  = 1e-12;
   if (lin_solver == 0)
   {
      S = new DSmoother(1, 1.0, max_lin_iter);
   }
   else if (lin_solver == 1)
   {
      CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(3);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver(MPI_COMM_WORLD);
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(rtol);
      minres->SetAbsTol(0.0);
      minres->SetPrintLevel(3);
      S = minres;
   }

   // 19. Compute the minimum det(J) of the starting mesh.
   tauval = numeric_limits<double>::infinity();
   const int NE = pmesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      ElementTransformation *transf = pmesh->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir->IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   double minJ0;
   MPI_Allreduce(&tauval, &minJ0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   tauval = minJ0;
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << tauval << endl; }

   // 20. Finally, perform the nonlinear optimization.
   NewtonSolver *newton = NULL;
   if (tauval > 0.0)
   {
      tauval = 0.0;
      newton = new RelaxedNewtonSolver(*ir, MPI_COMM_WORLD);
      if (myid == 0)
      { cout << "RelaxedNewtonSolver is used (as all det(J) > 0)." << endl; }
   }
   else
   {
      if ( (dim == 2 && metric_id != 22 && metric_id != 52) ||
           (dim == 3 && metric_id != 352) )
      {
         if (myid == 0)
         { cout << "The mesh is inverted. Use an untangling metric." << endl; }
         return 3;
      }
      double h0min = h0.Min(), h0min_all;
      MPI_Allreduce(&h0min, &h0min_all, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      tauval -= 0.01 * h0min_all; // Slightly below minJ0 to avoid div by 0.
      newton = new DescentNewtonSolver(*ir, MPI_COMM_WORLD);
      if (myid == 0)
      { cout << "DescentNewtonSolver is used (as some det(J) < 0)." << endl; }
   }
   newton->SetPreconditioner(*S);
   newton->SetMaxIter(newton_iter);
   newton->SetRelTol(rtol);
   newton->SetAbsTol(0.0);
   newton->SetPrintLevel(1);
   newton->SetOperator(a);
   Vector X(pfespace->TrueVSize());
   pfespace->GetRestrictionMatrix()->Mult(x, X);
   newton->Mult(b, X);
   if (myid == 0 && newton->GetConverged() == false)
   { cout << "NewtonIteration: rtol = " << rtol << " not achieved." << endl; }
   pfespace->Dof_TrueDof_Matrix()->Mult(X, x);
   delete newton;

   // 21. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized -np num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "optimized." << setfill('0') << setw(6) << myid;
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);
   }

   // 22. Compute the amount of energy decrease.
   const double fin_en = a.GetEnergy(x);
   if (myid == 0)
   {
      cout << "Final strain energy : " << fin_en << endl;
      cout << "The strain energy decreased by: " << setprecision(12)
           << (init_en - fin_en) * 100.0 / init_en << " %." << endl;
   }

   // 23. Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_metric(mesh_poly_deg, *model, *tj, *pmesh, title, 600);
   }

   // 23. Visualize the mesh displacement.
   if (visualization)
   {
      x0 -= x;
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      pmesh->PrintAsOne(sock);
      x0.SaveAsOne(sock);
      sock.send();
      sock << "window_title 'Displacements'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys JRmcl" << endl;
   }

   // 24. Free the used memory.
   delete S;
   delete model2;
   delete coeff1;
   delete model;
   delete pfespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();
   return 0;
}

// Defined with respect to the icf-pert mesh.
double weight_fun(const Vector &x)
{
   const double r = sqrt(x(0)*x(0) + x(1)*x(1) + 1e-12);
   const double den = 0.002;
   double l2 = 0.2 + 0.5*std::tanh((r-0.16)/den) - 0.5*std::tanh((r-0.17)/den)
               + 0.5*std::tanh((r-0.23)/den) - 0.5*std::tanh((r-0.24)/den);
   return l2;
}
