class CGOSolver : public IterativeSolver
{

private:
   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;
   ParFiniteElementSpace *pfes;
   mutable ParGridFunction x_gf;

   // GridFunction that has the latest mesh positions.
   ParGridFunction &mesh_nodes;

   // Advection related.
   ParGridFunction *x0, *ind0, *ind;
   AdvectorCG *advector;

   // Quadrature points that are checked for negative Jacobians etc.
//   const IntegrationRule &ir;
//   ParFiniteElementSpace *pfes;
//   mutable ParGridFunction x_gf;

protected:
   mutable Vector r, c, s;

public:
   CGOSolver(const IntegrationRule &irule,
                       ParFiniteElementSpace *pf, ParGridFunction &mn,
                       ParGridFunction *x0_, ParGridFunction *ind0_,
                       ParGridFunction *ind_, AdvectorCG *adv)
      : NewtonSolver(pf->GetComm()), ir(irule), pfes(pf), mesh_nodes(mn),
        x0(x0_), ind0(ind0_), ind(ind_), advector(adv) { }
//   CGOSolver(const IntegrationRule &irule, ParFiniteElementSpace *pf)
//      : IterativeSolver(pf->GetComm()),ir(irule), pfes(pf) { }

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

   virtual void ProcessNewState(const Vector &x) const
   {
      mesh_nodes.Distribute(x);

      if (x0 && ind0 && ind && advector)
      {
         // GridFunction with the current positions.
         Vector x_copy(x);
         x_gf.MakeTRef(pfes, x_copy, 0);

         // Reset the indicator to its values on the initial positions.
         *ind = *ind0;

         // Advect the indicator from the original to the new posiions.
         advector->Advect(*x0, x_gf, *ind);
      }
   }
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
   double norm0, norm, norm_goal, beta, num,den,num_all,den_all;
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

      if (it % 50 == 0)
      { 
        oper->Mult(x, r); // r = b-Ax
        if (have_b)
        {  
         r -= b;
        }
        c = r;
      }

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
      add(r,beta,c,c); // c = r + beta(c)

      norm = Norm(r);
   }

   final_iter = it;
   final_norm = norm;
}

double CGOSolver::ComputeScalingFactor(const Vector &x,
                                                 const Vector &b) const
{
   static double scalesav;
   const ParNonlinearForm *nlf = dynamic_cast<const ParNonlinearForm *>(oper);
   MFEM_VERIFY(nlf != NULL, "invalid Operator subclass");
   const bool have_b = (b.Size() == Height());

   const int NE = pfes->GetParMesh()->GetNE(), dim = pfes->GetFE(0)->GetDim(),
             dof = pfes->GetFE(0)->GetDof(), nsp = ir.GetNPoints();
   Array<int> xdofs(dof * dim);
   DenseMatrix Jpr(dim), dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);

   Vector x_out(x.Size());
   bool x_out_ok = false;
   const double energy_in = nlf->GetEnergy(x);
   double scale = 1.0, energy_out;
   double norm0 = Norm(r);
   x_gf.MakeTRef(pfes, x_out, 0);
   double redfac = 0.5;

   if (scalesav==0.)
    {
     scale = 1.;
    }
   else
    {
     scale = scalesav/redfac;
    }
     scale = 1.;

   // Decreases the scaling of the update until the new mesh is valid.
   for (int i = 0; i < 20; i++)
   {
      add(x, -scale, c, x_out);
      x_gf.SetFromTrueVector();
      ProcessNewState(x_out);

      energy_out = nlf->GetParGridFunctionEnergy(x_gf);
      if (energy_out > 1.2*energy_in || isnan(energy_out) != 0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Increasing energy." << endl; }
         scale *= 0.1; continue;
      }

      int jac_ok = 1;
      for (int i = 0; i < NE; i++)
      {
         pfes->GetElementVDofs(i, xdofs);
         x_gf.GetSubVector(xdofs, posV);
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

      if (jac_ok_all == 0)
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
      x_out_ok = true;
   }

   if (print_level >= 0)
   {
      cout << "Energy decrease: "
           << (energy_in - energy_out) / energy_in * 100.0
           << "% with " << scale << " scaling. " << energy_in << " " << energy_out <<  endl;
   }

   if (x_out_ok == false) { scale = 0.0; }

   scalesav = scale;
   return scale;
}
// Done CG Solver

