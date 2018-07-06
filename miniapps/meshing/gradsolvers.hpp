// BEGIN CGO SOLVER
//-------------- Begin CGO Solver
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
   mutable Vector r, c, sk, r2, dfdxo, yk;

public:
   CGOSolver(const IntegrationRule &irule,
                       ParFiniteElementSpace *pf, ParGridFunction &mn,
                       ParGridFunction *x0_, ParGridFunction *ind0_,
                       ParGridFunction *ind_, AdvectorCG *adv)
      : IterativeSolver(pf->GetComm()), ir(irule), pfes(pf), mesh_nodes(mn),
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
   dfdxo.SetSize(width);
   r2.SetSize(width);
   yk.SetSize(width);
   sk.SetSize(width);
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
   dfdxo = r; 

   norm0 = norm = Norm(r);
   norm_goal = std::max(rel_tol*norm, abs_tol);

   prec->iterative_mode = false;

   for (it = 0; true; it++)
   {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_level >= 0)
      {
         mfem::out << "CG iteration " << setw(2) << it
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
      
      r2 = r;    // Fletcher & Reeves has r2 = r;
//      r2 -= dfdxo; // r2 = r-dfdxo - uncomment this for Polak and Ribiere flavor
//    In a simple test on Blade, FR performed better than PR.

//    CG Opt - Fletcher&Reeves
/*
      num = Dot(r,r2); // g_{k+1}^T g_{k+1}
      den = Dot(dfdxo,dfdxo); // g_k ^T g_k 
      beta = num/den; //
      if (it % 10 == 0) beta = 0.; // restart CG
      if (print_level >= 0) {cout << "Fletcher & Reeves CG being used\n";}
*/
//
//    CG opt - Hager and Zhang
///*
      r2 = r;
      r2 -= dfdxo;
      yk = r2;

      double c1 = Dot(r2,r2)/Dot(c,r2);
      dfdxo = c; //temporarily overwrite dfdxo
      dfdxo *= 2.*c1;
      double c2 = Dot(c,r2);
      r2 -= dfdxo; //now overwrite r2 with useful stuff
      c1 = Dot(r2,r);
      beta = -c1/c2;

      sk = c; sk *= -c_scale; //sk
      double alpha1 = Dot(sk,yk);
      double e2t2 = 0.1*abs(Dot(sk,dfdxo)); //this gives 76
      if (it % 50) beta = 0.; // restart CG
      if (print_level >= 0) {cout << it << "  " << alpha1 << " " << e2t2 << " " << alpha1/e2t2 << " k10c\n";}
      if (print_level >= 0) {cout << "Hager & Zhang CG being used\n";}
//    HZ does better for some cases and FR does better for others.. not sure
//    what to do
//*/
//    END CG - HZ


      add(r,beta,c,c); // c = r + beta(c)
      dfdxo = r; 
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
      x_gf.MakeTRef(pfes, x_out, 0);
      x_gf.SetFromTrueVector();
      energy_out = nlf->GetEnergy(x_out);


      if (energy_out < 2.0*energy_in)
      {
        ProcessNewState(x_out);
        energy_out = nlf->GetEnergy(x_out);
      }
      else
      {
         if (print_level >= 0)
         {
            cout << "Scale = " << scale << " " << energy_in << " " << energy_out <<  " Increasing energy before remap." << endl;
         }
         scale *= redfac;
         continue;
      }

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
// BEGIN GDO SOLVER
//-------------- Begin GDO Solver
class GDOSolver : public IterativeSolver
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
   GDOSolver(const IntegrationRule &irule,
                       ParFiniteElementSpace *pf, ParGridFunction &mn,
                       ParGridFunction *x0_, ParGridFunction *ind0_,
                       ParGridFunction *ind_, AdvectorCG *adv)
      : IterativeSolver(pf->GetComm()), ir(irule), pfes(pf), mesh_nodes(mn),
        x0(x0_), ind0(ind0_), ind(ind_), advector(adv) { }
//   GDOSolver(const IntegrationRule &irule, ParFiniteElementSpace *pf)
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

void GDOSolver::SetOperator(const Operator &op)
{
   oper = &op;
   height = op.Height();
   width = op.Width();
   MFEM_ASSERT(height == width, "square Operator is required.");

   r.SetSize(width);
   c.SetSize(width);
}

void GDOSolver::Mult(const Vector &b, Vector &x) const
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
         mfem::out << "GD iteration " << setw(2) << it
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

      const double c_scale = ComputeScalingFactor(x, b);
      if (c_scale == 0.0)
      {
         converged = 0;
         break;
      }
      add(x, -c_scale, c, x);

      ProcessNewState(x); 

      oper->Mult(x, r);
      if (have_b)
      {
         r -= b;
      }

      c = r;
      norm = Norm(r);
   }

   final_iter = it;
   final_norm = norm;
}

double GDOSolver::ComputeScalingFactor(const Vector &x,
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
   double redfac = 0.1;

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
      x_gf.MakeTRef(pfes, x_out, 0);
      x_gf.SetFromTrueVector();
      energy_out = nlf->GetEnergy(x_out);


      if (energy_out < 1.2*energy_in)
      {
        ProcessNewState(x_out);
        energy_out = nlf->GetEnergy(x_out);
      }
      else
      {
         if (print_level >= 0)
         {
            cout << "Scale = " << scale << " " << energy_in << " " << energy_out <<  " Increasing energy before remap." << endl;
         }
         scale *= redfac;
         continue;
      }

      if (energy_out > 1.2*energy_in || isnan(energy_out) != 0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Increasing energy." << endl; }
         scale *= 0.1; continue;
      }

      x_gf.SetFromTrueVector();
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

// BFGS Stuff
//-------------- Begin BFGS Solver
class BFGSSolver : public IterativeSolver
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

protected:
   mutable Vector r, c, s, yk, sk, rksav,  ykA1;;
   mutable DenseMatrix A1,A2,A1A3,A4A1,Af;
public:
   BFGSSolver(const IntegrationRule &irule,
                       ParFiniteElementSpace *pf, ParGridFunction &mn,
                       ParGridFunction *x0_, ParGridFunction *ind0_,
                       ParGridFunction *ind_, AdvectorCG *adv)
      : IterativeSolver(pf->GetComm()), ir(irule), pfes(pf), mesh_nodes(mn),
        x0(x0_), ind0(ind0_), ind(ind_), advector(adv) { }

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

void BFGSSolver::SetOperator(const Operator &op)
{
   oper = &op;
   height = op.Height();
   width = op.Width();
   MFEM_ASSERT(height == width, "square Operator is required.");

   r.SetSize(width);
   c.SetSize(width);
   yk.SetSize(width);
   sk.SetSize(width);
   rksav.SetSize(width);
   ykA1.SetSize(width);

   A1.Diag(1.,width);
   Af.Diag(1.,width);
//   A2.SetSize(width,width);
//   A1A3.SetSize(width,width);
//   A4A1.SetSize(width,width);
}

void BFGSSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

   int it;
   double norm0, norm, norm_goal, beta, num,den,num_all,den_all;
   const bool have_b = (b.Size() == Height());
   double alpha1,alpha2,alpha3,const1,const2,e2t2;
// element by element bfgs
   Array<int> vdofs;
   const FiniteElement *fe;
   ElementTransformation *T;
   Mesh *mesh = pfes->GetMesh();
   fe = pfes->GetFE(0);
   int dof = fe->GetDof(), dim = fe->GetDim(), NE = pfes->GetNE();
   Vector csav;
//

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
   rksav = r;


   norm0 = norm = Norm(r);
   norm_goal = std::max(rel_tol*norm, abs_tol);

   prec->iterative_mode = false;

   for (it = 0; true; it++)
   {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_level >= 0)
      {
         mfem::out << "BFGS iteration " << setw(2) << it
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

      rksav = r;

      const double c_scale = ComputeScalingFactor(x, b);
      if (c_scale == 0.0)
      {
         converged = 0;
         break;
      }
      add(x, -c_scale, c, x);

      ProcessNewState(x); 

      oper->Mult(x, r);
      if (have_b)
      {
         r -= b;
      }
//    Regular - BFGS
///*
       subtract(r,rksav,yk);   //yk  
       sk = c; sk *= -c_scale; //sk

       alpha1 = Dot(sk,yk);
       alpha2 = Dot(yk,yk);
//     e2t2 = abs(Dot(sk,rksav)); // option 1
//     e2t2 = 0.1*Dot(c,rksav);
       e2t2 = 1.;                 //option 2
       double sc = (1./alpha1)*0.5*(1-std::tanh(10*(alpha1-1.5)));

       Af.Diag(sc,x.Size());A1+=Af;

       A1.Mult(yk,ykA1);

       alpha3 = Dot(yk,ykA1);
       const1 = (alpha1+alpha3)/(alpha1*alpha1);
       const2 = -1./alpha1;

       if (print_level >= 0)
      { 
       cout << alpha1 << " " << alpha3 << " " << e2t2 <<  " alphas \n";
      }
       
      for (int ii = 0; ii<A1.Height(); ii++)
      {
       for (int jj = 0; jj<A1.Width(); jj++)
       {
        A1(ii,jj) += const1*sk(ii)*sk(jj) + const2*(ykA1(ii)*sk(jj) + sk(ii)*ykA1(jj));
       }
      }
       
       A1.Mult(r,c);
//*/
//    Regular - BFGS - End
//    Another flavor
//    DFP stuff
/*
       subtract(r,rksav,yk);   //yk  
       sk = c; sk *= -c_scale; //sk

       if (alpha1 < 2.) {A1.Diag(1.,x.Size());}
       A1.MultTranspose(yk,ykA1);
       alpha1 = Dot(sk,yk);
       alpha2 = Dot(ykA1,yk);

       const1 = 1./alpha1;
       const2 = -1./alpha2;

       MultVVt(sk,A2);
       A2 *= const1;
    
       MultVVt(ykA1,A1A3);
       A1A3 *= const2;

       A1 += A2;
       A1 += A1A3;
       A1.Mult(r,c);
*/
//    End DFP stuff

      norm = Norm(r);
   }

   final_iter = it;
   final_norm = norm;
}

double BFGSSolver::ComputeScalingFactor(const Vector &x,
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
   Vector rksav(x.Size());
   Vector drk(x.Size());
   Vector sks(x.Size());
   bool x_out_ok = false;
   const double energy_in = nlf->GetEnergy(x);
   double scale = 1.0, energy_out;
   double norm0 = Norm(r);
   double redfac = 0.2;
   double alphasav;

   rksav = r;

   if (scalesav==0.)
    {
     scale = 1.;
    }
   else
    {
     scale = scalesav/(redfac*redfac);
    }
    scale = 1.;

   // Decreases the scaling of the update until the new mesh is valid.
   for (int i = 0; i < 20; i++)
   {
      add(x, -scale, c, x_out);
      x_gf.MakeTRef(pfes, x_out, 0);
      x_gf.SetFromTrueVector();
      energy_out = nlf->GetEnergy(x_out);

      oper->Mult(x_out, r);
      subtract(r,rksav,drk);
      sks = c; sks *= -scale;
      double curvat = Dot(sks,drk); //curvature check

      if (energy_out < 1.2*energy_in)
      {
        ProcessNewState(x_out);
        energy_out = nlf->GetEnergy(x_out);
      }
      else
      {
         if (print_level >= 0)
         {
            cout << "Scale = " << scale << " " << energy_in << " " << energy_out <<  " Increasing energy before remap." << endl;
         }
         scale *= redfac;
         continue;
      }

      if (energy_out > 1.2*energy_in || isnan(energy_out) != 0 || curvat < 0.)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Increasing energy." << endl; }
         scale *= redfac; continue;
      }

      x_gf.SetFromTrueVector();
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
         scale *= redfac; continue;
      }

      if (have_b) { r -= b; }
      double norm = Norm(r);

      if (norm > 1.2*norm0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Norm increased." << endl; }
         scale *= redfac; continue;
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
// End BFGS Stuff
// L-BFGS solver
class LBFGSSolver : public IterativeSolver
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

   //

protected:
   mutable Vector r, c, s, yk, sk, rksav,  ykA1, skt, ykt, rhov, qv, alphav;
   mutable DenseMatrix skM, ykM;
   int lbm;
public:
   LBFGSSolver(const IntegrationRule &irule,
                       ParFiniteElementSpace *pf, ParGridFunction &mn,
                       ParGridFunction *x0_, ParGridFunction *ind0_,
                       ParGridFunction *ind_, AdvectorCG *adv)
      : IterativeSolver(pf->GetComm()), ir(irule), pfes(pf), mesh_nodes(mn),
        x0(x0_), ind0(ind0_), ind(ind_), advector(adv) { }

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

void LBFGSSolver::SetOperator(const Operator &op)
{
   oper = &op;
   height = op.Height();
   width = op.Width();
   MFEM_ASSERT(height == width, "square Operator is required.");

   r.SetSize(width);
   c.SetSize(width);
   yk.SetSize(width);
   sk.SetSize(width);
   rksav.SetSize(width);
   ykA1.SetSize(width);
   skt.SetSize(width);
   ykt.SetSize(width);
   qv.SetSize(width);

   lbm = 10;
   skM.SetSize(width,lbm);
   ykM.SetSize(width,lbm);
   rhov.SetSize(lbm);
   alphav.SetSize(lbm);
}

void LBFGSSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

   int it;
   double norm0, norm, norm_goal, beta, num,den,num_all,den_all;
   const bool have_b = (b.Size() == Height());
   double alpha1,alpha2,alpha3,const1,const2,e2t2;
   const FiniteElement *fe;
   ElementTransformation *T;
   Mesh *mesh = pfes->GetMesh();
   fe = pfes->GetFE(0);
   int dof = fe->GetDof(), dim = fe->GetDim(), NE = pfes->GetNE();
   Vector csav;
/*
   Vector cntb, cnta;
   cnta.SetSize(x.Size());
   cnta = 0.;
   Array<int> vdofs;
   for (int i = 0; i < pfes->GetNE(); i++)
   {
      fe = pfes->GetFE(i);
      pfes->GetElementVDofs(i, vdofs);
      cnta.GetSubVector(vdofs, cntb);
      cntb = 1.;
      cnta.AddElementVector(vdofs, cntb);
      vdofs.Print();
   }

   double sumv = 0.;
   for (int i = 0; i < cnta.Size(); i++)
   {  
      cnta(i) = 1./cnta(i);
     sumv = sumv + cnta(i);
   }


   MPI_Barrier(pfes->GetComm());
   const int n_shared_faces = pfes->GetParMesh()->GetNSharedFaces();
   cout << n_shared_faces << " k10nshard\n";

   const FiniteElement *fe1, *fe2;
   FaceElementTransformations *tr;
   ParMesh *pmesh = pfes->GetParMesh();
   for (int i = 0; i < n_shared_faces; i++)
    {
      cout << i << " do shared faces\n";
      tr = pmesh->GetSharedFaceTransformations(i);
      cout << i << " do 1\n";
      fe1 = pfes->GetFE(tr->Elem1No);
      cout << i << " do 2\n";
      fe2 = pfes->GetFaceNbrFE(tr->Elem2No);
      cout << fe1 << " " << fe2 << " shared number\n";
    }

   cout << sumv << " say say\n";
   MPI_Barrier(pfes->GetComm());
   if (print_level>0) {cnta.Print();}
   if (print_level>0) {cout << cnta.Size() << " cntasize\n";}
   MPI_Barrier(pfes->GetComm());
   if (print_level<0) {cnta.Print();}
   if (print_level<0) {cout << cnta.Size() << " cntasize\n";}
   MPI_Barrier(pfes->GetComm());
   MFEM_ABORT("aborting");
*/


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
   rksav = r;


   norm0 = norm = Norm(r);
   norm_goal = std::max(rel_tol*norm, abs_tol);

   prec->iterative_mode = false;

   for (it = 0; true; it++)
   {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_level >= 0)
      {
         mfem::out << "LBFGS iteration " << setw(2) << it
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

      rksav = r;

      const double c_scale = ComputeScalingFactor(x, b);
      if (c_scale == 0.0)
      {
         converged = 0;
         break;
      }
      add(x, -c_scale, c, x);

      ProcessNewState(x); 

      oper->Mult(x, r);
      if (have_b)
      {
         r -= b;
      }

//    start machinery
       int k = it;
       int klim;
       subtract(r,rksav,yk);   //yk  
       sk = c; sk *= -c_scale; //sk
//       elp(sk,cnta,skt);
//       elp(yk,cnta,ykt);
       double hd = Dot(sk,yk)/Dot(yk,yk);
//       double hd = Dot(skt,yk)/Dot(ykt,yk);

       if (k<lbm) 
       { 
        skM.SetCol(k,sk);
        ykM.SetCol(k,yk);
        klim = k+1;
       }
       else
       { 
        for (int i = 0; i < lbm-1; i++)
        {
         skM.GetColumn(i+1, skt);
         ykM.GetColumn(i+1, ykt);
         skM.SetCol(i,skt);
         ykM.SetCol(i,ykt);
        }
        skM.SetCol(lbm-1,sk);
        ykM.SetCol(lbm-1,yk);
        klim = lbm;
       }
       
////    Now skM, ykM has the last k vectors saved
////    get the search direction now

      for (int i = 0; i < klim; i++)
        {
          skM.GetColumn(i,skt);
          ykM.GetColumn(i,ykt);
//          elp(skt,cnta,sk);
//          rhov(i) = 1./Dot(sk,ykt);
          rhov(i) = 1./Dot(skt,ykt);
        }

      qv = r;
      for (int i = klim-1; i > -1; i--)
        {
         skM.GetColumn(i,skt);
         ykM.GetColumn(i,ykt);
//          elp(skt,cnta,sk);
//         alphav(i) = rhov(i)*Dot(sk,qv);
         alphav(i) = rhov(i)*Dot(skt,qv);
         add(qv, -alphav(i), ykt, qv);
        }

        qv *= hd;   // scale search direction
      for (int i = 0; i < klim ; i++)
        {
         skM.GetColumn(i,skt);
         ykM.GetColumn(i,ykt);
//          elp(ykt,cnta,yk);
//         double betai = rhov(i)*Dot(yk,qv);
         double betai = rhov(i)*Dot(ykt,qv);
         add(qv, alphav(i)-betai, skt, qv);
        }

        c = qv;
        norm = Norm(r);
   }

   final_iter = it;
   final_norm = norm;
}

double LBFGSSolver::ComputeScalingFactor(const Vector &x,
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
   Vector rksav(x.Size());
   Vector drk(x.Size());
   Vector sks(x.Size());
   bool x_out_ok = false;
   const double energy_in = nlf->GetEnergy(x);
   double scale = 1.0, energy_out;
   double norm0 = Norm(r);
   double redfac = 0.5;
   double alphasav;

   rksav = r;

   if (scalesav==0.)
    {
     scale = 1.;
    }
   else
    {
     scale = scalesav/(redfac*redfac);
    }
    scale = 1.;

   // Decreases the scaling of the update until the new mesh is valid.
   for (int i = 0; i < 20; i++)
   {
      add(x, -scale, c, x_out);
      x_gf.MakeTRef(pfes, x_out, 0);
      x_gf.SetFromTrueVector();
      energy_out = nlf->GetEnergy(x_out);

      oper->Mult(x_out, r);
      subtract(r,rksav,drk);
      sks = c; sks *= -scale;
      double curvat = Dot(sks,drk); //curvature check

      if (energy_out < 1.2*energy_in)
      {
        ProcessNewState(x_out);
        energy_out = nlf->GetEnergy(x_out);
      }
      else
      {
         if (print_level >= 0)
         {
            cout << "Scale = " << scale << " " << energy_in << " " << energy_out <<  " Increasing energy before remap." << endl;
         }
         scale *= redfac;
         continue;
      }

      if (energy_out > 1.2*energy_in || isnan(energy_out) != 0 || curvat < 0.)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Increasing energy." << endl; }
         scale *= redfac; continue;
      }

      x_gf.SetFromTrueVector();
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
         scale *= redfac; continue;
      }

      if (have_b) { r -= b; }
      double norm = Norm(r);

      if (norm > 1.2*norm0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Norm increased." << endl; }
         scale *= redfac; continue;
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
// End L-BFGS

// Newtons method
class RelaxedNewtonSolver : public NewtonSolver
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

//   static double scalesav;
public:
   RelaxedNewtonSolver(const IntegrationRule &irule,
                       ParFiniteElementSpace *pf, ParGridFunction &mn,
                       ParGridFunction *x0_, ParGridFunction *ind0_,
                       ParGridFunction *ind_, AdvectorCG *adv)
      : NewtonSolver(pf->GetComm()), ir(irule), pfes(pf), mesh_nodes(mn),
        x0(x0_), ind0(ind0_), ind(ind_), advector(adv) { }

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

double RelaxedNewtonSolver::ComputeScalingFactor(const Vector &x,
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
   Vector x_outb(x.Size());
   Vector dx_out(x.Size());
   Vector csav(x.Size());
   Vector r1(x.Size());
   bool x_out_ok = false;
   double energy_dum;
   const double energy_in = nlf->GetEnergy(x);
   double scale, energy_out, energy_outd;
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
   csav = c;
   double norm0 = Norm(r);
   x_gf.MakeTRef(pfes, x_out, 0);

// Wolfe condition
   oper->Mult(x, r1);
   double prodr = Dot(r1,c);
   prodr *= 0.5; // c2 in wolfe
   prodr += energy_in; //f(xk) + c2 * dot(gradf(xk),pk) 

   // Decreases the scaling of the update until the new mesh is valid.
   for (int i = 0; i < 20; i++)
   {
      add(x, -scale, c, x_out);
      x_gf.MakeTRef(pfes, x_out, 0);
      x_gf.SetFromTrueVector();
      energy_out = nlf->GetEnergy(x_out);


      if (energy_out < 1.2*energy_in)
      {
        if (print_level >= 0){cout << "about to process new state" << endl;}
        ProcessNewState(x_out);
        if (print_level >= 0){cout << "done process new state" << endl;}
        energy_out = nlf->GetEnergy(x_out);
      } 
      else
      {
         if (print_level >= 0)
         { 
            cout << "Scale = " << scale << " " << energy_in << " " << energy_out <<  " Increasing energy before remap." << endl; 
         }
         scale *= redfac;
         continue;
      }


      if (energy_out > 1.2*energy_in || isnan(energy_out) != 0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Increasing energy." << endl; }
         scale *= redfac; continue;
      }

      x_gf.SetFromTrueVector();
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
         scale *= redfac; continue;
      }

      oper->Mult(x_out, r);
      if (have_b) { r -= b; }
      double prodl = Dot(r,c);
      double norm = Norm(r);

      if (norm > 1.2*norm0) // Norm condition
//       if (prodl < prodr) // Wolfe condition 2 - Choose either this or norm condition
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Norm increased." << endl; }
         scale *= redfac; 
         continue;
      }
      else { x_out_ok = true; break; }
   }

   double per_imp = (energy_in - energy_out) / energy_in * 100.0;
//   if (per_imp < 1.e-6) { x_out_ok = false;}

   if (print_level >= 0)
   {
      cout << "Energy decrease: "
           << (energy_in - energy_out) / energy_in * 100.0
           << "% with " << scale << " scaling. " << energy_in << " " << energy_out << endl;
   }

   if (x_out_ok == false) { scale = 0.0; }

   scalesav = scale;
   return scale;
}

// Allows negative Jacobians. Used in untangling metrics.
class DescentNewtonSolver : public NewtonSolver
{
private:
   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;
   ParFiniteElementSpace *pfes;
   mutable ParGridFunction x_gf;
   mutable ParGridFunction x_gff;
   double &tauval;

public:
   DescentNewtonSolver(const IntegrationRule &irule, ParFiniteElementSpace *pf, double &tauval)
      : NewtonSolver(pf->GetComm()), ir(irule), pfes(pf), tauval(tauval) { }

   virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const;
};

double DescentNewtonSolver::ComputeScalingFactor(const Vector &x,
                                                 const Vector &b) const
{
   const ParNonlinearForm *nlf = dynamic_cast<const ParNonlinearForm *>(oper);
   MFEM_VERIFY(nlf != NULL, "invalid Operator subclass");

   const int NE = pfes->GetParMesh()->GetNE(), dim = pfes->GetFE(0)->GetDim(),
             dof = pfes->GetFE(0)->GetDof(), nsp = ir.GetNPoints();
   Array<int> xdofs(dof * dim);
   DenseMatrix Jpr(dim), dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);

   Vector x_out(x.Size());
   x_gf.MakeTRef(pfes, x.GetData());
   x_gf.SetFromTrueVector();
   x_out = x;
   x_gff.MakeTRef(pfes, x_out.GetData());
   x_gff.SetFromTrueVector();

   double min_detJ = infinity();
   double tauvaln;
   int ninvo = 0;
   for (int i = 0; i < NE; i++)
   {
      pfes->GetElementVDofs(i, xdofs);
      x_gf.GetSubVector(xdofs, posV);
      int tchk = 0;
      for (int j = 0; j < nsp; j++)
      {
         pfes->GetFE(i)->CalcDShape(ir.IntPoint(j), dshape);
         MultAtB(pos, dshape, Jpr);
         if (Jpr.Det() < 0) tchk = 1;
         min_detJ = min(min_detJ, Jpr.Det());
      }
      if (tchk==1) ninvo += 1;
   }
   double min_detJ_all;
   MPI_Allreduce(&min_detJ, &min_detJ_all, 1, MPI_DOUBLE, MPI_MIN,
                 pfes->GetComm());
   tauval = min_detJ_all;
   double tauvals = tauval;
   if (tauval > 1.e-3) {tauval = 1.e-4;}
   else
   {tauval = min_detJ_all - 0.1;}
   if (print_level >= 0)
   { cout << "Minimum det(J) = " << min_detJ_all << endl; }
   int gninvo;
   MPI_Allreduce(&ninvo, &gninvo, 1, MPI_INT, MPI_SUM,
                 pfes->GetComm());
 

   bool x_out_ok = false;
   const double energy_in = nlf->GetParGridFunctionEnergy(x_gf);
   double scale = 1.0, energy_out;

   for (int i = 0; i < 15; i++)
   {
      add(x, -scale, c, x_out);

      energy_out = nlf->GetEnergy(x_out);
      if (energy_out > 1.1*energy_in || isnan(energy_out) != 0)
      {
        cout << i << " " << scale << " " << energy_in << " " << energy_out <<  " energy did not reduce\n";
         scale *= 0.1;continue;
      }

      min_detJ = infinity();
      x_gff.SetFromTrueVector();
      int ninvn = 0;
      for (int ii = 0; ii < NE; ii++)
      {
         pfes->GetElementVDofs(ii, xdofs);
         x_gff.GetSubVector(xdofs, posV);
         int tchk = 0;
         for (int j = 0; j < nsp; j++)
         {
            pfes->GetFE(ii)->CalcDShape(ir.IntPoint(j), dshape);
            MultAtB(pos, dshape, Jpr);
            if (Jpr.Det() < 0) tchk = 1;
            min_detJ = min(min_detJ, Jpr.Det());
         }
         if (tchk==1) ninvn += 1;
      }
      double min_detJ_all;
      MPI_Allreduce(&min_detJ, &min_detJ_all, 1, MPI_DOUBLE, MPI_MIN,
                 pfes->GetComm());
      tauvaln = min_detJ_all;
      int gninvn;
      MPI_Allreduce(&ninvn, &gninvn, 1, MPI_INT, MPI_SUM,
                 pfes->GetComm());

      if (print_level >= 0) {cout << i << " " << scale << " " << tauvals << " " << tauvaln << " " << gninvn << " " << gninvo <<  " " << energy_in << " " << energy_out <<  " k10\n";}
      if (tauvaln < 0 && tauvaln < 1.0*tauvals) //|| gninvn > gninvo) )
      { 
       scale *= 0.1;
      }
      else { x_out_ok = true; break; }
   }

   if (print_level >= 0)
   {
      cout << "Energy decrease: "
           << (energy_in - energy_out) / energy_in * 100.0
           << "% with " << scale << " scaling." << energy_out << endl;
   }

   if (x_out_ok == false) { return 0.0; }

   return scale;
}
