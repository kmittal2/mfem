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

#include "mfem.hpp"
#include "catch.hpp"
#include <cmath>

using namespace mfem;

TEST_CASE("Second order ODE methods",
          "[ODE2]")
{
   double tol = 0.1;

   /** Class for simple linear second order ODE.
    *
    *    du2/dt^2 + b du/dt  + a u = 0
    *
    */
   class ODE2 : public SecondOrderTimeDependentOperator
   {
   protected:
      double a,b;

   public:
      ODE2(double a_, double b_) :  SecondOrderTimeDependentOperator(1, 0.0),  a(a_), b(b_) {};

      virtual void Mult(const Vector &u, const Vector &dudt,
                              Vector &d2udt2)  const
      {
         d2udt2[0] = -a*u[0] - b*dudt[0];
      }

      virtual void ImplicitSolve(const double fac0, const double fac1,
                                 const Vector &u, const Vector &dudt, Vector &d2udt2)
      {
         double T = 1.0 + a*fac0 + fac1*b;
         d2udt2[0] = (-a*u[0] - b*dudt[0])/T;
      }

      virtual ~ODE2() {};
   };

   /** Class for checking order of convergence of first order ODE.
    */
   class CheckODE2
   {
   protected:
      int ti_steps,levels;
      Vector u0;
      Vector dudt0;
      double t_final,dt;
      ODE2 *oper;
   public:
      CheckODE2()
      {
         oper = new ODE2(1.0, 0.0);
         ti_steps = 20;
         levels   = 5;

         u0.SetSize(1);
         u0    = 1.0;

         dudt0.SetSize(1);
         dudt0  = 1.0;

         t_final = 2*M_PI;
         dt = t_final/double(ti_steps);
      };

      double order(SecondOrderODESolver* ode_solver)
      {
         double dt,t;
         Vector u(1);
         Vector du(1);
         Vector err_u(levels);
         Vector err_du(levels);
         int steps = ti_steps;

         t = 0.0;
         dt = t_final/double(steps);
         u = u0;
         du.Set(dt,dudt0);
         ode_solver->Init(*oper);
         for (int ti = 0; ti< steps; ti++)
         {
            ode_solver->Step(u, du, t, dt);
         }
         u -= u0;
         du.Add(-dt,dudt0);

         err_u[0] = u.Norml2();
         err_du[0] = du.Norml2();

         std::cout<<std::setw(12)<<"Error u"
                  <<std::setw(12)<<"Error du"
                  <<std::setw(12)<<"Ratio u"
                  <<std::setw(12)<<"Ratio du"
                  <<std::setw(12)<<"Order u"
                  <<std::setw(12)<<"Order du"<<std::endl;
         std::cout<<std::setw(12)<<err_u[0]
                  <<std::setw(12)<<err_du[0]<<std::endl;

         for (int l = 1; l< levels; l++)
         {
            t = 0.0;
            steps *=2;
            dt = t_final/double(steps);
            u = u0;
            du.Set(dt,dudt0);
            ode_solver->Init(*oper);
            for (int ti = 0; ti< steps; ti++)
            {
               ode_solver->Step(u, du, t, dt);
            }
            u -= u0;
            du.Add(-dt,dudt0);
            err_u[l] = u.Norml2();
            err_du[l] = du.Norml2();
            std::cout<<std::setw(12)<<err_u[l]
                     <<std::setw(12)<<err_du[l]
                     <<std::setw(12)<<err_u[l-1]/err_u[l]
                     <<std::setw(12)<<err_du[l-1]/err_du[l]
                     <<std::setw(12)<<log(err_u[l-1]/err_u[l])/log(2)
                     <<std::setw(12)<<log(err_du[l-1]/err_du[l])/log(2) <<std::endl;
         }
         delete ode_solver;

         return log(err_u[levels-2]/err_u[levels-1])/log(2);
      }
      virtual ~CheckODE2() {delete oper;};
   };
   CheckODE2 check;

   // Newmark-based solvers
   SECTION("Newmark")
   {
      std::cout <<"\nTesting NewmarkSolver" << std::endl;
      REQUIRE(check.order(new NewmarkSolver) + tol > 3.0 );
   }

   SECTION("LinearAcceleration")
   {
      std::cout <<"\nLinearAccelerationSolver" << std::endl;
      REQUIRE(check.order(new LinearAccelerationSolver) + tol > 3.0 );
   }

   SECTION("CentralDifference")
   {
      std::cout <<"\nTesting CentralDifference" << std::endl;
      REQUIRE(check.order(new CentralDifferenceSolver) + tol > 3.0 );
   }

   SECTION("FoxGoodwin")
   {
      std::cout <<"\nTesting FoxGoodwin" << std::endl;
      REQUIRE(check.order(new FoxGoodwinSolver) + tol > 4.0 );
   }

   //Generalized-alpha based solvers
   SECTION("GeneralizedAlpha")
   {
      std::cout <<"\nTesting GeneralizedAlpha" << std::endl;
      REQUIRE(check.order(new GeneralizedAlpha2Solver) + tol > 3.0 );
   }

   SECTION("AverageAcceleration")
   {
      std::cout <<"\nTesting AverageAcceleration" << std::endl;
      REQUIRE(check.order(new AverageAccelerationSolver) + tol > 3.0 );
   }

   SECTION("HHTAlpha")
   {
      std::cout <<"\nTesting HHTAlpha" << std::endl;
      REQUIRE(check.order(new HHTAlphaSolver) + tol > 3.0 );
   }

   SECTION("WBZAlphaAlpha")
   {
      std::cout <<"\nTesting WBZAlpha" << std::endl;
      REQUIRE(check.order(new WBZAlphaSolver) + tol > 3.0 );
   }
}


