// Shared implementation ex71p/ex71 for the AD integrands and the manually
// implemented integrators

#ifndef ADEXAMPLE_HPP
#define ADEXAMPLE_HPP

#include "mfem.hpp"
#include "admfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

namespace mfem
{

///Example: Implementation of the energy and the residual
/// for p-Laplacian problem. Both, the energy and the residual
/// are evaluated at the integration points for PDE parameters
/// vparam and state fields (derivatives with respect to x,y,z
/// and primal field) stored in vector uu.
template<typename TDataType, typename TParamVector, typename TStateVector,
         int residual_size, int state_size, int param_size>
class MyVFunctor
{
public:
   ///The operator returns the first derivative of the energy with respect
   /// to all state variables. These are set in vector uu and consist of the
   /// derivatives with respect to x,y,z and the primal field. The derivative
   /// is stored in vector rr with length equal to the length of vector uu.
   void operator()(TParamVector &vparam, TStateVector &uu, TStateVector &rr)
   {
      MFEM_ASSERT(residual_size==4,"PLaplacianResidual residual_size should be equal to 4!")
      double pp = vparam[0];
      double ee = vparam[1];
      double ff = vparam[2];

      TDataType norm2 = uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2];
      TDataType tvar = pow(ee * ee + norm2, (pp - 2.0) / 2.0);

      rr[0] = tvar * uu[0];
      rr[1] = tvar * uu[1];
      rr[2] = tvar * uu[2];
      rr[3] = -ff;
   }
};

///Defines template class (functor) for evaluating the energy
/// of the p-Laplacian problem. The input parameters vparam are:
/// vparam[0] - the p-Laplacian power, vparam[1] small value
/// ensuring exsitance of an unique solution, and vparam[2] -
/// the distributed extenal input to the PDE.
template<typename TDataType, typename TParamVector, typename TStateVector
         , int state_size, int param_size>
class MyQFunctor
{
public:
   ///Returns the energy of a  p-Laplacian for state field input
   /// provided in vector uu and parameters provided in vector
   /// vparam.
   TDataType operator()(TParamVector &vparam, TStateVector &uu)
   {
      MFEM_ASSERT(state_size==4,"MyQFunctor state_size should be equal to 4!");
      MFEM_ASSERT(param_size==3,"MyQFunctor param_size should be equal to 3!");
      double pp = vparam[0];
      double ee = vparam[1];
      double ff = vparam[2];

      TDataType u = uu[3];
      TDataType norm2 = uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2];

      TDataType rez = pow(ee * ee + norm2, pp / 2.0) / pp - ff * u;
      return rez;
   }
};


///Implements integrator for a p-Laplacian problem.
/// The integrator is based on a class QFunction utilized for
/// evaluating the energy, the first derivative (residual) and
/// the Hessian of the energy (the Jacobian of the residual).
template<class CQVectAutoDiff>
class pLaplaceAD : public NonlinearFormIntegrator
{
protected:
   Coefficient *pp;
   Coefficient *coeff;
   Coefficient *load;

   CQVectAutoDiff rdf;

public:
   pLaplaceAD()
   {
      coeff = nullptr;
      pp = nullptr;
   }

   pLaplaceAD(Coefficient &pp_) : pp(&pp_), coeff(nullptr), load(nullptr) {}

   pLaplaceAD(Coefficient &pp_, Coefficient &q, Coefficient &ld_)
      : pp(&pp_), coeff(&q), load(&ld_)
   {}

   virtual ~pLaplaceAD() {}

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &trans,
                                   const Vector &elfun) override
   {
      double energy = 0.0;
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      Vector shapef(ndof);
      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      Vector grad(spaceDim);

      Vector vparam(3); //[power, epsilon, load]
      Vector uu(4);     //[diff_x,diff_y,diff_z,u]

      uu = 0.0;
      vparam[0] = 2.0;  //default power
      vparam[1] = 1e-8; //default epsilon
      vparam[2] = 1.0;  //default load

      double w;
      double detJ;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w * w);
         w = ip.weight * w;

         el.CalcDShape(ip, dshape_iso);
         el.CalcShape(ip, shapef);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be divided by detJ for obtaining the real value
         // calculate the gradient
         dshape_xyz.MultTranspose(elfun, grad);

         // set the power
         if (pp != nullptr)
         {
            vparam[0] = pp->Eval(trans, ip);
         }

         // set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            vparam[1] = coeff->Eval(trans, ip);
         }
         // add the contribution from the load
         if (load != nullptr)
         {
            vparam[2] = load->Eval(trans, ip);
         }
         // fill the values of vector uu
         for (int jj = 0; jj < spaceDim; jj++)
         {
            uu[jj] = grad[jj] / detJ;
         }
         uu[3] = shapef * elfun;
         // the energy is taken directly the the templated function
         MyQFunctor<double,Vector,Vector,4,3> qfunc;
         energy = energy + w * qfunc(vparam,uu);
      }
      return energy;
   }

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect) override
   {
MFEM_PERF_BEGIN("AssembleElementVector");
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      Vector shapef(ndof);
      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      Vector lvec(ndof);
      elvect.SetSize(ndof);
      elvect = 0.0;

      DenseMatrix B(ndof, 4); //[diff_x,diff_y,diff_z, shape]
      Vector vparam(3);       //[power, epsilon, load]
      Vector uu(4);           //[diff_x,diff_y,diff_z,u]
      Vector du(4);
      B = 0.0;
      uu = 0.0;
      //initialize the parameters - keep the same order
      //utilized in the pLapIntegrator definition
      vparam[0] = 2.0;  //default power
      vparam[1] = 1e-8; //default epsilon
      vparam[2] = 1.0;  //default load

      double w;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         //detJ = (square ? w : w*w);
         w = ip.weight * w;

         el.CalcDShape(ip, dshape_iso);
         el.CalcShape(ip, shapef);
         Mult(dshape_iso, trans.InverseJacobian(), dshape_xyz);

         // set the matrix B
         for (int jj = 0; jj < spaceDim; jj++)
         {
            B.SetCol(jj, dshape_xyz.GetColumn(jj));
         }
         B.SetCol(3, shapef);

         // set the power
         if (pp != nullptr)
         {
            vparam[0] = pp->Eval(trans, ip);
         }
         // set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            vparam[1] = coeff->Eval(trans, ip);
         }
         // add the contribution from the load
         if (load != nullptr)
         {
            vparam[2] = load->Eval(trans, ip);
         }

         // calculate uu
         B.MultTranspose(elfun, uu);
         // calculate derivative of the energy with respect to uu
         rdf.QVectorFunc(vparam,uu,du);
         B.Mult(du, lvec);
         elvect.Add(w, lvec);
      } // end integration loop
MFEM_PERF_END("AssembleElementVector");
   }

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat) override
   {
MFEM_PERF_BEGIN("AssembleElementGrad");
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      Vector shapef(ndof);
      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      elmat.SetSize(ndof, ndof);
      elmat = 0.0;

      DenseMatrix B(ndof, 4); // [diff_x,diff_y,diff_z, shape]
      DenseMatrix A(ndof, 4);
      Vector vparam(3); // [power, epsilon, load]
      Vector uu(4);     // [diff_x,diff_y,diff_z,u]
      DenseMatrix duu(4, 4);
      B = 0.0;
      uu = 0.0;
      // initialize the parameters - keep the same order
      // utilized in the pLapIntegrator definition
      vparam[0] = 2.0;  // default power
      vparam[1] = 1e-8; // default epsilon
      vparam[2] = 1.0;  // default load

      double w;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         w = ip.weight * w;

         el.CalcDShape(ip, dshape_iso);
         el.CalcShape(ip, shapef);
         Mult(dshape_iso, trans.InverseJacobian(), dshape_xyz);

         // set the matrix B
         for (int jj = 0; jj < spaceDim; jj++)
         {
            B.SetCol(jj, dshape_xyz.GetColumn(jj));
         }
         B.SetCol(3, shapef);

         // set the power
         if (pp != nullptr)
         {
            vparam[0] = pp->Eval(trans, ip);
         }
         // set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            vparam[1] = coeff->Eval(trans, ip);
         }
         // add the contribution from the load
         if (load != nullptr)
         {
            vparam[2] = load->Eval(trans, ip);
         }

         // calculate uu
         B.MultTranspose(elfun, uu);
         // calculate derivative of the energy with respect to uu
         rdf.QJacobian(vparam,uu,duu);
         Mult(B, duu, A);
         AddMult_a_ABt(w, A, B, elmat);

      } // end integration loop
MFEM_PERF_END("AssembleElementGrad");
   }
};

///Implements hand-coded integrator for a p-Laplacian problem.
/// Utilized as alternative for the  pLaplaceAD class based on
/// automatic differentiation.
class pLaplace : public NonlinearFormIntegrator
{
protected:
   Coefficient *pp;
   Coefficient *coeff;
   Coefficient *load;

public:
   pLaplace()
   {
      coeff = nullptr;
      pp = nullptr;
   }

   pLaplace(Coefficient &pp_) : pp(&pp_), coeff(nullptr), load(nullptr) {}

   pLaplace(Coefficient &pp_, Coefficient &q, Coefficient &ld_)
      : pp(&pp_), coeff(&q), load(&ld_)
   {}

   virtual ~pLaplace() {}

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &trans,
                                   const Vector &elfun) override
   {
      double energy = 0.0;
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      Vector shapef(ndof);
      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      Vector grad(spaceDim);

      double w;
      double detJ;
      double nrgrad2;
      double ppp = 2.0;
      double eee = 0.0;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w * w);
         w = ip.weight * w;

         el.CalcDShape(ip, dshape_iso);
         el.CalcShape(ip, shapef);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be divided by detJ for obtaining the real value
         // calculate the gradient
         dshape_xyz.MultTranspose(elfun, grad);
         nrgrad2 = grad * grad / (detJ * detJ);

         // set the power
         if (pp != nullptr)
         {
            ppp = pp->Eval(trans, ip);
         }

         // set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            eee = coeff->Eval(trans, ip);
         }

         energy = energy + w * std::pow(nrgrad2 + eee * eee, ppp / 2.0) / ppp;

         // add the contribution from the load
         if (load != nullptr)
         {
            energy = energy - w * (shapef * elfun) * load->Eval(trans, ip);
         }
      }
      return energy;
   }

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect) override
   {

MFEM_PERF_BEGIN("AssembleElementVector");
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      Vector shapef(ndof);
      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      Vector grad(spaceDim);
      Vector lvec(ndof);
      elvect.SetSize(ndof);
      elvect = 0.0;

      double w;
      double detJ;
      double nrgrad;
      double aa;
      double ppp = 2.0;
      double eee = 0.0;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w * w);
         w = ip.weight * w; //w;

         el.CalcDShape(ip, dshape_iso);
         el.CalcShape(ip, shapef);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be divided by detJ for obtaining the real value

         // calculate the gradient
         dshape_xyz.MultTranspose(elfun, grad);
         nrgrad = grad.Norml2() / detJ;
         // grad is not scaled so far, i.e., grad=grad/detJ

         // set the power
         if (pp != nullptr)
         {
            ppp = pp->Eval(trans, ip);
         }

         // set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            eee = coeff->Eval(trans, ip);
         }

         aa = nrgrad * nrgrad + eee * eee;
         aa = std::pow(aa, (ppp - 2.0) / 2.0);
         dshape_xyz.Mult(grad, lvec);
         elvect.Add(w * aa / (detJ * detJ), lvec);

         // add loading
         if (load != nullptr)
         {
            elvect.Add(-w * load->Eval(trans, ip), shapef);
         }
      } // end integration loop
MFEM_PERF_END("AssembleElementVector");
   }

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat) override
   {
MFEM_PERF_BEGIN("AssembleElementGrad");
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      Vector grad(spaceDim);
      Vector lvec(ndof);
      elmat.SetSize(ndof, ndof);
      elmat = 0.0;

      double w;
      double detJ;
      double nrgrad;
      double aa0;
      double aa1;
      double ppp = 2.0;
      double eee = 0.0;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w * w);
         w = ip.weight * w;

         el.CalcDShape(ip, dshape_iso);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be divided by detJ for obtaining the real value
         // grad is not scaled so far,i.e., grad=grad/detJ

         //set the power
         if (pp != nullptr)
         {
            ppp = pp->Eval(trans, ip);
         }
         //set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            eee = coeff->Eval(trans, ip);
         }

         //calculate the gradient
         dshape_xyz.MultTranspose(elfun, grad);
         nrgrad = grad.Norml2() / detJ;
         aa0 = nrgrad * nrgrad + eee * eee;
         aa1 = std::pow(aa0, (ppp - 2.0) / 2.0);
         aa0 = (ppp - 2.0) * std::pow(aa0, (ppp - 4.0) / 2.0);
         dshape_xyz.Mult(grad, lvec);
         w = w / (detJ * detJ);
         AddMult_a_VVt(w * aa0 / (detJ * detJ), lvec, elmat);
         AddMult_a_AAt(w * aa1, dshape_xyz, elmat);

      } // end integration loop
MFEM_PERF_END("AssembleElementGrad");
   }
};

/// Implements AD enabled integrator for a p-Laplacian problem.
/// The tangent matrix is computed using the residual of the
/// element. The template argument should be equal to the size of
/// the residual vector (element vector), i.e., the user should
/// specify the sizeres to match the exact vector size for the
///  considered order of the shape functions.

template<int sizeres=10>
class pLaplaceSL : public NonlinearFormIntegrator
{
protected:
   Coefficient *pp;
   Coefficient *coeff;
   Coefficient *load;

public:
   pLaplaceSL()
   {
      coeff = nullptr;
      pp = nullptr;
   }

   pLaplaceSL(Coefficient &pp_) : pp(&pp_), coeff(nullptr), load(nullptr) {}

   pLaplaceSL(Coefficient &pp_, Coefficient &q, Coefficient &ld_)
      : pp(&pp_), coeff(&q), load(&ld_)
   {}

   virtual ~pLaplaceSL() {}

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &trans,
                                   const Vector &elfun) override
   {
      double energy = 0.0;
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      Vector shapef(ndof);
      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      Vector grad(spaceDim);

      double w;
      double detJ;
      double nrgrad2;
      double ppp = 2.0;
      double eee = 0.0;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w * w);
         w = ip.weight * w;

         el.CalcDShape(ip, dshape_iso);
         el.CalcShape(ip, shapef);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be divided by detJ for obtaining the real value
         // calculate the gradient
         dshape_xyz.MultTranspose(elfun, grad);
         nrgrad2 = grad * grad / (detJ * detJ);

         // set the power
         if (pp != nullptr)
         {
            ppp = pp->Eval(trans, ip);
         }

         // set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            eee = coeff->Eval(trans, ip);
         }

         energy = energy + w * std::pow(nrgrad2 + eee * eee, ppp / 2.0) / ppp;

         // add the contribution from the load
         if (load != nullptr)
         {
            energy = energy - w * (shapef * elfun) * load->Eval(trans, ip);
         }
      }
      return energy;
   }

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect) override
   {
MFEM_PERF_BEGIN("AssembleElementVector");
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      Vector shapef(ndof);
      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      Vector grad(spaceDim);
      Vector lvec(ndof);
      elvect.SetSize(ndof);
      elvect = 0.0;

      double w;
      double detJ;
      double nrgrad;
      double aa;
      double ppp = 2.0;
      double eee = 0.0;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w * w);
         w = ip.weight * w; //w;

         el.CalcDShape(ip, dshape_iso);
         el.CalcShape(ip, shapef);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be divided by detJ for obtaining the real value

         // calculate the gradient
         dshape_xyz.MultTranspose(elfun, grad);
         nrgrad = grad.Norml2() / detJ;
         // grad is not scaled so far, i.e., grad=grad/detJ

         // set the power
         if (pp != nullptr)
         {
            ppp = pp->Eval(trans, ip);
         }

         // set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            eee = coeff->Eval(trans, ip);
         }

         aa = nrgrad * nrgrad + eee * eee;
         aa = std::pow(aa, (ppp - 2.0) / 2.0);
         dshape_xyz.Mult(grad, lvec);
         elvect.Add(w * aa / (detJ * detJ), lvec);

         // add loading
         if (load != nullptr)
         {
            elvect.Add(-w * load->Eval(trans, ip), shapef);
         }
      } // end integration loop
MFEM_PERF_END("AssembleElementVector");
   }

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat) override
   {
MFEM_PERF_BEGIN("AssembleElementGrad");
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      elmat.SetSize(ndof, ndof);
      elmat = 0.0;

      double w;
      double detJ;
      double ppp = 2.0;
      double eee = 0.0;

      mfem::Vector param(3); param=0.0;

      // Computes the residual at an integration point.
      // The implementation is a copy of the integration loop
      // in AssembleElementVector.
      auto resfun = [&](mfem::Vector& vparam, mfem::ad::ADVectorType& uu, mfem::ad::ADVectorType& vres){

         vres.SetSize(uu.Size()); vres=0.0;
         mfem::ad::ADVectorType grad(spaceDim);
         mfem::ad::ADFloatType nrgrad;
         mfem::ad::ADFloatType aa;
         mfem::ad::ADVectorType lvec(ndof);

         for (int i = 0; i < ir->GetNPoints(); i++)
         {
             lvec=0.0;

             const IntegrationPoint &ip = ir->IntPoint(i);
             trans.SetIntPoint(&ip);
             w = trans.Weight();
             detJ = (square ? w : w * w);
             w = ip.weight * w;

             el.CalcDShape(ip, dshape_iso);
             // AdjugateJacobian = / adj(J),         if J is square
             //                    \ adj(J^t.J).J^t, otherwise
             Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
             // dshape_xyz should be divided by detJ for obtaining the real value
             // grad is not scaled so far,i.e., grad=grad/detJ

             //set the power
             if (pp != nullptr)
             {
                ppp = pp->Eval(trans, ip);
             }
             //set the coefficient ensuring positiveness of the tangent matrix
             if (coeff != nullptr)
             {
                eee = coeff->Eval(trans, ip);
             }

             grad=0.0;
             //calculate the gradient
             for(int i=0;i<spaceDim;i++){
             for(int j=0;j<ndof;j++){
                grad[i]= grad[i]+ dshape_xyz(j,i)*uu[j];
             }}


             nrgrad= (grad*grad)/(detJ*detJ);

             aa = nrgrad + eee * eee;
             aa = pow(aa, (ppp - 2.0) / 2.0);

             for(int i=0;i<spaceDim;i++){
             for(int j=0;j<ndof;j++){
                lvec[j] = lvec[j] + dshape_xyz(j,i) * grad[i];
             }}

             for(int j=0;j<ndof;j++)
             {
                vres[j]=vres[j] + lvec[j] * (w*aa/(detJ*detJ));
             }
         }
      };

      mfem::Vector bla(elfun);
      //calculate the gradient - only for a fixed ndof
      mfem::VectorFuncAutoDiff<sizeres,sizeres,3> fdr(resfun);
      fdr.QJacobian(param, bla, elmat);
MFEM_PERF_END("AssembleElementGrad");
   }

};

} // namespace mfem
#endif
