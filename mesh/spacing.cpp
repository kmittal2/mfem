// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "spacing.hpp"

namespace mfem
{

SpacingFunction* GetSpacingFunction(const SPACING_TYPE spacingType,
                                    Array<int> const& ipar,
                                    Vector const& dpar)
{
   Array<int> iparsub, relN;

   switch (spacingType)
   {
      case SPACING_TYPE::UNIFORM_SPACING:
         MFEM_VERIFY(ipar.Size() == 1 &&
                     dpar.Size() == 0, "Invalid spacing function parameters");
         return new UniformSpacingFunction(ipar[0]);
      case SPACING_TYPE::LINEAR:
         MFEM_VERIFY(ipar.Size() == 3 &&
                     dpar.Size() == 1, "Invalid spacing function parameters");
         return new LinearSpacingFunction(ipar[0], (bool) ipar[1], dpar[0],
                                          (bool) ipar[2]);
      case SPACING_TYPE::GEOMETRIC:
         MFEM_VERIFY(ipar.Size() == 3 &&
                     dpar.Size() == 1, "Invalid spacing function parameters");
         return new GeometricSpacingFunction(ipar[0], (bool) ipar[1], dpar[0],
                                             (bool) ipar[2]);
      case SPACING_TYPE::BELL:
         MFEM_VERIFY(ipar.Size() == 3 &&
                     dpar.Size() == 2, "Invalid spacing function parameters");
         return new BellSpacingFunction(ipar[0], (bool) ipar[1], dpar[0],
                                        dpar[1], (bool) ipar[2]);
      case SPACING_TYPE::GAUSSIAN:
         MFEM_VERIFY(ipar.Size() == 3 &&
                     dpar.Size() == 2, "Invalid spacing function parameters");
         return new GaussianSpacingFunction(ipar[0], (bool) ipar[1], dpar[0],
                                            dpar[1], (bool) ipar[2]);
      case SPACING_TYPE::LOGARITHMIC:
         MFEM_VERIFY(ipar.Size() == 3 &&
                     dpar.Size() == 1, "Invalid spacing function parameters");
         return new LogarithmicSpacingFunction(ipar[0], (bool) ipar[1],
                                               (bool) ipar[2], dpar[0]);
      case SPACING_TYPE::PIECEWISE:
         MFEM_VERIFY(ipar.Size() >= 3, "Invalid spacing function parameters");
         ipar.GetSubArray(3, ipar[1], relN);
         ipar.GetSubArray(3 + ipar[1], ipar.Size() - 3 - ipar[1], iparsub);
         return new PiecewiseSpacingFunction(ipar[0], ipar[1], (bool) ipar[2],
                                             relN, iparsub, dpar);
      default:
         MFEM_ABORT("Unknown spacing type \"" << spacingType << "\"");
         break;
   }

   MFEM_ABORT("Unknown spacing type");
   return nullptr;
}

SpacingFunction *SpacingFunction::Clone() const
{
   MFEM_ABORT("Base class SpacingFunction should not be cloned");
   return NULL;
}

void GeometricSpacingFunction::CalculateSpacing()
{
   // GeometricSpacingFunction requires more than 1 interval. If only 1
   // interval is requested, just use uniform spacing.
   if (n == 1) { return; }

   // Find the root of g(r) = s * (r^n - 1) - r + 1 by Newton's method.

   constexpr double convTol = 1.0e-8;
   constexpr int maxIter = 20;

   const double s_unif = 1.0 / ((double) n);

   r = s < s_unif ? 1.5 : 0.5;  // Initial guess

   bool converged = false;
   for (int iter=0; iter<maxIter; ++iter)
   {
      const double g = (s * (std::pow(r,n) - 1.0)) - r + 1.0;
      const double dg = (n * s * std::pow(r,n-1)) - 1.0;
      r -= g / dg;

      if (std::abs(g / dg) < convTol)
      {
         converged = true;
         break;
      }
   }

   MFEM_VERIFY(converged, "Convergence failure in GeometricSpacingFunction");
}

void BellSpacingFunction::CalculateSpacing()
{
   s.SetSize(n);

   // Bell spacing requires at least 3 intervals. If fewer than 3 are
   // requested, we simply use uniform spacing.
   if (n < 3)
   {
      s = 1.0 / ((double) n);
      return;
   }

   MFEM_VERIFY(s0 + s1 < 1.0, "Sum of first and last Bell spacings must be"
               << " less than 1");

   s[0] = s0;
   s[n-1] = s1;

   // If there are only 3 intervals, the calculation is linear and trivial.
   if (n == 3)
   {
      s[1] = 1.0 - s0 - s1;
      return;
   }

   // For more than 3 intervals, solve a system iteratively.
   double urk = 1.0;

   // Initialize unknown entries of s.
   double initialGuess = (1.0 - s0 - s1) / ((double) (n - 2));
   for (int i=1; i<n-1; ++i)
   {
      s[i] = initialGuess;
   }

   Vector wk(7);
   wk = 0.0;

   Vector s_new(n);

   Vector a(n+2);
   Vector b(n+2);
   Vector alpha(n+2);
   Vector beta(n+2);
   Vector gamma(n+2);

   a = 0.5;
   a[0] = 0.0;
   a[1] = 0.0;

   b = a;

   alpha = 0.0;
   beta = 0.0;
   gamma = 0.0;

   gamma[1] = s0;

   constexpr int maxIter = 100;
   constexpr double convTol = 1.0e-10;
   bool converged = false;
   for (int iter=0; iter<maxIter; ++iter)
   {
      int j;
      for (j = 1; j <= n - 3; j++)
      {
         wk[0] = (s[j] + s[j+1]) * (s[j] + s[j+1]);
         wk[1] = s[j-1];
         wk[2] = (s[j-1] + s[j]) * (s[j-1] + s[j]) * (s[j-1] + s[j]);
         wk[3] = s[j + 2];
         wk[4] = (s[j+2] + s[j+1]) * (s[j+2] + s[j+1]) * (s[j+2] + s[j+1]);
         wk[5] = wk[0] * wk[1] / wk[2];
         wk[6] = wk[0] * wk[3] / wk[4];
         a[j+1]  = a[j+1] + urk*(wk[5] - a[j+1]);
         b[j+1]  = b[j+1] + urk*(wk[6] - b[j+1]);
      }

      for (j = 2; j <= n - 2; j++)
      {
         wk[0] = a[j]*(1.0 - 2.0*alpha[j - 1] + alpha[j - 1]*alpha[j - 2]
                       + beta[j - 2]) + b[j] + 2.0 - alpha[j - 1];
         wk[1] = 1.0 / wk[0];
         alpha[j] = wk[1]*(a[j]*beta[j - 1]*(2.0 - alpha[j - 2]) +
                           2.0*b[j] + beta[j - 1] + 1.0);
         beta[j]  = -b[j]*wk[1];
         gamma[j] = wk[1]*(a[j]*(2.0*gamma[j - 1] - gamma[j - 2] -
                                 alpha[j - 2]*gamma[j - 1]) + gamma[j - 1]);
      }

      s_new[0] = s[0];
      for (j=1; j<n; ++j)
      {
         s_new[j] = s_new[j-1] + s[j];
      }

      for (j = n - 3; j >= 1; j--)
      {
         s_new[j] = alpha[j+1]*s_new[j + 1] +
                    beta[j+1]*s_new[j + 2] + gamma[j+1];
      }

      // Convert back from points to spacings
      for (j=n-1; j>0; --j)
      {
         s_new[j] = s_new[j] - s_new[j-1];
      }

      wk[5] = wk[6] = 0.0;
      for (j = n - 2; j >= 2; j--)
      {
         wk[5] = wk[5] + s_new[j]*s_new[j];
         wk[6] = wk[6] + pow(s_new[j] - s[j], 2);
      }

      s = s_new;

      const double res = sqrt(wk[6] / wk[5]);
      if (res < convTol)
      {
         converged = true;
         break;
      }
   }

   MFEM_VERIFY(converged, "Convergence failure in BellSpacingFunction");
}

void GaussianSpacingFunction::CalculateSpacing()
{
   s.SetSize(n);
   // Gaussian spacing requires at least 3 intervals. If fewer than 3 are
   // requested, we simply use uniform spacing.
   if (n < 3)
   {
      s = 1.0 / ((double) n);
      return;
   }

   s[0] = s0;
   s[n-1] = s1;

   // If there are only 3 intervals, the calculation is linear and trivial.
   if (n == 3)
   {
      s[1] = 1.0 - s0 - s1;
      return;
   }

   // For more than 3 intervals, solve a system iteratively.

   const double lnz01 = log(s0 / s1);

   const double h = 1.0 / ((double) n-1);

   // Determine concavity by first determining linear spacing and comparing
   // the total spacing to 1.
   // Linear formula: z_i = z0 + (i*h) * (z1-z0), 0 <= i <= n-1
   // \sum_{i=0}^{nzones-1} z_i = n * z0 + h * (z1-z0) * nz * (nz-1) / 2

   const double slinear = n * (s0 + (h * (s1 - s0) * 0.5 * (n-1)));

   MFEM_VERIFY(std::abs(slinear - 1.0) > 1.0e-8, "Bell distribution is too "
               << "close to linear.");

   const double u = slinear < 1.0 ? 1.0 : -1.0;

   double c = 0.3;  // Initial guess

   // Newton iterations
   constexpr int maxIter = 10;
   constexpr double convTol = 1.0e-8;
   bool converged = false;
   for (int iter=0; iter<maxIter; ++iter)
   {
      const double c2 = c * c;

      const double m = 0.5 * (1.0 - (u * c2 * lnz01));
      const double dmdc = -u * c * lnz01;

      double r = 0.0;  // Residual
      double drdc = 0.0;  // Derivative of residual

      for (int i=0; i<n; ++i)
      {
         const double x = i * h;
         const double ti = exp((-(x * x) + (2.0 * x * m)) * u / c2); // Gaussian
         r += ti;

         // Derivative of Gaussian
         drdc += ((-2.0 * (-(x * x) + (2.0 * x * m)) / (c2 * c)) +
                  ((2.0 * x * dmdc) / c2)) * ti;
      }

      r *= s0;
      r -= 1.0;  // Sum of spacings should equal 1.

      if (std::abs(r) < convTol)
      {
         converged = true;
         break;
      }

      drdc *= s0 * u;

      // Newton update is -r / drdc, limited by factors of 1/2 and 2.
      double dc = std::max(-r / drdc, -0.5*c);
      dc = std::min(dc, 2.0*c);

      c += dc;
   }

   MFEM_VERIFY(converged, "Convergence failure in GaussianSpacingFunction");

   const double c2 = c * c;
   const double m = 0.5 * (1.0 - (u * c2 * lnz01));
   const double q = s0 * exp(u*m*m / c2);

   for (int i=0; i<n; ++i)
   {
      const double x = (i * h) - m;
      s[i] = q * exp(-u*x*x / c2);
   }
}

void LogarithmicSpacingFunction::CalculateSpacing()
{
   MFEM_VERIFY(n > 0 && logBase > 1.0,
               "Invalid parameters in LogarithmicSpacingFunction");

   if (sym) { CalculateSymmetric(); }
   else { CalculateNonsymmetric(); }
}

void LogarithmicSpacingFunction::CalculateSymmetric()
{
   s.SetSize(n);

   const bool odd = (n % 2 == 1);

   const int M0 = n / 2;
   const int M = odd ? (M0 + 1) : M0;

   const double h = 1.0 / ((double) M);

   double p = 1.0;  // Initialize at right endpoint of [0,1].

   for (int i=M-2; i>=0; --i)
   {
      const double p_i = (pow(logBase, (i+1)*h) - 1.0) / (logBase - 1.0);
      s[i+1] = p - p_i;
      p = p_i;
   }

   s[0] = p;

   // Even case for spacing: [s[0], ..., s[M-1], s[M-1], s[M-2], ..., s[0]]
   //   covers interval [0,2]
   // Odd case for spacing: [s[0], ..., s[M-1], s[M-2], ..., s[0]]
   //   covers interval [0,2-s[M-1]]

   const double t = odd ? 1.0 / (2.0 - s[M-1]) : 0.5;

   for (int i=0; i<M; ++i)
   {
      s[i] *= t;

      if (i < (M-1) || !odd)
      {
         s[n - i - 1] = s[i];
      }
   }
}

void LogarithmicSpacingFunction::CalculateNonsymmetric()
{
   s.SetSize(n);

   const double h = 1.0 / ((double) n);

   double p = 1.0;  // Initialize at right endpoint of [0,1].

   for (int i=n-2; i>=0; --i)
   {
      const double p_i = (pow(logBase, (i+1)*h) - 1.0) / (logBase - 1.0);
      s[i+1] = p - p_i;
      p = p_i;
   }

   s[0] = p;
}

void PiecewiseSpacingFunction::SetupPieces(Array<int> const& ipar,
                                           Vector const& dpar)
{
   MFEM_VERIFY(partition.Size() == np - 1, "");
   bool validPartition = true;

   // Verify that partition has ascending numbers in (0,1).
   for (int i=0; i<np-1; ++i)
   {
      partition[i] = dpar[i];

      if (partition[i] <= 0.0 || partition[i] >= 1.0)
      {
         validPartition = false;
      }

      if (i > 0 && partition[i] <= partition[i-1])
      {
         validPartition = false;
      }
   }

   MFEM_VERIFY(validPartition, "");

   pieces.SetSize(np);

   Array<int> ipar_p;
   Vector dpar_p;

   int osi = 0;
   int osd = np - 1;
   int n_total = 0;
   for (int p=0; p<np; ++p)
   {
      // Setup piece p
      const SPACING_TYPE type = (SPACING_TYPE) ipar[osi];
      const int numIntParam = ipar[osi+1];
      const int numDoubleParam = ipar[osi+2];

      ipar_p.SetSize(numIntParam);
      dpar_p.SetSize(numDoubleParam);

      for (int i=0; i<numIntParam; ++i)
      {
         ipar_p[i] = ipar[osi + 3 + i];
      }

      for (int i=0; i<numDoubleParam; ++i)
      {
         dpar_p[i] = dpar[osd + i];
      }

      pieces[p] = GetSpacingFunction(type, ipar_p, dpar_p);

      osi += 3 + numIntParam;
      osd += numDoubleParam;
      n_total += npartition[p];

      MFEM_VERIFY(pieces[p]->Size() >= 1, "");
   }

   MFEM_VERIFY(osi == ipar.Size() && osd == dpar.Size(), "");
   n0 = n_total;
}

void PiecewiseSpacingFunction::ScaleParameters(double a)
{
   for (auto p : pieces)
   {
      p->ScaleParameters(a);
   }
}

void PiecewiseSpacingFunction::Print(std::ostream &os) const
{
   // SPACING_TYPE numIntParam numDoubleParam npartition {int params} {double params}
   int inum = 3 + np;
   int dnum = np-1;
   for (auto p : pieces)
   {
      // Add three for the type and the integer and double parameter counts.
      inum += p->NumIntParameters() + 3;
      dnum += p->NumDoubleParameters();
   }

   os << PIECEWISE << " " << inum << " " << dnum << " " << n << " " << np << " "
      << (int) reverse << "\n";

   for (auto n : npartition)
   {
      os << n << " ";
   }

   // Write integer parameters for all pieces.
   Array<int> ipar;
   for (auto p : pieces)
   {
      os << "\n" << p->SpacingType() << " " << p->NumIntParameters()
         << " " << p->NumDoubleParameters();

      p->GetIntParameters(ipar);

      for (auto ip : ipar)
      {
         os << " " << ip;
      }
   }

   os << "\n";
   for (auto p : partition)
   {
      os << p << " ";
   }

   // Write double parameters for all pieces.
   Vector dpar;
   for (auto p : pieces)
   {
      p->GetDoubleParameters(dpar);

      if (dpar.Size() > 0)
      {
         os << "\n";
         for (auto dp : dpar)
         {
            os << dp << " ";
         }
      }
   }

   os << "\n";
}

void PiecewiseSpacingFunction::CalculateSpacing()
{
   MFEM_VERIFY(n >= 1 && (n % n0 == 0 || n < n0), "");
   const int ref = n / n0;  // Refinement factor
   const int cf = n0 / n;  // Coarsening factor

   s.SetSize(n);

   bool coarsen = cf > 1 && n > 1;
   // If coarsening, check whether all pieces have size divisible by cf.
   if (coarsen)
   {
      for (int p=0; p<np; ++p)
      {
         const int csize = pieces[p]->Size() / cf;
         if (pieces[p]->Size() != cf * csize)
         {
            coarsen = false;
         }
      }
   }

   if (n == 1)
   {
      s[0] = 1.0;
      for (auto p : pieces) { p->SetSize(1); }
      return;
   }

   MFEM_VERIFY(coarsen || n >= n0,
               "Invalid case in PiecewiseSpacingFunction::CalculateSpacing");

   int n_total = 0;
   for (int p=0; p<np; ++p)
   {
      // Calculate spacing for piece p.

      if (coarsen)
      {
         pieces[p]->SetSize(npartition[p] / cf);
      }
      else
      {
         pieces[p]->SetSize(ref * npartition[p]);
      }

      const double p0 = (p == 0) ? 0.0 : partition[p-1];
      const double p1 = (p == np - 1) ? 1.0 : partition[p];
      const double h_p = p1 - p0;

      for (int i=0; i<pieces[p]->Size(); ++i)
      {
         s[n_total + i] = h_p * pieces[p]->Eval(i);
      }

      n_total += pieces[p]->Size();
   }

   MFEM_VERIFY(n_total == n, "");
}

SpacingFunction *PiecewiseSpacingFunction::Clone() const
{
   int osi = 0;
   int osd = np - 1;
   Array<int> ipar(osi);
   // Using Array<double> instead of Vector to increase size and keep data.
   Array<double> dpar(osd);

   for (int p=0; p<np-1; ++p)
   {
      dpar[p] = partition[p];
   }

   Array<int> ipar_p;
   Vector dpar_p;

   for (auto p : pieces)
   {
      // Setup piece p
      const SPACING_TYPE type = p->SpacingType();

      p->GetIntParameters(ipar_p);
      p->GetDoubleParameters(dpar_p);

      const int numIntParam = p->NumIntParameters();
      const int numDoubleParam = p->NumDoubleParameters();

      dpar.SetSize(osd + numDoubleParam);
      // Add three for the type and the integer and double parameter counts.
      ipar.SetSize(osi + numIntParam + 3);

      ipar[osi] = type;
      ipar[osi + 1] = numIntParam;
      ipar[osi + 2] = numDoubleParam;

      for (int i=0; i<numIntParam; ++i)
      {
         ipar[osi + 3 + i] = ipar_p[i];
      }

      for (int i=0; i<numDoubleParam; ++i)
      {
         dpar[osd + i] = dpar_p[i];
      }

      osi += numIntParam + 3;
      osd += numDoubleParam;
   }

   Vector dparvec(dpar.GetData(), dpar.Size());
   return new PiecewiseSpacingFunction(n, np, reverse, npartition,
                                       ipar, dparvec);
}

bool PiecewiseSpacingFunction::Nested() const
{
   for (auto p : pieces)
   {
      if (!p->Nested())
      {
         return false;
      }
   }

   return true;
}

}
