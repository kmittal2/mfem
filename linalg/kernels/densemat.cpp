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

#include "../../general/okina.hpp"

namespace mfem
{

// *****************************************************************************
template <class T> __device__ __host__
inline void Swap(T &a, T &b)
{
   T tmp(a);
   a = b;
   b = tmp;
}

// *****************************************************************************
/*__kernel__ void LSolve(const int m,
                       const int n,
                       const double *data,
                       const int *ipiv,
                       double *x)
{
   for (int k=0; k<n; k+=1)
   {
      double *mx = &x[k*m];
      // X <- P X
      for (int i = 0; i < m; i++)
      {
         Swap<double>(mx[i], mx[ipiv[i]]);
      }
      // X <- L^{-1} X
      for (int j = 0; j < m; j++)
      {
         const double mx_j = mx[j];
         for (int i = j+1; i < m; i++)
         {
            mx[i] -= data[i+j*m] * mx_j;
         }
      }
   }
   }*/


// *****************************************************************************
void kGetInverseMatrix(const int m,
                       const int *ipiv,
                       const double *data,
                       double *x){
   GET_CONST_ADRS(data);
   GET_CONST_ADRS_T(ipiv,int);
   GET_ADRS(x);

#warning kGetInverseMatrix
   MFEM_FORALL(_k_, 1,
   //for(int _k_=0;_k_<1;_k_+=1){
      for (int k = 0; k < m; k++)
      {
         double *d_mx = &d_x[k*m];
         const double minus_x_k = -( d_mx[k] = 1.0/d_data[k+k*m] );
         for (int i = 0; i < k; i++)
         {
            d_mx[i] = d_data[i+k*m] * minus_x_k;
         }
         for (int j = k-1; j >= 0; j--)
         {
            const double x_j = ( d_mx[j] /= d_data[j+j*m] );
            for (int i = 0; i < j; i++)
            {
               d_mx[i] -= d_data[i+j*m] * x_j;
            }
         }
         //d_x += m;
      }
      // X <- X L^{-1} (use input only from the upper triangular part of X)
      {
         int k = m-1;
         for (int j = 0; j < k; j++)
         {
            const double minus_L_kj = -d_data[k+j*m];
            for (int i = 0; i <= j; i++)
            {
               d_x[i+j*m] += d_x[i+k*m] * minus_L_kj;
            }
            for (int i = j+1; i < m; i++)
            {
               d_x[i+j*m] = d_x[i+k*m] * minus_L_kj;
            }
         }
      }
      for (int k = m-2; k >= 0; k--)
      {
         for (int j = 0; j < k; j++)
         {
            const double L_kj = d_data[k+j*m];
            for (int i = 0; i < m; i++)
            {
               d_x[i+j*m] -= d_x[i+k*m] * L_kj;
            }
         }
      }
      // X <- X P
      for (int k = m-1; k >= 0; k--) {
         const int piv_k = d_ipiv[k];
         if (k != piv_k) {
            for (int i = 0; i < m; i++) {
               Swap<double>(d_x[i+k*m], d_x[i+piv_k*m]);
            }
         }
      }
   );
}

// *****************************************************************************
void kLSolve( const int m,
              const int n,
              const double *data, const int *ipiv, double *x)
{
   GET_CONST_ADRS(data);
   GET_CONST_ADRS_T(ipiv,int);
   GET_ADRS(x);
   MFEM_FORALL(k, n,
   {
      double *d_mx = &d_x[k*m];
      // X <- P X
      for (int i = 0; i < m; i++)
      {
         Swap<double>(d_mx[i], d_mx[d_ipiv[i]]);
      }
      // X <- L^{-1} X
      for (int j = 0; j < m; j++)
      {
         const double d_mx_j = d_mx[j];
         for (int i = j+1; i < m; i++)
         {
            d_mx[i] -= d_data[i+j*m] * d_mx_j;
         }
      }
   });
}

// *****************************************************************************
void kUSolve(const int m, const int n, const double *data, double *x)
{
   GET_CONST_ADRS(data);
   GET_ADRS(x);
   MFEM_FORALL(k, n,
   {
      double *d_mx = &d_x[k*m];
      for (int j = m-1; j >= 0; j--)
      {
         const double x_j = ( d_mx[j] /= d_data[j+j*m] );
         for (int i = 0; i < j; i++)
         {
            d_mx[i] -= d_data[i+j*m] * x_j;
         }
      }
   });
}

// *****************************************************************************
void kFactorPrint(const int s, const double *data)
{
   GET_CONST_ADRS(data);
   MFEM_FORALL(i, s,
   {
      printf("\n\td_data[%ld]=%f",i,d_data[i]);
   });
}

// *****************************************************************************
void kFactorSet(const int s, const double *adata, double *ludata)
{
   GET_CONST_ADRS(adata);
   GET_ADRS(ludata);
   MFEM_FORALL(i, s,
   {
      d_ludata[i] = d_adata[i];
   });
}

// *****************************************************************************
void kFactor(const int m, int *ipiv, double *data)
{
   GET_ADRS_T(ipiv,int);
   GET_ADRS(data);
   MFEM_FORALL(i, m,
   {
      // pivoting
      {
         int piv = i;
         double a = fabs(d_data[piv+i*m]);
         for (int j = i+1; j < m; j++)
         {
            const double b = fabs(d_data[j+i*m]);
            if (b > a)
            {
               a = b;
               piv = j;
            }
         }
         d_ipiv[i] = piv;
         if (piv != (int) i)
         {
            // swap rows i and piv in both L and U parts
            for (int j = 0; j < m; j++)
            {
               Swap<double>(d_data[i+j*m], d_data[piv+j*m]);
            }
         }
      }
      const double diim = d_data[i+i*m];
      assert(diim != 0.0);
      const double a_ii_inv = 1.0/d_data[i+i*m];
      for (int j = i+1; j < m; j++)
      {
         d_data[j+i*m] *= a_ii_inv;
      }
      for (int k = i+1; k < m; k++)
      {
         const double a_ik = d_data[i+k*m];
         for (int j = i+1; j < m; j++)
         {
            d_data[j+k*m] -= a_ik * d_data[j+i*m];
         }
      }
   });
}

// **************************************************************************
void DenseMatrixSet(const double d,
                    const size_t size,
                    double *data)
{
   GET_ADRS(data);
   MFEM_FORALL(i, size, d_data[i] = d;);
}

// **************************************************************************
void DenseMatrixTranspose(const size_t height,
                          const size_t width,
                          double *data,
                          const double *mdata)
{
   GET_ADRS(data);
   GET_CONST_ADRS(mdata);
   MFEM_FORALL(i, height,
   {
      for (size_t j=0; j<width; j+=1)
      {
         d_data[i+j*height] = d_mdata[j+i*height];
      }
   });
}

// *****************************************************************************
void kMultAAt(const size_t height, const size_t width,
              const double *a, double *aat)
{
   GET_CONST_ADRS(a);
   GET_ADRS(aat);
   MFEM_FORALL(i, height,
   {
      for (size_t j=0; j<=i; j++)
      {
         double temp = 0.0;
         for (size_t k=0; k<width; k++)
         {
            temp += d_a[i+k*height] * d_a[j+k*height];
         }
         d_aat[j+i*height] = d_aat[i+j*height] = temp;
      }
   });
}

// *****************************************************************************
void kGradToDiv(const size_t n, const double *data, double *ddata)
{
   GET_CONST_ADRS(data);
   GET_ADRS(ddata);
   MFEM_FORALL(i, n, d_ddata[i] = d_data[i];);
}

// *****************************************************************************
void kAddMult_a_VVt(const size_t n, const double a, const double *v,
                    const size_t height, double *VVt)
{
   GET_CONST_ADRS(v);
   GET_ADRS(VVt);
   MFEM_FORALL(i, n,
   {
      double avi = a * d_v[i];
      for (size_t j = 0; j < i; j++)
      {
         double avivj = avi * d_v[j];
         d_VVt[i+j*height] += avivj;
         d_VVt[j+i*height] += avivj;
      }
      d_VVt[i+i*height] += avi * d_v[i];
   });

}

// *****************************************************************************
void kMultWidth0(const size_t height, double *y)
{
   GET_ADRS(y);
   MFEM_FORALL(row, height, d_y[row] = 0.0;);
}

// *****************************************************************************
void kMult(const size_t height, const size_t width,
           const double *data, const double *x, double *y)
{
   GET_CONST_ADRS(data);
   GET_CONST_ADRS(x);
   GET_ADRS(y);
   MFEM_FORALL(i, height,
   {
      double sum = 0.0;
      for (size_t j=0; j<width; j+=1)
      {
         sum += d_x[j]*d_data[i+j*height];
      }
      d_y[i] = sum;
   });
}

// *****************************************************************************
void kMult(const size_t ah, const size_t aw, const size_t bw,
           const double *bd, const double *cd, double *ad)
{
   GET_CONST_ADRS(bd);
   GET_CONST_ADRS(cd);
   GET_ADRS(ad);
   MFEM_FORALL(i, ah*aw, d_ad[i] = 0.0;);
   MFEM_FORALL(j, aw,
   {
      for (size_t k = 0; k < bw; k++)
      {
         for (size_t i = 0; i < ah; i++)
         {
            d_ad[i+j*ah] += d_bd[i+k*ah] * d_cd[k+j*bw];
         }
      }
   });
}

// *****************************************************************************
void kDiag(const size_t n, const size_t N, const double c, double *data)
{
   GET_ADRS(data);
   MFEM_FORALL(i, N, d_data[i] = 0.0;);
   MFEM_FORALL(i, n, d_data[i*(n+1)] = c;);
}

// *****************************************************************************
void kOpEq(const size_t hw, const double *m, double *data)
{
   GET_CONST_ADRS(m);
   GET_ADRS(data);
   MFEM_FORALL(i, hw, d_data[i] = d_m[i];);
}

// *****************************************************************************
double kDet2(const double *data)
{
   static double *result = mm::malloc<double>(1);
   GET_CONST_ADRS(data);
   GET_ADRS(result);
   MFEM_FORALL(k, 1,
               d_result[0] = d_data[0] * d_data[3] - d_data[1] * d_data[2];
               );
   mm::Get().Pull(d_result);
   return result[0];
}

// *****************************************************************************
double kDet3(const double *data)
{
   MFEM_GPU_CANNOT_PASS;
   GET_ADRS(data);
   return
      d_data[0] * (d_data[4] * d_data[8] - d_data[5] * d_data[7]) +
      d_data[3] * (d_data[2] * d_data[7] - d_data[1] * d_data[8]) +
      d_data[6] * (d_data[1] * d_data[5] - d_data[2] * d_data[4]);
}

// *****************************************************************************
double kFNormMax(const size_t hw, const double *data)
{
   MFEM_GPU_CANNOT_PASS;
   GET_ADRS(data);
   double max_norm = 0.0;
   for (size_t i = 0; i < hw; i++)
   {
      const double entry = fabs(d_data[i]);
      if (entry > max_norm)
      {
         max_norm = entry;
      }
   }
   return max_norm;
}

// *****************************************************************************
double kFNorm2(const size_t hw, const double max_norm, const double *data)
{
   MFEM_GPU_CANNOT_PASS;
   GET_ADRS(data);
   double fnorm2 = 0.0;
   for (size_t i = 0; i < hw; i++)
   {
      const double entry = d_data[i] / max_norm;
      fnorm2 += entry * entry;
   }
   return fnorm2;
}

// *****************************************************************************
void kCalcInverse2D(const double t, const double *a, double *inva)
{
   MFEM_GPU_CANNOT_PASS;
   GET_CONST_ADRS(a);
   GET_ADRS(inva);
   d_inva[0+2*0] =  d_a[1+2*1] * t ;
   d_inva[0+2*1] = -d_a[0+2*1] * t ;
   d_inva[1+2*0] = -d_a[1+2*0] * t ;
   d_inva[1+2*1] =  d_a[0+2*0] * t ;
}

// *****************************************************************************
void kCalcInverse3D(const double t, const double *a, double *inva)
{
   MFEM_GPU_CANNOT_PASS;
   GET_CONST_ADRS(a);
   GET_ADRS(inva);

   d_inva[0+3*0] = (d_a[1+3*1]*d_a[2+3*2]-d_a[1+3*2]*d_a[2+3*1])*t;
   d_inva[0+3*1] = (d_a[0+3*2]*d_a[2+3*1]-d_a[0+3*1]*d_a[2+3*2])*t;
   d_inva[0+3*2] = (d_a[0+3*1]*d_a[1+3*2]-d_a[0+3*2]*d_a[1+3*1])*t;

   d_inva[1+3*0] = (d_a[1+3*2]*d_a[2+3*0]-d_a[1+3*0]*d_a[2+3*2])*t;
   d_inva[1+3*1] = (d_a[0+3*0]*d_a[2+3*2]-d_a[0+3*2]*d_a[2+3*0])*t;
   d_inva[1+3*2] = (d_a[0+3*2]*d_a[1+3*0]-d_a[0+3*0]*d_a[1+3*2])*t;

   d_inva[2+3*0] = (d_a[1+3*0]*d_a[2+3*1]-d_a[1+3*1]*d_a[2+3*0])*t;
   d_inva[2+3*1] = (d_a[0+3*1]*d_a[2+3*0]-d_a[0+3*0]*d_a[2+3*1])*t;
   d_inva[2+3*2] = (d_a[0+3*0]*d_a[1+3*1]-d_a[0+3*1]*d_a[1+3*0])*t;
}

} // namespace mfem
