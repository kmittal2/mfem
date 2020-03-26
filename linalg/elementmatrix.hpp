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

#ifndef MFEM_ELEMMAT
#define MFEM_ELEMMAT

#include "../general/cuda.hpp"
#include "../general/forall.hpp"
#include "vector.hpp"

namespace mfem
{

class ElementMatrix
{
private:
   const int ne;
   const int ndofs;
   const Vector &data;

public:
   ElementMatrix(const Vector &vec, const int ne, const int ndofs)
   : ne(ne), ndofs(ndofs), data(vec)
   {
   }

   void AddMult(const Vector &x, Vector &y) const
   {
      const int NDOFS = ndofs;
      auto X = Reshape(x.Read(), ndofs, ne);
      auto Y = Reshape(y.ReadWrite(), ndofs, ne);
      auto A = Reshape(data.Read(), ndofs, ndofs, ne);
      MFEM_FORALL(glob_j, ne*ndofs,
      {
         const int e = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, e)*X(i, e);
         }
         Y(j, e) += res;
      });
   }
};

class FaceMatrixInt
{
private:
   const int nf;
   const int ndofs;// on the face
   const Vector &data;

public:
   FaceMatrixInt(const Vector &vec, const int nf, const int ndofs)
   : nf(nf), ndofs(ndofs), data(vec)
   {
   }

   void AddMult(const Vector &x, Vector &y) const
   {
      auto X = Reshape(x.Read(), ndofs, 2, nf);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, nf);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, nf);
      const int NDOFS = ndofs;
      MFEM_FORALL(glob_j, nf*ndofs,
      {
         const int f = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, 0, f)*X(i, 0, f);
         }
         Y(j, 0, f) += res;
         res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, 1, f)*X(i, 1, f);
         }
         Y(j, 1, f) += res;
      });
   }

   void Print()
   {
      for (size_t f = 0; f < nf; f++)
      {
         std::cout << "Face "<<f <<std::endl;
         for (size_t i = 0; i < ndofs; i++)
         {
            for (size_t j = 0; j < ndofs; j++)
            {
               std::cout << data[i+j*ndofs+2*f*ndofs*ndofs] << ", ";
            }
            std::cout << std::endl;
         }
         std::cout << std::endl;
         for (size_t i = 0; i < ndofs; i++)
         {
            for (size_t j = 0; j < ndofs; j++)
            {
               std::cout << data[i+j*ndofs+(2*f+1)*ndofs*ndofs] << ", ";
            }
            std::cout << std::endl;
         }
         std::cout << std::endl;
         std::cout << std::endl;
      }
   }
};

class FaceMatrixExt
{
private:
   const int nf;
   const int ndofs;// on the face
   const Vector &data;

public:
   FaceMatrixExt(const Vector &vec, const int nf, const int ndofs)
   : nf(nf), ndofs(ndofs), data(vec)
   {
   }

   void AddMult(const Vector &x, Vector &y) const
   {
      auto X = Reshape(x.Read(), ndofs, 2, nf);
      auto Y = Reshape(y.ReadWrite(), ndofs, 2, nf);
      auto A = Reshape(data.Read(), ndofs, ndofs, 2, nf);
      const int NDOFS = ndofs;
      MFEM_FORALL(glob_j, nf*ndofs,
      {
         const int f = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, 0, f)*X(i, 0, f);
         }
         Y(j, 1, f) += res;
         res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, 1, f)*X(i, 1, f);
         }
         Y(j, 0, f) += res;
      });
   }

   void Print()
   {
      for (size_t f = 0; f < nf; f++)
      {
         std::cout << "Face "<<f <<std::endl;
         for (size_t i = 0; i < ndofs; i++)
         {
            for (size_t j = 0; j < ndofs; j++)
            {
               std::cout << data[i+j*ndofs+2*f*ndofs*ndofs] << ", ";
            }
            std::cout << std::endl;
         }
         std::cout << std::endl;
         for (size_t i = 0; i < ndofs; i++)
         {
            for (size_t j = 0; j < ndofs; j++)
            {
               std::cout << data[i+j*ndofs+(2*f+1)*ndofs*ndofs] << ", ";
            }
            std::cout << std::endl;
         }
         std::cout << std::endl;
         std::cout << std::endl;
      }
   }
};

class FaceMatrixBdr
{
private:
   const int nf;
   const int ndofs;// on the face
   const Vector &data;

public:
   FaceMatrixBdr(const Vector &vec, const int nf, const int ndofs)
   : nf(nf), ndofs(ndofs), data(vec)
   {
   }

   void AddMult(const Vector &x, Vector &y) const
   {
      auto X = Reshape(x.Read(), ndofs, nf);
      auto Y = Reshape(y.ReadWrite(), ndofs, nf);
      auto A = Reshape(data.Read(), ndofs, ndofs, nf);
      const int NDOFS = ndofs;
      MFEM_FORALL(glob_j, nf*ndofs,
      {
         const int f = glob_j/NDOFS;
         const int j = glob_j%NDOFS;
         double res = 0.0;
         for (int i = 0; i < NDOFS; i++)
         {
            res += A(i, j, f)*X(i, f);
         }
         Y(j, f) += res;
      });
   }

   void Print()
   {
      for (size_t f = 0; f < nf; f++)
      {
         std::cout << "Face "<<f <<std::endl;
         for (size_t i = 0; i < ndofs; i++)
         {
            for (size_t j = 0; j < ndofs; j++)
            {
               std::cout << data[i+j*ndofs+f*ndofs*ndofs] << ", ";
            }
            std::cout << std::endl;
         }
         std::cout << std::endl;
         std::cout << std::endl;
      }
   }
};

}

#endif
