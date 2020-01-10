// Copyright (c) 2019, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "catch.hpp"
#include "mfem.hpp"

using namespace mfem;

TEST_CASE("ILU", "[ILU]")
{
   int N = 5;
   int Nb = 3;
   int nnz_blocks = 11;

   // Submatrix of sie Nb x Nb
   DenseMatrix Ab(Nb, Nb);

   // Matrix with N x N blocks of size Nb x Nb
   SparseMatrix A(N * Nb, N * Nb);

   // Initialize the submatrix
   int a = 1.0;
   for (int i = 0; i < Nb; ++i)
   {
      for (int j = 0; j < Nb; ++j)
      {
         Ab(i, j) = a;
         ++a;
      }
   }

   /** Create a SparseMatrix that has a block structure looking like

      {{1, 1, 0, 0, 1},
       {0, 1, 0, 1, 1},
       {0, 0, 1, 0, 0},
       {0, 1, 0, 1, 0},
       {1, 0, 0, 0, 1}}

      Where 1 represents a block of size Nb x Nb that is non zero.
  */

   // Lexographical pattern
   int p[] = {1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
              0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1};

   Array<int> pattern(p, N * N);

   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < N; ++j)
      {
         if (pattern[N * i + j] == 1)
         {
            Array<int> rows, cols;

            for (int ii = 0; ii < Nb; ++ii)
            {
               rows.Append(i * Nb + ii);
               cols.Append(j * Nb + ii);
            }

            A.SetSubMatrix(rows, cols, Ab);
         }
      }
   }

   A.Finalize();

   SECTION("Create block pattern from SparseMatrix")
   {
      ILU ilu(&A, Nb);
      ilu.CreateBlockPattern();

      int nnz_count = 0;

      for (int i = 0; i < N; ++i)
      {
         for (int k = ilu.IB[i]; k < ilu.IB[i + 1]; ++k)
         {
            // Check if the non zero block is expected
            REQUIRE(pattern[i * N + ilu.JB[k]] == 1);

            nnz_count++;
         }
      }

      // Check if the number of expected non zero blocks matches
      REQUIRE(nnz_count == nnz_blocks);
   }
}
