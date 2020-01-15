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

TEST_CASE("ILU Structure", "[ILU]")
{
   int N = 5;
   int Nb = 3;
   int nnz_blocks = 11;

   // Submatrix of sie Nb x Nb
   DenseMatrix Ab(Nb, Nb);

   // Matrix with N x N blocks of size Nb x Nb
   SparseMatrix A(N * Nb, N * Nb);
   // Create a SparseMatrix that has a block structure looking like
   //    {{1, 1, 0, 0, 1},
   //     {0, 1, 0, 1, 1},
   //     {0, 0, 1, 0, 0},
   //     {0, 1, 0, 1, 0},
   //     {1, 0, 0, 0, 1}}
   // Where 1 represents a block of size Nb x Nb that is non zero.

   // Lexographical pattern
   int p[] =
   {
      1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
      0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1
   };

   Array<int> pattern(p, N * N);
   int counter = 1;
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

            Vector Ab_data(Ab.GetData(), Nb * Nb);
            Ab_data.Randomize(counter);
            A.SetSubMatrix(rows, cols, Ab);
         }
      }
   }

   A.Finalize();

   SECTION("Create block pattern from SparseMatrix")
   {
      bool reorder = false;
      BlockILU0 ilu(A, Nb, reorder);

      int *IB = ilu.GetI();
      int *JB = ilu.GetJ();

      int nnz_count = 0;

      for (int i = 0; i < N; ++i)
      {
         for (int k = IB[i]; k < IB[i + 1]; ++k)
         {
            int j = JB[k];
            // Check if the non zero block is expected
            REQUIRE(pattern[i * N + j] == 1);
            nnz_count++;
         }
      }
      // Check if the number of expected non zero blocks matches
      REQUIRE(nnz_count == nnz_blocks);
   }
}

/*
TEST_CASE("ILU Factorization", "[ILU]")
{
   SparseMatrix A(6, 6);

   A.Set(0,0,1);
   A.Set(0,1,2);
   A.Set(0,2,3);
   A.Set(0,3,4);
   A.Set(0,4,5);
   A.Set(0,5,6);

   A.Set(1,0,7);
   A.Set(1,1,8);
   A.Set(1,2,9);
   A.Set(1,3,1);
   A.Set(1,4,2);
   A.Set(1,5,3);

   A.Set(2,0,4);
   A.Set(2,1,5);
   A.Set(2,2,6);
   A.Set(2,3,7);

   A.Set(3,0,8);
   A.Set(3,1,9);
   A.Set(3,2,1);
   A.Set(3,3,2);

   A.Set(4,0,3);
   A.Set(4,1,4);
   A.Set(4,4,5);
   A.Set(4,5,6);

   A.Set(5,0,7);
   A.Set(5,1,8);
   A.Set(5,4,9);
   A.Set(5,5,1);

   A.Finalize();

   bool reorder = true;
   BlockILU0 ilu(A, 2, reorder);

   DenseTensor AB;
   AB.UseExternalData(ilu.GetData(), 2, 2, 7);

   REQUIRE(AB(0,0,0) == Approx(5.0));
   REQUIRE(AB(1,0,0) == Approx(9.0));
   REQUIRE(AB(0,1,0) == Approx(6.0));
   REQUIRE(AB(1,1,0) == Approx(1.0));

   REQUIRE(AB(0,0,1) == Approx(3.0));
   REQUIRE(AB(1,0,1) == Approx(7.0));
   REQUIRE(AB(0,1,1) == Approx(4.0));
   REQUIRE(AB(1,1,1) == Approx(8.0));

   REQUIRE(AB(0,0,2) == Approx(6.0));
   REQUIRE(AB(1,0,2) == Approx(1.0));
   REQUIRE(AB(0,1,2) == Approx(7.0));
   REQUIRE(AB(1,1,2) == Approx(2.0));

   REQUIRE(AB(0,0,3) == Approx(4.0));
   REQUIRE(AB(1,0,3) == Approx(8.0));
   REQUIRE(AB(0,1,3) == Approx(5.0));
   REQUIRE(AB(1,1,3) == Approx(9.0));

   REQUIRE(AB(0,0,4) == Approx(1.0));
   REQUIRE(AB(1,0,4) == Approx(0.510204081632653));
   REQUIRE(AB(0,1,4) == Approx(0.0));
   REQUIRE(AB(1,1,4) == Approx(-0.06122448979591837));

   REQUIRE(AB(0,0,5) == Approx(0.4));
   REQUIRE(AB(1,0,5) == Approx(3.4));
   REQUIRE(AB(0,1,5) == Approx(0.6));
   REQUIRE(AB(1,1,5) == Approx(-11.4));

   REQUIRE(AB(0,0,6) == Approx(-8.4));
   REQUIRE(AB(1,0,6) == Approx(83.49795918367347));
   REQUIRE(AB(0,1,6) == Approx(-9.4));
   REQUIRE(AB(1,1,6) == Approx(92.04897959183674));
}
*/
