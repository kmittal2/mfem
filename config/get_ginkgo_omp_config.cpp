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

#include <ginkgo/core/base/version.hpp>
#include <cstdio>
#include <string.h>

int main()
{
   gko::version_info gko_version = gko::version_info::get();
   if (strcmp(gko_version.omp_version.tag, "not compiled") == 0)
   {
      printf("NO");
   }
   else
   {
      printf("YES");
   }

   return 0;
}
