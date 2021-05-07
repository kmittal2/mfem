// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#define CATCH_CONFIG_RUNNER
#include "mfem.hpp"
#include "unit_tests.hpp"

bool launch_all_non_regression_tests = false;

int main(int argc, char *argv[])
{
   mfem::Device device("cuda");

   // There must be exactly one instance.
   Catch::Session session;

   // Build a new command line parser on top of Catch's
   using namespace Catch::clara;
   auto cli = session.cli() |
              Opt(launch_all_non_regression_tests) ["--all"] ("all tests");
   session.cli(cli);

   // For floating point comparisons, print 8 digits for single precision
   // values, and 16 digits for double precision values.
   Catch::StringMaker<float>::precision = 8;
   Catch::StringMaker<double>::precision = 16;

   // Apply provided command line arguments.
   int r = session.applyCommandLine(argc, argv);
   if (r != 0) { return r; }

   auto cfg = session.configData();

   cfg.testsOrTags.push_back("[CUDA]");

#ifdef MFEM_USE_MPI
   // Exclude tests marked as Parallel in a serial run, even when compiled with
   // MPI. This is done because there is no MPI session initialized.
   cfg.testsOrTags.push_back("~[Parallel]");
#endif

   session.useConfigData(cfg);

   int result = session.run();

   return result;
}
