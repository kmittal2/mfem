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

#ifndef MFEM_CONFIG_HEADER
#define MFEM_CONFIG_HEADER

// MFEM version: integer of the form: (major*100 + minor)*100 + patch.
#define MFEM_VERSION 30301

// MFEM version type, see the MFEM_VERSION_TYPE_* constants below.
#define MFEM_VERSION_TYPE ((MFEM_VERSION)%2)

// MFEM version type constants.
#define MFEM_VERSION_TYPE_RELEASE 0
#define MFEM_VERSION_TYPE_DEVELOPMENT 1

// Separate MFEM version numbers for major, minor, and patch.
#define MFEM_VERSION_MAJOR ((MFEM_VERSION)/10000)
#define MFEM_VERSION_MINOR (((MFEM_VERSION)/100)%100)
#define MFEM_VERSION_PATCH ((MFEM_VERSION)%100)

// Build the parallel MFEM library.
// Requires an MPI compiler, and the libraries HYPRE and METIS.
#define MFEM_USE_MPI

// Enable debug checks in MFEM.
// #define MFEM_DEBUG

// Enable gzstream in MFEM.
// #define MFEM_USE_GZSTREAM

// Enable backtraces for mfem_error through libunwind.
// #define MFEM_USE_LIBUNWIND

// Enable this option if linking with METIS version 5 (parallel MFEM).
// #define MFEM_USE_METIS_5

// Use LAPACK routines for various dense linear algebra operations.
// #define MFEM_USE_LAPACK

// Use thread-safe implementation. This comes at the cost of extra memory
// allocation and de-allocation.
// #define MFEM_THREAD_SAFE

// Enable experimental OpenMP support. Requires MFEM_THREAD_SAFE.
// #define MFEM_USE_OPENMP

// Internal MFEM option: enable group/batch allocation for some small objects.
#define MFEM_USE_MEMALLOC

// Which library functions to use in class StopWatch for measuring time.
// For a list of the available options, see INSTALL.
// If not defined, an option is selected automatically.
#define MFEM_TIMER_TYPE 4

// Enable MFEM functionality based on the SUNDIALS libraries.
// #define MFEM_USE_SUNDIALS

// Enable MFEM functionality based on the Mesquite library.
// #define MFEM_USE_MESQUITE

// Enable MFEM functionality based on the SuiteSparse library.
// #define MFEM_USE_SUITESPARSE

// Enable MFEM functionality based on the SuperLU library.
// #define MFEM_USE_SUPERLU

// Enable MFEM functionality based on the STRUMPACK library.
// #define MFEM_USE_STRUMPACK

// Enable functionality based on the Gecko library
// #define MFEM_USE_GECKO

// Enable secure socket streams based on the GNUTLS library
// #define MFEM_USE_GNUTLS

// Enable Sidre support
// #define MFEM_USE_SIDRE

// Enable functionality based on the NetCDF library (reading CUBIT files)
// #define MFEM_USE_NETCDF

// Enable functionality based on the PETSc library
// #define MFEM_USE_PETSC

// Enable functionality based on the MPFR library.
// #define MFEM_USE_MPFR

// Windows specific options
#ifdef _WIN32
// Macro needed to get defines like M_PI from <cmath>. (Visual Studio C++ only?)
#define _USE_MATH_DEFINES
#endif

// Version of HYPRE used for building MFEM.
#define MFEM_HYPRE_VERSION 21000

#endif // MFEM_CONFIG_HEADER
