// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LIBCEED_CONV_HPP
#define MFEM_LIBCEED_CONV_HPP

#include "ceed.hpp"

#include "../fespace.hpp"

namespace mfem
{

/// Initialize a Convection Integrator using libCEED
void CeedPAConvectionAssemble(const FiniteElementSpace &fes,
                             const mfem::IntegrationRule &ir,
                             CeedData& ceedData);

void CeedMFConvectionAssemble(const FiniteElementSpace &fes,
                             const mfem::IntegrationRule &ir,
                             CeedData& ceedData);

}

#endif // MFEM_LIBCEED_CONV_HPP
