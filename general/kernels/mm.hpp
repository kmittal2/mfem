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

#ifndef MFEM_MM_KERNELS
#define MFEM_MM_KERNELS

namespace mfem
{

// *****************************************************************************
void* kH2D(void*, const void*, size_t, const bool =false);

// *****************************************************************************
void* kD2H(void*, const void*, size_t, const bool =false);

// *****************************************************************************
void* kD2D(void*, const void*, size_t, const bool =false);

} // namespace mfem

#endif // MFEM_MM_KERNELS
