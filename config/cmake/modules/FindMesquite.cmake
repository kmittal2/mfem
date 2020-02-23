# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-443211.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Defines the following variables:
#   - MESQUITE_FOUND
#   - MESQUITE_LIBRARIES
#   - MESQUITE_INCLUDE_DIRS

include(MfemCmakeUtilities)
mfem_find_package(Mesquite MESQUITE MESQUITE_DIR
  "include" "Mesquite_all_headers.hpp" "lib" "mesquite"
  "Paths to headers required by Mesquite." "Libraries required by Mesquite.")
