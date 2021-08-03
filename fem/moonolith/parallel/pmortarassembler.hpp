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

#ifndef MFEM_L2P_PAR_MORTAR_ASSEMBLER_HPP
#define MFEM_L2P_PAR_MORTAR_ASSEMBLER_HPP

#include <memory>
#include <vector>

#include "../../fem.hpp"
#include "../mortarintegrator.hpp"

namespace mfem
{

/*!
 * @brief This class implements the parallel variational transfer between finite
 * element spaces. Variational transfer has been shown to have better
 * approximation properties than standard interpolation. This facilities can be
 * used for supporting applications wich require the handling of non matching
 * meshes. For instance: General multi-physics problems, fluid structure
 * interaction, or even visulization of average quanties within subvolumes. This
 * particular code is also used with LLNL for large scale multilevel Monte Carlo
 * simulations.
 */
class ParMortarAssembler
{
public:
   /*!
    * @brief constructs the object with source and destination spaces
    * @param source the source space from where we want to transfer the discrete
    * field
    * @param destination the source space to where we want to transfer the
    * discrete field
    */
   ParMortarAssembler(const std::shared_ptr<ParFiniteElementSpace> &source,
                      const std::shared_ptr<ParFiniteElementSpace> &destination);

   ~ParMortarAssembler();

   /*!
    * @brief assembles the coupling matrix B. B : source -> destination If u is a
    * coefficient associated with source and v with destination Then v = M^(-1) *
    * B * u; where M is the mass matrix in destination. Works with
    * L2_FECollection, H1_FECollection and DG_FECollection (experimental with
    * RT_FECollection and ND_FECollection).
    * @param B the assembled coupling operator. B can be passed uninitialized.
    * @return true if there was an intersection and the operator has been
    * assembled. False otherwise.
    */
   bool Assemble(std::shared_ptr<HypreParMatrix> &B);

   /*!
    * @brief transfer a function from source to destination. if the transfer is
    * to be performed multiple times use Assemble or Init/Apply instead
    * @param src_fun the function associated with the source finite element space
    * @param[out] dest_fun the function associated with the destination finite
    * element space
    * @return true if there was an intersection and the output can be used.
    */
   bool Transfer(ParGridFunction &src_fun, ParGridFunction &dest_fun);

   /*!
    * @brief transfer a function from source to destination. It requires that
    * the Init function is called before
    * @param src_fun the function associated with the source finite element space
    * @param[out] dest_fun the function associated with the destination finite
    * element space
    * @return true if the transfer was succesfull, fale otherwise.
    */
   bool Apply(ParGridFunction &src_fun, ParGridFunction &dest_fun);

   /*!
    * @brief assembles the various components necessary for the transfer.
    * To before alling the Apply function. Works with
    * L2_FECollection, H1_FECollection and DG_FECollection (experimental with
    * RT_FECollection and ND_FECollection).
    * @param B the assembled coupling operator. B can be passed uninitialized.
    * @return true if there was an intersection and the operator has been
    * assembled. False otherwise.
    */
   bool Init();

   /*!
    * @brief This method must be called before Assemble or Transfer.
    * It will assemble the operator in all intersections found.
    * @param integrator the integrator object
    */
   void AddMortarIntegrator(const std::shared_ptr<MortarIntegrator> &integrator);

   /*!
    * @brief Expose process details with verbose output
    * @param verbose, set to true for verbose output
    */
   void SetVerbose(const bool verbose);

private:
   class Impl;
   std::unique_ptr<Impl> impl_;
};

} // namespace mfem

#endif // MFEM_L2P_PAR_MORTAR_ASSEMBLER_HPP
