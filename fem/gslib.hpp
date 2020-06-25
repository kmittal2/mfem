﻿// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_GSLIB
#define MFEM_GSLIB

#include "../config/config.hpp"
#include "gridfunc.hpp"

#ifdef MFEM_USE_GSLIB

struct comm;
struct findpts_data_2;
struct findpts_data_3;
struct array;
struct crystal;

namespace mfem
{

class FindPointsGSLIB
{
protected:


   Mesh *mesh;
   IntegrationRule *ir_simplex;
   struct findpts_data_2 *fdata2D;
   struct findpts_data_3 *fdata3D;
   int dim, points_cnt;
   Array<unsigned int> gsl_code, gsl_proc, gsl_elem;
   Vector gsl_mesh, gsl_ref, gsl_dist;
   bool setupflag;
   struct crystal *cr;
   struct comm *gsl_comm;
   Mesh *meshsplit;

   /// Get GridFunction from MFEM format to GSLIB format
   void GetNodeValues(const GridFunction &gf_in, Vector &node_vals);
   /// Get nodal coordinates from mesh to the format expected by GSLIB for quads
   /// and hexes
   void GetQuadHexNodalCoordinates();
   /// Convert simplices to quad/hexes and then get nodal coordinates for each
   /// split element into format expected by GSLIB
   void GetSimplexNodalCoordinates();

   /** Searches positions given in physical space by @a point_pos. All output
       Arrays and Vectors are expected to have the correct size.

       @param[in]  point_pos  Positions to be found. Must by ordered by nodes
                              (XXX...,YYY...,ZZZ).
       @param[out] codes      Return codes for each point: inside element (0),
                              element boundary (1), not found (2).
       @param[out] proc_ids   MPI proc ids where the points were found.
       @param[out] elem_ids   Element ids where the points were found.
       @param[out] ref_pos    Reference coordinates of the found point. Ordered
                              by vdim (XYZ,XYZ,XYZ...).
                              Note: the gslib reference frame is [-1,1].
       @param[out] dist       Distance between the sought and the found point
                              in physical space. */
   void FindPoints(const Vector &point_pos, Array<unsigned int> &codes,
                   Array<unsigned int> &proc_ids, Array<unsigned int> &elem_ids,
                   Vector &ref_pos, Vector &dist);

   /** Interpolation of field values at prescribed reference space positions.

       @param[in] codes       Return codes for each point: inside element (0),
                              element boundary (1), not found (2).
       @param[in] proc_ids    MPI proc ids where the points were found.
       @param[in] elem_ids    Element ids where the points were found.
       @param[in] ref_pos     Reference coordinates of the found point. Ordered
                              by vdim (XYZ,XYZ,XYZ...).
                              Note: the gslib reference frame is [-1,1].
       @param[in] field_in    Function values that will be interpolated on the
                              reference positions. Note: it is assumed that
                              @a field_in is in H1 and in the same space as the
                              mesh that was given to Setup().
       @param[out] field_out  Interpolated values. */
   void Interpolate(const Array<unsigned int> &codes,
                    const Array<unsigned int> &proc_ids,
                    const Array<unsigned int> &elem_ids, const Vector &ref_pos,
                    const GridFunction &field_in, Vector &field_out);

   /// Use GSLIB for communication and interpolation
   void InterpolateH1(const Array<unsigned int> &codes,
                      const Array<unsigned int> &proc_ids,
                      const Array<unsigned int> &elem_ids, const Vector &ref_pos,
                      const GridFunction &field_in, Vector &field_out);
   /// Uses GSLIB Crystal Router for communication followed by GetValue for
   /// interpolation
   void InterpolateGeneral(const Array<unsigned int> &codes,
                           const Array<unsigned int> &proc_ids,
                           const Array<unsigned int> &elem_ids,
                           const Vector &ref_pos, const GridFunction &field_in,
                           Vector &field_out);
   /// Map {r,s,t} coordinates from [-1,1] to [0,1] for MFEM. For simplices mesh
   /// find the original element number (that was split into micro quads/hexes
   /// by GetSimplexNodalCoordinates())
   void MapRefPosAndElemIndices(Array<unsigned int> &elem_ids, Vector &ref_pos);


public:
   FindPointsGSLIB();

#ifdef MFEM_USE_MPI
   FindPointsGSLIB(MPI_Comm _comm);
#endif

   ~FindPointsGSLIB();

   //enum avgtype {ARITHEMATIC, HARMONIC} avgt;
   enum mfem::GridFunction::AvgType avgtype;

   /** Initializes the internal mesh in gslib, by sending the positions of the
       Gauss-Lobatto nodes of the input Mesh object @a m.
       Note: not tested with periodic (DG meshes).
       Note: the input mesh @a m must have Nodes set.

       @param[in] m         Input mesh.
       @param[in] bb_t      Relative size of bounding box around each element.
       @param[in] newt_tol  Newton tolerance for the gslib search methods.
       @param[in] npt_max   Number of points for simultaneous iteration. This
                            alters performance and memory footprint. */
   void Setup(Mesh &m, const double bb_t = 0.1, const double newt_tol = 1.0e-12,
              const int npt_max = 256);

   /** Searches positions given in physical space by @a point_pos. */
   void FindPoints(const Vector &point_pos);
   /// Setup FindPoints and search positions
   void FindPoints(Mesh &m, const Vector &point_pos, const double bb_t = 0.1,
                   const double newt_tol = 1.0e-12,  const int npt_max = 256);

   /** Interpolation of field values at prescribed reference space positions. */
   void Interpolate(const GridFunction &field_in, Vector &field_out);
   /** Search positions and interpolate */
   void Interpolate(const Vector &point_pos, const GridFunction &field_in,
                    Vector &field_out);
   /** Setup FindPoints, search positions and interpolate */
   void Interpolate(Mesh &m, const Vector &point_pos,
                    const GridFunction &field_in, Vector &field_out);

   void SetL2AvgType(mfem::GridFunction::AvgType avgtype_) { avgtype = avgtype_; }

   /** Cleans up memory allocated internally by gslib.
       Note that in parallel, this must be called before MPI_Finalize(), as
       it calls MPI_Comm_free() for internal gslib communicators. */
   void FreeData();

   /// Return code for each point searched by FindPoints: inside element (0), on
   /// element boundary (1), or not found (2).
   const Array<unsigned int> &GetCode() const { return gsl_code; }
   /// Return element number for each point found by FindPoints.
   const Array<unsigned int> &GetElem() const { return gsl_elem; }
   /// Return MPI rank on which each point was found by FindPoints.
   const Array<unsigned int> &GetProc() const { return gsl_proc; }
   /// Return reference coordinates for each point found by FindPoints.
   const Vector &GetReferencePosition() const { return gsl_ref;  }
   /// Return distance Distance between the sought and the found point
   /// in physical space, for each point found by FindPoints.
   const Vector &GetDist()              const { return gsl_dist; }
};

} // namespace mfem

#endif //MFEM_USE_GSLIB

#endif //MFEM_GSLIB guard
