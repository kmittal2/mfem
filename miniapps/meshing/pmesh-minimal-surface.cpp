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
//
//              --------------------------------
//              Parallel Minimal Surface Miniapp
//              --------------------------------
//
// Description:  See mesh-minimal-surface.cpp description.
//
// Compile with: make pmesh-minimal-surface
//
// Sample runs:  mesh-minimal-surface -vis
//
// Device sample runs:
//               mesh-minimal-surface -d cuda

#include "mfem.hpp"
#include "general/forall.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Constant variables
const double PI = M_PI;
const double NRM = 1.e-4;
const double EPS = 1.e-24;

// Static variables for GLVis.
static int NRanks, MyRank;
static socketstream glvis;
const int  visport = 19916;
const char vishost[] = "localhost";

// Surface mesh class
template<typename T = nullptr_t>
class Surface: public Mesh
{
protected:
   T *S;
   Array<int> &bc;
   int order, nx, ny, nr, vdim;
   ParMesh *pmesh = nullptr;
   H1_FECollection *fec = nullptr;
   ParFiniteElementSpace *pfes = nullptr;
public:

   // Reading from mesh file
   Surface(Array<int> &bc, int order, const char *file, int nr, int vdim):
      Mesh(file, true), S(static_cast<T*>(this)),
      bc(bc), order(order), nr(nr), vdim(vdim)
   {
      //EnsureNodes();
      S->Postfix();
      S->Refine();
      GenFESpace();
      S->BoundaryConditions();
   }

   // Generate Quad surface mesh
   Surface(Array<int> &b, int order, int nx, int ny, int nr, int vdim):
      Mesh(nx, ny, Element::QUADRILATERAL, true, 1.0, 1.0, false),
      S(static_cast<T*>(this)),
      bc(b), order(order), nx(nx), ny(ny), nr(nr), vdim(vdim)
   {
      //EnsureNodes();
      S->Prefix();
      S->Create();
      S->Postfix();
      S->Refine();
      RemoveUnusedVertices();
      RemoveInternalBoundaries();
      SetCurvature(order, false, 3, Ordering::byVDIM);
      GridFunction &nodes = *GetNodes();
      for (int i = 0; i < nodes.Size(); i++)
      { if (std::abs(nodes(i)) < EPS) { nodes(i) = 0.0; } }
      SetCurvature(order, false, 3, Ordering::byNODES);
      GenFESpace();
      S->BoundaryConditions();
   }

   // Generated Cube surface mesh
   Surface(Array<int> &b, int order, int nr,
           int NVert, int NElem, int NBdrElem, int vdim):
      Mesh(2, NVert, NElem, NBdrElem, 3), S(static_cast<T*>(this)),
      bc(b), order(order), nx(NVert), ny(NElem), nr(nr), vdim(vdim)
   {
      //EnsureNodes();
      //S->Prefix();
      S->Create();
      //S->Postfix();
      //S->Refine();
      //RemoveUnusedVertices();
      //RemoveInternalBoundaries();
      GenFESpace();
      S->BoundaryConditions();
   }

   ~Surface() { delete fec; delete pmesh; delete pfes; }

   void Prefix() { SetCurvature(order, false, 3, Ordering::byNODES); }

   void Create() { Transform(T::Parametrization); }

   void Postfix() { SetCurvature(order, false, 3, Ordering::byNODES); }

   void Refine()
   {
      for (int l = 0; l < nr; l++) { UniformRefinement(); }
      // Adaptive mesh refinement
      //if (amr)  { for (int l = 0; l < 1; l++) { RandomRefinement(0.5); } }
      //PrintCharacteristics();
   }

   void BoundaryConditions()
   {
      if (bdr_attributes.Size())
      {
         Array<int> ess_bdr(bdr_attributes.Max());
         ess_bdr = 1;
         pfes->GetEssentialTrueDofs(ess_bdr, bc);
      }
   }

   void GenFESpace()
   {
      fec = new H1_FECollection(order, 2);
      pmesh = new ParMesh(MPI_COMM_WORLD, *this);
      pfes = new ParFiniteElementSpace(pmesh, fec, vdim);
   }

   ParMesh *GetMesh() const { return pmesh; }

   ParFiniteElementSpace *GetFESpace() const { return pfes; }
};

// Default surface mesh file
struct MeshFromFile: public Surface<MeshFromFile>
{
   MeshFromFile(Array<int> &BC, int o, const char *file, int r, int d):
      Surface(BC, o, file, r, d) {}
};

// #0: Catenoid surface
struct Catenoid: public Surface<Catenoid>
{
   Catenoid(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // u in [0,2π] and v in [-2π/3,2π/3]
      const double u = 2.0*PI*x[0];
      const double v = 2.0*PI*(2.0*x[1]-1.0)/3.0;
      p[0] = 3.2*cos(u); // cos(u)*cosh(v);
      p[1] = 3.2*sin(u); // sin(u)*cosh(v);
      p[2] = v;
   }
   // Prefix of the Catenoid surface
   void Prefix()
   {
      SetCurvature(order, false, 3, Ordering::byNODES);
      Array<int> v2v(GetNV());
      for (int i = 0; i < v2v.Size(); i++) { v2v[i] = i; }
      // identify vertices on vertical lines
      for (int j = 0; j <= ny; j++)
      {
         const int v_old = nx + j * (nx + 1);
         const int v_new =      j * (nx + 1);
         v2v[v_old] = v_new;
      }
      // renumber elements
      for (int i = 0; i < GetNE(); i++)
      {
         Element *el = GetElement(i);
         int *v = el->GetVertices();
         const int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         { v[j] = v2v[v[j]]; }
      }
      // renumber boundary elements
      for (int i = 0; i < GetNBE(); i++)
      {
         Element *el = GetBdrElement(i);
         int *v = el->GetVertices();
         const int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         { v[j] = v2v[v[j]]; }
      }
      RemoveUnusedVertices();
      RemoveInternalBoundaries();
   }
};

// #1: Helicoid surface
struct Helicoid: public Surface<Helicoid>
{
   Helicoid(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      const double a = 1.0;
      // u in [0,2π] and v in [-2π/3,2π/3]
      const double u = 2.0*PI*x[0];
      const double v = 2.0*PI*(2.0*x[1]-1.0)/3.0;
      p[0] = a*cos(u)*sinh(v);
      p[1] = a*sin(u)*sinh(v);
      p[2] = a*u;
   }
};

// #2: Enneper's surface
struct Enneper: public Surface<Enneper>
{
   Enneper(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // (u,v) in [-2, +2]
      const double u = 2.0*(2.0*x[0]-1.0);
      const double v = 2.0*(2.0*x[1]-1.0);
      p[0] = +u - u*u*u/3.0 + u*v*v;
      p[1] = -v - u*u*v + v*v*v/3.0;
      p[2] =  u*u - v*v;
   }
};

// #3: Parametrization of Scherk's doubly periodic surface
struct Scherk: public Surface<Scherk>
{
   Scherk(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      const double alpha = 0.49;
      // (u,v) in [-απ, +απ]
      const double u = alpha*PI*(2.0*x[0]-1.0);
      const double v = alpha*PI*(2.0*x[1]-1.0);
      p[0] = u;
      p[1] = v;
      p[2] = log(cos(u)/cos(v));
   }
};

// #4: Hold surface
struct Hold: public Surface<Hold>
{
   Hold(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // u in [0,2π] and v in [0,1]
      const double u = 2.0*PI*x[0];
      const double v = x[1];
      p[0] = cos(u)*(1.0 + 0.3*sin(3.*u + PI*v));
      p[1] = sin(u)*(1.0 + 0.3*sin(3.*u + PI*v));
      p[2] = v;
   }
   void Prefix()
   {
      SetCurvature(order, false, 3, Ordering::byNODES);

      Array<int> v2v(GetNV());
      for (int i = 0; i < v2v.Size(); i++) { v2v[i] = i; }
      // identify vertices on vertical lines
      for (int j = 0; j <= ny; j++)
      {
         const int v_old = nx + j * (nx + 1);
         const int v_new =      j * (nx + 1);
         v2v[v_old] = v_new;
      }
      // renumber elements
      for (int i = 0; i < GetNE(); i++)
      {
         Element *el = GetElement(i);
         int *v = el->GetVertices();
         const int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         { v[j] = v2v[v[j]]; }
      }
      // renumber boundary elements
      for (int i = 0; i < GetNBE(); i++)
      {
         Element *el = GetBdrElement(i);
         int *v = el->GetVertices();
         const int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         { v[j] = v2v[v[j]]; }
      }
      RemoveUnusedVertices();
      RemoveInternalBoundaries();
   }
};

// #5: 1/4th Peach street model
struct QuarterPeach: public Surface<QuarterPeach>
{
   QuarterPeach(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
   static void Parametrization(const Vector &X, Vector &p)
   {
      p = X;
      const double x = 2.0*X[0]-1.0;
      const double y = X[1];
      const double r = sqrt(x*x + y*y);
      const double t = (x==0.0) ? PI/2.0 :
                       (y==0.0 && x>0.0) ? 0. :
                       (y==0.0 && x<0.0) ? PI : acos(x/r);
      const double sqrtx = sqrt(1.0 + x*x);
      const double sqrty = sqrt(1.0 + y*y);
      const bool yaxis = PI/4.0<t && t < 3.0*PI/4.0;
      const double R = yaxis?sqrtx:sqrty;
      const double gamma = r/R;
      p[0] = gamma * cos(t);
      p[1] = gamma * sin(t);
      p[2] = 1.0 - gamma;
   }
   void Prefix() { SetCurvature(1, false, 3, Ordering::byNODES); }
   void Postfix()
   {
      for (int i = 0; i < GetNBE(); i++)
      {
         Element *el = GetBdrElement(i);
         const int fn = GetBdrElementEdgeIndex(i);
         MFEM_VERIFY(!FaceIsTrueInterior(fn),"");
         Array<int> vertices;
         GetFaceVertices(fn, vertices);
         const GridFunction *nodes = GetNodes();
         Vector nval;
         double R[2], X[2][3];
         for (int v = 0; v < 2; v++)
         {
            R[v] = 0.0;
            const int iv = vertices[v];
            for (int d = 0; d < 3; d++)
            {
               nodes->GetNodalValues(nval, d+1);
               const double x = X[v][d] = nval[iv];
               if (d < 2) { R[v] += x*x; }
            }
         }
         if (fabs(X[0][1])<=EPS && fabs(X[1][1])<=EPS &&
             (R[0]>0.1 || R[1]>0.1))
         { el->SetAttribute(1); }
         else { el->SetAttribute(2); }
      }
   }
};

// #6: Full Peach street model
struct FullPeach: public Surface<FullPeach>
{
   FullPeach(Array<int> &BC, int o, int r, int d):
      Surface(BC, o, r, 8, 6, 6, d) { }
   void Create()
   {
      const double quad_v[8][3] =
      {
         {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
         {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}
      };
      const int quad_e[6][4] =
      {
         {3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
         {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}

      };
      for (int j = 0; j < nx; j++) { AddVertex(quad_v[j]); }
      for (int j = 0; j < ny; j++) { AddQuad(quad_e[j], j+1); }
      for (int j = 0; j < ny; j++) { AddBdrQuad(quad_e[j], j+1); }
      //RemoveUnusedVertices();
      FinalizeQuadMesh(1, 1, true);
      EnsureNodes();
      //FinalizeTopology();
      UniformRefinement();
      for (int l = 0; l < nr; l++) { UniformRefinement(); }
      SetCurvature(order, false, 3, Ordering::byNODES);
      // Snap the nodes to the unit sphere
      const int sdim = SpaceDimension();
      GridFunction &nodes = *GetNodes();
      Vector node(sdim);
      for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
      {
         for (int d = 0; d < sdim; d++)
         { node(d) = nodes(nodes.FESpace()->DofToVDof(i, d)); }
         node /= node.Norml2();
         for (int d = 0; d < sdim; d++)
         { nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d); }
      }
   }

   void BoundaryConditions()
   {
      double X[3];
      Array<int> dofs;
      Array<int> ess_cdofs, ess_tdofs;
      ess_cdofs.SetSize(pfes->GetVSize());
      ess_cdofs = 0;
      for (int e = 0; e < pfes->GetNE(); e++)
      {
         pfes->GetElementDofs(e, dofs);
         for (int c = 0; c < dofs.Size(); c++)
         {
            int k = dofs[c];
            if (k < 0) { k = -1 - k; }
            pfes->GetMesh()->GetNode(k, X);
            const bool halfX = fabs(X[0]) < EPS && X[1] <= 0.0;
            const bool halfY = fabs(X[2]) < EPS && X[1] >= 0.0;
            const bool is_on_bc = halfX || halfY;
            for (int d = 0; d < vdim; d++)
            { ess_cdofs[pfes->DofToVDof(k, d)] = is_on_bc; }
         }
      }
      const SparseMatrix *R = pfes->GetRestrictionMatrix();
      if (!R) { ess_tdofs.MakeRef(ess_cdofs); }
      else { R->BooleanMult(ess_cdofs, ess_tdofs); }
      ParFiniteElementSpace::MarkerToList(ess_tdofs, bc);
   }
};

// #7: Full Peach street model
struct SlottedSphere: public Surface<SlottedSphere>
{
   SlottedSphere(Array<int> &BC, int o, int r, int d):
      Surface(BC, o, r, 64, 40, 0, d) { }
   void Create()
   {
      constexpr double delta = 0.15;
      constexpr int nv1d = 4;
      constexpr int nv = nv1d*nv1d*nv1d;
      constexpr int nel_per_face = (nv1d-1)*(nv1d-1);
      constexpr int nel_total = nel_per_face*6;
      const double vert1d[nv1d] = {-1.0, -delta, delta, 1.0};
      double quad_v[nv][3];

      for (int iv=0; iv<nv; ++iv)
      {
         int ix = iv % nv1d;
         int iy = (iv / nv1d) % nv1d;
         int iz = (iv / nv1d) / nv1d;

         quad_v[iv][0] = vert1d[ix];
         quad_v[iv][1] = vert1d[iy];
         quad_v[iv][2] = vert1d[iz];
      }

      int quad_e[nel_total][4];

      for (int ix=0; ix<nv1d-1; ++ix)
      {
         for (int iy=0; iy<nv1d-1; ++iy)
         {
            int el_offset = ix + iy*(nv1d-1);
            // x = 0
            quad_e[0*nel_per_face + el_offset][0] = nv1d*ix + nv1d*nv1d*iy;
            quad_e[0*nel_per_face + el_offset][1] = nv1d*(ix+1) + nv1d*nv1d*iy;
            quad_e[0*nel_per_face + el_offset][2] = nv1d*(ix+1) + nv1d*nv1d*(iy+1);
            quad_e[0*nel_per_face + el_offset][3] = nv1d*ix + nv1d*nv1d*(iy+1);
            // x = 1
            int x_off = nv1d-1;
            quad_e[1*nel_per_face + el_offset][3] = x_off + nv1d*ix + nv1d*nv1d*iy;
            quad_e[1*nel_per_face + el_offset][2] = x_off + nv1d*(ix+1) + nv1d*nv1d*iy;
            quad_e[1*nel_per_face + el_offset][1] = x_off + nv1d*(ix+1) + nv1d*nv1d*(iy+1);
            quad_e[1*nel_per_face + el_offset][0] = x_off + nv1d*ix + nv1d*nv1d*(iy+1);
            // y = 0
            quad_e[2*nel_per_face + el_offset][0] = nv1d*nv1d*iy + ix;
            quad_e[2*nel_per_face + el_offset][1] = nv1d*nv1d*iy + ix + 1;
            quad_e[2*nel_per_face + el_offset][2] = nv1d*nv1d*(iy+1) + ix + 1;
            quad_e[2*nel_per_face + el_offset][3] = nv1d*nv1d*(iy+1) + ix;
            // y = 1
            int y_off = nv1d*(nv1d-1);
            quad_e[3*nel_per_face + el_offset][0] = y_off + nv1d*nv1d*iy + ix;
            quad_e[3*nel_per_face + el_offset][1] = y_off + nv1d*nv1d*iy + ix + 1;
            quad_e[3*nel_per_face + el_offset][2] = y_off + nv1d*nv1d*(iy+1) + ix + 1;
            quad_e[3*nel_per_face + el_offset][3] = y_off + nv1d*nv1d*(iy+1) + ix;
            // z = 0
            quad_e[4*nel_per_face + el_offset][0] = nv1d*iy + ix;
            quad_e[4*nel_per_face + el_offset][1] = nv1d*iy + ix + 1;
            quad_e[4*nel_per_face + el_offset][2] = nv1d*(iy+1) + ix + 1;
            quad_e[4*nel_per_face + el_offset][3] = nv1d*(iy+1) + ix;
            // z = 1
            int z_off = nv1d*nv1d*(nv1d-1);
            quad_e[5*nel_per_face + el_offset][0] = z_off + nv1d*iy + ix;
            quad_e[5*nel_per_face + el_offset][1] = z_off + nv1d*iy + ix + 1;
            quad_e[5*nel_per_face + el_offset][2] = z_off + nv1d*(iy+1) + ix + 1;
            quad_e[5*nel_per_face + el_offset][3] = z_off + nv1d*(iy+1) + ix;
         }
      }

      // Delete on x = 0 face
      quad_e[0*nel_per_face + 1 + 2*(nv1d-1)][0] = -1;
      quad_e[0*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      // Delete on x = 1 face
      quad_e[1*nel_per_face + 1 + 2*(nv1d-1)][0] = -1;
      quad_e[1*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      // Delete on y = 1 face
      quad_e[3*nel_per_face + 1 + 0*(nv1d-1)][0] = -1;
      quad_e[3*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      // Delete on z = 1 face
      quad_e[5*nel_per_face + 0 + 1*(nv1d-1)][0] = -1;
      quad_e[5*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      quad_e[5*nel_per_face + 2 + 1*(nv1d-1)][0] = -1;
      // Delete on z = 0 face
      quad_e[4*nel_per_face + 1 + 0*(nv1d-1)][0] = -1;
      quad_e[4*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      quad_e[4*nel_per_face + 1 + 2*(nv1d-1)][0] = -1;
      // Delete on y = 0 face
      quad_e[2*nel_per_face + 1 + 0*(nv1d-1)][0] = -1;
      quad_e[2*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;

      for (int j = 0; j < nv; j++) { AddVertex(quad_v[j]); }
      for (int j = 0; j < nel_total; j++)
      {
         if (quad_e[j][0] < 0) { continue; }
         AddQuad(quad_e[j], j+1);
      }
      RemoveUnusedVertices();
      FinalizeQuadMesh(1, 1, true);
      EnsureNodes();
      FinalizeTopology();
      for (int l = 0; l < nr; l++) { UniformRefinement(); }
      SetCurvature(order, false, 3, Ordering::byNODES);
      // Snap the nodes to the unit sphere
      const int sdim = SpaceDimension();
      GridFunction &nodes = *GetNodes();
      Vector node(sdim);
      for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
      {
         for (int d = 0; d < sdim; d++)
         { node(d) = nodes(nodes.FESpace()->DofToVDof(i, d)); }
         node /= node.Norml2();
         for (int d = 0; d < sdim; d++)
         { nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d); }
      }
   }
};

// #8: Shell surface model
struct Shell: public Surface<Shell>
{
   Shell(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // u in [0,2π] and v in [-15, 6]
      const double u = 2.0*PI*x[0];
      const double v = 21.0*x[1]-15.0;
      p[0] = +1.0*pow(1.16,v)*cos(v)*(1.0+cos(u));
      p[1] = -1.0*pow(1.16,v)*sin(v)*(1.0+cos(u));
      p[2] = -2.0*pow(1.16,v)*(1.0+sin(u));
   }
};


// Visualize some solution on the given mesh
static void Visualize(ParMesh *pm, const int w, const int h,
                      const char *keys)
{
   glvis << "parallel " << NRanks << " " << MyRank << "\n";
   const GridFunction *x = pm->GetNodes();
   glvis << "solution\n" << *pm << *x;
   glvis << "window_size " << w << " " << h <<"\n";
   glvis << "keys " << keys << "\n";
   glvis.precision(8);
   glvis << flush;
}

static void Visualize(ParMesh *pm,  const bool pause)
{
   glvis << "parallel " << NRanks << " " << MyRank << "\n";
   const GridFunction *x = pm->GetNodes();
   glvis << "solution\n" << *pm << *x;
   if (pause) { glvis << "pause\n"; }
   glvis << flush;
}

// Surface solver class
template<class Type>
class SurfaceSolver
{
protected:
   bool pa, vis, pause, radial;
   int nmax, vdim, order;
   ParMesh *pmesh;
   Vector X, B;
   OperatorPtr A;
   ParFiniteElementSpace *pfes;
   ParBilinearForm a;
   Array<int> &bc;
   ParGridFunction x, x0, b;
   ConstantCoefficient one;
   Type *solver;
   Solver *M;
   const int print_iter = -1, max_num_iter = 2000;
   const double lambda = 0.0, RTOLERANCE = EPS, ATOLERANCE = 0.0;
public:
   SurfaceSolver(const bool pa, const bool vis,
                 const int n, const bool pause,
                 const int order, const double l, const bool radial,
                 ParMesh *pmesh, ParFiniteElementSpace *pfes,
                 Array<int> &bc):
      pa(pa), vis(vis), pause(pause), radial(radial), nmax(n),
      vdim(pfes->GetVDim()), order(order), pmesh(pmesh), pfes(pfes),
      a(pfes), bc(bc), x(pfes), x0(pfes), b(pfes), one(1.0),
      solver(static_cast<Type*>(this)), M(nullptr), lambda(l) { }
   ~SurfaceSolver() { delete M; }
   void Solve()
   {
      if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL);}
      for (int i=0; i<nmax; ++i)
      {
         if (MyRank == 0)
         {
            mfem::out << "Linearized iteration " << i << ": ";
         }
         Update();
         a.Assemble();
         if (solver->Loop()) { break; }
      }
   }
   bool Converged(const double rnorm)
   {
      if (rnorm < NRM)
      {
         if (MyRank==0) { mfem::out << "Converged!" << endl; }
         return true;
      }
      return false;
   }
   bool ParAXeqB(bool by_component)
   {
      b = 0.0;
      a.FormLinearSystem(bc, x, b, A, X, B);
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetPrintLevel(print_iter);
      cg.SetMaxIter(max_num_iter);
      cg.SetRelTol(sqrt(RTOLERANCE));
      cg.SetAbsTol(sqrt(ATOLERANCE));
      if (!pa) { M = new HypreBoomerAMG; }
      if (M) { cg.SetPreconditioner(*M); }
      cg.SetOperator(*A);
      cg.Mult(B, X);
      a.RecoverFEMSolution(X, b, x);
      GridFunction *nodes = by_component ? &x0 : pfes->GetMesh()->GetNodes();
      double rnorm = nodes->DistanceTo(x) / nodes->Norml2();
#ifdef MFEM_USE_MPI
      double glob_norm;
      MPI_Allreduce(&rnorm, &glob_norm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      rnorm = glob_norm;
#endif
      if (MyRank == 0) { mfem::out << "rnorm = " << rnorm << endl; }
      if (by_component)
      {
         MFEM_VERIFY(lambda == 0.0,"'By component' assumes lambda == 0.0");
         MFEM_VERIFY(!radial,"'By component' solver can't use 'radial option");
         return Converged(rnorm);
      }
      if (!radial) { add(lambda, *nodes, 1.0-lambda, x, *nodes); }
      else
      {
         GridFunction delta(pfes); // x = nodes + delta
         subtract(x,*nodes,delta);
         // position and delta vectors at point i
         Vector ni(3), di(3);
         for (int i = 0; i < delta.Size()/3; i++)
         {
            // extract local vectors
            const int ndof = pfes->GetNDofs();
            for (int d = 0; d < 3; d++)
            {
               ni(d) = (*nodes)(d*ndof + i);
               di(d) = delta(d*ndof + i);
            }
            // project the delta vector in radial direction
            const double ndotd = (ni*di) / (ni*ni);
            di.Set(ndotd,ni);
            // set global vectors
            for (int d = 0; d < 3; d++) { delta(d*ndof + i) = di(d); }
         }
         add(lambda, delta, 1.0-lambda, *nodes, *nodes);
      }
      return Converged(rnorm);
   }
   void Update()
   {
      if (vis) { Visualize(pmesh, pause); }
      pmesh->DeleteGeometricFactors();
      a.Update();
   }
};

// Surface solver 'by compnents'
class ByComponent: public SurfaceSolver<ByComponent>
{
public:
   void SetNodes(const GridFunction &Xi, const int c)
   {
      auto d_Xi = Xi.Read();
      auto d_nodes  = pfes->GetMesh()->GetNodes()->Write();
      const int ndof = pfes->GetNDofs();
      MFEM_FORALL(i, ndof, d_nodes[c*ndof + i] = d_Xi[i]; );
   }

   void GetNodes(GridFunction &Xi, const int c)
   {
      auto d_Xi = Xi.Write();
      const int ndof = pfes->GetNDofs();
      auto d_nodes  = pfes->GetMesh()->GetNodes()->Read();
      MFEM_FORALL(i, ndof, d_Xi[i] = d_nodes[c*ndof + i]; );
   }

public:
   ByComponent(bool PA, bool glvis, int n, bool wait, int o, double l, bool rad,
               ParMesh *xmesh, ParFiniteElementSpace *xfes, Array<int> &BC):
      SurfaceSolver(PA, glvis, n, wait, o, l, rad, xmesh, xfes, BC)
   { a.AddDomainIntegrator(new DiffusionIntegrator(one)); }
   bool Loop()
   {
      bool cvg[3] {false};
      for (int c=0; c < 3; ++c)
      {
         this->GetNodes(x, c);
         x0 = x;
         cvg[c] = this->ParAXeqB(true);
         this->SetNodes(x, c);
      }
      const bool converged = cvg[0] && cvg[1] && cvg[2];
      return converged ? true : false;
   }
};

// Surface solver 'by vector'
class ByVector: public SurfaceSolver<ByVector>
{
public:
   ByVector(bool PA, bool glvis, int n, bool wait, int o, double l, bool rad,
            ParMesh *xmsh, ParFiniteElementSpace *xfes, Array<int> &BC):
      SurfaceSolver(PA, glvis, n, wait, o, l, rad, xmsh, xfes, BC)
   { a.AddDomainIntegrator(new VectorDiffusionIntegrator(one)); }
   bool Loop()
   {
      x = *pfes->GetMesh()->GetNodes();
      bool converge = this->ParAXeqB(false);
      pmesh->SetNodes(x);
      return converge ? true : false;
   }
};

Mesh *NewMeshFromSurface(const int surface, Array<int> &bc,
                         int o, int x, int y, int r, int d)
{
   switch (surface)
   {
      case 0: return new Catenoid(bc, o, x, y, r, d);
      case 1: return new Helicoid(bc, o, x, y, r, d);
      case 2: return new Enneper(bc, o, x, y, r, d);
      case 3: return new Scherk(bc, o, x, y, r, d);
      case 4: return new Hold(bc, o, x, y, r, d);
      case 5: return new QuarterPeach(bc, o, x, y, r, d);
      case 6: return new FullPeach(bc, o, r, d);
      case 7: return new SlottedSphere(bc, o, r, d);
      case 8: return new Shell(bc, o, x, y, r, d);
      default: ;
   }
   mfem_error("Unknown surface (0 <= surface <= 8)!");
   return nullptr;
}

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   NRanks = num_procs; MyRank = myid;

   // Parse command-line options.
   int o = 4;
   int x = 4;
   int y = 4;
   int r = 2;
   int nmax = 16;
   int surface = -1;
   bool pa = true;
   bool vis = true;
   bool amr = false;
   bool wait = false;
   bool rad = false;
   double lambda = 0.0;
   bool solve_by_components = false;
   const char *keys = "gAmmaaa";
   const char *device_config = "cpu";
   const char *mesh_file = "../../data/mobius-strip.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&wait, "-w", "--wait", "-no-w", "--no-wait",
                  "Enable or disable a GLVis pause.");
   args.AddOption(&rad, "-rad", "--radial", "-no-rad", "--no-radial",
                  "Enable or disable radial constraints in solver.");
   args.AddOption(&x, "-x", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&y, "-y", "--num-elements-y",
                  "Number of elements in y-direction.");
   args.AddOption(&o, "-o", "--order", "Finite element order.");
   args.AddOption(&r, "-r", "--ref-levels", "Refinement");
   args.AddOption(&nmax, "-n", "--niter-max", "Max number of iterations");
   args.AddOption(&surface, "-s", "--surface", "Choice of the surface.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&lambda, "-l", "--lambda", "Lambda step toward solution.");
   args.AddOption(&amr, "-amr", "--adaptive-mesh-refinement", "-no-amr",
                  "--no-adaptive-mesh-refinement", "Enable AMR.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&keys, "-k", "--keys", "GLVis configuration keys.");
   args.AddOption(&vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable visualization.");
   args.AddOption(&solve_by_components, "-c", "--components",
                  "-no-c", "--no-components",
                  "Enable or disable the 'by component' solver");

   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   if (myid == 0) { args.PrintOptions(cout); }
   MFEM_VERIFY(!amr, "AMR not yet supported!");

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // Initialize GLVis server if 'visualization' is set.
   if (vis) { vis = glvis.open(vishost, visport) == 0; }

   // Initialize our surface mesh from command line option and determine
   // the list of true (i.e. conforming) essential boundary dofs.
   Mesh *mesh;
   Array<int> bc;
   const int d = solve_by_components ? 1 : 3;
   mesh = (surface < 0) ? new MeshFromFile(bc, o, mesh_file, r, d) :
          NewMeshFromSurface(surface, bc, o, x, y, r, d);
   MFEM_VERIFY(mesh, "Not a valid surface number!");

   // Grab back the pmesh & pfes from the Surface object.
   Surface<> &S = *static_cast<Surface<>*>(mesh);
   ParMesh *pmesh = S.GetMesh();
   ParFiniteElementSpace *pfes = S.GetFESpace();

   // Send to GLVis the first mesh and set the 'keys' options.
   if (vis) { Visualize(pmesh, 800, 800, keys); }

   // Create and launch the surface solver.
   if (solve_by_components)
   { ByComponent(pa, vis, nmax, wait, o, lambda, rad, pmesh, pfes, bc).Solve(); }
   else
   { ByVector(pa, vis, nmax, wait, o, lambda, rad, pmesh, pfes, bc).Solve(); }

   // Free the used memory.
   delete mesh;
   MPI_Finalize();
   return 0;
}
