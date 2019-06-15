//                                MFEM Example 24
//
// Compile with: make ex24
//
// Sample runs:  ex24
//               ex24 -c
//               ex24 -pa
//               ex24 -pa -c
//               ex24 -p 0
//               ex24 -p 0 -c
//               ex24 -p 0 -pa
//               ex24 -p 0 -pa -c
//               ex24 -p 1
//               ex24 -p 1 -c
//               ex24 -p 1 -pa
//               ex24 -p 1 -pa -c
//               ex24 -p 2
//               ex24 -p 2 -c
//               ex24 -p 2 -pa
//               ex24 -p 2 -pa -c
//               ex24 -p 3
//               ex24 -p 3 -c
//               ex24 -p 3 -pa
//               ex24 -p 3 -pa -c
//               ex24 -p 4
//               ex24 -p 4 -c
//               ex24 -p 4 -pa
//               ex24 -p 4 -pa -c
//               ex24 -p 5
//               ex24 -p 5 -c
//               ex24 -p 5 -pa
//               ex24 -p 5 -pa -c
//               ex24 -p 6
//               ex24 -p 6 -c
//               ex24 -p 6 -pa
//               ex24 -p 6 -pa -c
//               ex24 -p 7
//               ex24 -p 7 -c
//               ex24 -p 7 -pa
//               ex24 -p 7 -pa -c
//
// Device sample runs:
//               ex24 -d cuda -pa
//               ex24 -d cuda -pa -c
//               ex24 -d cuda -p 0 -pa
//               ex24 -d cuda -p 0 -pa -c
//               ex24 -d cuda -p 1 -pa
//               ex24 -d cuda -p 1 -pa -c
//               ex24 -d cuda -p 2 -pa
//               ex24 -d cuda -p 2 -pa -c
//               ex24 -d cuda -p 3 -pa
//               ex24 -d cuda -p 3 -pa -c
//               ex24 -d cuda -p 4 -pa
//               ex24 -d cuda -p 4 -pa -c
//               ex24 -d cuda -p 5 -pa
//               ex24 -d cuda -p 5 -pa -c
//               ex24 -d cuda -p 6 -pa
//               ex24 -d cuda -p 6 -pa -c
//               ex24 -d cuda -p 7 -pa
//               ex24 -d cuda -p 7 -pa -c

#include "mfem.hpp"
#include "../general/dbg.hpp"
#include <cassert>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Constant variables
const double pi = M_PI;
const double eps = 1.e-12;
static socketstream glvis;
const int  visport   = 19916;
const char vishost[] = "localhost";

// Forward declaration
void SnapNodes(Mesh *mesh);

// Surface mesh class
template<class Type = nullptr_t>
class Surface: public Mesh
{
protected:
   const int Order, Nx, Ny, Rl;
   const double Sx;
   const double Sy;
   const int SpaceDim;
   const Element::Type ElType;
   const bool GenerateEdges;
   const bool SpaceFillingCurves;
   const bool Discontinuous;
   Array<int> &dbc;
   Type *S;
public:
   // Reading from mesh file
   Surface(Array<int> &bc, int order, const char *fn, int rl):
      Mesh(fn, true), Order(order), Nx(), Ny(), Rl(rl), Sx(), Sy(),
      SpaceDim(), ElType(), GenerateEdges(true),
      SpaceFillingCurves(), Discontinuous(),
      dbc(bc), S(static_cast<Type*>(this))  { S->Postfix(); S->BC(); }

   // Generate Quad surface mesh
   Surface(Array<int> &bc,
           const int order,
           const int nx = 4,
           const int ny = 4,
           const int rl = 2,
           const double sx = 1.0,
           const double sy = 1.0,
           const int sdim = 3,
           const Element::Type type = Element::QUADRILATERAL,
           const bool edges = true,
           const bool space_filling_curves = true,
           const bool discontinuous = false):
      Mesh(nx, ny, type, edges, sx, sy, space_filling_curves),
      Order(order), Nx(nx), Ny(ny), Rl(rl), Sx(sx), Sy(sy),
      SpaceDim(sdim), ElType(type), GenerateEdges(edges),
      SpaceFillingCurves(space_filling_curves), Discontinuous(discontinuous),
      dbc(bc), S(static_cast<Type*>(this))
   {
      EnsureNodes();
      S->Prefix();
      S->Equation();
      S->Postfix();
      RemoveUnusedVertices();
      RemoveInternalBoundaries();
      SetCurvature(order, discontinuous, sdim, Ordering::byVDIM);
      GridFunction &nodes = *GetNodes();
      for (int i = 0; i < nodes.Size(); i++)
      { if (std::abs(nodes(i)) < eps) { nodes(i) = 0.0; } }
      SetCurvature(order, discontinuous, sdim, Ordering::byNODES);
      S->BC();
   }

   // Generated Cube surface mesh
   Surface(const bool snap,
           Array<int> &bc,
           const int order,
           const int rl,
           const int dim,
           const int NVert,
           const int NElem,
           const int NBdrElem,
           const int sdim):
      Mesh(dim, NVert, NElem, NBdrElem, sdim),
      Order(order), Nx(NVert), Ny(NBdrElem), Rl(rl), Sx(), Sy(),
      SpaceDim(sdim), ElType(Element::QUADRILATERAL), GenerateEdges(true),
      SpaceFillingCurves(true), Discontinuous(false),
      dbc(bc), S(static_cast<Type*>(this))
   { S->Equation(); S->Postfix(); S->BC(); }

   void Prefix()
   {
      dbg("Root Prefix");
      SetCurvature(Order, Discontinuous, SpaceDim, Ordering::byNODES);
   }

   void Equation() { dbg("Root Equation"); Transform(Type::Parametrization); }

   void Postfix()
   {
      dbg("Root Postfix");
      SetCurvature(Order, Discontinuous, SpaceDim, Ordering::byNODES);
      for (int l = 0; l < Rl; l++) { UniformRefinement(); }
      // Adaptive mesh refinement
      //if (amr)  { for (int l = 0; l < 1; l++) { RandomRefinement(0.5); } }
   }

   void BC()
   {
      dbg("Root DirichletBC");
      if (bdr_attributes.Size())
      {
         Array<int> ess_bdr(bdr_attributes.Max());
         ess_bdr = 1;
         const_cast<FiniteElementSpace*>
         (GetNodalFESpace())->GetEssentialTrueDofs(ess_bdr, dbc);
      }
   }

};

// Mesh file surface
struct File: public Surface<File>
{ File(Array<int> &bc, int o, const char *f, int r): Surface(bc,o,f,r) {} };

// Catenoid surface
struct Catenoid: public Surface<Catenoid>
{
   Catenoid(Array<int> &bc, int o, int x, int y, int r): Surface(bc,o,x,y,r) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // u in [0,2π] and v in [-2π/3,2π/3]
      const double u = 2.0*pi*x[0];
      const double v = 2.0*pi*(2.0*x[1]-1.0)/3.0;
      p[0] = cos(u)*cosh(v);
      p[1] = sin(u)*cosh(v);
      p[2] = v;
   }
   // Postfix of the Catenoid surface
   void Postfix()
   {
      Array<int> v2v(GetNV());
      for (int i = 0; i < v2v.Size(); i++) { v2v[i] = i; }
      // identify vertices on vertical lines
      for (int j = 0; j <= Ny; j++)
      {
         const int v_old = Nx + j * (Nx + 1);
         const int v_new =      j * (Nx + 1);
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
      for (int l = 0; l < Rl; l++) { UniformRefinement(); }
   }
};

// Helicoid surface
struct Helicoid: public Surface<Helicoid>
{
   Helicoid(Array<int> &bc, int o, int x, int y, int r): Surface(bc,o,x,y,r) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      const double a = 1.0;
      // u in [0,2π] and v in [-2π/3,2π/3]
      const double u = 2.0*pi*x[0];
      const double v = 2.0*pi*(2.0*x[1]-1.0)/3.0;
      p[0] = a*cos(u)*sinh(v);
      p[1] = a*sin(u)*sinh(v);
      p[2] = a*u;
   }
};

// Enneper's surface
struct Enneper: public Surface<Enneper>
{
   Enneper(Array<int> &bc, int o, int x, int y, int r): Surface(bc,o,x,y,r) {}
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

// Parametrization of Scherk's surface
struct Scherk: public Surface<Scherk>
{
   Scherk(Array<int> &bc, int o, int x, int y, int r): Surface(bc,o,x,y,r) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      const double alpha = 0.49;
      // (u,v) in [-απ, +απ]
      const double u = alpha*pi*(2.0*x[0]-1.0);
      const double v = alpha*pi*(2.0*x[1]-1.0);
      p[0] = u;
      p[1] = v;
      p[2] = log(cos(u)/cos(v));
   }
};

// Shell surface model
struct Shell: public Surface<Shell>
{
   Shell(Array<int> &bc, int o, int x, int y, int r): Surface(bc,o,x,y,r) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // u in [0,2π] and v in [-15, 6]
      const double u = 2.0*pi*x[0];
      const double v = 21.0*x[1]-15.0;
      p[0] = +1.0*pow(1.16,v)*cos(v)*(1.0+cos(u));
      p[1] = -1.0*pow(1.16,v)*sin(v)*(1.0+cos(u));
      p[2] = -2.0*pow(1.16,v)*(1.0+sin(u));
   }
};

// Hold surface
struct Hold: public Surface<Hold>
{
   Hold(Array<int> &bc, int o, int x, int y, int r): Surface(bc,o,x,y,r) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // u in [0,2π] and v in [0,1]
      const double u = 2.0*pi*x[0];
      const double v = x[1];
      p[0] = cos(u)*(1.0 + 0.3*sin(5.*u + pi*v));
      p[1] = sin(u)*(1.0 + 0.3*sin(5.*u + pi*v));
      p[2] = v;
   }
};

// 1/4th Peach street model
// Could set BC: ess_bdr[0] = false; with attribute set to '1' from postfix
struct QPeach: public Surface<QPeach>
{
   QPeach(Array<int> &bc, int o, int x, int y, int r): Surface(bc,o,x,y,r) {}
   static void Parametrization(const Vector &X, Vector &p)
   {
      p = X;
      const double x = 2.0*X[0]-1.0;
      const double y = X[1];
      const double r = sqrt(x*x + y*y);
      const double t = (x==0.0) ? pi/2.0 :
                       (y==0.0 && x>0.0) ? 0. :
                       (y==0.0 && x<0.0) ? pi : acos(x/r);
      const double sqrtx = sqrt(1.0 + x*x);
      const double sqrty = sqrt(1.0 + y*y);
      const bool yaxis = pi/4.0<t && t < 3.0*pi/4.0;
      const double R = yaxis?sqrtx:sqrty;
      const double gamma = r/R;
      p[0] = gamma * cos(t);
      p[1] = gamma * sin(t);
      p[2] = fabs(t)<eps?1.0:0.0;
      p[2] = 1.0 - gamma;
   }
   void Prefix()
   {
      SetCurvature(1, Discontinuous, SpaceDim, Ordering::byNODES);
   }
   void Postfix()
   {
      PrintCharacteristics();
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
         if (fabs(X[0][1])<=eps && fabs(X[1][1])<=eps &&
             (R[0]>0.1 || R[1]>0.1))
         { el->SetAttribute(1); }
         else { el->SetAttribute(2); }
      }
      for (int l = 0; l < Rl; l++) { UniformRefinement(); }
   }
};

// Full Peach street model
struct FPeach: public Surface<FPeach>
{
   FPeach(Array<int> &bc, int o, int r):
      // Snap, order, ref_level, dim, Nvert, Nelem, NBdrElem, sdim
      Surface<FPeach>(true, bc, o, r, 2, 8, 6, 6, 3) { }
   void Equation()
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
      // Nx == NVert and Ny == NBdrElem
      for (int j = 0; j < Nx; j++) { AddVertex(quad_v[j]); }
      for (int j = 0; j < Ny; j++) { AddQuad(quad_e[j], j+1); }
      for (int j = 0; j < Ny; j++) { AddBdrQuad(quad_e[j], j+1); }
      FinalizeQuadMesh(1, 1, true);
      SetCurvature(Order, Discontinuous, SpaceDim, Ordering::byNODES);
      UniformRefinement();
      SnapNodes(this);
   }

   void BC()
   {
      double X[3];
      Array<int> cdofs;
      Array<int> ess_cdofs, ess_tdofs;
      const FiniteElementSpace *nfes = GetNodalFESpace();
      ess_cdofs.SetSize(nfes->GetVSize());
      ess_cdofs = 0;
      for (int e = 0; e < nfes->GetNE(); e++)
      {
         nfes->GetElementDofs(e, cdofs);
         for (int c = 0; c < cdofs.Size(); c++)
         {
            int k = cdofs[c];
            if (k < 0) { k = -1 - k; }
            GetNode(k, X);
            const bool hX =  fabs(X[0]) <= eps && X[1] <= 0;
            const bool hY =  fabs(X[2]) <= eps && X[1] >= 0;
            const bool is_on_bc = hX || hY;
            for (int d = 0; d < SpaceDim; d++)
            { ess_cdofs[nfes->DofToVDof(k, d)] = is_on_bc; }
         }
      }
      const SparseMatrix *R = nfes->GetConformingRestriction();
      if (!R) { ess_tdofs.MakeRef(ess_cdofs); }
      else { R->BooleanMult(ess_cdofs, ess_tdofs); }
      FiniteElementSpace::MarkerToList(ess_tdofs, dbc);
   }
};

// Visualize some solution on the given mesh
static void Visualize(Mesh *mesh, const int order, const bool pause,
                      const char *keys = NULL,
                      const int width = 0, const int height = 0)
{
   const H1_FECollection fec(2, 2);
   FiniteElementSpace *sfes = new FiniteElementSpace(mesh, &fec);
   GridFunction K(sfes);
   const int NE = mesh->GetNE();
   const Element::Type type = Element::QUADRILATERAL;
   const IntegrationRule *ir = &IntRules.Get(type, order);
   for (int i = 0; i < NE; i++)
   {
      ElementTransformation *tr = mesh->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         tr->SetIntPoint(&ir->IntPoint(j));
         K(i) = tr->Jacobian().Weight();
      }
   }
   //glvis << "mesh\n" << *mesh << flush;
   glvis.precision(8);
   glvis << "solution\n" << *mesh << K << flush;
   if (keys) { glvis << "keys " << keys << "\n" << flush; }
   if (width * height > 0)
   { glvis << "window_size " << width << " " << height <<"\n" << flush; }
   if (pause) { glvis << "pause\n" << flush; }
}

// Surface solver class
template<class Type>
class SurfaceSolver
{
protected:
   bool pa, visualization, pause;
   int niter, sdim, order;
   Mesh *mesh;
   Vector X, B;
   OperatorPtr A;
   FiniteElementSpace *fes;
   BilinearForm a;
   Array<int> bc;
   GridFunction x, b, *nodes, solution;
   ConstantCoefficient one;
   Type *solver;
public:
   SurfaceSolver(const bool p, const bool v,
                 const int n, const bool w,
                 const int o, Mesh *m,
                 FiniteElementSpace *f,
                 Array<int> dbc):
      pa(p), visualization(v), pause(w), niter(n),
      sdim(m->SpaceDimension()), order(o), mesh(m), fes(f),
      a(fes), bc(dbc), x(fes), b(fes),
      nodes(mesh->GetNodes()), solution(*nodes), one(1.0),
      solver(static_cast<Type*>(this)) { Solve(); }
   void Solve() { solver->Solve(); }
};

// Surface solver 'by compnents'
class ByComponent: public SurfaceSolver<ByComponent>
{
public:
   ByComponent(const bool pa,  const bool vis,
               const int niter, const bool wait,
               const int order, Mesh *mesh,
               FiniteElementSpace *fes,
               Array<int> dbc):
      SurfaceSolver(pa, vis, niter, wait, order, mesh, fes, dbc) { dbg(""); }
   void Solve()
   {
      if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      for (int iiter=0; iiter<niter; ++iiter)
      {
         a.Assemble();
         solution = *nodes;
         for (int i=0; i<sdim; ++i)
         {
            b = 0.0;
            GetComponent(*nodes, x, i);
            a.FormLinearSystem(bc, x, b, A, X, B);
            if (!pa)
            {
               // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
               GSSmoother M(static_cast<SparseMatrix&>(*A));
               PCG(*A, M, B, X, 3, 2000, eps, 0.0);
            }
            else { CG(*A, B, X, 3, 2000, eps, 0.0); }
            // Recover the solution as a finite element grid function.
            a.RecoverFEMSolution(X, b, x);
            SetComponent(solution, x, i);
         }
         *nodes = solution;
         // Send the solution by socket to a GLVis server.
         if (visualization) { Visualize(mesh, order, pause); }
         a.Update();
      }
   }
private:
   void SetComponent(GridFunction &X, const GridFunction &Xi, const int d)
   {
      const int ndof = fes->GetNDofs();
      for (int i = 0; i < ndof; i++)
      { X[d*ndof + i] = Xi[i]; }
   }

   void GetComponent(const GridFunction &X, GridFunction &Xi, const int d)
   {
      const int ndof = fes->GetNDofs();
      for (int i = 0; i < ndof; i++)
      { Xi[i] = X[d*ndof + i]; }
   }
};

// Surface solver 'by vector'
class ByVector: public SurfaceSolver<ByVector>
{
public:
   ByVector(const bool pa, const bool vis,
            const int niter, const bool wait,
            const int order, Mesh *mesh,
            FiniteElementSpace *fes,
            Array<int> dbc):
      SurfaceSolver(pa, vis, niter, wait, order, mesh, fes, dbc) { dbg(""); }
   void Solve()
   {
      if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      a.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
      for (int iiter=0; iiter<niter; ++iiter)
      {
         a.Assemble();
         b = 0.0;
         x = *nodes; // should only copy the BC
         a.FormLinearSystem(bc, x, b, A, X, B);
         if (!pa)
         {
            // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
            GSSmoother M(static_cast<SparseMatrix&>(*A));
            PCG(*A, M, B, X, 3, 2000, eps, 0.0);
         }
         else { CG(*A, B, X, 3, 2000, eps, 0.0); }
         // Recover the solution as a finite element grid function.
         a.RecoverFEMSolution(X, b, x);
         *nodes = x;
         // Send the solution by socket to a GLVis server.
         if (visualization) { Visualize(mesh, order, pause); }
         a.Update();
      }
   }
};

int main(int argc, char *argv[])
{
   int nx = 4;
   int ny = 4;
   int order = 3;
   int niter = 4;
   int surface = 7;
   int rl = 2;
   bool pa = true;
   bool vis = true;
   bool amr = false;
   bool byc = false;
   bool wait = false;
   const char *keys = "gAmaaa";
   const char *device_config = "cpu";
   const char *mesh_file = "../data/mobius-strip.mesh";

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&surface, "-s", "--surface",
                  "Choice of the surface.");
   args.AddOption(&wait, "-w", "--wait", "-no-w", "--no-wait",
                  "Enable or disable a GLVis pause.");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in y-direction.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&rl, "-r", "--ref-levels", "Refinement");
   args.AddOption(&niter, "-n", "--niter", "Number of iterations");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&amr, "-amr", "--adaptive-mesh-refinement", "-no-amr",
                  "--no-adaptive-mesh-refinement", "Enable AMR.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&keys, "-k", "--keys", "GLVis configuration keys.");
   args.AddOption(&vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable visualization.");
   args.AddOption(&byc, "-c", "--components", "-no-c", "--no-components",
                  "Enable or disable the 'by component' solver");

   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);
   MFEM_VERIFY(!amr, "WIP");

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // Initialize GLVis server if 'visualization' is set.
   if (vis) { glvis.open(vishost, visport); }

   // Initialize our surface mesh from command line option.
   // Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> bc;
   Mesh *mesh = nullptr;
   if (surface < 0)  { mesh = new File(bc, order, mesh_file, rl); }
   if (surface == 0) { mesh = new Catenoid(bc, order, nx, ny, rl); }
   if (surface == 1) { mesh = new Helicoid(bc, order, nx, ny, rl); }
   if (surface == 2) { mesh = new Enneper(bc, order, nx, ny, rl); }
   if (surface == 3) { mesh = new Scherk(bc, order, nx, ny, rl); }
   if (surface == 4) { mesh = new Shell(bc, order, nx, ny, rl); }
   if (surface == 5) { mesh = new Hold(bc, order, nx, ny, rl); }
   if (surface == 6) { mesh = new QPeach(bc, order, nx, ny, rl); }
   if (surface == 7) { mesh = new FPeach(bc, order, rl); }
   MFEM_VERIFY(mesh!=nullptr, "Not a valid surface number!");

   const int mdim = mesh->Dimension();
   const int sdim = mesh->SpaceDimension();
   const int vdim = byc ? 1 : sdim;

   // Define a finite element space on the mesh.
   const H1_FECollection fec(order, mdim);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh, &fec, vdim);
   cout << "Number of true DOFs: " << fes->GetTrueVSize() << endl;

   // Send to GLVis the first mesh and set the 'keys' options.
   if (vis) { Visualize(mesh, order, wait, keys, 800, 800); }

   // Instanciate and launch the surface solver.
   if (byc) { ByComponent Solve(pa, vis, niter, wait, order, mesh, fes, bc); }
   else        { ByVector Solve(pa, vis, niter, wait, order, mesh, fes, bc); }

   // Free the used memory.
   delete fes;
   delete mesh;
   return 0;
}

void SnapNodes(Mesh *mesh)
{
   const int sdim = mesh->SpaceDimension();
   GridFunction &nodes = *mesh->GetNodes();
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
