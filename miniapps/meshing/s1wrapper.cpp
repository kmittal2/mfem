// r-adapt shape+size:
// ./cfp -m square.mesh -qo 4
// ./s1wrapper -m RT2D.mesh -qo 8 -o 3
// TO DO: Add checks inside wrapper for array sizes etc...
//
#include "mfem.hpp"
extern "C" {
# include "3rd_party/gslib/gslib-1.0.1/src/cpp_ser/findpts_h.h"
}

#include <fstream>
#include <iostream>
#include <ctime>
#include <sstream>
#include <string>

using namespace mfem;
using namespace std;

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);

#include "fpt_ser_wrapper.hpp"

int main (int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   double jitter         = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   bool visualization    = true;
   int verbosity_level   = 0;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   cout << "Mesh curvature: ";
   if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
   else { cout << "(NONE)"; }
   cout << endl;

   // 4. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   // 3. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   FiniteElementCollection *fec;
   if (mesh_poly_deg <= 0)
   {
      fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else { fec = new H1_FECollection(mesh_poly_deg, dim); }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);

   // 4. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   mesh->SetNodalFESpace(fespace);

   // 6. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   GridFunction *x = mesh->GetNodes();

   // 7. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in fespace.
   Vector h0(fespace->GetNDofs());
   h0 = infinity();
   Array<int> dofs;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      fespace->GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), mesh->GetElementSize(i));
      }
   }

   // 9. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   GridFunction rdm(fespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < fespace->GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(fespace->DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < fespace->GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      fespace->GetBdrElementVDofs(i, vdofs);
      // Set the boundary values to zero.
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   *x -= rdm;

   // 9. Save the starting (prior to the optimization) mesh to a file. This
   //    output can be viewed later using GLVis: "glvis -m perturbed.mesh".
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }

   // 10. Store the starting (prior to the optimization) positions.
   GridFunction x0(fespace);
   x0 = *x;

   // 12. Setup the quadrature rule for the non-linear form integrator.
   TMOP_Integrator *he_nlf_integ;
   TMOP_QualityMetric *metric = NULL;
   TargetConstructor::TargetType target_t;
   metric = new TMOP_Metric_001;
   TargetConstructor *target_c;
   target_c = new TargetConstructor(target_t);
   target_c->SetNodes(x0);
   he_nlf_integ = new TMOP_Integrator(metric, target_c);
   const IntegrationRule *ir = NULL;
   const IntegrationRule *irb = NULL;
   const int geom_type = fespace->GetFE(0)->GetGeomType();
   int quad_eval = quad_order;
   int myid = 0;
//   cout << quad_order << " the quad_order" << endl;
   if (quad_order > 4) 
   {
    if (quad_order % 2 == 0) {quad_order = 2*quad_order - 4;}
    else {quad_order = 2*quad_order - 3;}
   }
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order), irb = &IntRulesLo.Get(geom_type, quad_eval);
   }
    int quad_order_ac;
    if ( quad_order % 2 == 0)
     {
        quad_order_ac = (quad_order+4)/2;
     }
    else
     {
        quad_order_ac = (quad_order+3)/2;
     }
   if (myid==0) {cout << "Quadrature points per cell: " << ir->GetNPoints() << endl;}
   he_nlf_integ->SetIntegrationRule(*ir);

   // 11. write out all dofs
   const int NE = fespace->GetMesh()->GetNE(),
   dof = fespace->GetFE(0)->GetDof(), nsp = ir->GetNPoints();
   const int nspb = irb->GetNPoints();
   
   GridFunction nodes(fespace);
   mesh->GetNodes(nodes);


   int NR = sqrt(nsp);
   int NRb = sqrt(nspb);
   if (dim==3) {NR = cbrt(nsp); NRb = cbrt(nspb);}

   int sz1 = NR*NR, szb = NRb*NRb;
   if (dim==3) {sz1 *= NR; szb *= NRb;}
   double fx[dim*NE*sz1], fxb[dim*NE*szb];
   double dumfield[NE*sz1], dumfieldb[NE*szb];
   int np;

   np = 0;
   int tnp = NE*nsp;
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < nsp; j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
        fx[np] = nodes.GetValue(i, ip, 1); 
        fx[tnp+np] =nodes.GetValue(i, ip, 2);
        dumfield[np] = pow(fx[np],2)+pow(fx[tnp+np],2);
        if (dim==3) {fx[2*tnp+np] =nodes.GetValue(i, ip, 3);
                    dumfield[np] += pow(fx[2*tnp+np],2);}
        np = np+1;
      }
   }

   np = 0;
   tnp = NE*nspb;
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < nspb; j++)
      {
         const IntegrationPoint &ipb = irb->IntPoint(j);
        fxb[np] = nodes.GetValue(i, ipb, 1);
        fxb[tnp+np] =nodes.GetValue(i, ipb, 2);
        dumfieldb[np] = pow(fxb[np],2)+pow(fxb[tnp+np],2);
        if (dim==3) {fxb[2*tnp+np] =nodes.GetValue(i, ipb, 3);
                     dumfieldb[np] += pow(fxb[2*tnp+np],2);}
        np = np+1;
      }
   }
//kkkk
   findpts_gslib *gsfl=NULL;
//   findpts_gslib *gsflb=NULL;
   gsfl = new findpts_gslib(fespace,mesh,quad_order);
//   gsflb = new findpts_gslib(pfespace,pmesh,quad_eval);

   gsfl->gslib_findpts_setup();
//   gsflb->gslib_findpts_setup();
//   if (myid==0) {cout <<  "Done findpts_setup\n";}

// Read x,y,z
  int nlim = 50000;
  double *vrx = new double[nlim];
  double *vry = new double[nlim];
  double *vrz = new double[nlim];
  int it = 0;
  double r1,r2,r3;
  int nxyz;
//  std::ifstream infile("randsqm.txt"); //from 0 to 1 for square mesh
//   std::ifstream infile("randRT2d2.txt"); //for curvy 2D mesh
   std::ifstream infile("randRT3d.txt"); //for curvy 2D mesh
//  std::ifstream infile("randrst.txt"); // collection of r,s,t between -1 and 1 
 
  std::string line;
 
  std::getline(infile, line);
  std::istringstream iss(line);
  iss >> nxyz;
  int num_procs = 1;
  int npp = nxyz/num_procs;

// make sure not all procs are finding the same point
  it = 0;
  int nn = 0;
  int lc = (myid)*npp;
  int uc = (myid+1)*npp;
  while (std::getline(infile, line))
  {
    std::istringstream iss(line);
      iss >>  r1 >>  r2 >>  r3;
      if (nn>lc && nn < uc)
      {
      vrx[it] = r1;
      vry[it] = r2;
      vrz[it] = r3;
      it += 1;
      }
      nn += 1;
   }
   nxyz = it+1;
   if (myid==0) {cout << "Points to be found: " << nxyz*num_procs << " \n";}

    uint pcode[nxyz];
    uint pproc[nxyz];
    uint pel[nxyz];
    double pr[nxyz*dim];
    double pd[nxyz];
    double fout[nxyz];
    int start_s=clock();
    gsfl->gslib_findpts(pcode,pproc,pel,pr,pd,vrx,vry,vrz,nxyz);
    int stop_s=clock();
    if (myid==0) {cout << "findpts order: " << NR << " \n";}
    if (myid==0) {cout << "findpts time (sec): " << (stop_s-start_s)/1000000. << endl;}
// FINDPTS_EVAL
    start_s=clock();
    gsfl->gslib_findpts_eval(fout,pcode,pproc,pel,pr,dumfield,nxyz);
    stop_s=clock();
    if (myid==0) {cout << "findpts_eval time (sec): " << (stop_s-start_s)/1000000. << endl;}
//    gsflb->gslib_findpts_eval(fout,pcode,pproc,pel,pr,dumfieldb,nxyz);
    gsfl->gslib_findpts_free();
//    gsflb->gslib_findpts_free();

    int nbp = 0;
    int nnpt = 0;
    int nerrh = 0;
    double maxv = -100.;
    for (it = 0; it < nxyz; it++)
    {
    if (pcode[it] < 2) {
    double val = pow(vrx[it],2)+pow(vry[it],2);
    if (dim==3) val += pow(vrz[it],2);
    double delv = abs(val-fout[it]);
    if (delv > maxv) {maxv = delv;}
    if (pcode[it] == 1) {nbp += 1;}
    if (delv > 1.e-10) {nerrh += 1;}
//    cout << it << " " << vrx[it] << " " << vry[it] << " " << fout[it] << " k10a\n";
    }
    else
    {
     nnpt += 1;
    }
    }
  double glob_maxerr=maxv;
  int glob_nnpt=nnpt;
  int glob_nbp=nbp;
  int glob_nerrh=nerrh;
  cout << setprecision(16);
  if (myid==0) {cout << "maximum error: " << glob_maxerr << " \n";}
  if (myid==0) {cout << "points not found: " << glob_nnpt << " \n";}
  if (myid==0) {cout << "points on element border: " << glob_nbp << " \n";}
  if (myid==0) {cout << "points with error > 1.e-10: " << glob_nerrh << " \n";}

   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
