﻿// r-adapt shape+size:
// ./cfp -m square.mesh -qo 4
//mpirun -np 2 p1wrapper -m RT2D.mesh -qo 14 -qe 12 -o 3
// TO DO: Add checks inside wrapper for array sizes etc...
//
#include "mfem.hpp"
extern "C" {
# include "3rd_party/gslib/gslib-1.0.1/src/cpp/custom2.h"
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

struct handle { void *data; unsigned ndim; };

class gslib_findpts_lib
{
protected: 
//        int dim, nel, qo, msz;
private:
        IntegrationRule ir;
        double *fmesh;
        struct findpts_data_2 *fda;
        struct findpts_data_3 *fdb;
        struct comm cc;
        int dim, nel, qo, msz;

public:
      gslib_findpts_lib (ParFiniteElementSpace *pfes, ParMesh *pmesh, int QORDER);

      void gslib_findpts_setup();

      void gslib_findpts(uint *pcode, uint *pproc, uint *pel,
      double *pr,double *pd,double *xp, double *yp, double *zp, int nxyz);

      void gslib_findpts_eval (double *fieldout, uint *pcode, uint *pproc, uint *pel,
            double *pr,double *fieldin, int nxyz);

      void gslib_findpts_free ();

      ~gslib_findpts_lib();
};

gslib_findpts_lib::gslib_findpts_lib (ParFiniteElementSpace *pfes, ParMesh *pmesh, int QORDER)
// gslib_findpts_lib(int DD,int NE,int QOR)
{
   const int geom_type = pfes->GetFE(0)->GetGeomType();
   this->ir = IntRulesLo.Get(geom_type, QORDER); 
   dim = pmesh->Dimension();
   nel = pmesh->GetNE();
   qo = sqrt(ir.GetNPoints());
   if (dim==3) qo = cbrt(ir.GetNPoints());
   if (dim==2) 
     {msz = nel*qo*qo;}
   else
     {msz = nel*qo*qo*qo;}
   this->fmesh = new double[dim*msz];

   const int NE = nel, nsp = this->ir.GetNPoints(), NR = qo;
   int np;
   ParGridFunction nodes(pfes);
   pmesh->GetNodes(nodes);

   if (dim==2)
   {
    np = 0;
    int npt = NE*nsp;
    for (int i = 0; i < NE; i++)
    {
       for (int j = 0; j < nsp; j++)
       {
         const IntegrationPoint &ip = this->ir.IntPoint(j);
         this->fmesh[0+np] = nodes.GetValue(i, ip, 1);
         this->fmesh[npt+np] =nodes.GetValue(i, ip, 2);
         np = np+1;
       }
    }
   } //end dim==2
  else
  {
   np = 0; 
    int npt = NE*nsp;
    for (int i = 0; i < NE; i++)
    {  
       for (int j = 0; j < nsp; j++)
       { 
         const IntegrationPoint &ip = this->ir.IntPoint(j);
         this->fmesh[0+np] = nodes.GetValue(i, ip, 1);
         this->fmesh[npt+np] =nodes.GetValue(i, ip, 2);
         this->fmesh[2*npt+np] =nodes.GetValue(i, ip, 3);
         np = np+1;
       }
    }
   }
}

void gslib_findpts_lib::gslib_findpts_setup()
{
   const int NE = nel, nsp = this->ir.GetNPoints(), NR = qo;
   comm_init(&this->cc,MPI_COMM_WORLD);
   double bb_t = 0.05;
   int npt_max = 256;
   double tol = 1.e-12;
   int ntot = pow(NR,dim)*NE;
   int npt = NE*NR*NR;
   if (dim==3) {npt *= NR;}

   if (dim==2)
   {
    unsigned nr[2] = {NR,NR};
    unsigned mr[2] = {2*NR,2*NR};
    double *const elx[2] = {&this->fmesh[0],&this->fmesh[npt]};
    this->fda=findpts_setup_2(&this->cc,elx,nr,NE,mr,bb_t,ntot,ntot,npt_max,tol);
   }
   else
   {
    unsigned nr[3] = {NR,NR,NR};
    unsigned mr[3] = {2*NR,2*NR,2*NR};
    double *const elx[3] = {&this->fmesh[0],&this->fmesh[npt],&this->fmesh[2*npt]};
    this->fdb=findpts_setup_3(&this->cc,elx,nr,NE,mr,bb_t,ntot,ntot,npt_max,tol);
   }
}

void gslib_findpts_lib::gslib_findpts(uint *pcode, uint *pproc, uint *pel,double *pr,double *pd,double *xp, double *yp, double *zp, int nxyz)
{
    if (dim==2)
    {
    int npt = nel*qo*qo;
    const double *const elx[2] = {&fmesh[0],&fmesh[npt]};
    const double *xv_base[2];
    xv_base[0]=xp, xv_base[1]=yp;
    unsigned xv_stride[2];
    xv_stride[0] = sizeof(double),
    xv_stride[1] = sizeof(double);
    uint *const code_base = pcode;
    uint *const proc_base = pproc;
    uint *const el_base = pel;
    double *const r_base = pr;
    double *const dist_base = pd;
    findpts_2(
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr,sizeof(double)*dim,
      dist_base,sizeof(double),
      xv_base,     xv_stride,
      nxyz,this->fda);
    }
   else
   {
    int npt = nel*qo*qo*qo;
    const double *const elx[3] = {&fmesh[0],&fmesh[npt],&fmesh[2*npt]};
    const double *xv_base[3];
    xv_base[0]=xp, xv_base[1]=yp;xv_base[2]=zp;
    unsigned xv_stride[3];
    xv_stride[0] = sizeof(double),
    xv_stride[1] = sizeof(double);
    xv_stride[2] = sizeof(double);
    uint *const code_base = pcode;
    uint *const proc_base = pproc;
    uint *const el_base = pel;
    double *const r_base = pr;
    double *const dist_base = pd;
    findpts_3(
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr,sizeof(double)*dim,
      dist_base,sizeof(double),
      xv_base,     xv_stride,
      nxyz,this->fdb);
   }
//   if (this->cc.id==0) {cout <<  "Done findpts\n";}
}

void gslib_findpts_lib::gslib_findpts_eval(
                double *fieldout, uint *pcode, uint *pproc, uint *pel, double *pr,
                   double *fieldin, int nxyz)
{
    if (dim==2)
    {
    int npt = nel*qo*qo;
    const double *const elx[2] = {&fmesh[0],&fmesh[npt]};
    uint *const code_base = pcode;
    uint *const proc_base = pproc;
    uint *const el_base = pel;
    double *const r_base = pr;
    double *const out_base = fieldout;
    double *const in_base = fieldin;
    findpts_eval_2(out_base,sizeof(double),
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr,sizeof(double)*dim,
      nxyz,fieldin,this->fda);
    }
   else
   {
    int npt = nel*qo*qo;
    const double *const elx[3] = {&fmesh[0],&fmesh[npt],&fmesh[2*npt]};
    uint *const code_base = pcode;
    uint *const proc_base = pproc;
    uint *const el_base = pel;
    double *const r_base = pr;
    double *const out_base = fieldout;
    double *const in_base = fieldin;
    findpts_eval_3(out_base,sizeof(double),
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr,sizeof(double)*dim,
      nxyz,fieldin,this->fdb);
   }
//   if (this->cc.id==0) {cout <<  "Done findpts_eval\n";}
}

void gslib_findpts_lib::gslib_findpts_free ()
{
 if (dim==2)
 {
  findpts_free_2(this->fda);
 }
 else
 {
  findpts_free_3(this->fdb);
 }
}

//IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
//IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);

int main (int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
// JL STUFF
   struct comm cc;
   comm_init(&cc,MPI_COMM_WORLD);

   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   int rp_levels         = 0;
   double jitter         = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int quad_eval         = 8;
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
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&quad_eval, "-qe", "--quad_eval",
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
   if (myid == 0) {args.PrintOptions(cout);}

   // 3. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   if (myid == 0)
   {
      cout << "Mesh curvature: ";
      if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
      else { cout << "(NONE)"; }
      cout << endl;
   }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   // 4. Define a finite element space on the mesh. Here we use vector finite
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
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim);

   // 5. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   pmesh->SetNodalFESpace(pfespace);

   // 6. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   ParGridFunction x(pfespace);
   pmesh->SetNodalGridFunction(&x);

   // 8. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in pfespace.
   Vector h0(pfespace->GetNDofs());
   h0 = infinity();
   Array<int> dofs;
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      pfespace->GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), pmesh->GetElementSize(i));
      }
   }

   // 9. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   ParGridFunction rdm(pfespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < pfespace->GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(pfespace->DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < pfespace->GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      pfespace->GetBdrElementVDofs(i, vdofs);
      // Set the boundary values to zero.
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   x -= rdm;
   // Set the perturbation of all nodes from the true nodes.
   x.SetTrueVector();
   x.SetFromTrueVector();

   // 10. Save the starting (prior to the optimization) mesh to a file. This
   //     output can be viewed later using GLVis: "glvis -m perturbed -np
   //     num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "perturbed." << setfill('0') << setw(6) << myid;
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);
   }

   // 10. Store the starting (prior to the optimization) positions.
   // 11. Store the starting (prior to the optimization) positions.
   ParGridFunction x0(pfespace);
   x0 = x;


   // 12. Setup the quadrature rule for the non-linear form integrator.
   TMOP_Integrator *he_nlf_integ;
   TMOP_QualityMetric *metric = NULL;
   TargetConstructor::TargetType target_t;
   metric = new TMOP_Metric_001;
   TargetConstructor *target_c;
   target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   target_c->SetNodes(x0);
   he_nlf_integ = new TMOP_Integrator(metric, target_c);
   const IntegrationRule *ir = NULL;
   const IntegrationRule *irb = NULL;
   const int geom_type = pfespace->GetFE(0)->GetGeomType();
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
   const int NE = pfespace->GetMesh()->GetNE(),
   dof = pfespace->GetFE(0)->GetDof(), nsp = ir->GetNPoints();
   const int nspb = irb->GetNPoints();
   
   ParGridFunction nodes(pfespace);
   pmesh->GetNodes(nodes);


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
   gslib_findpts_lib *gsfl=NULL;
//   gslib_findpts_lib *gsflb=NULL;
   gsfl = new gslib_findpts_lib(pfespace,pmesh,quad_order);
//   gsflb = new gslib_findpts_lib(pfespace,pmesh,quad_eval);

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
    MPI_Barrier(MPI_COMM_WORLD);
    int stop_s=clock();
    if (myid==0) {cout << "findpts order: " << NR << " \n";}
    if (myid==0) {cout << "findpts time (sec): " << (stop_s-start_s)/1000000. << endl;}
// FINDPTS_EVAL
    MPI_Barrier(MPI_COMM_WORLD);
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
  MPI_Barrier(MPI_COMM_WORLD);
  double glob_maxerr;
  MPI_Allreduce(&maxv, &glob_maxerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  int glob_nnpt;
  MPI_Allreduce(&nnpt, &glob_nnpt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  int glob_nbp;
  MPI_Allreduce(&nbp, &glob_nbp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  int glob_nerrh;
  MPI_Allreduce(&nerrh, &glob_nerrh, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  cout << setprecision(16);
  if (myid==0) {cout << "maximum error: " << glob_maxerr << " \n";}
  if (myid==0) {cout << "points not found: " << glob_nnpt << " \n";}
  if (myid==0) {cout << "points on element border: " << glob_nbp << " \n";}
  if (myid==0) {cout << "points with error > 1.e-10: " << glob_nerrh << " \n";}

   delete pfespace;
   delete fec;
   delete pmesh;
   MPI_Finalize();

   return 0;
}
