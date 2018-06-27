// r-adapt shape+size:
// ./cfp -m square.mesh -qo 4
//
#include "mfem.hpp"
extern "C" {
# include "3rd_party/gslib/gslib-1.0.1/src/cpp/custom2.h"
# include "3rd_party/gslib/gslib-1.0.1/src/cpp/findpts_mfem.h"
}
#include <fstream>
#include <iostream>
#include <ctime>
#include <sstream>
#include <string>

#define D 2

#if D==3
#define INITD(a,b,c) {a,b,c}
#define MULD(a,b,c) ((a)*(b)*(c))
#define INDEXD(a,na, b,nb, c) (((c)*(nb)+(b))*(na)+(a))
#define findpts_data  findpts_data_3
#define findpts_setup findpts_setup_3
#define findpts_free  findpts_free_3
#define findpts       findpts_3
#define findpts_eval  findpts_eval_3
#elif D==2
#define INITD(a,b,c) {a,b}
#define MULD(a,b,c) ((a)*(b))
#define INDEXD(a,na, b,nb, c) ((b)*(na)+(a))
#define findpts_data  findpts_data_2
#define findpts_setup findpts_setup_2
#define findpts_free  findpts_free_2
#define findpts       findpts_2
#define findpts_eval  findpts_eval_2
#endif

static uint np, id;
struct pt_data { double x[D], r[D], dist2, ex[D]; uint code, proc, el; };
static struct array testp;

using namespace mfem;
using namespace std;

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);

double valfun(double xv, double yv)
{
  double xvp = xv;
  double yvp = yv;
//  double val = sin(M_PI*xvp*yvp) + cos(M_PI*xvp*yvp)*sin(M_PI*xvp*yvp);
  double val = pow(xvp,2) + pow(yvp,2);
  return val;
}

double ind_values(const Vector &x)
{
 double indval = valfun(x(0),x(1));
 return indval;
}

void normalize(Vector &v)
{
   const double max = v.Max();
   const double min = v.Min();
   v -= min;
   v /= max;
}

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
   int mesh_poly_deg     = 3;
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
                  "Order of the quadrature rule for findpts_eval.");
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
   const int geom_type = pfespace->GetFE(0)->GetGeomType();

//  OUtput function
   // Indicator function.
   // Copy of the initial mesh.
   ParMesh mesh0(*pmesh);
   FunctionCoefficient ind_coeff(ind_values);
   L2_FECollection ind_fec(0, dim);
   ParFiniteElementSpace ind_fes(&mesh0, &ind_fec);
   ParGridFunction ind_gf(&ind_fes);
   ind_gf.ProjectCoefficient(ind_coeff);
   normalize(ind_gf);

   osockstream sock(19916, "localhost");
   sock << "solution\n";
   mesh0.PrintAsOne(sock);
   ind_gf.SaveAsOne(sock);
   sock.send();
   sock << "window_title 'Mesh and Function'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;

//
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order); break;
      case 2: ir = &IntRules.Get(geom_type, quad_order); break;
      case 3: ir = &IntRulesCU.Get(geom_type, quad_order); break;
      default: cout << "Unknown quad_type: " << quad_type << endl;
        delete he_nlf_integ; return 3;
   }
   if (myid==0) {cout << "Quadrature points per cell: " << ir->GetNPoints() << endl;}
   const IntegrationRule *ire = NULL;
   switch (quad_type)
   {
      case 1: ire = &IntRulesLo.Get(geom_type, quad_eval); break;
      case 2: ire = &IntRules.Get(geom_type, quad_eval); break;
      case 3: ire = &IntRulesCU.Get(geom_type, quad_eval); break;
      default: cout << "Unknown quad_type: " << quad_type << endl;
        delete he_nlf_integ; return 3;
   }

//   he_nlf_integ->SetIntegrationRule(*ir);

   // 11. write out all dofs
   const int NE = pfespace->GetMesh()->GetNE(),
   dof = pfespace->GetFE(0)->GetDof(), nsp = ir->GetNPoints();
   const int nspe = ire->GetNPoints();
   if (myid==0) {cout << dof << " " << nsp << " " << nspe <<  " dof and nsp for findpts and eval\n";}

//  Check number of inverted elements
   double tauval = infinity();
   for (int i = 0; i < NE; i++)
   { 
      ElementTransformation *transf = pmesh->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      { 
         transf->SetIntPoint(&ir->IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   double minJ0;
   MPI_Allreduce(&tauval, &minJ0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   tauval = minJ0;
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << tauval << endl; }
   
   ParGridFunction nodes(pfespace);
   pmesh->GetNodes(nodes);

   int NR = sqrt(nsp);
   int NS = NR,NT=NR;
   static const unsigned nr[D] = INITD(NR,NS,NT);
   static const unsigned mr[D] = INITD(2*NR,2*NS,2*NT);
   double fmesh[D][NE*MULD(NR,NS,NT)];
   double dumfield[NE*MULD(NR,NS,NT)];
   int np;

   int NRe = sqrt(nspe);
   int NSe = NRe,NTe=NRe;
   static const unsigned nre[D] = INITD(NRe,NSe,NTe);
   static const unsigned mre[D] = INITD(2*NRe,2*NSe,2*NTe);
   double fmeshe[D][NE*MULD(NRe,NSe,NTe)];
   double dumfielde[NE*MULD(NRe,NSe,NTe)];
   int npe;

   np = 0;
   npe = 0;
   if (dim==2) 
   {
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < nsp; j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
        fmesh[0][np] = nodes.GetValue(i, ip, 1); //valsi); 
        fmesh[1][np] =nodes.GetValue(i, ip, 2); //valsi1);
        dumfield[np] = valfun(fmesh[0][np],fmesh[1][np]);
        np = np+1;
      }
   }
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < nspe; j++)
      {
         const IntegrationPoint &ipe = ire->IntPoint(j);
        fmeshe[0][npe] = nodes.GetValue(i, ipe, 1); //valsi); 
        fmeshe[1][npe] =nodes.GetValue(i, ipe, 2); //valsi1);
        dumfielde[npe] = valfun(fmeshe[0][npe],fmeshe[1][npe]);
        npe = npe+1;
      }
   }
   }
   else
   {
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < nsp; j++)
      {
        const IntegrationPoint &ip = ir->IntPoint(j);
        fmesh[0][np] = nodes.GetValue(i, ip, 1); //valsi); 
        fmesh[1][np] =nodes.GetValue(i, ip, 2); //valsi1);
        fmesh[2][np] =nodes.GetValue(i, ip, 3); //valsi1);
        np = np+1;
      }
   }
   }

//  Setup findpts
   struct findpts_data *fd;
   int ldim = dim;
   int nel = NE;
   double bb_t = 0.01;
   int npt_max = 256;
   double tol = 1.e-12;
   int ntot = pow(NR,ldim)*nel;

   struct findpts_data *fde;
   int ntote = pow(NRe,ldim)*nel;

// Setup findpts
   static const double *const elx[D] = INITD(fmesh[0],fmesh[1],fmesh[2]);
   static const double *const elxe[D] = INITD(fmeshe[0],fmeshe[1],fmeshe[2]);
   if (myid==0) {printf("calling findpts_setup\n");}

   fd=findpts_setup(&cc,elx,nr,NE,mr,bb_t,ntot,ntot,npt_max,tol);
   fde=findpts_setup(&cc,elxe,nre,NE,mre,bb_t,ntote,ntote,npt_max,tol);

// Read x,y,z
  Vector vrx(50000),vry(50000),vrz(50000);
  int it = 0;
  double r1,r2,r3;
  int nxyz;
//  std::ifstream infile("randrst.txt");
  std::ifstream infile("randRT2d2.txt");
  std::string line;
 
  std::getline(infile, line);
  std::istringstream iss(line);
  iss >> nxyz;

  int npp = nxyz/num_procs;

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
      vrx(it) = r1;
      vry(it) = r2;
      vrz(it) = r3;
      it += 1;
      }
      nn += 1;
   }
// fpt stuff
   nxyz = it+1;
   if (myid==0) {cout << "Points to be found: " << nxyz << " \n";}

#define TN nxyz
   array_init(struct pt_data,&testp,TN);
   struct pt_data *pt = (pt_data*) testp.ptr;
   memset(testp.ptr,0,TN*sizeof(struct pt_data));

   for (it = 0; it < TN; it++)
   {
   pt->x[0] = vrx(it);
   pt->x[1] = vry(it);
   ++pt;
   }
   pt = (pt_data*) testp.ptr;

   const double *x_base[D];
   const unsigned x_stride[D] = INITD(sizeof(struct pt_data),
                                     sizeof(struct pt_data),
                                     sizeof(struct pt_data));
   uint npt = TN;
   x_base[0]=pt->x, x_base[1]=pt->x+1;

// findpts
  MPI_Barrier(MPI_COMM_WORLD);
  int start_s=clock();
  findpts(&pt->code , sizeof(struct pt_data),
          &pt->proc , sizeof(struct pt_data),
          &pt->el   , sizeof(struct pt_data),
           pt->r    , sizeof(struct pt_data),
          &pt->dist2, sizeof(struct pt_data),
           x_base   , x_stride, npt, fd);
   MPI_Barrier(MPI_COMM_WORLD);
   int stop_s=clock();
   if (myid==0) {cout << "findpts time (sec): " << (stop_s-start_s)/1000000. << endl;}
// Print Results

// Do findpts eval
  pt = (pt_data*) testp.ptr;
  start_s=clock();

  findpts_eval(&pt->ex[0], sizeof(struct pt_data),
               &pt->code , sizeof(struct pt_data),
               &pt->proc , sizeof(struct pt_data),
               &pt->el   , sizeof(struct pt_data),
               pt->r    , sizeof(struct pt_data),
               npt, &dumfielde[0], fde);

  MPI_Barrier(MPI_COMM_WORLD);
  stop_s=clock();
  if (myid==0) {cout << "findpts_eval time (sec): " << (stop_s-start_s)/1000000. << endl;}

// Print findpts_eval results
  double maxv = -100.;
  pt = (pt_data*) testp.ptr;
  int nnpt = 0; // number of points not found
  int nbp = 0;
  int nerrh = 0; //points with high error
  for (it = 0; it < TN; it++)
  {
  double val = valfun(pt->x[0],pt->x[1]);
  double delv = abs(val-pt->ex[0]);
//  cout << pt->x[0] << " " << pt->x[1] << " " << pt->ex[0] << " " << abs(val-pt->ex[0]) << " K10c\n";
  if (pt->code < 2) 
  {
  if (delv > maxv) {maxv = delv;}
  if (pt->code == 1) {nbp += 1;}
//  if (delv > 1.e-10) {cout << pt->code << " " << pt->r[0] << " " << pt->r[1] << " " << pt->x[0] << " " << pt->x[1] << " " << pt->dist2 << " " << abs(val-pt->ex[0]) << " K10c\n";}
  if (delv > 1.e-10) {nerrh += 1;}
  }
  else
  {
   nnpt += 1;
  }
  ++pt;
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
  if (myid==0) {cout << "maximum error is: " << glob_maxerr << " \n";}
  if (myid==0) {cout << "points not found: " << glob_nnpt << " \n";}
  if (myid==0) {cout << "points on element border: " << glob_nbp << " \n";}
  if (myid==0) {cout << "points with high error: " << glob_nerrh << " \n";}


   delete pfespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
