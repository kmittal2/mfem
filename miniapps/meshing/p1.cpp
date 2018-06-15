// r-adapt shape+size:
// ./cfp -m square.mesh -qo 4
//
#include "mfem.hpp"
extern "C" {
# include "3rd_party/gslib/gslib-1.0.1/src/cpp/custom2.h"
}
    // C Function call
//    struct findpts_setup()
//hndl,MPI_COMM_WORLD,num_procs, ldim, xvals,yvals,zvals,lx1,lx1,lx1,nel,2*lx1,2*lx1,2*lx1,bb_t,ntot,ntot,npt_max,tol);
//#include "3rd_party/gslib/gslib-1.0.1/src/gslib.h" 
//}
#include <fstream>
#include <iostream>
#include <ctime>

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

int main (int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
// JL STUFF
   comm_ext world;
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
//   cout << quad_order << " the quad_order" << endl;
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order); break;
      case 2: ir = &IntRules.Get(geom_type, quad_order); break;
      case 3: ir = &IntRulesCU.Get(geom_type, quad_order); break;
      default: cout << "Unknown quad_type: " << quad_type << endl;
        delete he_nlf_integ; return 3;
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
//   cout << quad_order_ac  << " Expected " << endl;
   cout << "Quadrature points per cell: " << ir->GetNPoints() << endl;
   he_nlf_integ->SetIntegrationRule(*ir);

   // 11. write out all dofs
   const int NE = pfespace->GetMesh()->GetNE(),
   dof = pfespace->GetFE(0)->GetDof(), nsp = ir->GetNPoints();
//   cout << NE << " k10ne" << endl;
//   const ParGridFunction &nodes = pmesh->GetNodes();
   
   ParGridFunction nodes(pfespace);
   pmesh->GetNodes(nodes);

   int NR = sqrt(nsp);
   int NS = NR,NT=NR;
   static const unsigned nr[D] = INITD(NR,NS,NT);
   static const unsigned mr[D] = INITD(2*NR,2*NS,2*NT);
   double zr[NR], zs[NS], zt[NT];
   double fmesh[D][NE*MULD(NR,NS,NT)];
   double dumfield[NE*MULD(NR,NS,NT)];
   int np;

   np = 0;
   if (dim==2) 
   {
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < nsp; j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
        fmesh[0][np] = nodes.GetValue(i, ip, 1); //valsi); 
        fmesh[1][np] =nodes.GetValue(i, ip, 2); //valsi1);
        dumfield[np] = pow(fmesh[0][np],2)+pow(fmesh[1][np],2);
        np = np+1;
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
/*
   cout << NE << " " << np << " NEL, npts" <<  endl;
   cout << NE << " " << myid << " " << num_procs << endl;
*/

//  Setup findpts
   struct findpts_data *fd;
   int ldim = dim;
   int nel = NE;
   double bb_t = 0.1;
   int npt_max = 256;
   double tol = 1.e-12;
   int ntot = pow(NR,ldim)*nel;

// Setup findpts
   static const double *const elx[D] = INITD(fmesh[0],fmesh[1],fmesh[2]);
  printf("calling findpts_setup\n");
   fd=findpts_setup_2(&cc,elx,nr,NE,mr,bb_t,
                   ntot,ntot,npt_max,tol);


// fpt stuff
#define TN 2  //number of test points 
   array_init(struct pt_data,&testp,TN);
   struct pt_data *out = (pt_data*) testp.ptr;
   memset(testp.ptr,0,TN*sizeof(struct pt_data));

   out->x[0] = 0.125;
   out->x[1] = 0.25;
   ++out;
   out->x[0] = 0.25;
   out->x[1] = 0.75;

   const double *x_base[D];
   const unsigned x_stride[D] = INITD(sizeof(struct pt_data),
                                     sizeof(struct pt_data),
                                     sizeof(struct pt_data));
   struct pt_data *pt = (pt_data*) testp.ptr;;
   uint npt = TN;
   x_base[0]=pt->x, x_base[1]=pt->x+1;

// Setup findpts
  printf("calling findpts\n");
  findpts(&pt->code , sizeof(struct pt_data),
          &pt->proc , sizeof(struct pt_data),
          &pt->el   , sizeof(struct pt_data),
           pt->r    , sizeof(struct pt_data),
          &pt->dist2, sizeof(struct pt_data),
           x_base   , x_stride, npt, fd);

// Print Results

  if(myid==0) printf("done findpts\n");
  if (myid==0) 
  {
 struct pt_data *pr = (pt_data*) testp.ptr;;
  cout << myid << " " << pr->x[0] << " " << pr->x[1] << " k10xy\n";
  cout << myid << " " << pr->code << " " << pr->el << " " << pr->proc << " " << pr->dist2 << " " << pr->r[0] << " " << pr->r[1] << " k10rst\n";
  cout << " DO NEXT POINT NOW\n";
  ++pr;
  cout << myid << " " << pr->x[0] << " " << pr->x[1] << " k10xy\n";
  cout << myid << " " << pr->code << " " << pr->el << " " << pr->proc << " " << pr->dist2 << " " << pr->r[0] << " " << pr->r[1] << " k10rst\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);
  cout << " DO PROC 1 NOW\n";
  if (myid==1) 
  {
 struct pt_data *pr = (pt_data*) testp.ptr;;
  cout << myid << " " << pr->x[0] << " " << pr->x[1] << " k10xy\n";
  cout << myid << " " << pr->code << " " << pr->el << " " << pr->proc << " " << pr->dist2 << " " << pr->r[0] << " " << pr->r[1] << " k10rst\n";
  cout << " DO NEXT POINT NOW\n";
  ++pr;
  cout << myid << " " << pr->x[0] << " " << pr->x[1] << " k10xy\n";
  cout << myid << " " << pr->code << " " << pr->el << " " << pr->proc << " " << pr->dist2 << " " << pr->r[0] << " " << pr->r[1] << " k10rst\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);

// Do findpts eval
  if(myid==0) printf("doing findpts_eval\n");
  findpts_eval(&pt->ex[0], sizeof(struct pt_data),
                 &pt->code , sizeof(struct pt_data),
                 &pt->proc , sizeof(struct pt_data),
                 &pt->el   , sizeof(struct pt_data),
                  pt->r    , sizeof(struct pt_data),
                  npt, &dumfield[0], fd);
  if(myid==0) printf("done findpts_eval\n");

// Print findpts_eval results
  if (myid==0)
  {
 struct pt_data *pr = (pt_data*) testp.ptr;;
  double val = pow(pr->x[0],2)+pow(pr->x[1],2);
  cout << myid << " " << pr->ex[0] << " " << val << " " << val-pr->ex[0] << " k10comp\n";
  ++pr;
  val = pow(pr->x[0],2)+pow(pr->x[1],2);
  cout << myid << " " << pr->ex[0] << " " << val << " " << val-pr->ex[0] << " k10comp\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);
  cout << " DO PROC 1 NOW\n";
  if (myid==1)
  {
 struct pt_data *pr = (pt_data*) testp.ptr;; 
  double val = pow(pr->x[0],2)+pow(pr->x[1],2);
  cout << myid << " " << pr->ex[0] << " " << val << " " << val-pr->ex[0] << " k10comp\n";
  ++pr;
  val = pow(pr->x[0],2)+pow(pr->x[1],2);
  cout << myid << " " << pr->ex[0] << " " << val << " " << val-pr->ex[0] << " k10comp\n";
  }




   delete pfespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
