// r-adapt shape+size:
// ./mesh-optimizer -m square.mesh -rs 2 -o 3 -mid 9 -tid 7 -ls 2 -bnd -vl 2 -ni 200 -li 100 -qo 4 -qt 2
// .cfp -m square.mesh -qo 4
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <ctime>
using namespace mfem;
using namespace std;

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);

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
   FiniteElementSpace *fes = new FiniteElementSpace(mesh, fec, dim);

   // 4. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   mesh->SetNodalFESpace(fes);

   // 5. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 6. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   GridFunction *x = mesh->GetNodes();

   // 7. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in fespace.
   Vector h0(fes->GetNDofs());
   h0 = infinity();
   Array<int> dofs;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      fes->GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), mesh->GetElementSize(i));
      }
   }

   // 8. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   GridFunction rdm(fes);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < fes->GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(fes->DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < fes->GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      fes->GetBdrElementVDofs(i, vdofs);
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
   GridFunction x0(fes);
   x0 = *x;


   TMOP_QualityMetric *metric = NULL;
   metric = new TMOP_Metric_001;
   TargetConstructor::TargetType target_t;
   target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   TargetConstructor *target_c = new TargetConstructor(target_t);
   target_c->SetNodes(x0);
   TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, target_c);
   // 12. Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = NULL;
   const int geom_type = fes->GetFE(0)->GetGeomType();
   cout << quad_order << " the quad_order" << endl;
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order); break;
      case 2: ir = &IntRules.Get(geom_type, quad_order); break;
      case 3: ir = &IntRulesCU.Get(geom_type, quad_order); break;
      default: cout << "Unknown quad_type: " << quad_type << endl;
        delete he_nlf_integ; return 3;
   }
   cout << pow((quad_order+3)/2,2)  << " Expected for odd " << endl;
   cout << pow((quad_order+4)/2,2)  << " Expected for even " << endl;
   cout << "Quadrature points per cell: " << ir->GetNPoints() << endl;
   he_nlf_integ->SetIntegrationRule(*ir);

   // 11. write out all dofs
   const int NE = fes->GetMesh()->GetNE(),
   dof = fes->GetFE(0)->GetDof(), nsp = ir->GetNPoints();
   const GridFunction &nodes = *mesh->GetNodes();
   Vector xvals((nsp+1)*(NE+1));
   Vector yvals((nsp+1)*(NE+1));
   Vector zvals((nsp+1)*(NE+1));
   int np;

   np = 0;
   if (dim==2) 
   {
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < nsp; j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
        xvals[np] = nodes.GetValue(i, ip, 1); //valsi);
        yvals[np] = nodes.GetValue(i, ip, 2); //valsi);
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
        xvals[np] = nodes.GetValue(i, ip, 1); //valsi);
        yvals[np] = nodes.GetValue(i, ip, 2); //valsi);
        zvals[np] = nodes.GetValue(i, ip, 3); //valsi);
        np = np+1;
      }
   }
   }
//   cout << NE << " " << np << " NEL, npts" <<  endl;

/* 
   for (int i = 0; i < np; i++)
   {  
     cout << i << " " << xvals[i] << " " << yvals[i] <<   endl;
   }
*/


   delete fes;
   delete fec;
   delete mesh;

   return 0;
}
