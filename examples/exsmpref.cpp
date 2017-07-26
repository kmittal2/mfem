//                               ETHOS Example 1
//
// Compile with: make exsm
//
// Sample runs:  exsm -m tipton.mesh
//
// Description: This example code performs a simple mesh smoothing based on a
//              topologically defined "mesh Laplacian" matrix.
//
//              The example highlights meshes with curved elements, the
//              assembling of a custom finite element matrix, the use of vector
//              finite element spaces, the definition of different spaces and
//              grid functions on the same mesh, and the setting of values by
//              iterating over the interior and the boundary elements.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>

using namespace mfem;

using namespace std;

#define BIG_NUMBER 1e+100 // Used when a matrix is outside the metric domain.
#define NBINS 25          // Number of intervals in the metric histogram.
#define GAMMA 0.9         // Used for composite metrics 73, 79, 80.
#define BETA0 0.01        // Used for adaptive pseudo-barrier metrics.
#define TAU0_EPS 0.001    // Used for adaptive shifted-barrier metrics.

//IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);

int main (int argc, char *argv[])
{
    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    Mesh *mesh;
    char vishost[] = "localhost";
    int  visport   = 19916;
    vector<double> logvec (100);
    double weight_fun(const Vector &x);
    double tstart_s=clock();
    
    
    // 3. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral or hexahedral elements with the same code.
    const char *mesh_file = "../data/tipton.mesh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;
    
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }
    mesh = new Mesh(mesh_file, 1, 1,false);
    
    int dim = mesh->Dimension();
    
    // 4. Refine the mesh to increase the resolution. In this example we do
    //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    //    largest number that gives a final mesh with no more than 1000
    //    elements.
    {
        int ref_levels = (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
        if (myid == 0)
        {
           cout << "enter refinement levels [" << ref_levels << "] --> " << flush;
           cin >> ref_levels;
        }
        int ref_levels_g = 0;
        MPI_Allreduce(&ref_levels, &ref_levels_g, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        
        for (int l = 0; l < ref_levels_g; l++)
        {
            mesh->UniformRefinement();
        }
        
        logvec[0]=ref_levels;
    }
    
    // 5. Define a finite element space on the mesh. Here we use vector finite
    //    elements which are tensor products of quadratic finite elements. The
    //    dimensionality of the vector finite element space is specified by the
    //    last parameter of the FiniteElementSpace constructor.
    if (myid == 0)
    {
       cout << "Mesh curvature: ";
       if (mesh->GetNodes())
       {
          cout << mesh->GetNodes()->OwnFEC()->Name();
       }
       else
       {
          cout << "(NONE)";
       }
       cout << endl;
    }
    
    int mesh_poly_deg = 1;
    if (myid == 0)
    {
       cout << "Enter polynomial degree of mesh finite element space:\n"
               "0) QuadraticPos (quads only)\n"
               "p) Degree p >= 1\n"
               " --> " << flush;
       cin >> mesh_poly_deg;
    }
    MPI_Bcast(&mesh_poly_deg, num_procs, MPI_INT, 0, MPI_COMM_WORLD);

    FiniteElementCollection *fec;
    if (mesh_poly_deg <= 0)
    {
        fec = new QuadraticPosFECollection;
        mesh_poly_deg = 2;
    }
    else
    {
        fec = new H1_FECollection(mesh_poly_deg, dim);
    }
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    int rp_levels = 2;
    for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }
    
    ostringstream mesh_name, velo_name, ee_name;
    mesh_name << "refined." << setfill('0') << setw(6) << myid;
    ofstream mesh_ofs(mesh_name.str().c_str());
    mesh_ofs.precision(8);
    pmesh->Print(mesh_ofs);
    
    // 17. Free the used memory.
    MPI_Finalize();
    
    delete fec;
    delete pmesh;
    
    return 0;
    
}
double weight_fun(const Vector &x)
{
    double r2 = x(0)*x(0) + x(1)*x(1);
    double l2;
    if (r2>0)
    {
        r2 = sqrt(r2);
    }
    l2 = 0;
    //This is for tipton
    if (r2 >= 0.10 && r2 <= 0.15 )
    {
        l2 = 1;
    }
    //l2 = 0.01+0.5*std::tanh((r2-0.13)/0.01)-(0.5*std::tanh((r2-0.14)/0.01))
    //        +0.5*std::tanh((r2-0.21)/0.01)-(0.5*std::tanh((r2-0.22)/0.01));
    l2 = 0.1+0.5*std::tanh((r2-0.12)/0.005)-(0.5*std::tanh((r2-0.13)/0.005))
    +0.5*std::tanh((r2-0.18)/0.005)-(0.5*std::tanh((r2-0.19)/0.005));
    //l2 = 10*r2;
    /*
    //This is for blade
    int l4 = 0, l3 = 0;
    double xmin, xmax, ymin, ymax,dx ,dy;
    xmin = 0.9; xmax = 1.;
    ymin = -0.2; ymax = 0.;
    dx = (xmax-xmin)/2;dy = (ymax-ymin)/2;
    
    if (abs(x(0)-xmin)<dx && abs(x(0)-xmax)<dx) {
        l4 = 1;
    }
    if (abs(x(1)-ymin)<dy && abs(x(1)-ymax)<dy) {
        l3 = 1;
    }
    l2 = l4*l3;
     */
    
    // This is for square perturbed
    /*
     if (r2 < 0.5) {
        l2 = 1;
    }
     */
    
    return l2;
}
