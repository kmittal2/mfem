﻿// mpirun -np 2 ./exp -m RT2D.mesh -o 3

#include "mfem.hpp"

#include <fstream>
#include <ctime>

using namespace mfem;
using namespace std;

// Initial condition
double field_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d = 0; d < dim; d++) { res += x(d) * x(d); }
   return res;
}

int main (int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   int rp_levels         = 0;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Initialize and refine the starting mesh.
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

   // Mesh bounding box (for the full serial mesh).
   Vector pos_min, pos_max;
   MFEM_VERIFY(mesh_poly_deg > 0, "The order of the mesh must be a positive.");
   mesh->GetBoundingBox(pos_min, pos_max, mesh_poly_deg);
   if (myid == 0)
   {
      std::cout << "x in [" << pos_min(0) << ", " << pos_max(0) << "]\n";
      std::cout << "y in [" << pos_min(1) << ", " << pos_max(1) << "]\n";
      if (dim == 3)
      {
         std::cout << "z in [" << pos_min(2) << ", " << pos_max(2) << "]\n";
      }
   }

   // Distribute the mesh.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }

   // Curve the mesh based on the chosen polynomial degree.
   H1_FECollection fec(mesh_poly_deg, dim);
   ParFiniteElementSpace pfespace(&pmesh, &fec, dim);
   pmesh.SetNodalFESpace(&pfespace);

   // Define a scalar function on the mesh.
   ParFiniteElementSpace sc_fes(&pmesh, &fec, 1);
   GridFunction field_vals(&sc_fes);
   FunctionCoefficient fc(field_func);
   field_vals.ProjectCoefficient(fc);

   // Setup the gslib mesh.
   findpts_gslib *gsfl = new findpts_gslib(MPI_COMM_WORLD);
   const double rel_bbox_el = 0.05;
   const double newton_tol  = 1.0e-12;
   const int npts_at_once   = 256;
   gsfl->gslib_findpts_setup(pmesh, rel_bbox_el, newton_tol, npts_at_once);

   // Generate equidistant points in physical coordinates over the whole mesh.
   // Note that some points might be outside, if the mesh is not a box.
   // Note that all tasks search the same points (not mandatory).
   const int pts_cnt_1D = 5;
   const int pts_cnt = std::pow(pts_cnt_1D, dim);
   Vector vxyz(pts_cnt * dim);
   if (dim == 2)
   {
      L2_QuadrilateralElement el(pts_cnt_1D - 1, BasisType::ClosedUniform);
      const IntegrationRule &ir = el.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         vxyz(i)           = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
         vxyz(pts_cnt + i) = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
      }
   }
   else
   {
      L2_HexahedronElement el(pts_cnt_1D - 1, BasisType::ClosedUniform);
      const IntegrationRule &ir = el.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         vxyz(i)             = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
         vxyz(pts_cnt + i)   = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
         vxyz(2*pts_cnt + i) = pos_min(2) + ip.z * (pos_max(2)-pos_min(2));
      }
   }

   Array<uint> el_id_out(pts_cnt), code_out(pts_cnt), task_id_out(pts_cnt);
   Vector pos_r_out(pts_cnt * dim), dist_p_out(pts_cnt);

   // Finds points stored in vxyz.
   gsfl->gslib_findpts(vxyz, code_out, task_id_out,
                       el_id_out, pos_r_out, dist_p_out);

   // Interpolate FE function values on the found points.
   Vector interp_vals(pts_cnt);
   gsfl->gslib_findpts_eval(code_out, task_id_out, el_id_out,
                            pos_r_out, field_vals, interp_vals);

   gsfl->gslib_findpts_free();

   int face_pts = 0, not_found = 0, found_loc = 0, found_away = 0;
   double max_err = 0.0, max_dist = 0.0;
   Vector pos(dim);
   for (int i = 0; i < pts_cnt; i++)
   {
      (task_id_out[i] == myid) ? found_loc++ : found_away++;

      if (code_out[i] < 2)
      {
         for (int d = 0; d < dim; d++) { pos(d) = vxyz(d * pts_cnt + i); }
         const double exact_val = field_func(pos);

         max_err  = std::max(max_err, fabs(exact_val - interp_vals[i]));
         max_dist = std::max(max_dist, dist_p_out(i));
         if (code_out[i] == 1) { face_pts++; }
      }
      else { not_found++; }
   }

   // We print only the task 0 result (other tasks should be identical except
   // the number of points found locally).
   if (myid == 0)
   {
      cout << setprecision(16) << "--- Task " << myid << ": "
           << "\nSearched points:      " << pts_cnt
           << "\nFound on local mesh:  " << found_loc
           << "\nFound on other tasks: " << found_away
           << "\nMax interp error:     " << max_err
           << "\nMax dist (of found):  " << max_dist
           << "\nPoints not found:     " << not_found
           << "\nPoints on faces:      " << face_pts << std::endl;
   }

   MPI_Finalize();
   return 0;
}
