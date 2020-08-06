#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *device_config = "cpu";
   bool refine = false;
   bool rebalance = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&refine, "-r", "--refine", "-no-r", "--no-refine",
                  "Test random parallel refinement.");
   args.AddOption(&rebalance, "-b", "--rebalance", "-no-b", "--no-rebalance",
                  "Test mesh load balancing.");
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

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 6. Load the parallel mesh from a file
   ParMesh *pmesh;
   {
      ifstream ifs(MakeParFilename("mesh.", myid));
      MFEM_VERIFY(ifs.good(), "Mesh file not found.");
      pmesh = new ParMesh(MPI_COMM_WORLD, ifs);
   }

   // 7. Load the ParGridFunction
   ParGridFunction *pgf;
   {
      ifstream ifs(MakeParFilename("sol.", myid));
      MFEM_VERIFY(ifs.good(), "Solution file not found.");
      pgf = new ParGridFunction(pmesh, ifs);
   }

   // 8. Test refinement of the loaded mesh
   {
      if (myid == 0) { mfem::out << "Refining..." << std::endl; }
      pmesh->RandomRefinement(0.5);
      pgf->FESpace()->Update();
      pgf->Update();
   }

   // 9. Test rebalancing
   if (rebalance)
   {
      if (myid == 0) { mfem::out << "Rebalancing..." << std::endl; }
      pmesh->Rebalance();
      pgf->FESpace()->Update();
      pgf->Update();
   }

   // 10. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << *pgf << flush;
   }

   // 11. Free the used memory.
   delete pgf;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
