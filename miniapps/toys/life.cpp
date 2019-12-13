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
//               ----------------------------------------
//               Life Miniapp:  Model of the Game of Life
//               ----------------------------------------
//
// This miniapp implements Conway's Game of Life.  A few simple starting
// positions are available as well as a default random initial state.  The
// game will terminate only if two successive iterations are identical.
//
// See the output of 'life -h' for more options.
//
// Compile with: make life
//
// Sample runs: life
//              life -nx 100 -ny 100 -r 0.3

#include "mfem.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <bitset>
#include <vector>
#include <stdlib.h> // for random number functions

using namespace std;
using namespace mfem;

bool GameStep(vector<bool> * b[], int nx, int ny);
void ProjectStep(const vector<bool> & b, GridFunction & x, int n);

void InitSketchPad(vector<bool> & b, int nx, int ny, const Array<int> & params);
void InitBlinker(vector<bool> & b, int nx, int ny, const Array<int> & params);
void InitGlider(vector<bool> & b, int nx, int ny, const Array<int> & params);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int nx = 20;
   int ny = 20;
   int rs = -1;
   double r = 0.1;
   Array<int> sketch_pad_params(0);
   Array<int> blinker_params(0);
   Array<int> glider_params(0);
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--num-elems-x",
                  "Number of elements in the x direction.");
   args.AddOption(&ny, "-ny", "--num-elems-y",
                  "Number of elements in the y direction.");
   args.AddOption(&r, "-r", "--random-fraction",
                  "Fraction of randomly chosen live cells.");
   args.AddOption(&rs, "-rs", "--random-seed",
                  "Seed for the random number generator.");
   args.AddOption(&sketch_pad_params, "-sp", "--sketch-pad",
                  "Specify the starting coordinates and values on a grid"
                  " of cells.  The values can be 0, 1, or 2.  Where 0 and 1"
                  " indicate cells that are off or on and 2 represents a"
                  " newline character.");
   args.AddOption(&blinker_params, "-b", "--blinker",
                  "Specify the starting coordinates and orientation (0 or 1)"
                  " of the blinker. Multiple blinkers can be specified as "
                  "'x0 y0 o0 x1 y1 o1 ...'.");
   args.AddOption(&glider_params, "-g", "--glider",
                  "Specify the starting coordinates and "
                  "orientation (0,1,2, or 3) of the glider. "
                  "Multiple gliders can be specified as "
                  "'x0 y0 o0 x1 y1 o1 ...'.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Build a rectangular mesh of quadrilateral elements.
   Mesh *mesh = new Mesh(nx, ny, Element::QUADRILATERAL, 0, nx, ny, false);

   // 3. Define a finite element space on the mesh. Here we use discontinuous
   //    Lagrange finite elements of order zero i.e. piecewise constant basis
   //    functions.
   FiniteElementCollection *fec = new L2_FECollection(0, 2);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   // 4. Initialize a pair of bit arrays to store two copies of the
   //    playing field.
   int len = nx * ny;

   vector<bool> * vbp[2];
   vector<bool> vb0(len);
   vector<bool> vb1(len);

   vbp[0] = &vb0;
   vbp[1] = &vb1;

   if ( r > 0.0 )
   {
      long seed;
      if ( rs < 0 )
      {
         srandom(time(NULL));
         seed = random();
         srand48(seed);
      }
      else
      {
         seed = (long)rs;
      }
      cout << "Using random seed:  " << seed << endl;
   }

   for (int i=0; i<len; i++)
   {
      double rv = drand48();
      vb0[i] = (rv <= r);
      vb1[i] = false;
   }
   if ( sketch_pad_params.Size() > 2 )
   {
      InitSketchPad(vb0, nx, ny, sketch_pad_params);
   }
   if ( blinker_params.Size() > 0 && (blinker_params.Size() % 3 == 0 ) )
   {
      InitBlinker(vb0, nx, ny, blinker_params);
   }
   if ( glider_params.Size() > 0 && (glider_params.Size() % 3 == 0 ) )
   {
      InitGlider(vb0, nx, ny, glider_params);
   }

   // 5. Define the vector x as a finite element grid function corresponding
   //    to fespace which will be used to visualize the playing field.
   //    Initialize x with the starting layout set above.
   GridFunction x(fespace);

   ProjectStep(*vbp[0], x, len);

   // 6. Open a socket to GLVis
   socketstream sol_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sol_sock.open(vishost, visport);
   }

   // 7. Apply the rule iteratively
   cout << endl << "Running the Game of Life..." << flush;

   bool is_good = true;
   bool is_stable = false;
   while ( is_good && visualization && !is_stable )
   {
      is_stable = GameStep(vbp, nx, ny);
      ProjectStep(*vbp[1], x, len);

      // Swap bit arrays
      std::swap(vbp[0], vbp[1]);

      // 8. Send the solution by socket to a GLVis server.
      is_good = sol_sock.good();

      if (visualization && is_good )
      {
         sol_sock << "solution\n" << *mesh << x << flush;
         {
            static int once = 1;
            if (once)
            {
               sol_sock << "keys Ajlm\n";
               sol_sock << "view 0 0\n";
               sol_sock << "zoom 1.9\n";
               sol_sock << "palette 24\n";
               once = 0;
            }
         }
      }
   }
   cout << "done." << endl;

   // 9. Save the mesh and the final state of the game. This output can be
   //    viewed later using GLVis: "glvis -m life.mesh -g life.gf".
   ofstream mesh_ofs("life.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("life.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 10. Free the used memory.
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}

inline int index(int i, int j, int nx, int ny)
{
   return ((j + ny) % ny) * nx + ((i + nx) % nx);
}

bool GameStep(vector<bool> * b[], int nx, int ny)
{
   bool is_stable = true;
   for (int j=0; j<ny; j++)
   {
      for (int i=0; i<nx; i++)
      {
         int c =
            (int)(*b[0])[index(i+0,j+0,nx,ny)] +
            (int)(*b[0])[index(i+1,j+0,nx,ny)] +
            (int)(*b[0])[index(i+1,j+1,nx,ny)] +
            (int)(*b[0])[index(i+0,j+1,nx,ny)] +
            (int)(*b[0])[index(i-1,j+1,nx,ny)] +
            (int)(*b[0])[index(i-1,j+0,nx,ny)] +
            (int)(*b[0])[index(i-1,j-1,nx,ny)] +
            (int)(*b[0])[index(i+0,j-1,nx,ny)] +
            (int)(*b[0])[index(i+1,j-1,nx,ny)];
         switch (c)
         {
            case 3:
               (*b[1])[index(i,j,nx,ny)] = true;
               break;
            case 4:
               (*b[1])[index(i,j,nx,ny)] = (*b[0])[index(i,j,nx,ny)];
               break;
            default:
               (*b[1])[index(i,j,nx,ny)] = false;
               break;
         }
         is_stable &= (*b[1])[index(i,j,nx,ny)] == (*b[0])[index(i,j,nx,ny)];
      }
   }
   return is_stable;
}

void ProjectStep(const vector<bool> & b, GridFunction & x, int n)
{
   for (int i=0; i<n; i++)
   {
      x[i] = (double)b[i];
   }
}

void InitBlinker(vector<bool> & b, int nx, int ny, const Array<int> & params)
{
   for (int i=0; i<params.Size()/3; i++)
   {
      int cx   = params[3 * i + 0];
      int cy   = params[3 * i + 1];
      int ornt = params[3 * i + 2];

      switch (ornt % 2)
      {
         case 0:
            b[index(cx+0,cy+1,nx,ny)] = true;
            b[index(cx+0,cy+0,nx,ny)] = true;
            b[index(cx+0,cy-1,nx,ny)] = true;
            break;
         case 1:
            b[index(cx+1,cy+0,nx,ny)] = true;
            b[index(cx+0,cy+0,nx,ny)] = true;
            b[index(cx-1,cy+0,nx,ny)] = true;
            break;
      }
   }
}

void InitGlider(vector<bool> & b, int nx, int ny, const Array<int> & params)
{
   for (int i=0; i<params.Size()/3; i++)
   {
      int cx   = params[3 * i + 0];
      int cy   = params[3 * i + 1];
      int ornt = params[3 * i + 2];

      switch (ornt % 4)
      {
         case 0:
            b[index(cx-1,cy+0,nx,ny)] = true;
            b[index(cx+0,cy+1,nx,ny)] = true;
            b[index(cx+1,cy-1,nx,ny)] = true;
            b[index(cx+1,cy+0,nx,ny)] = true;
            b[index(cx+1,cy+1,nx,ny)] = true;
            break;
         case 1:
            b[index(cx+0,cy-1,nx,ny)] = true;
            b[index(cx-1,cy+0,nx,ny)] = true;
            b[index(cx-1,cy+1,nx,ny)] = true;
            b[index(cx+0,cy+1,nx,ny)] = true;
            b[index(cx+1,cy+1,nx,ny)] = true;
            break;
         case 2:
            b[index(cx+1,cy+0,nx,ny)] = true;
            b[index(cx+0,cy-1,nx,ny)] = true;
            b[index(cx-1,cy-1,nx,ny)] = true;
            b[index(cx-1,cy+0,nx,ny)] = true;
            b[index(cx-1,cy+1,nx,ny)] = true;
            break;
         case 3:
            b[index(cx+0,cy+1,nx,ny)] = true;
            b[index(cx+1,cy+0,nx,ny)] = true;
            b[index(cx-1,cy-1,nx,ny)] = true;
            b[index(cx+0,cy-1,nx,ny)] = true;
            b[index(cx+1,cy-1,nx,ny)] = true;
            break;
      }
   }
}

void InitSketchPad(vector<bool> & b, int nx, int ny, const Array<int> & params)
{
   int cx   = params[0];
   int cy   = params[1];

   int ox = 0;
   int oy = 0;

   for (int i=2; i<params.Size(); i++)
   {
      if ( params[i]/2 == 1 )
      {
         ox = 0;
         oy--;
      }
      else
      {
         b[index(cx+ox,cy+oy,nx,ny)] = (bool)params[i];
         ox++;
      }
   }
}
