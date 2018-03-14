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

#include "pumi.hpp"

#ifdef MFEM_USE_PUMI
#ifdef MFEM_USE_MPI

#include "mesh_headers.hpp"
#include "../fem/fem.hpp"
#include "../general/sort_pairs.hpp"
#include "../general/text.hpp"
#include "../general/sets.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <cmath>
#include <cstring>
#include <ctime>

using namespace std;

namespace mfem
{

PumiMesh::PumiMesh(apf::Mesh2* apf_mesh, int generate_edges, int refine,
                   bool fix_orientation)
{
   SetEmpty();
   Load(apf_mesh, generate_edges, refine, fix_orientation);
}

Element *PumiMesh::ReadElement( apf::MeshEntity* Ent, const int geom,
                                apf::Downward Verts,
                                const int Attr, apf::Numbering* vert_num)
{
   Element *el;
   int nv, *v;

   // Create element in MFEM
   el = NewElement(geom);
   nv = el->GetNVertices();
   v  = el->GetVertices();

   // Fill the connectivity
   for (int i = 0; i < nv; ++i)
   {
      v[i] = apf::getNumber(vert_num, Verts[i], 0, 0);
   }

   // Assign attribute
   el->SetAttribute(Attr);

   return el;
}

void PumiMesh::CountBoundaryEntity( apf::Mesh2* apf_mesh, const int BcDim,
                                    int &NumBc)
{
   apf::MeshEntity* ent;
   apf::MeshIterator* itr = apf_mesh->begin(BcDim);

   while ((ent=apf_mesh->iterate(itr)))
   {
      apf::ModelEntity* mdEnt = apf_mesh->toModel(ent);
      if (apf_mesh->getModelType(mdEnt) == BcDim)
      {
         NumBc++;
      }
   }
   apf_mesh->end(itr);

   // Check if any boundary is detected
   if (NumBc==0)
   {
      MFEM_ABORT("In CountBoundaryEntity; no boundary is detected!");
   }
}

void PumiMesh::Load(apf::Mesh2* apf_mesh, int generate_edges, int refine,
                    bool fix_orientation)
{
   int  curved = 0, read_gf = 1;

   // Add a check on apf_mesh just in case
   Clear();

   // First number vertices
   apf::Field* apf_field_crd = apf_mesh->getCoordinateField();
   apf::FieldShape* crd_shape = apf::getShape(apf_field_crd);
   apf::Numbering* v_num_loc = apf::createNumbering(apf_mesh, "VertexNumbering",
                                                    crd_shape, 1);
   // Check if it is a curved mesh
   curved = (crd_shape->getOrder() > 1) ? 1 : 0;

   // Read mesh
   ReadSCORECMesh(apf_mesh, v_num_loc, curved);
   cout<< "After ReadSCORECMesh" <<endl;
   // at this point the following should be defined:
   //  1) Dim
   //  2) NumOfElements, elements
   //  3) NumOfBdrElements, boundary
   //  4) NumOfVertices, with allocated space in vertices
   //  5) curved
   //  5a) if curved == 0, vertices must be defined
   //  5b) if curved != 0 and read_gf != 0,
   //         'input' must point to a GridFunction
   //  5c) if curved != 0 and read_gf == 0,
   //         vertices and Nodes must be defined

   // FinalizeTopology() will:
   // - assume that generate_edges is true
   // - assume that refine is false
   // - does not check the orientation of regular and boundary elements
   FinalizeTopology();

   if (curved && read_gf)
   {
      // Check it to be only Quadratic if higher order
      cout << "Is Curved?: "<< curved << "\n" <<read_gf <<endl;
      Nodes = new GridFunctionPumi(this, apf_mesh, v_num_loc, crd_shape->getOrder());
      edge_vertex = NULL;
      own_nodes = 1;
      spaceDim = Nodes->VectorDim();
      // if (ncmesh) { ncmesh->spaceDim = spaceDim; }
      // Set the 'vertices' from the 'Nodes'
      for (int i = 0; i < spaceDim; i++)
      {
         Vector vert_val;
         Nodes->GetNodalValues(vert_val, i+1);
         for (int j = 0; j < NumOfVertices; j++)
         {
            vertices[j](i) = vert_val(j);
         }
      }
   }

   // Delete numbering
   apf::destroyNumbering(v_num_loc);

   Finalize(refine, fix_orientation);
}

void PumiMesh::ReadSCORECMesh(apf::Mesh2* apf_mesh, apf::Numbering* v_num_loc,
                              const int curved)
{
   // Here fill the element table from SCOREC MESH
   // The vector of element pointers are generated with attr and connectivity

   apf::MeshIterator* itr = apf_mesh->begin(0);
   apf::MeshEntity* ent;
   NumOfVertices = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
      // ids start from 0
      apf::number(v_num_loc, ent, 0, 0, NumOfVertices);
      NumOfVertices++;
   }
   apf_mesh->end(itr);

   Dim = apf_mesh->getDimension();
   NumOfElements = countOwned(apf_mesh,Dim);
   elements.SetSize(NumOfElements);

   // Get the attribute tag
   apf::MeshTag* attTag = apf_mesh->findTag("attribute");

   // read elements from SCOREC Mesh
   itr = apf_mesh->begin(Dim);
   unsigned int j=0;
   while ((ent = apf_mesh->iterate(itr)))
   {
      // Get vertices
      apf::Downward verts;
      int num_vert =  apf_mesh->getDownward(ent,0,verts);
      // Get attribute Tag vs Geometry
      int attr = 1;
      /*if (apf_mesh->hasTag(ent,atts)){
          attr = apf_mesh->getIntTag(ent,attTag,&attr);
      }*/
      apf::ModelEntity* me = apf_mesh->toModel(ent);
      attr = 1; // apf_mesh->getModelTag(me);
      int geom_type = apf_mesh->getType(ent); // Make sure this works!!!
      elements[j] = ReadElement(ent, geom_type, verts, attr, v_num_loc);
      j++;
   }
   // End iterator
   apf_mesh->end(itr);

   // Read Boundaries from SCOREC Mesh
   // First we need to count them
   int BCdim = Dim - 1;
   NumOfBdrElements = 0;
   CountBoundaryEntity(apf_mesh, BCdim, NumOfBdrElements);
   boundary.SetSize(NumOfBdrElements);
   j=0;

   // Read boundary from SCOREC mesh
   itr = apf_mesh->begin(BCdim);
   while ((ent = apf_mesh->iterate(itr)))
   {
      // check if this mesh entity is on the model boundary
      apf::ModelEntity* mdEnt = apf_mesh->toModel(ent);
      if (apf_mesh->getModelType(mdEnt) == BCdim)
      {
         apf::Downward verts;
         int num_verts = apf_mesh->getDownward(ent, 0, verts);
         int attr = 1 ; // apf_mesh->getModelTag(mdEnt);
         int geom_type = apf_mesh->getType(ent);
         boundary[j] = ReadElement( ent, geom_type, verts, attr, v_num_loc);
         j++;
      }
   }
   apf_mesh->end(itr);

   // Fill vertices
   vertices.SetSize(NumOfVertices);


   if (!curved)
   {
      apf::MeshIterator* itr = apf_mesh->begin(0);
      spaceDim = Dim;

      while ((ent = apf_mesh->iterate(itr)))
      {
         unsigned int id = apf::getNumber(v_num_loc, ent, 0, 0);
         apf::Vector3 Crds;
         apf_mesh->getPoint(ent,0,Crds);

         for (unsigned int ii=0; ii<spaceDim; ii++)
         {
            vertices[id](ii) = Crds[ii];
         }
      }
      apf_mesh->end(itr);

      // initialize vertex positions in NCMesh
      // if (ncmesh) { ncmesh->SetVertexPositions(vertices); }
   }
}


// ParPumiMesh implementation

Element *ParPumiMesh::ReadElement( apf::MeshEntity* Ent, const int geom,
                                   apf::Downward Verts,
                                   const int Attr, apf::Numbering* vert_num)
{
   Element *el;
   int nv, *v;

   // Create element in MFEM
   el = NewElement(geom);
   nv = el->GetNVertices();
   v  = el->GetVertices();

   // Fill the connectivity
   for (int i = 0; i < nv; ++i)
   {
      v[i] = apf::getNumber(vert_num, Verts[i], 0, 0);
   }

   // Assign attribute
   el->SetAttribute(Attr);

   return el;
}


ParPumiMesh::ParPumiMesh(MPI_Comm comm, apf::Mesh2* apf_mesh)
{
   // Set the communicator for gtopo
   gtopo.SetComm(comm);

   int i, j;
   Array<int> vert;

   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   Mesh::SetEmpty();
   // The ncmesh part is deleted

   Dim = apf_mesh->getDimension();
   spaceDim = Dim;// mesh.spaceDim;

   // Iterator to get type
   apf::MeshIterator* itr = apf_mesh->begin(Dim);
   BaseGeom = apf_mesh->getType( apf_mesh->iterate(itr) );
   apf_mesh->end(itr);

   itr = apf_mesh->begin(Dim - 1);
   BaseBdrGeom = apf_mesh->getType( apf_mesh->iterate(itr) );
   apf_mesh->end(itr);

   ncmesh = pncmesh = NULL;

   // Global numbering of vertices
   // This is necessary to build a local numbering that has the same ordering in
   // each process
   apf::FieldShape* v_shape = apf::getConstant(0);
   apf::Numbering* vLocNum = apf::createNumbering(apf_mesh, "AuxVertexNumbering",
                                                  v_shape, 1);
   // Number
   itr = apf_mesh->begin(0);
   apf::MeshEntity* ent;
   int owned_num = 0;
   int all_num = 0;
   int shared_num = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
      all_num++;
      if (apf_mesh->isOwned(ent))
      {
         apf::number(vLocNum, ent, 0, 0, owned_num++);
      }
      if (apf_mesh->isShared(ent))
      {
         shared_num++;
      }
   }
   apf_mesh->end(itr);

   // Make it global
   apf::GlobalNumbering* VertexNumbering = apf::makeGlobal(vLocNum, true);
   apf::synchronize(VertexNumbering);

   // Take this process global ids and sort
   Array<int> thisIds(all_num);
   Array<int> SharedVertIds(shared_num);
   itr = apf_mesh->begin(0);
   all_num = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
      unsigned int id = apf::getNumber(VertexNumbering, ent, 0, 0);
      thisIds[all_num++] = id;
   }
   apf_mesh->end(itr);
   thisIds.Sort();

   // Create local numbering that respects the global ordering
   apf::Field* apf_field_crd = apf_mesh->getCoordinateField();
   apf::FieldShape* crd_shape = apf::getShape(apf_field_crd);
   apf::Numbering* v_num_loc = apf::createNumbering(apf_mesh,
                                                    "LocalVertexNumbering",
                                                    crd_shape, 1);

   NumOfVertices = 0;
   shared_num = 0;
   itr = apf_mesh->begin(0);
   while ((ent = apf_mesh->iterate(itr)))
   {
      // Id from global numbering
      unsigned int id = apf::getNumber(VertexNumbering, ent, 0, 0);
      // Find its position at sorted list
      int ordered_id = thisIds.Find(id);
      // Assign as local number
      apf::number(v_num_loc, ent, 0, 0, ordered_id);
      NumOfVertices++;

      // Add to shared vertices list
      if (apf_mesh->isShared(ent))
      {
         SharedVertIds[shared_num++] = ordered_id;
      }

   }
   apf_mesh->end(itr);
   SharedVertIds.Sort();
   apf::destroyGlobalNumbering(VertexNumbering);


   vertices.SetSize(NumOfVertices);
   // Set vertices for non-curved mesh
   int curved = (crd_shape->getOrder() > 1) ? 1 : 0;

   // if (!curved)
   // {
   itr = apf_mesh->begin(0);
   while ((ent = apf_mesh->iterate(itr)))
   {
      unsigned int id = apf::getNumber(v_num_loc, ent, 0, 0);
      apf::Vector3 Crds;
      apf_mesh->getPoint(ent,0,Crds);

      for (unsigned int ii=0; ii<spaceDim; ii++)
      {
         vertices[id](ii) =
            Crds[ii];   // !! I am assuming the ids are ordered and from 0
      }
   }
   apf_mesh->end(itr);
   // }

   // Fill the elements
   NumOfElements = countOwned(apf_mesh,Dim);
   elements.SetSize(NumOfElements);

   // Get the attribute tag
   apf::MeshTag* attTag = apf_mesh->findTag("attribute");

   // Read elements from SCOREC Mesh
   itr = apf_mesh->begin(Dim);
   j=0;
   while ((ent = apf_mesh->iterate(itr)))
   {
      // Get vertices
      apf::Downward verts;
      int num_vert =  apf_mesh->getDownward(ent,0,verts);
      // Get attribute Tag vs Geometry
      int attr = 1;
      /*if (apf_mesh->hasTag(ent,atts)){
         apf_mesh->getIntTag(ent,attTag,&attr);
      }*/
      // apf::ModelEntity* me = apf_mesh->toModel(ent);
      // attr = 1; // apf_mesh->getModelTag(me);

      int geom_type = BaseGeom;// apf_mesh->getType(ent); // Make sure this works!!!
      elements[j] = ReadElement(ent, geom_type, verts, attr, v_num_loc);
      j++;
   }
   // End iterator
   apf_mesh->end(itr);

   Table *edge_element = NULL;
   /*if (mesh.NURBSext)
   {
      activeBdrElem.SetSize(mesh.GetNBE());
      activeBdrElem = false;
   }*/

   // Count number of boundaries by classification
   int BcDim = Dim - 1;
   itr = apf_mesh->begin(BcDim);
   NumOfBdrElements = 0;

   while ((ent=apf_mesh->iterate(itr)))
   {
      apf::ModelEntity* mdEnt = apf_mesh->toModel(ent);
      if (apf_mesh->getModelType(mdEnt) == BcDim)
      {
         NumOfBdrElements++;
      }
   }
   apf_mesh->end(itr);

   boundary.SetSize(NumOfBdrElements);
   int bdr_ctr=0;
   // Read boundary from SCOREC mesh
   itr = apf_mesh->begin(BcDim);
   while ((ent = apf_mesh->iterate(itr)))
   {
      // Check if this mesh entity is on the model boundary
      apf::ModelEntity* mdEnt = apf_mesh->toModel(ent);
      if (apf_mesh->getModelType(mdEnt) == BcDim)
      {
         apf::Downward verts;
         int num_verts = apf_mesh->getDownward(ent, 0, verts);
         int attr = 1 ;// apf_mesh->getModelTag(mdEnt);
         /*if (apf_mesh->hasTag(ent,atts)){
             apf_mesh->getIntTag(ent,attTag,&attr);
           }*/

         int geom_type = BaseBdrGeom;// apf_mesh->getType(ent);
         boundary[bdr_ctr] = ReadElement( ent, geom_type, verts, attr, v_num_loc);
         bdr_ctr++;
      }
   }
   apf_mesh->end(itr);

   Mesh::SetMeshGen();
   Mesh::SetAttributes();

   // This is called by the default Mesh constructor
   Mesh::InitTables();
   bool refine = false;
   bool fix_orientation = true;
   this->FinalizeTopology();
   Mesh::Finalize(refine, fix_orientation);
   if (Dim > 1)
   {
      el_to_edge = new Table;
      NumOfEdges = Mesh::GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   else
   {
      NumOfEdges = 0;
   }

   STable3D *faces_tbl = NULL;
   if (Dim == 3)
   {
      faces_tbl = GetElementToFaceTable(1);
   }
   else
   {
      NumOfFaces = 0;
   }

   GenerateFaces();

   ListOfIntegerSets  groups;
   IntegerSet         group;

   // The first group is the local one
   group.Recreate(1, &MyRank);
   groups.Insert(group);

#ifdef MFEM_DEBUG
   if (Dim < 3 && GetNFaces() != 0)
   {
      cerr << "ParMesh::ParMesh (proc " << MyRank << ") : "
           "(Dim < 3 && mesh.GetNFaces() != 0) is true!" << endl;
      mfem_error();
   }
#endif

   // Determine shared faces
   int sface_counter = 0;
   Array<int> face_group(GetNFaces());
   apf::FieldShape* fc_shape =apf::getConstant(2);
   apf::Numbering* faceNum = apf::createNumbering(apf_mesh, "FaceNumbering",
                                                  fc_shape, 1);
   Array<int> SharedFaceIds;
   if (Dim > 2)
   {
      // Number Faces
      apf::Numbering* AuxFaceNum = apf::numberOwnedDimension(apf_mesh,
                                                             "AuxFaceNumbering", 2);
      apf::GlobalNumbering* GlobalFaceNum = apf::makeGlobal(AuxFaceNum, true);
      apf::synchronize(GlobalFaceNum);

      // Take this process global ids and sort
      Array<int> thisFaceIds(GetNFaces());

      itr = apf_mesh->begin(2);
      all_num = 0;
      shared_num = 0;
      while ((ent = apf_mesh->iterate(itr)))
      {
         unsigned int id = apf::getNumber(GlobalFaceNum, ent, 0, 0);
         thisFaceIds[all_num++] = id;
         if (apf_mesh->isShared(ent))
         {
            shared_num++;
         }
      }
      apf_mesh->end(itr);
      thisFaceIds.Sort();

      // Create local numbering that respects the global ordering
      SharedFaceIds.SetSize(shared_num);
      shared_num = 0;
      itr = apf_mesh->begin(2);
      while ((ent = apf_mesh->iterate(itr)))
      {
         // Id from global numbering
         unsigned int id = apf::getNumber(GlobalFaceNum, ent, 0, 0);
         // Find its position at sorted list
         int ordered_id = thisFaceIds.Find(id);
         // Assign as local number
         apf::number(faceNum, ent, 0, 0, ordered_id);

         if (apf_mesh->isShared(ent))
         {
            SharedFaceIds[shared_num++] = ordered_id;
         }
      }
      apf_mesh->end(itr);
      SharedFaceIds.Sort();
      apf::destroyGlobalNumbering(GlobalFaceNum);

      itr = apf_mesh->begin(2);
      while ((ent = apf_mesh->iterate(itr)))
      {
         int faceId = apf::getNumber(faceNum, ent, 0, 0);
         face_group[faceId] = -1;
         if (apf_mesh->isShared(ent))
         {
            // Number of adjacent element
            int thisNumAdjs = 2;
            int eleRanks[thisNumAdjs];

            // Get the Ids
            apf::Parts res;
            apf_mesh->getResidence(ent, res);
            int kk = 0;
            for (std::set<int>::iterator itr = res.begin(); itr != res.end(); ++itr)
            {
               eleRanks[kk++] = *itr;
            }

            group.Recreate(2, eleRanks);
            face_group[faceId] = groups.Insert(group) - 1;
            sface_counter++;
         }
      }
      apf_mesh->end(itr);

   }

   // Determine shared edges
   int sedge_counter = 0;
   if (!edge_element)
   {
      edge_element = new Table;
      if (Dim == 1)
      {
         edge_element->SetDims(0,0);
      }
      else
      {
         edge_element->SetSize(GetNEdges(), 1);
      }
   }

   // Number Edges
   apf::Numbering* AuxEdgeNum = apf::numberOwnedDimension(apf_mesh,
                                                          "EdgeNumbering", 1);
   apf::GlobalNumbering* GlobalEdgeNum = apf::makeGlobal(AuxEdgeNum, true);
   apf::synchronize(GlobalEdgeNum);

   // Take this process global ids and sort
   Array<int> thisEdgeIds(GetNEdges());

   itr = apf_mesh->begin(1);
   all_num = 0;
   shared_num = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
      unsigned int id = apf::getNumber(GlobalEdgeNum, ent, 0, 0);
      thisEdgeIds[all_num++] = id;
      if (apf_mesh->isShared(ent))
      {
         shared_num++;
      }
   }
   apf_mesh->end(itr);
   thisEdgeIds.Sort();

   // Create local numbering that respects the global ordering
   apf::FieldShape* ed_shape =apf::getConstant(1);
   apf::Numbering* edgeNum = apf::createNumbering(apf_mesh, "EdgeNumbering",
                                                  ed_shape, 1);

   Array<int> SharedEdgeIds(shared_num);
   shared_num = 0;
   itr = apf_mesh->begin(1);
   while ((ent = apf_mesh->iterate(itr)))
   {
      // Id from global numbering
      unsigned int id = apf::getNumber(GlobalEdgeNum, ent, 0, 0);
      // Find its position at sorted list
      int ordered_id = thisEdgeIds.Find(id);
      // Assign as local number
      apf::number(edgeNum, ent, 0, 0, ordered_id);

      if (apf_mesh->isShared(ent))
      {
         SharedEdgeIds[shared_num++] = ordered_id;
      }
   }
   apf_mesh->end(itr);
   SharedEdgeIds.Sort();
   apf::destroyGlobalNumbering(GlobalEdgeNum);

   itr = apf_mesh->begin(1);
   i = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
      apf::Downward verts;
      int num_verts = apf_mesh->getDownward(ent,0,verts);
      int ed_ids[2];
      ed_ids[0] = apf::getNumber(v_num_loc, verts[0], 0, 0);
      ed_ids[1] = apf::getNumber(v_num_loc, verts[1], 0, 0);

      int edId = apf::getNumber(edgeNum, ent, 0, 0);

      edge_element->GetRow(edId)[0] = -1;

      if (apf_mesh->isShared(ent))
      {
         sedge_counter++;

         // Number of adjacent element
         apf::Parts res;
         apf_mesh->getResidence(ent, res);
         int thisNumAdjs = res.size();
         int eleRanks[thisNumAdjs];

         // Get the Ids
         int kk = 0;
         for ( std::set<int>::iterator itr = res.begin(); itr != res.end(); itr++)
         {
            eleRanks[kk++] = *itr;
         }

         // Generate the group
         group.Recreate(thisNumAdjs, eleRanks);
         edge_element->GetRow(edId)[0] = groups.Insert(group) - 1;
         // edge_element->GetRow(i)[0] = groups.Insert(group) - 1;

      }
      i++;
   }
   apf_mesh->end(itr);

   // Determine shared vertices
   int svert_counter = 0;
   Table *vert_element = new Table;
   vert_element->SetSize(GetNV(), 1);

   itr = apf_mesh->begin(0);
   while ((ent = apf_mesh->iterate(itr)))
   {
      int vtId = apf::getNumber(v_num_loc, ent, 0, 0);
      vert_element->GetRow(vtId)[0] = -1;


      if (apf_mesh->isShared(ent))
      {
         svert_counter++;
         // Number of adjacent element
         apf::Parts res;
         apf_mesh->getResidence(ent, res);
         int thisNumAdjs = res.size();
         int eleRanks[thisNumAdjs];

         // Get the Ids
         int kk = 0;
         for (std::set<int>::iterator itr = res.begin(); itr != res.end(); itr++)
         {
            eleRanks[kk++] = *itr;
         }

         group.Recreate(thisNumAdjs, eleRanks);
         vert_element->GetRow(vtId)[0]= groups.Insert(group) - 1;
      }
   }
   apf_mesh->end(itr);

   // Build group_sface
   group_sface.MakeI(groups.Size()-1);

   for (i = 0; i < face_group.Size(); i++)
   {
      if (face_group[i] >= 0)
      {
         group_sface.AddAColumnInRow(face_group[i]);
      }
   }

   group_sface.MakeJ();

   sface_counter = 0;
   for (i = 0; i < face_group.Size(); i++)
   {
      if (face_group[i] >= 0)
      {
         group_sface.AddConnection(face_group[i], sface_counter++);
      }
   }

   group_sface.ShiftUpI();

   // Build group_sedge
   group_sedge.MakeI(groups.Size()-1);

   for (i = 0; i < edge_element->Size(); i++)
   {
      if (edge_element->GetRow(i)[0] >= 0)
      {
         group_sedge.AddAColumnInRow(edge_element->GetRow(i)[0]);
      }
   }

   group_sedge.MakeJ();

   sedge_counter = 0;
   for (i = 0; i < edge_element->Size(); i++)
   {
      if (edge_element->GetRow(i)[0] >= 0)
      {
         group_sedge.AddConnection(edge_element->GetRow(i)[0], sedge_counter++);
      }
   }

   group_sedge.ShiftUpI();

   // Build group_svert
   group_svert.MakeI(groups.Size()-1);

   for (i = 0; i < vert_element->Size(); i++)
   {
      if (vert_element->GetRow(i)[0] >= 0)
      {
         group_svert.AddAColumnInRow(vert_element->GetRow(i)[0]);
      }
   }

   group_svert.MakeJ();

   svert_counter = 0;
   for (i = 0; i < vert_element->Size(); i++)
   {
      if (vert_element->GetRow(i)[0] >= 0)
      {
         group_svert.AddConnection(vert_element->GetRow(i)[0], svert_counter++);
      }
   }
   group_svert.ShiftUpI();

   // Build shared_faces and sface_lface
   shared_faces.SetSize(sface_counter);
   sface_lface. SetSize(sface_counter);

   if (Dim == 3)
   {
      sface_counter = 0;
      itr = apf_mesh->begin(2);
      while ((ent = apf_mesh->iterate(itr)))
      {
         if (apf_mesh->isShared(ent))
         {
            // Generate the face
            int fcId = apf::getNumber(faceNum, ent, 0, 0);
            int ctr = SharedFaceIds.Find(fcId);

            apf::Downward verts;
            int num_vert =  apf_mesh->getDownward(ent,0,verts);
            int geom = BaseBdrGeom;
            int attr = 1;
            shared_faces[ctr] = ReadElement(ent, geom, verts, attr,
                                            v_num_loc);

            int *v = shared_faces[ctr]->GetVertices();
            switch ( geom )
            {
               case Element::TRIANGLE:
                  sface_lface[ctr] = (*faces_tbl)(v[0], v[1], v[2]);
                  // The marking for refinement is omitted. All done in PUMI
                  break;
               case Element::QUADRILATERAL:
                  sface_lface[ctr] =
                     (*faces_tbl)(v[0], v[1], v[2], v[3]);
                  break;
            }
            sface_counter++;
         }
      }
      apf_mesh->end(itr);
      delete faces_tbl;
   }

   // Build shared_edges and sedge_ledge
   shared_edges.SetSize(sedge_counter);
   sedge_ledge. SetSize(sedge_counter);

   {
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      sedge_counter = 0;
      itr = apf_mesh->begin(1);
      while ((ent = apf_mesh->iterate(itr)))
      {
         if (apf_mesh->isShared(ent))
         {
            int edId = apf::getNumber(edgeNum, ent, 0, 0);
            int ctr = SharedEdgeIds.Find(edId);
            apf::Downward verts;
            apf_mesh->getDownward(ent, 0, verts);
            int id1, id2;
            id1 = apf::getNumber(v_num_loc, verts[0], 0, 0);
            id2 = apf::getNumber(v_num_loc, verts[1], 0, 0);
            if (id1 > id2) { swap(id1,id2); }

            shared_edges[ctr] = new Segment(id1, id2, 1);
            if ((sedge_ledge[ctr] = v_to_v(id1,id2)) < 0)
            {
               cerr << "\n\n\n" << MyRank << ": ParMesh::ParMesh: "
                    << "ERROR in v_to_v\n\n" << endl;
               mfem_error();
            }

            sedge_counter++;
         }
      }
   }
   apf_mesh->end(itr);

   delete edge_element;

   // Build svert_lvert
   svert_lvert.SetSize(svert_counter);

   svert_counter = 0;
   itr = apf_mesh->begin(0);
   while ((ent = apf_mesh->iterate(itr)))
   {
      if (apf_mesh->isShared(ent))
      {
         int vt_id = apf::getNumber(v_num_loc, ent, 0, 0);
         int ctr = SharedVertIds.Find(vt_id);
         svert_lvert[ctr] = vt_id;
         svert_counter++;
      }
   }
   apf_mesh->end(itr);

   delete vert_element;

   // Build the group communication topology
   gtopo.Create(groups, 822);


   if (curved) // curved mesh
   {
      GridFunctionPumi* auxNodes = new GridFunctionPumi(this, apf_mesh, v_num_loc,
                                                        crd_shape->getOrder());
      Nodes = new ParGridFunction(this, auxNodes);
      Nodes->SetData(auxNodes->GetData());
      this->edge_vertex = NULL;
      own_nodes = 1;
   }

   //pumi_ghost_delete(apf_mesh);
   apf::destroyNumbering(v_num_loc);
   apf::destroyNumbering(edgeNum);
   apf::destroyNumbering(faceNum);
   have_face_nbr_data = false;
}


// GridFunctionPumi Implementation

GridFunctionPumi::GridFunctionPumi(Mesh* m, apf::Mesh2* PumiM,
                                   apf::Numbering* v_num_loc,
                                   const int mesh_order)
{
   // Set to zero
   SetDataAndSize(NULL, 0);
   int ec;
   int spDim = m->SpaceDimension();
   // Needs to be modified for other orders
   if (mesh_order == 1)
   {
      mfem_error("GridFunction::GridFunction : First order mesh!");
   }
   else if (mesh_order == 2)
   {
      fec =  FiniteElementCollection::New("Quadratic");
   }
   else
   {
      fec = new H1_FECollection(mesh_order, m->Dimension());
   }
   int ordering = 1; // x1y1z1/x2y2z2/...
   fes = new FiniteElementSpace(m, fec, spDim, ordering);
   int data_size = fes->GetVSize();

   // Read Pumi mesh data
   this->SetSize(data_size);
   double* PumiData = this->GetData();

   apf::MeshEntity* ent;
   apf::MeshIterator* itr;


   // Assume all element type are the same i.e. tetrahedral
   const FiniteElement* H1_elem = fes->GetFE(1);
   const IntegrationRule &All_nodes = H1_elem->GetNodes();
   int num_vert = m->GetElement(1)->GetNVertices();
   int nnodes = All_nodes.Size();

   // loop over elements
   apf::Field* crd_field = PumiM->getCoordinateField();

   int nc = apf::countComponents(crd_field);
   int iel = 0;
   itr = PumiM->begin(m->Dimension());
   while ((ent = PumiM->iterate(itr)))
   {
      Array<int> vdofs;
      fes->GetElementVDofs(iel, vdofs);

      // create Pumi element to interpolate
      apf::MeshElement* mE = apf::createMeshElement(PumiM, ent);
      apf::Element* elem = apf::createElement(crd_field, mE);

      // Vertices are already interpolated
      for (int ip = 0; ip < nnodes; ip++)// num_vert
      {
         // Take parametric coordinates of the node
         apf::Vector3 param;
         param[0] = All_nodes.IntPoint(ip).x;
         param[1] = All_nodes.IntPoint(ip).y;
         param[2] = All_nodes.IntPoint(ip).z;


         // Compute the interpolating coordinates
         apf::DynamicVector phCrd(nc);
         apf::getComponents(elem, param, &phCrd[0]);

         // Fill the nodes list
         for (int kk = 0; kk < spDim; ++kk)
         {
            int dof_ctr = ip + kk * nnodes;
            PumiData[vdofs[dof_ctr]] = phCrd[kk];
         }

      }
      iel++;
      apf::destroyElement(elem);
      apf::destroyMeshElement(mE);
   }
   PumiM->end(itr);

   sequence = 0;
}


}

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SCOREC
