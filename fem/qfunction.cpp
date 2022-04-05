// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "qfunction.hpp"
#include "quadinterpolator.hpp"

namespace mfem
{

QuadratureFunction::QuadratureFunction(Mesh *mesh, std::istream &in)
{
   const char *msg = "invalid input stream";
   std::string ident;

   qspace = new QuadratureSpace(mesh, in);
   own_qspace = true;

   in >> ident; MFEM_VERIFY(ident == "VDim:", msg);
   in >> vdim;

   Load(in, vdim*qspace->GetSize());
}

void QuadratureFunction::SetSpace(QuadratureSpace *qspace_, int vdim_)
{
   if (qspace_ != qspace)
   {
      if (own_qspace) { delete qspace; }
      qspace = qspace_;
      own_qspace = false;
   }
   vdim = (vdim_ < 0) ? vdim : vdim_;
   SetSize(vdim*qspace->GetSize());
}

void QuadratureFunction::SetSpace(
   QuadratureSpace *qspace_, double *qf_data, int vdim_)
{
   if (qspace_ != qspace)
   {
      if (own_qspace) { delete qspace; }
      qspace = qspace_;
      own_qspace = false;
   }
   vdim = (vdim_ < 0) ? vdim : vdim_;
   NewDataAndSize(qf_data, vdim*qspace->GetSize());
}

QuadratureFunction &QuadratureFunction::operator=(double value)
{
   Vector::operator=(value);
   return *this;
}

QuadratureFunction &QuadratureFunction::operator=(const Vector &v)
{
   MFEM_ASSERT(qspace && v.Size() == this->Size(), "");
   Vector::operator=(v);
   return *this;
}

QuadratureFunction &QuadratureFunction::operator=(const QuadratureFunction &v)
{
   return this->operator=((const Vector &)v);
}

void QuadratureFunction::Save(std::ostream &os) const
{
   GetSpace()->Save(os);
   os << "VDim: " << vdim << '\n'
      << '\n';
   Vector::Print(os, vdim);
   os.flush();
}

void QuadratureFunction::ProjectGridFunction(const GridFunction &gf)
{
   SetVDim(gf.VectorDim());

   const FiniteElementSpace &gf_fes = *gf.FESpace();
   const bool use_tensor_products = UsesTensorBasis(gf_fes);
   const ElementDofOrdering ordering = use_tensor_products ?
                                       ElementDofOrdering::LEXICOGRAPHIC :
                                       ElementDofOrdering::NATIVE;

   // Use element restriction to go from L-vector to E-vector
   const Operator *R = gf_fes.GetElementRestriction(ordering);
   Vector e_vec(R->Height());
   R->Mult(gf, e_vec);

   // Use quadrature interpolator to go from E-vector to Q-vector
   const QuadratureInterpolator *qi(gf_fes.GetQuadratureInterpolator(*GetSpace()));
   qi->SetOutputLayout(QVectorLayout::byVDIM);
   qi->DisableTensorProducts(!use_tensor_products);
   qi->Values(e_vec, *this);
}

void QuadratureFunction::ProjectCoefficient(Coefficient &coeff)
{
   coeff.Project(*this);
}

void QuadratureFunction::ProjectCoefficient(VectorCoefficient &coeff)
{
   // Should we automatically resize, or check the vdim?
   // MFEM_ASSERT(vdim == coeff.GetVDim(), "Wrong sizes.");
   SetVDim(coeff.GetVDim());
   coeff.Project(*this);
}

void QuadratureFunction::ProjectSymmetricCoefficient(
   SymmetricMatrixCoefficient &coeff)
{
   coeff.ProjectSymmetric(*this);
}

void QuadratureFunction::ProjectCoefficient(MatrixCoefficient &coeff,
                                            bool transpose)
{
   coeff.Project(*this, transpose);
}

std::ostream &operator<<(std::ostream &os, const QuadratureFunction &qf)
{
   qf.Save(os);
   return os;
}

void QuadratureFunction::SaveVTU(std::ostream &os, VTKFormat format,
                                 int compression_level) const
{
   os << R"(<VTKFile type="UnstructuredGrid" version="0.1")";
   if (compression_level != 0)
   {
      os << R"( compressor="vtkZLibDataCompressor")";
   }
   os << " byte_order=\"" << VTKByteOrder() << "\">\n";
   os << "<UnstructuredGrid>\n";

   const char *fmt_str = (format == VTKFormat::ASCII) ? "ascii" : "binary";
   const char *type_str = (format != VTKFormat::BINARY32) ? "Float64" : "Float32";
   std::vector<char> buf;

   Mesh &mesh = *qspace->GetMesh();

   int np = qspace->GetSize();
   int ne = mesh.GetNE();
   int sdim = mesh.SpaceDimension();

   // For quadrature functions, each point is a vertex cell, so number of cells
   // is equal to number of points
   os << "<Piece NumberOfPoints=\"" << np
      << "\" NumberOfCells=\"" << np << "\">\n";

   // print out the points
   os << "<Points>\n";
   os << "<DataArray type=\"" << type_str
      << "\" NumberOfComponents=\"3\" format=\"" << fmt_str << "\">\n";

   Vector pt(sdim);
   for (int i = 0; i < ne; i++)
   {
      ElementTransformation &T = *mesh.GetElementTransformation(i);
      const IntegrationRule &ir = GetElementIntRule(i);
      for (int j = 0; j < ir.Size(); j++)
      {
         T.Transform(ir[j], pt);
         WriteBinaryOrASCII(os, buf, pt[0], " ", format);
         if (sdim > 1) { WriteBinaryOrASCII(os, buf, pt[1], " ", format); }
         else { WriteBinaryOrASCII(os, buf, 0.0, " ", format); }
         if (sdim > 2) { WriteBinaryOrASCII(os, buf, pt[2], "", format); }
         else { WriteBinaryOrASCII(os, buf, 0.0, "", format); }
         if (format == VTKFormat::ASCII) { os << '\n'; }
      }
   }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>\n";
   os << "</Points>\n";

   // Write cells (each cell is just a vertex)
   os << "<Cells>\n";
   // Connectivity
   os << R"(<DataArray type="Int32" Name="connectivity" format=")"
      << fmt_str << "\">\n";

   for (int i=0; i<np; ++i) { WriteBinaryOrASCII(os, buf, i, "\n", format); }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>\n";
   // Offsets
   os << R"(<DataArray type="Int32" Name="offsets" format=")"
      << fmt_str << "\">\n";
   for (int i=0; i<np; ++i) { WriteBinaryOrASCII(os, buf, i, "\n", format); }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>\n";
   // Types
   os << R"(<DataArray type="UInt8" Name="types" format=")"
      << fmt_str << "\">\n";
   for (int i = 0; i < np; i++)
   {
      uint8_t vtk_cell_type = VTKGeometry::POINT;
      WriteBinaryOrASCII(os, buf, vtk_cell_type, "\n", format);
   }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>\n";
   os << "</Cells>\n";

   os << "<PointData>\n";
   os << "<DataArray type=\"" << type_str << "\" Name=\"u\" format=\""
      << fmt_str << "\" NumberOfComponents=\"" << vdim << "\">\n";
   for (int i = 0; i < ne; i++)
   {
      DenseMatrix vals;
      GetElementValues(i, vals);
      for (int j = 0; j < vals.Size(); ++j)
      {
         for (int vd = 0; vd < vdim; ++vd)
         {
            WriteBinaryOrASCII(os, buf, vals(vd, j), " ", format);
         }
         if (format == VTKFormat::ASCII) { os << '\n'; }
      }
   }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>\n";
   os << "</PointData>\n";

   os << "</Piece>\n";
   os << "</UnstructuredGrid>\n";
   os << "</VTKFile>" << std::endl;
}

void QuadratureFunction::SaveVTU(const std::string &filename, VTKFormat format,
                                 int compression_level) const
{
   std::ofstream f(filename + ".vtu");
   SaveVTU(f, format, compression_level);
}


FaceQuadratureFunction &FaceQuadratureFunction::operator=(double value)
{
   Vector::operator=(value);
   return *this;
}

FaceQuadratureFunction &FaceQuadratureFunction::operator=(const Vector &v)
{
   MFEM_ASSERT(qspace && v.Size() == this->Size(), "");
   Vector::operator=(v);
   return *this;
}

FaceQuadratureFunction &FaceQuadratureFunction::operator=(
   const FaceQuadratureFunction &v)
{
   return this->operator=((const Vector &)v);
}

}
