// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ATTRIBUTE_SETS
#define MFEM_ATTRIBUTE_SETS

#include "../config/config.hpp"
#include "../general/arrays_by_name.hpp"

#include <iostream>
#include <map>
#include <set>
#include <string>

namespace mfem
{

class AttributeSets
{
private:
   const int def_width = 10;

public:
   /// Named sets of element attributes
   ArraysByName<int> attr_sets;

   /// Named sets of boundary attributes
   ArraysByName<int> bdr_attr_sets;

   AttributeSets() {}

   /// @brief Create a copy of the internal data to the provided @a copy.
   void Copy(AttributeSets &copy) const;

   /// @brief Return true if any named sets are currently been defined
   bool SetsExist() const;

   /// @brief Return all attribute set names as an STL set
   std::set<std::string> GetAttributeSetNames() const;

   /// @brief Return all boundary attribute set names as an STL set
   std::set<std::string> GetBdrAttributeSetNames() const;

   /// @brief Return true is the named attribute set is present
   bool AttributeSetExists(const std::string &name) const;
   /// @brief Return true is the named boundary attribute set is present
   bool BdrAttributeSetExists(const std::string &name) const;

   /// @brief Create an empty named attribute set
   Array<int> & CreateAttributeSet(const std::string &set_name);
   /// @brief Create an empty named boundary attribute set
   Array<int> & CreateBdrAttributeSet(const std::string &set_name);

   /// @brief Delete a named attribute set
   void DeleteAttributeSet(const std::string &set_name);
   /// @brief Delete a named boundary attribute set
   void DeleteBdrAttributeSet(const std::string &set_name);

   /// @brief Create a new attribute set
   /**
       @param[in] set_name The name of the new set
       @param[in] attr An array of attribute numbers making up the new set

       @note If an attribute set matching this name already exists, that set
       will be replaced with this new attribute set.

       @note The attribute numbers are not checked for validity or
       existence within the mesh.
    */
   void SetAttributeSet(const std::string &set_name, const Array<int> &attr);
   /// @brief Create a new boundary attribute set
   /**
       @param[in] set_name The name of the new set
       @param[in] attr An array of attribute numbers making up the new set

       @note If a boundary attribute set matching this name already exists,
       that set will be replaced with this new boundary attribute set.

       @note The boundary attribute numbers are not checked for validity or
       existence within the mesh.
    */
   void SetBdrAttributeSet(const std::string &set_name, const Array<int> &attr);

   /// @brief Add a single entry to an existing attribute set
   /**
       @param[in] set_name The name of the set being augmented
       @param[in] attr A single attribute number to be inserted in the set

       @note If the named set does not exist an error message will be printed
       and execution will halt.
       @note Duplicate entries will be ignored and the resulting sets will be
       sorted.
    */
   void AddToAttributeSet(const std::string &set_name, int attr);

   /// @brief Add an array of entries to an existing attribute set
   /**
       @param[in] set_name The name of the set being augmented
       @param[in] attr Array of attribute numbers to be inserted in the set

       @note If the named set does not exist an error message will be printed
       and execution will halt.
       @note Duplicate entries will be ignored and the resulting sets will be
       sorted.
    */
   void AddToAttributeSet(const std::string &set_name,
                          const Array<int> &attr);

   /// @brief Add a single entry to an existing boundary attribute set
   /**
       @param[in] set_name The name of the set being augmented
       @param[in] attr A single attribute number to be inserted in the set

       @note If the named set does not exist an error message will be printed
       and execution will halt.
       @note Duplicate entries will be ignored and the resulting sets will be
       sorted.
    */
   void AddToBdrAttributeSet(const std::string &set_name, int attr);

   /// @brief Add an array of entries to an existing boundary attribute set
   /**
       @param[in] set_name The name of the set being augmented
       @param[in] attr Array of attribute numbers to be inserted in the set

       @note If the named set does not exist an error message will be printed
       and execution will halt.
       @note Duplicate entries will be ignored and the resulting sets will be
       sorted.
    */
   void AddToBdrAttributeSet(const std::string &set_name,
                             const Array<int> &attr);

   /// @brief Remove a single entry from an existing attribute set
   /**
       @param[in] set_name The name of the set being modified
       @param[in] attr A single attribute number to be removed from the set

       @note If the named set does not exist an error message will be printed
       and execution will halt.
       @note If @a attr is not a member of the named set the set will not
       be modified and no error will occur.
    */
   void RemoveFromAttributeSet(const std::string &set_name, int attr);

   /// @brief Remove a single entry from an existing boundary attribute set
   /**
       @param[in] set_name The name of the set being modified
       @param[in] attr A single attribute number to be removed from the set

       @note If the named set does not exist an error message will be printed
       and execution will halt.
       @note If @a attr is not a member of the named set the set will not
       be modified and no error will occur.
    */
   void RemoveFromBdrAttributeSet(const std::string &set_name, int attr);

   /// @brief Print the contents of the container to an output stream
   ///
   /// @note The two types of named attribute sets will be printed with
   /// hearders "attribute sets" and "bdr_attribute_sets". The array entries
   /// will contain 10 entries per line. A specific number of entries per line
   /// can be used by changing the @a width argument.
   void Print(std::ostream &out = mfem::out, int width = -1) const;

   /// @brief Print the contents of only the domain attribute container
   ///
   /// @note The array entries will contain 10 entries per line. A specific
   /// number of entries per line can be used by changing the @a width argument.
   void PrintAttributeSets(std::ostream &out = mfem::out,
                           int width = -1) const;

   /// @brief Print the contents of only the boundary attribute container
   ///
   /// @note The array entries will contain 10 entries per line. A specific
   /// number of entries per line can be used by changing the @a width argument.
   void PrintBdrAttributeSets(std::ostream &out = mfem::out,
                              int width = -1) const;

   /// @brief Access a named attribute set
   /**
       @param[in] set_name The name of the set being accessed

       @note If the named set does not exist an error message will be printed
       and execution will halt.

       @note The reference returned by this method can be invalidated by
       subsequent calls to SetAttributeSet, ClearAttributeSet, or
       RemoveFromAttributeSet. AddToAttributeSet should not invalidate this
       reference.
    */
   Array<int> & GetAttributeSet(const std::string & set_name);

   /// @brief Access a named boundary attribute set
   /**
       @param[in] set_name The name of the set being accessed

       @note If the named set does not exist an error message will be printed
       and execution will halt.

       @note The reference returned by this method can be invalidated by
       subsequent calls to SetBdrAttributeSet, ClearBdrAttributeSet, or
       RemoveFromBdrAttributeSet. AddToBdrAttributeSet should not invalidate
       this reference.
    */
   Array<int> & GetBdrAttributeSet(const std::string & set_name);
};

} // namespace mfem

#endif
