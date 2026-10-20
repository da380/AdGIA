/**
 * @file manifest.hpp
 * @brief MFEM glue over planetmodel's mesh-manifest reader: opening the
 * mesh and the fields a manifest names, and the manifest's attribute lists
 * as the arrays and markers the problem classes take.
 *
 * The manifest itself (schema `planetmodel.mesh.manifest/5`) is read and
 * checked by `planetmodel::MeshManifest`, which knows no MFEM and reports
 * faults as `planetmodel::Error` exceptions — a driver catches them once,
 * around its setup. What is MFEM's lives here: LoadMesh opens the mesh
 * with the manifest's read options (the options the fields' dof numbering
 * was written under), LoadField a field beside it, and Marker and ToArray
 * turn attribute lists into mfem::Array form.
 */

#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "planetmodel/manifest.hpp"

namespace AdGIA {

/** @brief An attribute list as the mfem::Array the problem classes take. */
mfem::Array<int> ToArray(const std::vector<int>& attributes);

/** @brief A 0/1 marker of length @p size with the entries of
 * @p attributes set. @p size is `attributes.Max()` or
 * `bdr_attributes.Max()` of the mesh the marker is for; an attribute
 * above it is ignored, so that a marker for a SubMesh can be made from
 * the attributes of the parent.
 * @throws planetmodel::Error when an attribute is below one. */
mfem::Array<int> Marker(const std::vector<int>& attributes, int size);
mfem::Array<int> Marker(const mfem::Array<int>& attributes, int size);

/** @brief The manifest's mesh, opened with its read options. The dof
 * numbering of the manifest's fields is that of the mesh opened this way.
 * @throws planetmodel::Error when the mesh file cannot be opened. */
mfem::Mesh LoadMesh(const planetmodel::MeshManifest& manifest);

/** @brief The named field on @p mesh, which must be LoadMesh()'s. The
 * GridFunction owns its space and collection.
 * @throws planetmodel::Error when the manifest has no such field, the
 * file cannot be opened, or it holds another space or vdim than the
 * manifest says. */
std::unique_ptr<mfem::GridFunction> LoadField(
    const planetmodel::MeshManifest& manifest, mfem::Mesh& mesh,
    const std::string& name);

/** @brief A readable summary: layers, interfaces, fields, scales. */
void Print(const planetmodel::MeshManifest& manifest, std::ostream& os);

}  // namespace AdGIA
