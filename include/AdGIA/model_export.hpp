/**
 * @file model_export.hpp
 * @brief Fields on a mesh written out as a planetmodel model file: the
 * interchange form of the sensitivity kernels (the CSV samplers of
 * examples/kernel_sampling.hpp stay for quick looks).
 *
 * The file is laid out for values kept as they are: radial elements of
 * two nodes (linear between), a Gauss–Legendre angular grid, and the
 * bilinear lateral rule, which never overshoots a peak the way the
 * bicubic rule would. Each field is sampled at the file's own points
 * (batched FindPoints on the field's mesh; a point the discrete mesh
 * just misses — a curved facet at a layer boundary — is nudged radially
 * towards the layer's middle and retried). The caller states each
 * field's dimensions and meaning: the library ships no kernel names.
 *
 * The skeleton is the manifest's body layers, so a file written here
 * passes CheckModelMatchesManifest and reads back onto the same mesh
 * through ModelScalarCoefficient. Sampling is serial — gather a
 * parallel kernel to the root first, the way the kernel examples
 * already do for the CSVs. Faults are planetmodel::Error exceptions.
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "planetmodel/manifest.hpp"
#include "planetmodel/modelfile.hpp"

namespace AdGIA {

/** @brief One field to write: its values, and what the caller says it
 * is. The field's mesh must be the manifest's, or a SubMesh of it
 * keeping the parent's attributes (a displacement SubMesh). */
struct ExportedField {
  std::string name;
  /** @brief The field, on a serial mesh. */
  const mfem::GridFunction* field = nullptr;
  /** @brief Its dimensions (mass, length, time exponents), as the
   * caller states them. */
  planetmodel::Dimensions dimensions{};
  /** @brief What it is, e.g. "derivative of the objective with respect
   * to the viscosity, per unit volume". */
  std::string meaning;
  /** @brief The body attributes to sample on; empty samples every body
   * layer present in the field's mesh. */
  std::vector<int> attributes{};
};

/** @brief How the file is laid out. */
struct ModelExportOptions {
  /** @brief The radial size of the file's two-node elements. */
  double radial_size = 0.05;
  /** @brief The angular grid (GaussLegendre(lmax)). */
  int lmax = 64;
  /** @brief The file's name attribute; empty leaves it unset. */
  std::string name;
};

/** @brief Write @p fields, sampled on the manifest's body layers, as a
 * model file at @p path.
 * @throws planetmodel::Error when a field names a layer its mesh does
 * not hold, a sample point cannot be found on the mesh, or the file
 * cannot be written. */
void WriteKernelModelFile(const std::filesystem::path& path,
                          const planetmodel::MeshManifest& manifest,
                          const std::vector<ExportedField>& fields,
                          const ModelExportOptions& options = {});

}  // namespace AdGIA
