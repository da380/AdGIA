/**
 * @file model_coefficient.hpp
 * @brief Model-file fields as MFEM coefficients: the variables of the
 * netCDF model file that planetmodel writes beside a mesh
 * (`export_mfem(..., model_file=...)`), evaluated at an integration
 * point's physical position on the layer of its element's attribute.
 *
 * The export's contract binds the two: layer n of the file is element
 * attribute n + 1 of the mesh, the shells outside the body have no model
 * data, and the mesh's coordinates and the file's values share the
 * model's units. LoadModelFile reads the file the manifest records and
 * checks that contract against the manifest's body layers; the
 * coefficients then evaluate the file's fields through planetmodel's
 * interpolants (thread-safe, cheap to copy), so a coefficient built once
 * serves serial and parallel assembly alike.
 *
 * Positions are taken as physical: r = |x|, clamped into the layer's
 * radial interval (the mesh conforms to the material interfaces, so an
 * excursion is the geometric error of a curved facet, not a different
 * layer); a 2-D mesh is read as the equatorial plane. Faults are
 * planetmodel::Error exceptions, to be caught at the driver level.
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "planetmodel/manifest.hpp"
#include "planetmodel/modelfile.hpp"

namespace AdGIA {

/** @brief Check a model file against a mesh's manifest: layer n of the
 * file must be attribute n + 1 of the mesh (count and radii, to the
 * file's tolerance).
 * @throws planetmodel::Error when the skeleton does not match the
 * manifest's body layers. */
void CheckModelMatchesManifest(const planetmodel::ModelFile& file,
                               const planetmodel::MeshManifest& manifest);

/** @brief The model file the manifest records, read and checked against
 * the manifest's body layers (count and radii, to the file's tolerance).
 * @throws planetmodel::Error when the manifest records no model, the
 * file cannot be read, or its skeleton does not match the layers. */
planetmodel::ModelFile LoadModelFile(const planetmodel::MeshManifest& manifest);

/** @brief A scalar variable of a model file as a Coefficient on the mesh
 * the file was exported beside: at an integration point, the field of the
 * element's attribute's layer at the point's physical position.
 *
 * The variable need not live on every layer; evaluating on an element of
 * a layer without it (a shell among them) throws, so restrict the
 * coefficient to the layers that hold it (markers, composite rheologies)
 * or ask HasLayer first. */
class ModelScalarCoefficient : public mfem::Coefficient {
 public:
  /** @throws planetmodel::Error when no layer holds the variable or it
   * is not a scalar. */
  ModelScalarCoefficient(const planetmodel::ModelFile& file,
                         const std::string& name);

  /** @brief Whether the layer of @p attribute holds the variable. */
  bool HasLayer(int attribute) const {
    return attribute >= 1 &&
           attribute <= static_cast<int>(fields_.size()) &&
           fields_[static_cast<std::size_t>(attribute - 1)].has_value();
  }

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  std::string name_;
  std::vector<std::optional<planetmodel::ScalarField>> fields_;  // [attr - 1]
  mfem::Vector x_;
};

}  // namespace AdGIA
