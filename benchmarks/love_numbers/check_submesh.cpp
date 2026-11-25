// ============================================================================
// check_submesh.cpp
//
// Are the entities that a ParSubMesh shares between ranks matched, on a
// given mesh and partition? mfem can pair the shared vertices and edges
// of a ParSubMesh wrongly on some partitions (SharedEntityMismatch() in
// AdGIA/submesh.hpp), and every space on the SubMesh is then wrong
// without an error. This program makes the ParSubMesh of sets of the
// mesh's domain attributes and checks each.
//
// The mesh is read from parts, <dir>/mesh.<rank> in mfem's parallel
// format (those of partition_case, or of mfem's mesh-explorer), on as
// many ranks as there are parts; or a serial mesh is read and cut for the
// ranks of the run.
//
// Options:
//   -p    directory of the parts.
//   -m    a serial mesh file, in place of -p.
//   -a    the domain attributes of one SubMesh, as "1 3". Without it,
//         every set of the mesh's attributes that leaves some out is
//         checked, on a mesh of at most eight attributes.
//
// It prints each set whose SubMesh is mismatched and returns 1 if there
// is one, 0 if every SubMesh checked is matched.
//
// Sample runs:
//    mpiexec -np 32 ./check_submesh -p cases/inner_core/parts_32
//    mpiexec -np 32 ./check_submesh -p cases/prem_6/parts_32 -a "1 3 4 5 6"
//    mpiexec -np 8 ./check_submesh -m mesh.msh
// ============================================================================

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const auto root = Mpi::Root();

  const char* parts = "";
  const char* mesh_file = "";
  auto chosen = Array<int>();
  OptionsParser args(argc, argv);
  args.AddOption(&parts, "-p", "--parts",
                 "Directory of the parts of the mesh, mesh.<rank>.");
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "A serial mesh, cut for the ranks of the run.");
  args.AddOption(&chosen, "-a", "--attributes",
                 "The domain attributes of one SubMesh; every set of them "
                 "if not given.");
  args.Parse();
  if (!args.Good() || (!*parts == !*mesh_file)) {
    if (root) {
      args.PrintUsage(std::cout);
      std::cout << "Give one of -p and -m.\n";
    }
    return 2;
  }

  auto mesh = std::unique_ptr<ParMesh>();
  if (*parts) {
    const auto name = [&](int rank) {
      return MakeParFilename(std::string(parts) + "/mesh.", rank);
    };
    auto missing = std::ifstream(name(Mpi::WorldRank())).good() ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &missing, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    const auto extra = std::ifstream(name(Mpi::WorldSize())).good();
    if (missing || extra) {
      if (root) {
        std::cout << parts << " does not hold the parts of "
                  << Mpi::WorldSize() << " ranks.\n";
      }
      return 2;
    }
    auto in = std::ifstream(name(Mpi::WorldRank()));
    mesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, in, false, 1, false);
  } else {
    auto serial = Mesh(mesh_file, 1, 1);
    mesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
  }

  const auto n_attributes = mesh->attributes.Size();
  auto sets = std::vector<Array<int>>();
  if (chosen.Size() > 0) {
    for (auto a : chosen) {
      if (mesh->attributes.Find(a) < 0) {
        if (root) {
          std::cout << "The mesh has no domain attribute " << a << ".\n";
        }
        return 2;
      }
    }
    sets.push_back(chosen);
  } else {
    if (n_attributes > 8) {
      if (root) {
        std::cout << "The mesh has " << n_attributes
                  << " domain attributes: name those of a SubMesh with -a.\n";
      }
      return 2;
    }
    // Every set but the empty one and the whole mesh, whose SubMesh
    // leaves nothing out.
    for (auto set = 1; set + 1 < (1 << n_attributes); set++) {
      auto attributes = Array<int>();
      for (auto i = 0; i < n_attributes; i++) {
        if (set & (1 << i)) {
          attributes.Append(mesh->attributes[i]);
        }
      }
      sets.push_back(attributes);
    }
  }

  auto mismatched = 0;
  for (auto& attributes : sets) {
    auto submesh = ParSubMesh(ParSubMesh::CreateFromDomain(*mesh, attributes));
    auto mismatch = real_t{0};
    if (!SharedEntitiesMatch(submesh, &mismatch)) {
      mismatched++;
      if (root) {
        std::cout << "MISMATCHED: the SubMesh of attributes";
        for (auto a : attributes) {
          std::cout << " " << a;
        }
        std::cout << " (a linear function changes by " << mismatch << ")\n";
      }
    }
  }
  const auto elements = mesh->GetGlobalNE();
  if (root) {
    std::cout << Mpi::WorldSize() << " ranks, " << elements
              << " elements, " << n_attributes << " domain attributes: "
              << sets.size() - mismatched << " of " << sets.size()
              << " SubMeshes matched.\n";
  }
  return mismatched ? 1 : 0;
}
