/*
  Parallel test for SharedEntityMismatch() and SharedEntitiesMatch()
  (submesh.hpp), on four ranks; a standalone MPI program that fails if
  any check does.

  A 2 x 2 x 2 mesh of unit hexahedra. Rank 2 holds every element but
  three of the upper layer, which ranks 1, 3 and 0 hold one each; the
  four elements with y < 1 are the SubMesh. Of the vertices that ranks 1
  and 2 share on the SubMesh, the parent has some in its group {1, 2},
  some in {0, 1, 2} and some in {0, 1, 2, 3}: the arrangement in which
  mfem can list them in a different order on the two ranks.

  The ParMesh is made twice: from the serial mesh and the partitioning,
  whose SubMesh must be matched; and from the parts that mfem's
  MeshPartitioner makes, whose SubMesh is matched or not as mfem has it.
  In both the answer of SharedEntitiesMatch() must be that of a direct
  comparison, between the ranks of every group, of the positions of the
  group's shared vertices in the order each rank holds them.
*/

#include <mpi.h>

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

int num_checks = 0;
int num_fails = 0;

void Check(bool ok, const std::string& what) {
  num_checks++;
  if (!ok) {
    num_fails++;
    if (Mpi::Root()) {
      std::cout << "FAIL: " << what << "\n";
    }
  }
}

// Whether every rank of every group holds the group's shared vertices in
// the same order: their positions are sent to the root and compared.
bool SharedVerticesAgree(ParMesh& mesh) {
  // For each group: the number of its ranks, the ranks in increasing
  // order, the number of its vertices, and their positions.
  auto mine = std::vector<double>();
  for (auto g = 1; g < mesh.GetNGroups(); g++) {
    auto ranks = Array<int>();
    const auto* members = mesh.gtopo.GetGroup(g);
    for (auto i = 0; i < mesh.gtopo.GetGroupSize(g); i++) {
      ranks.Append(mesh.gtopo.GetNeighborRank(members[i]));
    }
    ranks.Sort();
    mine.push_back(ranks.Size());
    for (auto r : ranks) {
      mine.push_back(r);
    }
    mine.push_back(mesh.GroupNVertices(g));
    for (auto v = 0; v < mesh.GroupNVertices(g); v++) {
      const auto* x = mesh.GetVertex(mesh.GroupVertex(g, v));
      mine.insert(mine.end(), x, x + mesh.SpaceDimension());
    }
  }

  const auto size = Mpi::WorldSize();
  auto count = static_cast<int>(mine.size());
  auto counts = std::vector<int>(size), offsets = std::vector<int>(size);
  MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  auto all = std::vector<double>();
  if (Mpi::Root()) {
    auto total = 0;
    for (auto r = 0; r < size; r++) {
      offsets[r] = total;
      total += counts[r];
    }
    all.resize(total);
  }
  MPI_Gatherv(mine.data(), count, MPI_DOUBLE, all.data(), counts.data(),
              offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  auto agree = 1;
  if (Mpi::Root()) {
    auto seen = std::map<std::vector<double>, std::vector<double>>();
    for (auto r = 0; r < size; r++) {
      auto at = all.begin() + offsets[r];
      const auto end = at + counts[r];
      while (at != end) {
        const auto n_ranks = static_cast<int>(*at);
        const auto group = std::vector<double>(at + 1, at + 1 + n_ranks);
        at += 1 + n_ranks;
        const auto n_values = static_cast<int>(*at) * mesh.SpaceDimension();
        const auto positions = std::vector<double>(at + 1, at + 1 + n_values);
        at += 1 + n_values;
        const auto [entry, first] = seen.emplace(group, positions);
        if (!first && entry->second != positions) {
          agree = 0;
        }
      }
    }
  }
  MPI_Bcast(&agree, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return agree == 1;
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  MFEM_VERIFY(Mpi::WorldSize() == 4, "This test runs on four ranks.");

  auto serial =
      Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 2.0, 2.0, 2.0);
  auto partitioning = Array<int>(serial.GetNE());
  for (auto e = 0; e < serial.GetNE(); e++) {
    auto c = Vector();
    serial.GetElementCenter(e, c);
    const auto left = c[0] < 1.0, front = c[1] < 1.0, lower = c[2] < 1.0;
    serial.SetAttribute(e, front ? 1 : 2);
    partitioning[e] = (lower || (left && front)) ? 2
                      : (front ? 1 : (left ? 3 : 0));
  }
  serial.SetAttributes();
  auto attributes = Array<int>(1);
  attributes[0] = 1;

  {
    auto mesh = ParMesh(MPI_COMM_WORLD, serial, partitioning.GetData());
    auto submesh = ParSubMesh(ParSubMesh::CreateFromDomain(mesh, attributes));
    const auto direct = SharedVerticesAgree(submesh);
    Check(direct, "from the serial mesh: shared vertices agree");
    Check(SharedEntitiesMatch(submesh) == direct,
          "from the serial mesh: SharedEntitiesMatch is the comparison");
  }
  {
    auto partitioner = MeshPartitioner(serial, 4, partitioning.GetData());
    auto part = MeshPart();
    partitioner.ExtractPart(Mpi::WorldRank(), part);
    auto stream = std::stringstream();
    stream.precision(16);
    part.Print(stream);
    auto mesh = ParMesh(MPI_COMM_WORLD, stream);
    Check(SharedVerticesAgree(mesh), "from parts: the parent's vertices agree");
    auto submesh = ParSubMesh(ParSubMesh::CreateFromDomain(mesh, attributes));
    const auto direct = SharedVerticesAgree(submesh);
    auto mismatch = real_t{0};
    const auto matched = SharedEntitiesMatch(submesh, &mismatch);
    Check(matched == direct,
          "from parts: SharedEntitiesMatch is the comparison");
    Check(matched == (mismatch < 1e-10), "from parts: the mismatch returned");
    if (Mpi::Root()) {
      std::cout << "From parts, this mfem's SubMesh is "
                << (direct ? "matched" : "mismatched") << ".\n";
    }
  }

  if (Mpi::Root()) {
    std::cout << num_checks - num_fails << " of " << num_checks
              << " checks passed.\n";
  }
  return num_fails == 0 ? 0 : 1;
}
