// ============================================================================
// partition_case.cpp
//
// Partition a case of the benchmark for a given number of ranks, so that a
// parallel run reads on each rank its own part of the mesh and of the fields
// rather than the whole of them. A serial program, run once per case and
// number of ranks; it writes, beside the manifest,
//
//   parts_<N>/mesh.<rank>      the parts of the mesh, in MFEM's parallel format
//   parts_<N>/<field>.<rank>   the fields on them
//
// The drivers look for parts_<N> when started on N ranks, and read the mesh
// and the fields in serial on every rank when there is none, which is
// simpler and costs each rank the memory of the whole mesh.
//
// Sample run:
//    ./partition_case -c cases/earth_like/case.json -np 128
// ============================================================================

#include <filesystem>
#include <fstream>
#include <iostream>

#include "benchmark_case.hpp"

using namespace mfem;
using namespace AdGIA;

int main(int argc, char* argv[]) {
  const char* manifest_file = "case.json";
  int parts = 8;
  OptionsParser args(argc, argv);
  args.AddOption(&manifest_file, "-c", "--case", "Manifest of the case.");
  args.AddOption(&parts, "-np", "--parts", "Number of ranks.");
  args.Parse();
  if (!args.Good() || parts < 1) {
    args.PrintUsage(std::cout);
    return 1;
  }

  const MeshManifest manifest(manifest_file);
  Mesh mesh = manifest.LoadMesh();
  std::vector<std::string> names(std::begin(benchmark::kFields),
                                 std::end(benchmark::kFields));
  if (manifest.HasField("p0")) {
    names.push_back("p0");
  }
  std::vector<std::unique_ptr<GridFunction>> fields;
  for (const std::string& name : names) {
    fields.push_back(manifest.LoadField(mesh, name));
  }

  const std::string directory = benchmark::PartsDirectory(manifest, parts);
  std::filesystem::create_directories(directory);
  MeshPartitioner partitioner(mesh, parts);
  MeshPart part;
  for (int rank = 0; rank < parts; rank++) {
    partitioner.ExtractPart(rank, part);
    {
      std::ofstream os(MakeParFilename(directory + "/mesh.", rank));
      os.precision(16);
      part.Print(os);
    }
    std::size_t k = 0;
    for (const std::string& name : names) {
      auto fes = partitioner.ExtractFESpace(part, *fields[k]->FESpace());
      auto local = partitioner.ExtractGridFunction(part, *fields[k], *fes);
      std::ofstream os(
          MakeParFilename(directory + "/" + name + ".", rank));
      os.precision(16);
      local->Save(os);
      k++;
    }
  }
  std::cout << "Wrote " << parts << " parts of " << mesh.GetNE()
            << " elements to " << directory << "\n";
  return 0;
}
