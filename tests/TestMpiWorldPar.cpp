/*
  Guard: the launcher really wires its processes into one MPI world. On a
  launcher/PMI mismatch (for instance Ubuntu 24.04's MPICH package,
  Launchpad #2072338) every launched process comes up as rank 0 of an
  MPI_COMM_WORLD of size 1, and every computation-only parallel test
  silently degenerates to independent serial runs that pass. This test
  takes the launched rank count as its argument and fails when the world
  it finds does not match.
*/
#include <mpi.h>

#include <cstdlib>
#include <iostream>

#include "mfem.hpp"

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);
  if (argc < 2) {
    std::cout << "usage: TestMpiWorldPar <ranks the launcher was given>\n";
    return 1;
  }
  const int expected = std::atoi(argv[1]);
  const int size = mfem::Mpi::WorldSize();
  if (size != expected) {
    std::cout << "FAIL: launched with " << expected
              << " ranks but MPI_COMM_WORLD has " << size
              << " (rank " << mfem::Mpi::WorldRank()
              << "): the launcher did not wire the processes into one "
                 "world.\n";
    return 1;
  }
  if (mfem::Mpi::Root()) {
    std::cout << "World of " << size << " ranks as launched.\n";
  }
  return 0;
}
