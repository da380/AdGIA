#!/bin/bash
#
# Build and install the MFEM that AdGIA is tested against.
#
#   install_mfem.sh serial   <prefix>
#   install_mfem.sh parallel <prefix>
#
# The parallel build compiles hypre and MFEM with one pair of MPI compiler
# wrappers (MPICC, MPICXX; default mpicc and mpicxx on PATH), so that every
# MPI-dependent library, AdGIA and the mpiexec that runs the tests come from
# the same MPI. Mixing MPIs (e.g. hypre from one, mpiexec from another) is
# the usual cause of link errors and of launch failures at run time. METIS
# is serial and is taken from METIS_DIR (default /usr, Ubuntu's
# libmetis-dev); PETSc is not needed.
#
# Environment (defaults in brackets):
#   MFEM_VERSION    MFEM release tag [v4.10]
#   HYPRE_VERSION   hypre release [3.1.0]
#   METIS_DIR       METIS 5 install prefix [/usr]
#   MPICC, MPICXX   MPI compiler wrappers [mpicc, mpicxx]
#   WORK_DIR        where sources are downloaded and built [<prefix>-work]
#   JOBS            parallel build jobs [nproc]
#
# Used by .github/workflows/ci.yml; it runs equally well on a workstation.

set -euo pipefail

MFEM_VERSION=${MFEM_VERSION:-v4.10}
HYPRE_VERSION=${HYPRE_VERSION:-3.1.0}
METIS_DIR=${METIS_DIR:-/usr}
MPICC=${MPICC:-mpicc}
MPICXX=${MPICXX:-mpicxx}
JOBS=${JOBS:-$(nproc)}

if [[ $# -ne 2 || ( $1 != serial && $1 != parallel ) ]]; then
    echo "usage: $(basename "$0") serial|parallel <prefix>" >&2
    exit 1
fi
MODE=$1
PREFIX=$(realpath -m "$2")
WORK_DIR=$(realpath -m "${WORK_DIR:-$PREFIX-work}")
mkdir -p "$WORK_DIR"

fetch() {  # fetch <url> <directory>: download a tarball and unpack it
    local url=$1 dir=$2
    if [[ ! -d $dir ]]; then
        mkdir -p "$dir"
        curl -fsSL "$url" | tar -xz -C "$dir" --strip-components=1
    fi
}

if [[ $MODE == parallel ]]; then
    echo "== hypre $HYPRE_VERSION with $(command -v "$MPICC")"
    fetch "https://github.com/hypre-space/hypre/archive/refs/tags/v$HYPRE_VERSION.tar.gz" \
          "$WORK_DIR/hypre"
    cmake -S "$WORK_DIR/hypre/src" -B "$WORK_DIR/hypre-build" \
          -DCMAKE_C_COMPILER="$MPICC" \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="$PREFIX" \
          -DCMAKE_INSTALL_LIBDIR=lib
    cmake --build "$WORK_DIR/hypre-build" -j "$JOBS"
    cmake --install "$WORK_DIR/hypre-build"

    MFEM_ARGS=(-DCMAKE_C_COMPILER="$MPICC"
               -DCMAKE_CXX_COMPILER="$MPICXX"
               -DMFEM_USE_MPI=ON
               -DMFEM_USE_METIS=ON
               -DHYPRE_DIR="$PREFIX"
               -DMETIS_DIR="$METIS_DIR")
else
    MFEM_ARGS=(-DMFEM_USE_MPI=OFF)
fi

echo "== MFEM $MFEM_VERSION ($MODE)"
fetch "https://github.com/mfem/mfem/archive/refs/tags/$MFEM_VERSION.tar.gz" "$WORK_DIR/mfem"
cmake -S "$WORK_DIR/mfem" -B "$WORK_DIR/mfem-build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      "${MFEM_ARGS[@]}"
cmake --build "$WORK_DIR/mfem-build" -j "$JOBS"
cmake --install "$WORK_DIR/mfem-build"

echo "== installed in $PREFIX"
