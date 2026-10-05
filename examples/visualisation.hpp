// ============================================================================
// visualisation.hpp
//
// The two outputs the examples share:
//
//   GLVisWindow  a field sent to a running GLVis server (start `glvis` in
//                a terminal first), once or repeatedly (an animation). In
//                parallel every rank sends its own piece, announced by the
//                "parallel <size> <rank>" line, and GLVis assembles them.
//   CsvTable     a self-describing table of numbers for the curves an
//                example produces (histories, Love numbers by degree,
//                convergence). Written by the root rank only, into the
//                working directory: run the examples from the build's
//                examples/ directory, and plot with
//                    python3 plot_csv.py <file>.csv [--show]
//                (plot_csv.py is copied there by the build). The "# key:
//                value" lines at the top tell plot_csv.py what to draw;
//                see that script for the keys.
// ============================================================================

#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mfem.hpp"

namespace examples {

inline bool IsRoot() {
#ifdef MFEM_USE_MPI
  return mfem::Mpi::Root();
#else
  return true;
#endif
}

// A GLVis window, opened on the first Send. The keys (GLVis keystrokes,
// e.g. "Rjl" to view a 2-D field from above with light off) and the
// window title are sent with the first frame only, so an animation keeps
// whatever view the user has set since.
class GLVisWindow {
 public:
  explicit GLVisWindow(std::string title, std::string keys = "")
      : title_(std::move(title)), keys_(std::move(keys)) {}

  // Fix the colour scale of an animation at [lo, hi] (sent with the first
  // frame; by default GLVis rescales every frame).
  void SetValueRange(double lo, double hi) {
    std::ostringstream os;
    os << "autoscale off\nvaluerange " << lo << " " << hi << "\n";
    commands_ = os.str();
  }

  // Send one frame: every rank calls this (collective in parallel).
  void Send(mfem::Mesh& mesh, const mfem::GridFunction& field) {
    if (failed_) {
      return;
    }
    if (!sock_.is_open()) {
      sock_.open("localhost", 19916);
      if (!sock_.is_open()) {
        // Every rank fails alike when no server is running.
        failed_ = true;
        if (IsRoot()) {
          std::cout << "(GLVis: no server on localhost:19916 for '" << title_
                    << "'; start glvis to see it)\n";
        }
        return;
      }
      sock_.precision(8);
    }
#ifdef MFEM_USE_MPI
    // Parallel: GLVis must be told how many pieces make up the frame.
    sock_ << "parallel " << mfem::Mpi::WorldSize() << " "
          << mfem::Mpi::WorldRank() << "\n";
#endif
    sock_ << "solution\n" << mesh << field;
    if (first_) {
      sock_ << "window_title '" << title_ << "'\n";
      if (!commands_.empty()) {
        sock_ << commands_;
      }
      if (!keys_.empty()) {
        sock_ << "keys " << keys_ << "\n";
      }
      first_ = false;
    }
    sock_ << std::flush;
  }

 private:
  std::string title_, keys_, commands_;
  mfem::socketstream sock_;
  bool first_ = true, failed_ = false;
};

// GLVis keys for a first view (those of elastogravity_layered): 2-D
// fields from above, no lighting, mesh boundary and colour bar; 3-D ones
// turned and cut open.
inline std::string DefaultKeys(int dim) {
  return dim == 2 ? "Rjlbc" : "RRRilc";
}

// A table of numbers written as CSV with "# key: value" header lines for
// plot_csv.py. Rows are buffered and written on Write (or on destruction);
// only the root rank writes. An empty file name disables the table.
class CsvTable {
 public:
  CsvTable(std::string file, std::vector<std::string> columns)
      : file_(std::move(file)), columns_(std::move(columns)) {}
  ~CsvTable() { Write(); }

  bool Enabled() const { return !file_.empty(); }

  // A plotting hint: title, note, x, y, group, xlabel, ylabel, logx, logy
  // (see plot_csv.py).
  CsvTable& Meta(const std::string& key, const std::string& value) {
    meta_.emplace_back(key, value);
    return *this;
  }

  void Row(const std::vector<double>& values) {
    MFEM_VERIFY(values.size() == columns_.size(), "CsvTable: row size");
    std::ostringstream os;
    os << std::setprecision(10);
    for (std::size_t i = 0; i < values.size(); i++) {
      os << (i ? "," : "") << values[i];
    }
    rows_.push_back(os.str());
  }

  // A row with a leading text field (the first column is a label, e.g. a
  // scheme name, used to group the rows into curves).
  void Row(const std::string& label, const std::vector<double>& values) {
    MFEM_VERIFY(values.size() + 1 == columns_.size(), "CsvTable: row size");
    std::ostringstream os;
    os << std::setprecision(10) << label;
    for (double v : values) {
      os << "," << v;
    }
    rows_.push_back(os.str());
  }

  void Write() {
    if (written_ || !Enabled() || !IsRoot()) {
      return;
    }
    written_ = true;
    std::ofstream os(file_);
    for (const auto& [k, v] : meta_) {
      os << "# " << k << ": " << v << "\n";
    }
    for (std::size_t i = 0; i < columns_.size(); i++) {
      os << (i ? "," : "") << columns_[i];
    }
    os << "\n";
    for (const auto& r : rows_) {
      os << r << "\n";
    }
    std::cout << "Wrote " << file_ << " (plot: python3 plot_csv.py " << file_
              << ")\n";
  }

 private:
  std::string file_;
  std::vector<std::string> columns_;
  std::vector<std::pair<std::string, std::string>> meta_;
  std::vector<std::string> rows_;
  bool written_ = false;
};

}  // namespace examples
