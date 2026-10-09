#ifndef ADGIA_RIESZ_HPP
#define ADGIA_RIESZ_HPP

#include <memory>

#include "mfem.hpp"

namespace AdGIA {

/**
 * @brief Riesz maps: a derivative (a dual vector, assembled against a
 * space's basis) into a gradient in a chosen inner product. The choice
 * of metric is the implicit preconditioner of a descent loop
 * (doc/equilibrium_figures.tex §6): the raw derivative is a functional,
 * not a direction, and identifying the two smuggles in the L2 metric.
 *
 * The maps act on true-dof vectors of one finite-element space; they
 * are mfem Operators, so loops can hold any of them behind one pointer
 * and the metric becomes a runtime switch. Vector spaces (vdim > 1,
 * e.g. a mapping control field) get the component-wise metric through
 * the vector mass and diffusion integrators. Serial and parallel.
 */
class RieszMap : public mfem::Operator {
 public:
  explicit RieszMap(mfem::FiniteElementSpace& fes)
      : mfem::Operator(fes.GetTrueVSize()), fes_(&fes) {}

  /** @brief The pairing of a dual with a field, @f$j\cdot x@f$ (global
   * in parallel). Every metric quantity the descent loops need is such
   * a pairing — @f$\langle g, g\rangle = j\cdot g@f$ for the gradient
   * g = Mult(j), directional slopes, the mass-projection scalar — so
   * no metric inner product is assembled. */
  mfem::real_t Pair(const mfem::Vector& dual, const mfem::Vector& x) const;

 protected:
  mfem::FiniteElementSpace* fes_;
};

/**
 * @brief The L2 identification: g solves M g = j. The baseline of the
 * equilibrium-figures programme — acceptable when the derivative is
 * itself smooth, with the caveats documented in the reference (mesh-
 * scale assembly noise passes straight through, cumulatively).
 */
class L2RieszMap : public RieszMap {
 public:
  explicit L2RieszMap(mfem::FiniteElementSpace& fes);
  ~L2RieszMap();

  void Mult(const mfem::Vector& dual, mfem::Vector& g) const override;

 private:
  std::unique_ptr<mfem::BilinearForm> m_;
  mfem::OperatorHandle M_;
  std::unique_ptr<mfem::Solver> prec_;
  std::unique_ptr<mfem::CGSolver> cg_;
};

/**
 * @brief Sobolev metrics of integer order on an H1 space: the Gram
 * operator of
 * @f$\langle u, v\rangle = \int (\alpha\,u v + \beta\,\nabla u\cdot
 * \nabla v)@f$, applied @p order times with mass multiplies between —
 * discretely @f$g = (\tilde A^{-1} M)^{s-1} \tilde A^{-1} j@f$ with
 * @f$\tilde A = \alpha M + \beta K@f$ — so each order costs one more
 * solve with the one assembled operator. The smoothing length is
 * @f$\sqrt{\beta/\alpha}@f$; order 2 is the working default of the
 * equilibrium-figures programme (H1 does not embed in C0 in 3-D;
 * H2 does, with margin).
 *
 * Posed on an extension domain (the whole mesh of @p fes, typically
 * body and buffer) with homogeneous Dirichlet conditions on
 * @p dirichlet_bdr (the outer boundary) and natural conditions
 * elsewhere; the composition is spectrally @f$(\alpha-\beta
 * \Delta_D)^s@f$, whose difference from the boundary-condition-free
 * H^s norm decays away from the Dirichlet boundary — the point of the
 * extension domain. A dual with content on the Dirichlet dofs has it
 * annihilated (those dofs are not part of the metric space).
 */
class SobolevRieszMap : public RieszMap {
 public:
  /**
   * @param fes H1 space on the extension domain, scalar or vector
   * (component-wise metric); not owned.
   * @param alpha Mass weight.
   * @param beta Gradient weight; @f$\sqrt{\beta/\alpha}@f$ is the
   * smoothing length.
   * @param order The Sobolev order s (1 or more); s solves per Mult.
   * @param dirichlet_bdr Boundary-attribute marker of the homogeneous
   * Dirichlet (outer) boundary; natural conditions when null.
   */
  SobolevRieszMap(mfem::FiniteElementSpace& fes, mfem::real_t alpha,
                  mfem::real_t beta, int order = 2,
                  const mfem::Array<int>* dirichlet_bdr = nullptr);
  ~SobolevRieszMap();

  void Mult(const mfem::Vector& dual, mfem::Vector& g) const override;

 private:
  void EliminateEssential(mfem::Vector& v) const;

  int order_;
  mfem::Array<int> ess_tdofs_;
  std::unique_ptr<mfem::BilinearForm> a_, m_;
  mfem::OperatorHandle A_, M_;
  std::unique_ptr<mfem::Solver> prec_;
  std::unique_ptr<mfem::CGSolver> cg_;
};

}  // namespace AdGIA

#endif  // ADGIA_RIESZ_HPP
