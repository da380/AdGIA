// ============================================================================
// relabelled_identity.cpp
//
// The discrete change-of-variables identity at the level of the FULL
// self-gravitating SOLVE (leg B of the relabelling family,
// doc/benchmarks.tex; doc/mappings.md, "The discrete change-of-variables
// identity"): the same problem class solves
//
//   side A  the REFERENCE mesh, phi_e = the nodal interpolant of the
//           interior relabelling (relabelling.hpp), the constitutive
//           coefficients as the case's L2 fields with the pulled-back
//           chain (C~, S~ = -p0 I pulled back, rho J);
//   side B  the MAPPED mesh (the element-by-element nodal image), the
//           identity mapping, the SAME L2 dof data transplanted.
//
// FE interpolation commutes with the nodal transplant, and the mapped
// integrator overloads share their standard twins' quadrature rules, so
// the two assemblies are the same linear system and the true-dof
// solutions must agree to solver tolerance — a certification that the
// mapped assembly IS the standard assembly under a change of variables,
// for every term of the coupled problem at once (stiffness, equilibrium
// stress, gravity couplings, Poisson block, DtN, vacuum extension,
// loads and projectors). With -method slip_broken, the slipping
// four-block machinery (B_Sigma, G_Sigma, both jump constraints) joins
// the certification.
//
// The physical model the two sides share is the transplanted one — a
// slightly-off-spherical body — which is irrelevant here: this is a
// change-of-variables identity, not a physics comparison (that is
// love_benchmark -map, against pyslfp).
//
// Two pieces bound the achievable agreement, and the driver treats
// them differently:
//
//  - The DtN centres itself on the MESH centroid, which the mapped
//    interior shifts by O(A): the truncated operators differ slightly
//    (7e-6 in action at A = 0.02 on the coarse case, ~1e-7 in the
//    solutions, superlinear in A — an UNMAPPED assembly term would
//    show up linearly, which is what this driver detects). The vacuum
//    extension, which is gauge DATA, is shared between the sides.
//
//  - The fluid GAUGE PENALTY is assembled covariantly — the
//    pulled-back deviatoric tensor through ElasticTensorIntegrator's
//    mapped form; it must be that integrator, since the
//    material-stiffness form expects a RELABELLED tensor, not the
//    plain pull-back — so the two sides gauge-fix identically and
//    every WELDED case is STRICT, the gauged fluid included
//    (u 1.6e-7 at A = 0.02, h = 0.3, the DtN-centring floor).
//    Through the SLIPPING interface the run is informational (u 1.2e-2,
//    zeta 8e-3 at A = 0.02, h = 0.3 on fluid_core): the identity does
//    not hold there, and the slipping-interface forms are not certified
//    covariant. For the welded organisation the driver prints the
//    operator-action probes and the per-attribute breakdown; for
//    slip_broken it prints, after the solves, the sixteen broken blocks
//    and the stored constraint kernels of both sides, which locate the
//    differing terms (doc/benchmarks.tex, leg B, gives the measured
//    pattern). (The single-valued slip organisation refuses non-identity
//    maps outright, so it cannot be run here.)
//
// Both parallel meshes are built from the same serial partition, so the
// true-dof numbering coincides and the solutions compare entrywise.
//
// Sample run:
//    mpiexec -np 8 ./relabelled_identity -c case.json -o 2 -map 0.02
//    mpiexec -np 8 ./relabelled_identity -c case.json -method slip_broken
// ============================================================================

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "benchmark_case.hpp"
#include "relabelling.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace benchmark;

namespace {

constexpr double kEps = 1.0e-2;
constexpr int kRefinements = 3;
constexpr double kTheta = 1.0e2;
constexpr int kALIterations = 6;

// Degree-2-flavoured surface load (any smooth load will do: the
// identity is data-independent).
double SurfaceLoad(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[x.Size() - 1] / r;
  return 0.02 * (2.0 * c * c - 1.0) + 0.01 * x[0] * x[1] / (r * r);
}

// One side of the identity: everything a solve needs, built on one
// ParMesh with one mapping.
struct Side {
  std::unique_ptr<ParMesh> parent;
  std::unique_ptr<ParSubMesh> solid, fluid, buffer, outer;
  std::unique_ptr<ParFiniteElementSpace> fes_u, fes_f, fes_buffer,
      fes_zeta, fes_zo;
  std::unique_ptr<ParGridFunction> rho, kappa, mu, p0;
  std::vector<std::unique_ptr<ParFiniteElementSpace>> spaces;
  std::vector<std::unique_ptr<ParGridFunction>> fields;
  std::unique_ptr<GridFunctionCoefficient> rho_c, kappa_c, mu_c, p0_c;
  std::unique_ptr<TwoRegionCoefficient> rho2, kappa2, mu2, p02;
  std::unique_ptr<GridFunctionCoefficient> kappa_gauge;
  std::unique_ptr<MultiMeshDiffeomorphism> xi;
  std::unique_ptr<IdentityDiffeomorphism> id;
  std::unique_ptr<JacobianCoefficient> jac;
  std::unique_ptr<ProductCoefficient> rho_tilde;
  std::optional<IsotropicElasticTensorCoefficient> C_eff;
  std::unique_ptr<BareElasticTensorCoefficient> C_bare;
  ConstantCoefficient minus_one{-1.0};
  std::unique_ptr<ProductCoefficient> neg_p0;
  std::unique_ptr<IdentityMatrixCoefficient> id_mat;
  std::unique_ptr<ScalarMatrixProductCoefficient> S_e;
  std::unique_ptr<RelabelledElasticTensorCoefficient> C_rel;
  std::unique_ptr<PullbackStressCoefficient> S_rel;
  std::unique_ptr<ReferentialElasticRheology> rheology;
  std::unique_ptr<PWConstCoefficient> pi_c;
  std::unique_ptr<HypreParMatrix> Evac;
  std::unique_ptr<LinearQuasiStaticReferentialSelfGravitatingProblem> problem;

  // On a SubMesh, the transplant of a parent field.
  ParGridFunction* OnMesh(const ParGridFunction& f, ParSubMesh& sub) {
    spaces.push_back(std::make_unique<ParFiniteElementSpace>(
        &sub, f.ParFESpace()->FEColl()));
    fields.push_back(
        std::make_unique<ParGridFunction>(spaces.back().get()));
    ParSubMesh::Transfer(f, *fields.back());
    return fields.back().get();
  }
};

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const bool root = Mpi::Root();

  const char* manifest_file = "case.json";
  const char* method = "referential";
  int order = 2;
  double amplitude = 0.02;
  double rel_tol = 1e-11;
  int dtn_degree = 8;

  OptionsParser args(argc, argv);
  args.AddOption(&manifest_file, "-c", "--case", "Manifest of the case.");
  args.AddOption(&method, "-method", "--method",
                 "referential (welded) or slip_broken.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&amplitude, "-map", "--map-amplitude",
                 "Amplitude of the interior relabelling.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Solver tolerance.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree", "DtN degree.");
  args.Parse();
  if (!args.Good()) {
    if (root) args.PrintUsage(std::cout);
    return 1;
  }
  const std::string m(method);
  MFEM_VERIFY(m == "referential" || m == "slip_broken",
              "-method must be referential or slip_broken.");
  const bool slip = m == "slip_broken";

  const MeshManifest manifest(manifest_file);
  if (root) manifest.Print(std::cout);
  const real_t G = manifest.G();
  const real_t radius = manifest.SurfaceRadius();

  // The serial reference mesh, its nodal-image mesh, one partition.
  Mesh serial = manifest.LoadMesh();
  const int dim = serial.Dimension();
  std::vector<real_t> boundaries;
  for (const auto& layer : manifest.Layers()) {
    if (boundaries.empty()) {
      boundaries.push_back(layer.r_inner);
    }
    boundaries.push_back(layer.r_outer);
  }
  auto xi_analytic = InteriorRelabelling(dim, boundaries, amplitude);
  Mesh mapped_serial = MappedMesh(serial, xi_analytic);
  std::unique_ptr<int[]> partitioning(
      serial.GeneratePartitioning(Mpi::WorldSize()));

  auto rho_s = manifest.LoadField(serial, "rho");
  auto kappa_s = manifest.LoadField(serial, "kappa");
  auto mu_s = manifest.LoadField(serial, "mu");
  MFEM_VERIFY(manifest.HasField("p0"),
              "The identity driver needs the p0 field: re-make the case.");
  auto p0_s = manifest.LoadField(serial, "p0");

  const Array<int> solid_attributes = manifest.SolidAttributes();
  const Array<int> fluid_attributes = manifest.FluidAttributes();
  Array<int> buffer_attributes, outer_attributes(solid_attributes);
  for (int a = 1; a <= serial.attributes.Max(); a++) {
    if (solid_attributes.Find(a) < 0 && fluid_attributes.Find(a) < 0) {
      buffer_attributes.Append(a);
    }
  }
  outer_attributes.Append(buffer_attributes);
  outer_attributes.Sort();
  Array<int> u_attributes(solid_attributes);
  if (!slip) {
    u_attributes.Append(fluid_attributes);
    u_attributes.Sort();
  }
  if (slip) {
    MFEM_VERIFY(fluid_attributes.Size() == 1,
                "slip_broken: one fluid core.");
  }

  H1_FECollection fec(order, dim);
  FunctionCoefficient sigma(SurfaceLoad);
  ConstantCoefficient mu_gauge(1.0);

  // The vacuum extension is GAUGE: any admissible E serves. The identity
  // needs the SAME gauge data on both sides — the builder samples the
  // body side of the surface, whose interior geometry differs at O(A)
  // between the meshes, so per-side extensions differ (by ~20 percent at
  // A = 0.02) and would break the identity through the buffer folds.
  const HypreParMatrix* shared_E = nullptr;
  auto build = [&](Mesh& smesh, bool mapped_side) {
    auto side = std::make_unique<Side>();
    side->parent = std::make_unique<ParMesh>(MPI_COMM_WORLD, smesh,
                                             partitioning.get());
    auto distribute = [&](const GridFunction& f) {
      return std::make_unique<ParGridFunction>(side->parent.get(), &f,
                                               partitioning.get());
    };
    side->rho = distribute(*rho_s);
    side->kappa = distribute(*kappa_s);
    side->mu = distribute(*mu_s);
    side->p0 = distribute(*p0_s);
    side->solid = std::make_unique<ParSubMesh>(
        ParSubMesh::CreateFromDomain(*side->parent, u_attributes));
    side->buffer = std::make_unique<ParSubMesh>(
        ParSubMesh::CreateFromDomain(*side->parent, buffer_attributes));
    side->fes_u = std::make_unique<ParFiniteElementSpace>(
        side->solid.get(), &fec, dim);
    side->fes_buffer = std::make_unique<ParFiniteElementSpace>(
        side->buffer.get(), &fec, dim);
    side->fes_zeta =
        std::make_unique<ParFiniteElementSpace>(side->parent.get(), &fec);

    // The coefficients, from the SAME dof data on either mesh; on the
    // displacement SubMesh (and, slipping, the fluid one).
    auto* rho_u = side->OnMesh(*side->rho, *side->solid);
    auto* kappa_u = side->OnMesh(*side->kappa, *side->solid);
    auto* mu_u = side->OnMesh(*side->mu, *side->solid);
    auto* p0_u = side->OnMesh(*side->p0, *side->solid);
    side->rho_c = std::make_unique<GridFunctionCoefficient>(rho_u);
    side->kappa_c = std::make_unique<GridFunctionCoefficient>(kappa_u);
    side->mu_c = std::make_unique<GridFunctionCoefficient>(mu_u);
    side->p0_c = std::make_unique<GridFunctionCoefficient>(p0_u);
    Coefficient *rho_scalar = side->rho_c.get(),
                *kappa_scalar = side->kappa_c.get(),
                *mu_scalar = side->mu_c.get(),
                *p0_scalar = side->p0_c.get();
    if (slip) {
      // The slipping class evaluates the constitutive coefficients on
      // the fluid SubMesh as well: two-region dispatch on the same dof
      // data, as in the benchmark case.
      side->fluid = std::make_unique<ParSubMesh>(
          ParSubMesh::CreateFromDomain(*side->parent, fluid_attributes));
      auto* rho_f = side->OnMesh(*side->rho, *side->fluid);
      auto* kappa_f = side->OnMesh(*side->kappa, *side->fluid);
      auto* mu_f = side->OnMesh(*side->mu, *side->fluid);
      auto* p0_f = side->OnMesh(*side->p0, *side->fluid);
      side->kappa_gauge = std::make_unique<GridFunctionCoefficient>(kappa_f);
      side->rho2 = std::make_unique<TwoRegionCoefficient>(*rho_u, *rho_f);
      side->kappa2 =
          std::make_unique<TwoRegionCoefficient>(*kappa_u, *kappa_f);
      side->mu2 = std::make_unique<TwoRegionCoefficient>(*mu_u, *mu_f);
      side->p02 = std::make_unique<TwoRegionCoefficient>(*p0_u, *p0_f);
      rho_scalar = side->rho2.get();
      kappa_scalar = side->kappa2.get();
      mu_scalar = side->mu2.get();
      p0_scalar = side->p02.get();
    }

    side->C_eff.emplace(IsotropicElasticTensorCoefficient::FromBulkModulus(
        dim, *kappa_scalar, *mu_scalar));
    side->C_bare = std::make_unique<BareElasticTensorCoefficient>(
        dim, *side->C_eff, *p0_scalar);
    side->neg_p0 = std::make_unique<ProductCoefficient>(side->minus_one,
                                                        *p0_scalar);
    side->id_mat = std::make_unique<IdentityMatrixCoefficient>(dim);
    side->S_e = std::make_unique<ScalarMatrixProductCoefficient>(
        *side->neg_p0, *side->id_mat);

    MatrixCoefficient* C_use = side->C_bare.get();
    MatrixCoefficient* S_use = side->S_e.get();
    Coefficient* rho_use = rho_scalar;
    Diffeomorphism* phi_e = nullptr;
    if (mapped_side) {
      side->xi = std::make_unique<MultiMeshDiffeomorphism>(xi_analytic,
                                                           *side->parent);
      side->xi->AddMesh(*side->solid);
      side->xi->AddMesh(*side->buffer);
      side->jac = std::make_unique<JacobianCoefficient>(*side->xi);
      side->rho_tilde = std::make_unique<ProductCoefficient>(
          *side->jac, *rho_scalar);
      side->C_rel = std::make_unique<RelabelledElasticTensorCoefficient>(
          dim, *side->C_bare, *side->xi);
      side->S_rel = std::make_unique<PullbackStressCoefficient>(
          dim, *side->S_e, *side->xi);
      C_use = side->C_rel.get();
      S_use = side->S_rel.get();
      rho_use = side->rho_tilde.get();
      phi_e = side->xi.get();
    } else {
      side->id = std::make_unique<IdentityDiffeomorphism>(dim);
      phi_e = side->id.get();
    }
    side->rheology = std::make_unique<ReferentialElasticRheology>(
        dim, *C_use, *S_use, *phi_e);

    const int n_bdr = side->solid->bdr_attributes.Max();
    Array<int> surface_marker(n_bdr);
    surface_marker = 0;
    surface_marker[manifest.SurfaceAttribute() - 1] = 1;

    if (!slip) {
      side->problem =
          std::make_unique<LinearQuasiStaticReferentialSelfGravitatingProblem>(
              side->fes_u.get(), side->fes_zeta.get(), *side->rheology,
              *rho_use, G, dtn_degree);
      if (fluid_attributes.Size() > 0) {
        Array<int> gauge_marker(side->solid->attributes.Max());
        gauge_marker = 0;
        for (const int a : fluid_attributes) {
          gauge_marker[a - 1] = 1;
        }
        side->problem->SetGaugedFluid(gauge_marker, *side->kappa_c, kEps,
                                      kRefinements);
      }
    } else {
      side->fes_f = std::make_unique<ParFiniteElementSpace>(
          side->fluid.get(), &fec, dim);
      if (mapped_side) {
        side->xi->AddMesh(*side->fluid);
      }
      side->outer = std::make_unique<ParSubMesh>(
          ParSubMesh::CreateFromDomain(*side->parent, outer_attributes));
      side->fes_zo = SubMeshDofInjection::MakeShadowSpace(*side->fes_zeta,
                                                          *side->outer);
      if (mapped_side) {
        side->xi->AddMesh(*side->outer);
      }
      const int fluid_attr = fluid_attributes[0];
      const int b_itf = manifest.FluidSolidInterfaces(fluid_attr)[0];
      const int first = manifest.Interfaces().front().attribute;
      Vector pi_values(n_bdr);
      pi_values = 0.0;
      pi_values[b_itf - 1] =
          manifest.Interfaces()[b_itf - first].ValueBeside("p0",
                                                           fluid_attr);
      side->pi_c = std::make_unique<PWConstCoefficient>(pi_values);
      Array<int> interface_marker =
          MeshManifest::Marker(Array<int>({b_itf}), n_bdr);
      auto slip_problem =
          std::make_unique<LinearQuasiStaticReferentialSelfGravitatingSlipProblem>(
              side->fes_u.get(), side->fes_f.get(), side->fes_zeta.get(),
              *side->rheology, *rho_use, *side->pi_c, interface_marker, G,
              dtn_degree);
      slip_problem->SetFluidGauge(*side->kappa_gauge, kEps);
      slip_problem->SetConstraint(kTheta, kALIterations);
      slip_problem->EnableBrokenZeta(side->fes_zo.get(), kTheta);
      side->problem = std::move(slip_problem);
    }
    if (shared_E == nullptr) {
      Vector bb_min, bb_max;
      side->parent->GetBoundingBox(bb_min, bb_max);
      side->Evac = NewRadialVacuumExtension(
          *side->fes_u, *side->fes_buffer, radius, bb_max.Normlinf());
      shared_E = side->Evac.get();
    }
    side->problem->SetPrescribedVacuumExtension(*side->fes_buffer,
                                                *shared_E);
    side->problem->SetSurfaceLoad(sigma, surface_marker);
    side->problem->SetRelTol(rel_tol);
    side->problem->AssembleForce(0.0);
    return side;
  };

  auto A = build(serial, true);
  auto B = build(mapped_serial, false);

  // Operator-action probes: one smooth field, restricted per attribute,
  // projected ONCE (the sides share a dof layout, so the same true-dof
  // vector is fed to both) and pushed through both assembled block
  // operators — a non-pulled-back block shows up in the probe whose
  // support its columns touch.
  if (!slip) {
    struct Probe : public VectorCoefficient {
      int attr;  // 0: everywhere
      Probe(int d, int attribute) : VectorCoefficient(d), attr(attribute) {}
      void Eval(Vector& V, ElementTransformation& T,
                const IntegrationPoint& ip) override {
        V.SetSize(vdim);
        if (attr != 0 && T.Attribute != attr) {
          V = 0.0;
          return;
        }
        Vector x(3);
        x = 0.0;
        T.Transform(ip, x);
        for (int i = 0; i < vdim; i++) {
          V[i] = std::sin(3.0 * x[0] + i) * std::cos(2.0 * x[1]) +
                 0.3 * x[2] + 0.1 * (i + 1);
        }
      }
    };
    const int d = A->fes_u->GetVDim();
    Vector zu(A->fes_zeta->GetTrueVSize());
    zu = 0.0;
    Vector zero_u(A->fes_u->GetTrueVSize());
    zero_u = 0.0;
    FunctionCoefficient zc([](const Vector& x) {
      return std::sin(2.0 * x[0]) * std::cos(x[1]) + 0.3 * x[2];
    });
    auto probe = [&](const char* name, const Vector& u, const Vector& z) {
      Vector ruA, rzA, ruB, rzB;
      A->problem->ApplyBlockOperator(u, z, ruA, rzA);
      B->problem->ApplyBlockOperator(u, z, ruB, rzB);
      real_t s[4] = {0.0, 0.0, 0.0, 0.0};
      for (int i = 0; i < ruA.Size(); i++) {
        s[0] += (ruA[i] - ruB[i]) * (ruA[i] - ruB[i]);
        s[1] += ruB[i] * ruB[i];
      }
      for (int i = 0; i < rzA.Size(); i++) {
        s[2] += (rzA[i] - rzB[i]) * (rzA[i] - rzB[i]);
        s[3] += rzB[i] * rzB[i];
      }
      MPI_Allreduce(MPI_IN_PLACE, s, 4, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
      if (Mpi::Root()) {
        std::cout << "  probe " << name << ": |dr_u|/|r_u| "
                  << std::sqrt(s[0] / std::max(s[1], real_t{1e-300}))
                  << ", |dr_z|/|r_z| "
                  << std::sqrt(s[2] / std::max(s[3], real_t{1e-300}))
                  << "\n";
      }
    };
    if (Mpi::Root()) {
      std::cout << "\noperator-action probes (A vs B):\n";
    }
    ParGridFunction gu(A->fes_u.get());
    Vector tu;
    for (int a = 0; a <= A->solid->attributes.Max(); a++) {
      Probe pc(d, a);
      gu.ProjectCoefficient(pc);
      gu.GetTrueDofs(tu);
      std::string name = a == 0 ? "u everywhere"
                                : "u attribute " + std::to_string(a);
      probe(name.c_str(), tu, zu);
      Vector qA, qB;
      A->problem->ApplyGaugePenalty(tu, qA);
      B->problem->ApplyGaugePenalty(tu, qB);
      real_t q[2] = {0.0, 0.0};
      for (int i = 0; i < qA.Size(); i++) {
        q[0] += (qA[i] - qB[i]) * (qA[i] - qB[i]);
        q[1] += qB[i] * qB[i];
      }
      MPI_Allreduce(MPI_IN_PLACE, q, 2, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
      if (Mpi::Root()) {
        std::cout << "    penalty alone: |dQu|/|Qu| "
                  << std::sqrt(q[0] / std::max(q[1], real_t{1e-300}))
                  << "\n";
      }
    }
    ParGridFunction gz(A->fes_zeta.get());
    Vector tz;
    gz.ProjectCoefficient(zc);
    gz.GetTrueDofs(tz);
    probe("zeta", zero_u, tz);
  }

  const bool okA = A->problem->Solve();
  const bool okB = B->problem->Solve();

  // Per-block probes of the broken-zeta organisation (after the solves:
  // the sixteen solver blocks are assembled by SetupSolverBroken): one
  // smooth field per slot (u_s, u_f, zeta_o, zeta_f), projected once on
  // side A's spaces — the sides share the dof layout — and pushed
  // through both sides' blocks. A non-covariant assembly term shows up
  // in its block's column; the (2,2) entry excludes the DtN fold, which
  // differs by the mesh-centroid centring alone.
  if (slip) {
    using SlipProblem = LinearQuasiStaticReferentialSelfGravitatingSlipProblem;
    auto* SA = dynamic_cast<SlipProblem*>(A->problem.get());
    auto* SB = dynamic_cast<SlipProblem*>(B->problem.get());
    MFEM_VERIFY(SA && SB, "broken-block probes: unexpected problem type.");
    Vector x[4];
    for (int j = 0; j < 4; j++) {
      auto* pfes =
          dynamic_cast<ParFiniteElementSpace*>(&SA->BrokenSpace(j));
      MFEM_VERIFY(pfes, "broken-block probes: parallel spaces expected.");
      ParGridFunction g(pfes);
      if (pfes->GetVDim() > 1) {
        VectorFunctionCoefficient c(
            pfes->GetVDim(), [](const Vector& p, Vector& v) {
              for (int k = 0; k < v.Size(); k++) {
                v[k] = std::sin(3.0 * p[0] + k) * std::cos(2.0 * p[1]) +
                       0.3 * p[2] + 0.1 * (k + 1);
              }
            });
        g.ProjectCoefficient(c);
      } else {
        FunctionCoefficient c([](const Vector& p) {
          return std::sin(2.0 * p[0]) * std::cos(p[1]) + 0.3 * p[2];
        });
        g.ProjectCoefficient(c);
      }
      g.GetTrueDofs(x[j]);
    }
    const char* slots[4] = {"u_s", "u_f", "z_o", "z_f"};
    if (Mpi::Root()) {
      std::cout << "\n  broken-block probes, |(S_A - S_B) x| / |S_B x| "
                   "(columns: x in u_s, u_f, z_o, z_f):\n";
    }
    for (int i = 0; i < 4; i++) {
      std::ostringstream line;
      for (int j = 0; j < 4; j++) {
        Vector yA, yB;
        SA->ApplyBrokenBlock(i, j, x[j], yA);
        SB->ApplyBrokenBlock(i, j, x[j], yB);
        real_t s[2] = {0.0, 0.0};
        for (int k = 0; k < yA.Size(); k++) {
          const real_t d = yA[k] - yB[k];
          s[0] += d * d;
          s[1] += yB[k] * yB[k];
        }
        MPI_Allreduce(MPI_IN_PLACE, s, 2, MPITypeMap<real_t>::mpi_type,
                      MPI_SUM, MPI_COMM_WORLD);
        line << "  " << std::setw(9) << std::setprecision(2)
             << std::scientific
             << std::sqrt(s[0] / std::max(s[1], real_t{1e-300}))
             << std::defaultfloat;
      }
      if (Mpi::Root()) {
        std::cout << "    row " << slots[i] << line.str() << "\n";
      }
    }
    // Second level: the stored constraint kernels, separated by their
    // mapping ingredient — Bn carries the Nanson nu alone (invariant
    // under shears of F that fix the face, by nu = adj(F)^T n), Pb and
    // Kvz carry the mapped b = F^{-T} grad zeta0 (its normal component
    // is NOT invariant under those shears), Mz carries no mapping.
    if (Mpi::Root()) {
      std::cout << "  kernel probes, |(K_A - K_B) x| / |K_B x| (x the u_s "
                   "probe):\n";
    }
    for (const char* k : {"Bn", "Pb", "KvzT"}) {
      Vector yA, yB;
      SA->ApplyBrokenKernel(k, x[0], yA);
      SB->ApplyBrokenKernel(k, x[0], yB);
      real_t s[2] = {0.0, 0.0};
      for (int m = 0; m < yA.Size(); m++) {
        const real_t d = yA[m] - yB[m];
        s[0] += d * d;
        s[1] += yB[m] * yB[m];
      }
      MPI_Allreduce(MPI_IN_PLACE, s, 2, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
      if (Mpi::Root()) {
        std::cout << "    " << std::setw(5) << k << "  "
                  << std::setprecision(2) << std::scientific
                  << std::sqrt(s[0] / std::max(s[1], real_t{1e-300}))
                  << std::defaultfloat << "\n";
      }
    }
  }

  auto rel_diff = [&](const GridFunction& a, const GridFunction& b) {
    Vector ta, tb;
    a.GetTrueDofs(ta);
    b.GetTrueDofs(tb);
    Vector d(ta);
    d -= tb;
    real_t n2[2] = {d * d, tb * tb};
    MPI_Allreduce(MPI_IN_PLACE, n2, 2, MPITypeMap<real_t>::mpi_type,
                  MPI_SUM, MPI_COMM_WORLD);
    return std::sqrt(n2[0] / std::max(n2[1], real_t{1e-300}));
  };
  const real_t du = rel_diff(A->problem->Displacement(),
                             B->problem->Displacement());
  const real_t dz =
      rel_diff(A->problem->Potential(), B->problem->Potential());

  // Localise a gauge-bearing mismatch: the relative difference per
  // mesh attribute (fluid vs solid) says whether it is a gauge
  // representative shift (fluid only) or a physical leak.
  Vector attr_sums;
  int n_attr = 0;
  if (!slip && fluid_attributes.Size() > 0) {
    auto& a_gf = A->problem->Displacement();
    auto& b_gf = B->problem->Displacement();
    Vector ta, tb;
    a_gf.GetTrueDofs(ta);
    b_gf.GetTrueDofs(tb);
    ta -= tb;
    ParGridFunction d_gf(const_cast<ParFiniteElementSpace*>(
        static_cast<const ParFiniteElementSpace*>(a_gf.FESpace())));
    d_gf.SetFromTrueDofs(ta);
    const int d = a_gf.FESpace()->GetVDim();
    VectorFunctionCoefficient zero(
        d, [](const Vector&, Vector& v) { v = 0.0; });
    Mesh* mesh = a_gf.FESpace()->GetMesh();
    Vector ed(mesh->GetNE()), eb(mesh->GetNE());
    ed = 0.0;
    eb = 0.0;
    d_gf.ComputeElementLpErrors(2.0, zero, ed);
    b_gf.ComputeElementLpErrors(2.0, zero, eb);
    n_attr = mesh->attributes.Max();
    attr_sums.SetSize(2 * n_attr);
    attr_sums = 0.0;
    for (int e = 0; e < mesh->GetNE(); e++) {
      const int a = mesh->GetAttribute(e) - 1;
      attr_sums[2 * a] += ed(e) * ed(e);
      attr_sums[2 * a + 1] += eb(e) * eb(e);
    }
    MPI_Allreduce(MPI_IN_PLACE, attr_sums.GetData(), 2 * n_attr,
                  MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
  }

  // Strict wherever every assembled term is covariant: all welded
  // cases, the gauged fluid included, since the gauge penalty is
  // assembled as the exact pull-back (ElasticTensorIntegrator with the
  // pulled-back deviatoric tensor and the map). The slipping interface
  // is informational: its forms are not certified covariant.
  const bool strict = !slip;
  const bool pass = okA && okB && (!strict || (du < 1e-5 && dz < 1e-5));
  if (root) {
    std::cout << "\nrelabelled identity (" << m << ", order " << order
              << ", amplitude " << amplitude << "):\n  converged "
              << (okA && okB ? "yes" : "NO") << "\n  |u_A - u_B| / |u_B|  "
              << du << "\n  |z_A - z_B| / |z_B|  " << dz << "\n";
    for (int a = 0; a < n_attr; a++) {
      if (attr_sums[2 * a + 1] <= 0.0) {
        continue;
      }
      const bool is_fluid = std::find(fluid_attributes.begin(),
                                      fluid_attributes.end(), a + 1) !=
                            fluid_attributes.end();
      std::cout << "    attribute " << a + 1
                << (is_fluid ? " (fluid)" : " (solid)") << ": u diff "
                << std::sqrt(attr_sums[2 * a] /
                             std::max(attr_sums[2 * a + 1],
                                      real_t{1e-300}))
                << "\n";
    }
    if (strict) {
      std::cout << (pass ? "  IDENTITY HOLDS (to the DtN-centring floor)\n"
                         : "  MISMATCH: an unmapped assembly term\n");
    } else {
      std::cout << "  informational: the slip interface forms are not "
                   "certified covariant (the welded runs, gauged "
                   "included, are strict)\n";
    }
  }
  return pass ? 0 : 1;
}
