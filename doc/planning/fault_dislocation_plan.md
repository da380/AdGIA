# Meshed-fault coseismic deformation — implementation plan

*Status: decided with David, 16 October 2026; **delivered in 2-D and
3-D** the same day (all four work packages, gates G0–G5 passing — see
"Where it stands"), uncommitted, for David's review. A basic coseismic
example with the fault meshed explicitly and the slip imposed as a
displacement discontinuity, plus the linear map from fault slip to
surface displacement and its adjoint. No gravity, no fluid regions;
generalisations only where they come essentially free.*

## Method: the split-node load on the welded mesh

- **Mesh.** The box of `examples/coseismic_deformation.cpp`, fragmented
  in gmsh by the fault plane extended to the box boundary: footwall and
  hanging wall are two volumes, and the cut is the fault patch Σ_f plus
  the welded extension Σ_w. The extension is scaffolding: it supplies
  the hanging-wall attribute and the fault-edge dofs (those shared by Σ_f
  and Σ_w) with no geometry in C++. Refined toward the fault and more
  strongly toward its tips/edge (a-priori grading, no adaptive
  refinement).
- **Unknown.** The displacement u_w in the ordinary continuous space on
  the whole mesh.
- **Slip.** ŝ, the slip s interpolated at the dofs of Σ_f, set to zero at
  the dofs of Σ_w (the fault edge, where the displacement is
  single-valued).
- **Load.** b(v) = −a⁺(ŝ, v): the stiffness restricted to hanging-wall
  elements, acting on ŝ. A linear-form integrator that borrows the
  problem's own stiffness integrators, so anisotropic and composite
  rheologies (and relaxation weights) follow. It sits in the problem's
  `ExternalLoad()`, so the `AssembleForce`/`Solve` protocol is unchanged.
- **Physical displacement.** u = u_w + ŝ on hanging-wall elements and u_w
  elsewhere, so [[u]] = u_HW − u_FW = ŝ exactly. This is the exact
  solution in the cracked finite-element space (the split-node method,
  Melosh & Raefsky 1981), identical to a Lagrange-multiplier solution
  with an equal-order multiplier.
- **Fault reaction.** r = A⁺(u_w + ŝ) on the fault-interior rows equals
  −A⁻u_w there (action–reaction). It is the consistent nodal form of
  −τ, where τ = σν is the traction change on the fault plane and ν
  points into the hanging wall.
- **Fault-to-surface map** G: a vector space on the fault's boundary
  SubMesh maps to the free-surface boundary SubMesh, through the true-dof
  injection, the load, one solve and the surface restriction. Gᵀ uses the
  same symmetric solve.
- **Adjoint for surface observables (in scope, David 16 Oct).** Gᵀ is
  "apply a traction w on the free surface, and return the traction it
  induces on the fault". Concretely:
  - solve the welded problem y = K⁻¹ w with zero slip;
  - the hanging-wall reaction of y on the fault-interior rows,
    −A⁺y = ∫ σ(y)ν·φᵢ, is that fault traction in consistent nodal form;
  - its tangential part is the sensitivity of the observable to
    tangential slip, by Volterra reciprocity:
    ∫_surface w·u = ∫_Σ s·σ(y)ν.

  w enters as a dual vector on the surface space (for an observable
  ∫ w_obs·u dS, w = M_surface w_obs). The output is a dual vector on the
  fault space, which a Riesz map of the fault space turns into a field.

Sign conventions are those of `examples/fault_box.hpp`: ν is the upward
fault normal (pointing into the hanging wall), and the slip direction
û is the hanging wall's motion relative to the footwall. So the
stress-glut moment density μ(ûνᵀ + νûᵀ) and the meshed slip describe
the same source.

## Work packages

1. **Mesh** — `meshes/faulted_box.py`, plain gmsh (OCC), writing
   `faulted_box_{2d,3d}.msh` and a JSON sidecar with the box, the fault
   geometry and sizes, and the attribute dictionary. Domain: footwall 1,
   hanging wall 2. Boundary: fault 1, fault extension 2, free surface 3,
   clamped footwall side 4, clamped hanging-wall side 5. Registered in
   `meshes/CMakeLists.txt`, `make_all.py` and the README table.
2. **Problem class** — `LinearQuasiStaticClampedFaultProblem` (derived from
   the Clamped class) in `fault_problem.hpp/.cpp`, and the load integrator
   in `lininteg.hpp`. API: `SetSlip(VectorCoefficient&)`, slip from a
   parent true-dof vector, the slip field, the physical (discontinuous)
   displacement, and the fault reaction.
3. **Example** — `examples/coseismic_fault.cpp`. It reads the mesh and
   sidecar and builds the cos²-tapered slip of `fault_box.hpp`. Outputs:
   surface-profile and fault-profile CSVs, and GLVis windows for the
   displacement with its jump visible. `-compare` overlays the smoothed
   stress glut of the same fault on the same mesh.
4. **Fault-to-surface map and its adjoint** — `FaultToSurfaceMap` (an
   `mfem::Operator`): `Mult` is the forward map (slip to surface
   displacement); `MultTranspose` is the adjoint (surface traction to
   fault traction). Also its reciprocity test, and in the example a
   Green's-function demo plus an adjoint demo: a surface observable (for
   example the uplift at one station, or a window average) and its fault
   sensitivity, the tangential traction profile along the fault,
   checked against the forward map.

## Where it stands (16 Oct 2026)

All four work packages are delivered in 2-D and 3-D; gates G0–G5 pass
in serial and on 1, 2 and 4 ranks (and 8 for the empty-rank paths).
Uncommitted.

| Piece | Location |
|---|---|
| Mesh | `meshes/faulted_box.py --all` → `faulted_box_2d` (1854 triangles) and `faulted_box_3d` (16077 tetrahedra) under a mountain-range topography over the hanging wall (David, 16 Oct: built into the gmsh geometry, so the fault stays planar), and `faulted_box_{2d,3d}_flat` with a flat surface; `--scale` scales the size field uniformly |
| Load integrator | `DomainLFFormActionIntegrator` (`lininteg.hpp`) |
| Class and map | `LinearQuasiStaticClampedFaultProblem` and `FaultToSurfaceMap` (`fault_problem.hpp`) |
| Tests | `TestFaultProblem` (5 gtests over 2-D orders 1–2 and 3-D order 1) and `TestFaultProblemPar` (39 checks per rank count) |
| Example | `examples/coseismic_fault.cpp` (2-D or 3-D by the mesh; rake in 3-D) |
| Reference doc | `doc/quasi_static_models.tex`, "A meshed fault with prescribed slip" |

Measured on the flat box (the first canned meshes, before the topography; the `_flat` meshes reproduce them within discretisation), with the default tapered thrust of peak 0.01. With the topography the fault lies deeper below the raised surface and the peak uplift drops to 0.00173 (2-D, order 2) and 0.00062 (3-D, order 1); the checks below hold unchanged on it (glut gap 0.25 % in 2-D, J forward = adjoint to 1e-12, serial = 4 ranks to 3e-10).
- **Moment:** equal to the glut's (2-D 0.0075, 3-D 0.005625).
- **Glut gap**, meshed slip against the smoothed glut on the same mesh;
  it is the glut's smoothing bias, falling as σ²:
  - 2-D, order 2: 0.45 % → 0.115 % → 0.032 % of the peak at
    `--scale 1, 0.5, 0.25`;
  - 3-D, order 1: 6.1 % → 1.6 % at `--scale 1, 0.5` (order 2: 5.6 %,
    1.45 %).
- **Peak uplift against `coseismic_deformation`** on its structured mesh:
  - 2-D: 0.00223165 against 0.00223047;
  - 3-D, order 2: 0.001072 at `--scale 0.5` against 0.001058 on the
    coarser structured mesh.
- **Adjoint:** forward J against adjoint J agree to 1e-12 (2-D and 3-D).
- **Green's functions:** the sub-fault sum matches the whole fault to
  1e-11.
- **Serial against 4 ranks:** the example's tables agree to 4e-10 in 2-D
  and 3-D.
- **Mutation check:** flipping the sign of the slip load, or dropping the
  edge masking, fails the corresponding tests.
- **Cost:** 3-D on 8 ranks, order 2: 5.8 s on the canned mesh (61k
  dofs), 58 s at `--scale 0.5` (371k dofs); serial order 1 on the canned
  mesh, 1.1 s.

Refinement and relief (16 Oct): the example takes MFEM's `-rs`/`-rp` uniform refinements (glut width follows the refined fault size: 2-D gap 0.25 % → 0.064 % at `-rs 1`, 3-D 2.3 % → 0.53 % at `-rp 1`; serial `-rs 1` = 4 ranks `-rp 1` to 3e-10); the 3-D relief defaults to 2 (crest 0.72), the 2-D to 1.

GLVis (16 Oct): in 3-D the volume windows are cut open along the dip section through the fault centre and seen along the strike (the cutting plane turned by the example's keys, the camera set by a `view` command, `GLVisWindow::SetView`; checked against a model of GLVis's plane arithmetic), and the fault windows are seen face on.

Duals to fields (David, 16 Oct): the fault reaction and Gᵀw are dual
vectors. The example maps them to fields with the Riesz maps of
`riesz.hpp`, on the fault SubMesh space with the slip pinned on the fault
edge (the SubMesh's boundary, attribute max + 1):
- **Traction:** the L² map, giving the L² projection of σν onto the slip
  space.
- **Kernel:** its dual first projected onto the fault plane (the slip is
  tangential; on a planar fault the projection commutes with the
  component-wise metric), then the H^s map (`-sl` 0.1, `-so` 2, the
  gia_kernels convention), giving the gradient of J with respect to
  tangential slip.

Measured trade-off on the canned meshes:
- **The L² traction is faithful but noisy.** It matches the old lumped
  profile to 1.6 % of the peak in 2-D away from the edge, but carries
  mesh-scale noise on the unstructured 3-D mesh: about 10 % of the peak
  at order 2 and worse at order 1. The reaction is assembled from the
  stress of the hanging-wall element layer, and the consistent mass
  passes that noise through.
- **An H² map is the wrong metric for the traction.** It smooths the
  noise but damps the stress drop (2-D: −0.016 → −0.012), and the pinned
  edge erases the tip concentrations.
- **Open (David):** whether a recovery from the stress itself would serve
  the display better.


## Gates

- **G0** Each mesh loads in serial and parallel, with the advertised
  attributes.
- **G1** Rigid block. The whole cut is treated as fault, only the
  footwall side is clamped, and the slip is uniform. The hanging wall
  must translate rigidly by s with zero strain, to solver tolerance.
  This pins the signs, the lifting and the parallel slip construction.
- **G2** The fault reaction is symmetric and positive in the slip:
  ⟨r(s₁), s₂⟩ = ⟨r(s₂), s₁⟩ and ⟨r(s), s⟩ > 0. It is the Schur complement
  of the cracked stiffness.
- **G3** Reciprocity ⟨G s, w⟩ = ⟨s, Gᵀ w⟩ to solver tolerance, G s
  equals a direct solve, and Gᵀ w equals the hanging-wall reaction of the
  surface-traction solve.
- **G4** Serial ≡ parallel.
- **G5** Meshed slip against the smoothed stress glut on the same mesh:
  the surface profiles agree away from the fault and converge together
  under refinement.

## Gotchas

- **Parallel slip.** A rank can hold a hanging-wall element touching a
  fault node without holding the fault face, so build ŝ only through
  rank-consistent paths: `ParGridFunction::ProjectBdrCoefficient`
  (reduce + broadcast), true dofs then `SetFromTrueDofs`, and
  synchronised essential-dof markers. Never a local injection of a
  SubMesh L-vector.
- **SubMesh factories** take attribute lists, not markers.
- **Edge dofs** carry zero slip by construction (Σ_w dofs). The fault
  space of the map has them in its domain, but G ignores them.
- **Mesh constraints.** The extended plane must cross the box cleanly
  (not through a box edge or corner), and the fault must be strictly
  inside it. Surface-rupturing faults are out of scope for now.
- **House rules.** One source for serial and parallel, `std::numbers`,
  GLVis through `visualisation.hpp`, outputs in the build tree, no
  narrative comments.

## Deferred (not built)

- **AL/KKT on a broken submesh pair.** The use case is friction and
  inequality constraints; the comparison of the two paths themselves was
  already settled in the fluid–solid slip class.
- **gmsh Crack-plugin meshes.** Verified viable: the plugin is in the
  gmsh 4.15.2 wheel and MFEM reads the cracked MSH, with the crack faces
  as one-element boundary faces. Not needed for the split-node load.
- **Gravity, fluid regions, a lumped local ocean.** The ocean would be a
  boundary term added through `StiffnessIntegrators()`, which composes
  with a buried fault.
- **Postseismic stepping.** Consistent as it stands if the elements next
  to the fault are elastic (the lid of `postseismic_deformation`).
  Otherwise the strain map must also see the strain of the hanging-wall
  lifting.
- **An independent analytic reference.** Okada (1992) in 3-D, imposed as
  Dirichlet data on the box's artificial boundaries.
