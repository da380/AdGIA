# Documentation review, 13 October 2026 — findings for discussion

Five-agent sweep of every document (tex, method notes, READMEs,
planning) for typos, LaTeX defects and correctness. Mechanical fixes
were applied directly (committed 13 Oct 2026, d9a8708): ~60 sites across
the tex files (notation, dashes, bibliography order, `\texorpdfstring`
titles), the READMEs/INSTALL (17 path and table fixes) and two method
notes; **the headline defect was the `\eps` macro in
quasi_static_models.tex**, corrupted by a global replace in commit
dbed7e1 (`\varepsilon` caught by a `\va` → `\mathbf{a}` sweep), which
rendered every ε in the PDF — 94 sites — as a bold "a" + "repsilon"
and caused both overfull equations. All tex documents now compile
warning-free (one harmless overfull box in static_fluid_core.tex).
Two stale API names were also fixed (`SetMaxwellFluid`,
`ClearGaugedFluid` → the real `SetFluid`/`ClearFluid`).

The items below were FLAGGED, not fixed — each needs a judgement or a
rewrite. Line numbers are current as of this tree. Delete entries (or
the file) as they are resolved.

Resolved 13 Oct 2026: the method-hierarchy gaps (former items 1–8;
Table 1 + abstract/vocabulary/§4 restructure with the promoted
`sec:maxwell` subsection, R4, the degree-0 reference qualifiers, and the
slip_interface/equilibrium_figures/gauge_penalty_iteration/gauged_fluid
qualifiers). Alongside item 1, `FluidRegion::Treatment` was delivered
(the CMB ladder as API) with the default-drop of fluid-enclosed inner
cores on non-full rungs (`-cmb-keep-inner` keeps them; the recorded
`prem_4` CMB numbers predate the default — re-measure queued in
`open_issues.md`).

Resolved 13 Oct 2026: the mathematical/notational items (former 9–22;
the ∂M mass row, the φ¹ definition, the Nanson-symbol alignment, the
dangling normal, the ε/C symbol-list entries, the \tilde c slip, the
equilibrium_figures notation pass — functional renamed \mathscr{J},
shape coefficients unified to s, overloaded-symbols note, quartic
re-point — cond(A_ε), the solve-vs-refinement counting, the
static_fluid_core Laplace-variable unification to s, §maxwell table
re-point, seven-architecture counts, the large-fixed-β Tikhonov
statement, the dynamic-argument paragraph stripped to its qualitative
essential-spectrum point per David, and the slip class refusing both
SetFluid forms).

Resolved 13 Oct 2026: the stale gauge-refinement algorithm
descriptions (former item 23; gauged_fluid.md, viscoelasticity.md and
gauge_penalty_iteration.tex now describe the non-virtual warm-started
full-solve refinement through the virtual `SolveLinearSystem`, no
block overrides — verified against `src/quasi_static_problem.cpp`
before rewriting).

Resolved 13 Oct 2026: the smaller correctness flags (former items
24–31; the two relabelling classes stated with the Friedman–Schutz
reference, OrthoSolver's users, the MappedRotation distinction, the
five-line count, the covariant form widened to both SetFluid
treatments and any non-null map, the `SlipProblem.` suite name, and
the benchmarks.tex bundle — disc/sea_level in the abstract and table,
`aw_core` in the model table/stratification/shiftable entries and in
love_numbers/README per David, `outside_source` guard wording, the
pert_* reproduce row fixed AND the campaign's perturbation stage now
passes `-geps 0.1 -gref 7` itself, the talk_figures/talk_render
source-tree invocations, "1400–13000 solves" matching the table, also
propagated to solvers.md). DEFERRED with the `prem_4` CMB re-measure
(David, 13 Oct: wait for new figures, then update): the winkler 17 %
sentence and the two error-measure slips in the same CMB-results
bullets — both recorded on the open_issues re-measure entry.

Resolved 13 Oct 2026: the verification items (former 32–33). The 2-D
moment cos/sin convention was indeed SWAPPED against poisson.hpp —
fixed, the doc now credits the header's convention. D's third entry
−C₃ verified against `centrifugal.hpp` (diag(C₃−C₁, C₃−C₂, −C₃)).
"Valette 1991" was a phantom entry — GJI 104(3):555–563 is Vermeersen
& Vlaar; David confirmed Valette (1986) alone is the intended
Weingarten citation, applied everywhere. The equilibrium_figures
`% TODO: verify` bibliography comments stay (spot-checks looked
right; they are harmless comments).

Resolved 13 Oct 2026: the house-style items (former 34–37). The
session-measured narrative compressed to mechanism + gate/benchmark
pointers (the monolithic 3.8×→2.3× story, the guard's five-passes
story, the "no longer a thought experiment"/"now measured"/"next
study"/"Still to come" register, the in-text 7–8 October dates); the
"Crawford:" attribution dropped (no citable entry); "not physics
anyone needs" neutralised and the log-DtN jargon replaced by a
\S-reference to the 2-D compatibility treatment; "carried for the
mapped gates" spelt out; static_fluid_core's planning pointers
re-pointed at the reference content; equilibrium_figures' planning
pointer and publication musing removed; the slip tidal-refusal note
de-dated and completed (welded referential and Maxwell solve tides).
The one deliberately-kept planning reference is resolved (David,
13 Oct): the rotation derivation graduated to
`doc/generalised_rotation.md` (de-planned register, Verification
section from the gates table, open items moved to future_work's
residue — where the fluid-core frame choice, settled and gated in the
note, no longer appears as open), and every pointer re-aimed.

All 37 items are now resolved or explicitly deferred (the winkler
sentence and the two error-measure slips ride the `prem_4` CMB
re-measure; see `open_issues.md`). The Notes below stay as recorded
observations.

## Notes

- Thirteen unreferenced labels in quasi_static_models.tex (harmless;
  list in the review transcript).
- `aspherical_buffer_3d_eE.mesh` naming does not follow the new
  `<geometry>_<dim>d[_variant]` pattern; consistent between
  aspherical_body.py, campaign.py and benchmarks.tex, so renaming is a
  coordinated change if wanted.
- Stale pre-rename meshes linger in build trees' data/ (not tracked).
- planning/equilibrium_figures.md:127 keeps old script/mesh names in a
  historical "Done:" record — left as history.
