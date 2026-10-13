# Documentation review, 13 October 2026 — findings for discussion

Five-agent sweep of every document (tex, method notes, READMEs,
planning) for typos, LaTeX defects and correctness. Mechanical fixes
were applied directly (uncommitted, in this tree): ~60 sites across
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

## Method-hierarchy gaps (Maxwell default / gauged = neutral-only / Dahlen workhorse)

1. quasi_static_models.tex 259–266 (Table 1): the class grid's fluid
   column lists only Dahlen/gauged variants; the Maxwell default
   (`SetFluid`'s front door) appears nowhere in the grid.
2. quasi_static_models.tex 2009–2011 (R4): "fluid regions in this
   class are treated by the welded gauged formulation" — stale; the
   referential class inherits both `SetFluid` overloads.
3. quasi_static_models.tex 71–73, 188–191, 1480–1492: the abstract,
   vocabulary and §4 structure present Dahlen-vs-gauged as the choice;
   the general (Maxwell) method appears only as a §4.1 paragraph.
4. quasi_static_models.tex 934–937, 991–993: the gauged treatment as
   "the reference … at degree 0" needs the neutral-stratification
   qualifier.
5. slip_interface.tex 84–87: reads as endorsing the welded gauged
   formulation for spherically symmetric models regardless of
   stratification.
6. equilibrium_figures.tex 197–201: "the slipping-interface
   formulation is required" for the perturbation layer — contradicts
   slip/welded as instruments and Maxwell as the general method.
7. gauge_penalty_iteration.tex 242–250: "Gauged fluid: the condition
   holds … the benign case" needs the N² = 0 qualifier (the same
   document admits the violation at 301–307).
8. gauged_fluid.md 89–94 vs 105–111: the validated ε window is
   evidenced on `fluid_core` (N² < 0) but called "the neutrally
   stratified one"; cite neutral models (`aw_core`, `prem_4`) or
   qualify.

## Mathematical / notational (quasi_static_models.tex unless noted)

9. 1353: the ocean mass row written over ∂B (the DtN sphere); should
   be ∂M (the body surface).
10. 2534, 2729: φ¹ used but never defined here (slip_interface.tex:690
    has the definition).
11. 2075 vs 292: Nanson-formula normal symbols (N reference vs n
    physical) swapped between §1.4 and §3.5.
12. 772: "With n the upward unit normal," dangles.
13. 199 vs 2754–2759: bare ε declared "a small parameter" in the
    overloaded-symbols list but used as strain in §6.
14. 2099: \tilde c where the text calls the tensor c / c′.
15. 1332/1360: C = ocean function vs C = coupling matrix, not in the
    overloaded-symbols list.
16. equilibrium_figures.tex: h vs s for interface-shape coefficients
    (1360–1364 vs 1463+); s also Sobolev order and LM step; J both
    functional and Jacobian (1372/1387); r residual/Riesz/radius.
17. equilibrium_figures.tex 1991: "the exact quartic line search of
    §sec:second" — the quartic lives in sec:derivative.
18. gauge_penalty_iteration.tex 157: κ(A_ε) clashes with κ = bulk
    modulus.
19. gauge_penalty_iteration.tex 201–202: refinement-vs-solve counting
    looseness ("two or three refinements reach 1e-4–1e-6").
20. static_fluid_core.tex: the Laplace-variable symbol is s/σ/p by
    turns (490, 508, 799, 835–851), p also pressure; u₀, Z, Q, B, D at
    834–850 undefined; table row 718–719 cites §regular where
    §maxwell is likely meant; 639–640 "the seventh architecture" and
    691/716 "six architectures" are mutually stale against
    examples/self_gravitating_solvers (Maxwell is number 5 of seven).
21. static_fluid_core.tex 564–566: "the β → ∞ limit is iterated
    Tikhonov" is loose (the memory update tends to w = u while γ → 0;
    iterated Tikhonov holds ε fixed).
22. slip_interface.tex 1463: the slip class refuses BOTH base
    treatments (Maxwell and gauge), not just the gauge-penalty form.

## Stale algorithm descriptions (the gauge refinement)

23. gauged_fluid.md 185–188, 254–271 and viscoelasticity.md 172–173:
    the refinement described as cold-started increments with
    mixed/referential `GaugeRefine()` overrides. The code's refinement
    is the warm-started full solve through the virtual
    `SolveLinearSystem`, non-virtual, no overrides (companion loads
    reapplied; equivalent by linearity). gauge_penalty_iteration.tex
    423–427 carries the same stale "block overrides" sentence.

## Smaller correctness flags

24. gravitating_elasticity.md 459–464: barotropic relabelling class
    stated as the two-parameter (N² ≠ 0) one; the barotropic class is
    larger.
25. mfem_notes.md 28–30: `OrthoSolver` also used by
    src/equilibrium_figures.cpp, not only the Poisson example.
26. mappings.md 233: "the last three … relabelling coefficients" —
    `MappedRotation` doesn't transform data; "the two".
27. self_gravitation.md 53: "the first six lines are the solid body"
    miscounts (five, one of them not solid-body).
28. elastic_tensors.md 194–197: covariant-penalty condition stated
    narrower than the code (any non-null map, identity included; also
    the Maxwell engine).
29. mappings.md 394: gtest ID written with the file prefix
    (`TestSlipProblem.…` vs suite `SlipProblem.…`).
30. benchmarks.tex: disc/ and sea_level/ families missing from the
    abstract and organisation table (50–59, 84–96); `aw_core` missing
    from the model table and the shiftable list (180–197, 211–214)
    though relied on at 576; `outputs.py` guard description stale
    (113); winkler 17 % sentence mixes `fluid_core` and `prem_4`
    numbers (1137–1140); the pert_* reproduce command can't produce
    the definitive table (460–462 vs 1559; needs
    `--program-args "-geps 0.1 -gref 7"` and order 3);
    talk_figures/talk_render commands written as if launchable from
    the build tree (437–477); two error-measure slips at 1019 and
    1025–1027; "1400–23000 solves" vs the table's 13008 peak (2208).
31. love_numbers/README.md 21–23, 142–153: `aw_core` runs under
    `run all` but is not listed (the standing "aw_core still to
    discuss" item).
32. quasi_static_models.tex 1092–1094 (unverified): the 2-D moment
    cos/sin convention vs poisson.hpp; 2325: D's third entry −C₃
    matches the planning note but was not independently verified.
33. slip_interface.tex 1716–1718: the Valette (1991) GJI 104 entry
    could not be confirmed; equilibrium_figures.tex retains its
    `% TODO: verify` bibliography comments (spot-checks looked right).

## House style (narrative, planning references, jargon)

34. quasi_static_models.tex: planning references at 2330–2331, 2353;
    narrative/measured-data passages at 1421–1427, 1429–1449;
    "Crawford:" attribution without citation (1390); "not physics
    anyone needs" and the undefined "log-DtN monopole gauge" (1408);
    "carried for the mapped gates" jargon (2291).
35. static_fluid_core.tex: planning references (206, 760) and the
    dated/narrative passages (78, 84–86, 312, 492, 533–534, 566–568,
    657, 661, 684, 727, 858–866).
36. equilibrium_figures.tex: planning reference (65–66); narrative at
    1409, 1975–1976, 2253, 2270–2272, 2329–2330, 2345.
37. gravitating_elasticity.md 348–350: "refuse the tidal load for
    now", "(measured)".

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
