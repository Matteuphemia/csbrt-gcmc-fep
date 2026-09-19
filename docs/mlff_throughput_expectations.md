# What the MACE surrogate costs, and what it buys

**Read this before planning a campaign around the MACE surrogate.**
See also [`mlff_delivery_report.md`](mlff_delivery_report.md) for what was
built and verified, and [`mlff_decisions.md`](mlff_decisions.md) D1 for this
argument in its shortest form.

The implementation plan sets a target of roughly **50% less simulation time**
from replacing classical force-field evaluations with a MACE surrogate. That
target does not follow from the hybrid ML/MM architecture the same plan
specifies, and the two papers the plan is grounded in say so directly. This
note sets out why, what the surrogate does buy, and how to measure both on your
own system rather than taking either claim on trust.

---

## 1. Hybrid ML/MM adds work; it does not remove it

The speedup argument would hold if MACE *replaced* the classical force field.
It does not, and it cannot: evaluating a pure MLFF over a 30,000–60,000 atom
solvated complex is about **100x slower** than GPU-accelerated MM and risks
running the GPU out of memory. That is the conclusion of Wang, Eastman,
Tuckerman et al. (2024) — `docs/wang2024_design_space_mm_mlff.pdf`, summarised
in `docs/wang2024_design_space_report.md` — and it is exactly why the plan
specifies hybrid ML/MM partitioning instead.

But partitioning means the MM side still runs. In the mixed system:

- the protein and all bulk solvent keep their Amber ff14SB / TIP3P terms;
- PME still runs over every charge in the box;
- the ligand's interactions *with* that environment are still computed by the
  MM force field (that is what mechanical embedding means);
- **and** MACE is evaluated on top, every step, for the ML region.

What the ML region removes from the MM side is the ligand's internal bonded
terms and its internal nonbonded pairs: for a 40-atom ligand in a 40,000-atom
box, well under 0.1% of the classical work. What it adds is a forward and
backward pass through an equivariant GNN at every timestep.

The numbers from the same paper, per the report in `docs/`:

| regime | throughput |
|---|---|
| classical MM | 100–1000 ns/day |
| **hybrid ML/MM** | **10–50 ns/day** |
| pure MLFF on a macro-complex | 0.1–1 ns/day |

Hybrid ML/MM is roughly **an order of magnitude slower** than the classical
simulation this pipeline runs today. There is no configuration of this
architecture in which enabling the surrogate makes a leg finish sooner.

Measured on this repository's own fixture (`csbrt-mace-benchmark`, 26-particle
test system, CPU platform, MACE-OFF23-small):

```
classical MM          :  65.90 ns/day
mixed, fallback (l=0) :   5.64 ns/day
mixed, surrogate (l=1):   2.15 ns/day
speedup vs classical  : 0.033x
```

That fixture exaggerates the effect — it has almost no MM work to amortise the
ML cost against, and it runs on a CPU. A real complex on an A100 will land far
closer to the paper's 10–50 ns/day. It will not land above the classical rate.

## 2. The fallback does not save time either

`lambda_interpolate` is implemented, as openmm-ml implements it, with a
`CustomCVForce`:

```
U = λ·U_ML/MM + (1 − λ)·U_MM
```

A `CustomCVForce` evaluates **every** collective variable on every step,
whatever its coefficient. So at λ = 0 the MACE forward pass still runs; its
result is simply multiplied by zero. The benchmark row above shows this: the
mixed system at λ = 0 is 5.64 ns/day against the classical system's 65.90, even
though the two compute the same energy to eight significant figures.

The consequence is worth stating plainly:

> The physics fallback is a **correctness** mechanism, not a performance one.
> It guarantees that no frame is integrated on a potential the model is not
> confident about. It does not make the fallback cheap.

A corollary: when the controller latches to classical mode (because the
surrogate spent more than `fallback_abort_fraction` of the run falling back),
the run keeps paying the full ML cost for no benefit. It logs a warning saying
so. The right response is to stop the run and restart without the surrogate,
not to let it finish.

The alternative design — separate force groups plus
`Integrator.setIntegrationForceGroups()`, which genuinely skips the ML
evaluation — would fix that, at the cost of hand-building the interpolation
openmm-ml already provides and verifying it ourselves. It is not worth it while
the fallback is expected to be a few percent of steps. It would become worth it
if the fallback rate were high, and at that point the surrogate is not fit for
the system anyway.

## 3. What the surrogate actually buys

Accuracy in the ligand's internal energetics, which is a real and measurable
thing in this pipeline.

MACE-OFF23 is fitted to wB97M-D3(BJ)/def2-TZVPPD. GAFF2 with AM1-BCC charges —
what `csbrt` uses for ligands today — is fitted to neither, and its weakest
point is exactly the part of the Hamiltonian that alchemical FEP is most
sensitive to: torsion profiles and intramolecular strain in the perturbable
region. Wang et al. show hybrid alchemical RBFE reaching chemical accuracy
(< 1 kcal/mol) with the perturbable region on a neural potential.

This repository already has evidence that ligand torsions are a live problem in
its own results: `csbrt.torsion_diagnostics` exists because an edge can have
healthy window overlap and still be wrong if a rotatable bond never crossed its
barrier, and the analysis stage runs it on every edge. A better description of
that barrier is directly relevant.

So the honest framing is a trade, not a win:

| | classical | MACE ML/MM |
|---|---|---|
| ligand internal energetics | GAFF2/AM1-BCC | ωB97M-D3(BJ) quality |
| throughput | 1x | ~0.1x (measure it) |
| ΔΔG accuracy | current baseline | to be established per target |

## 4. Where a 50% reduction could actually come from

If the goal is wall-clock, these are the levers in this pipeline, none of which
involve MACE:

1. **Replica exchange**, already implemented (`--replica-exchange`). Raises
   adjacent-window overlap directly, so fewer windows and shorter runs reach
   the same statistical precision. Measured overhead is negligible.
2. **Hydrogen mass repartitioning**, already in use — `somd2_config.yaml` runs
   a 4 fs timestep.
3. **Fewer λ windows per edge**, justified by the overlap diagnostics the
   analysis stage already produces, rather than by the fixed `num_lambda: 11`.
4. **Shorter runs where the convergence diagnostics say they are converged.**
   `runtime: 5 ns` is described in `somd2_config.yaml` itself as "a
   conservative starting point, not a convergence guarantee"; some edges will
   be converged well before it and some will not be converged by it.
5. **Pruning the edge network.** 52 edges is a design choice; cycle-closure
   residuals show which edges carry information.

Each of those can be measured against the existing checkpoints without any new
physics.

## 5. Measure it yourself

Do not take either the plan's figure or this note's figure for your system.

```bash
# On the GPU node you intend to run on, with the real complex:
csbrt-mace-benchmark \
    --prmtop RUN/endpoint/LIG/rep1/production/LIG-production-final.prmtop \
    --rst7   RUN/endpoint/LIG/rep1/production/LIG-production-final.rst7 \
    --steps 2000 --json mace_benchmark.json
```

The report gives ns/day for the classical system, the mixed system in both
modes, and the committee's inference cost at the configured UQ interval, plus
the resulting speedup factor. A factor below 1.0 is reported as a slowdown, in
those words.

## 6. Licensing

`mace-off23-*` models are distributed under the **Academic Software Licence
(ASL)**: academic use only, **commercial use is not permitted**. openmm-ml logs
a warning to that effect every time one is loaded. For commercial work use a
model whose licence permits it (`mace-mpa-0-medium` is unrestricted but is a
materials model, not fitted to molecular conformational energies), or train
your own from the active-learning loop. Confirm the licence of any foundation
model before it touches a commercial project.

---

## References

- Wang, Eastman, Tuckerman et al. (2024), *On the Design Space Between
  Molecular Mechanics and Machine Learning Force Fields*, arXiv:2409.02861 —
  `docs/wang2024_design_space_mm_mlff.pdf`, `docs/wang2024_design_space_report.md`
- Duignan (2024), *The Potential of Neural Network Potentials*, ACS Phys. Chem.
  Au 4, 232–241 — `docs/duignan2024_potential_nnps.pdf`,
  `docs/duignan2024_potential_of_nnps_report.md`
