# Instructions for the Claude instance on the HPC — finish the FEP frame fix

You are picking up a relative-FEP (RBFE) pipeline that has a **fixed** correctness bug
and **one remaining** environment blocker. Your job is to land the fix on the cluster,
clear the environment blocker, regenerate the invalidated inputs, and produce the
first trustworthy ΔΔG network.

Read this whole file before running anything. Then read
`handover/06_Relative_FEP_over_Loch_Waters.ipynb` §3–3d, which has the measurements
this document summarises.

---

## 0 · Context in six sentences

The pipeline runs Loch GCMC endpoint MD (CRY1–AN139 and EV-A71 2A protease), then
relative FEP with SOMD2 seeded from the equilibrated waters. A 52-edge EV71 network
was submitted on 2026-07-24 and **every bound leg died** on
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)`; no ΔΔG has ever been produced. While
investigating that, a **coordinate-frame bug** was found in `prepare_fep.py` that made
the free leg physically invalid in every run ever built. It was partially fixed on
2026-07-31 (commit `9fc64e8`, branch `fix/free-leg-frame-mismatch`) and **fully fixed
on 2026-08-10**, which is the change you are landing. The endpoint MD data — 12 CRY1
replicas and the 192-run EV71 matrix — is unaffected and does **not** need rerunning.

---

## 1 · What the bug was, so you can recognise a regression

Three coordinate frames are in play, and **neither FEP leg keeps the parameterisation
frame**:

| Object | Frame | Offset from the ligand input frame |
|---|---|---|
| `ligand.prmtop` / `.rst7` / `.mol2`, `ligand_input.sdf` | ligand **input** | 0 |
| bound system — prepared complex *or* production restart | **receptor** | **28.97 Å** |
| free system after `solvateOct` | tLEaP's **re-centred box** | **12.30 Å** |

The third one is the part that was missed until 2026-08-10: `solvateOct` translates
the solute when it builds the octahedral box, so the free leg is *not* in the mol2
frame either.

Measured from retained streams, what pre-fix runs actually contained:

| Built | Code | Free-leg ligand sat in | Closest water O |
|---|---|---|---|
| ≤ 2026-07-30 | single merge, `align=true` | the **receptor** frame — outside the water box | **31.08 Å** — solvated by nothing |
| 2026-07-31 → 08-10 | two merges | the **mol2** frame, box elsewhere | **0.56–1.85 Å** — overlapping water |

Both give a wrong ΔG_free, hence a wrong ΔΔG. **Neither raises.**

### The fix

`build_leg_merge()` builds **one merge per leg, RMSD-aligned onto the ligand it
replaces in that leg** — one rule, no flags, all three frames. Both merges are built
*after* both boxes exist, because the free box's ligand position is tLEaP's, not the
mol2's. `--align-to-bound-pose` is now **deprecated and ignored**; it is still accepted
so `fep_edge.slurm` and `fep_prepare.slurm` work unchanged.

`verify_leg_frame()` enforces it on **both** legs and writes `frame_check.*` into
`fep_preparation.complete.json`.

### ⚠ Why it survived its own self-test — do not repeat this mistake

The 2026-07-31 change was validated with a **null perturbation** (x7259a → x7259a,
identity mapping). That test is **mathematically incapable of detecting a frame
error**: with state A identical to state B, any environment error — vacuum,
overlapping water, wrong pocket — contributes identically at λ=0 and λ=1 and cancels
exactly in ΔΔG.

That run returned **ΔΔG = −0.0001 kcal/mol, 11/11 windows, zero minimisation
failures**, through a free leg with **33 of 57 atoms inside 2.0 Å of a water oxygen**.
It read as a clean pass.

**So: use null perturbations for the run path only** (does a leg complete, are all
Parquets written). Use `frame_check` for geometry. It works precisely because it does
not depend on the perturbation being non-trivial.

---

## 2 · Land the fix

The fix is on branch **`fix/fep-leg-frame-registration`** of
`git@github.com:BenCree/csbrt-gcmc-fep.git`, branched from `main`. It changes one
file, `csbrt/src/csbrt/prepare_fep.py` (+214 / −51).

```bash
cd <your csbrt-gcmc-fep checkout>
git fetch origin
git log --oneline main..origin/fix/fep-leg-frame-registration
git diff main origin/fix/fep-leg-frame-registration -- csbrt/src/csbrt/prepare_fep.py | less
```

**Do not merge it blindly.** Read the diff, then either check out the branch or
cherry-pick the commit onto whatever the cluster actually runs.

### ⚠ There are multiple copies of `prepare_fep.py`

On the workstation there were **seven**, all byte-identical. Find every copy on the
cluster and update all of them, or you will fix the one you are reading and run a
different one:

```bash
find "$HOME" -name prepare_fep.py -not -path '*/__pycache__/*' 2>/dev/null \
  | while read f; do echo "$(md5sum "$f" | cut -c1-8)  $f"; done | sort
```

They must all end up with the same hash. Then remove stale bytecode, which has
misled in this project before:

```bash
find "$HOME" -name 'prepare_fep*.pyc' -delete
```

Confirm the fix is the one loaded, not a cached copy:

```bash
python -c "import prepare_fep; print(prepare_fep.__file__); print(hasattr(prepare_fep,'build_leg_merge'), hasattr(prepare_fep,'verify_leg_frame'))"
# expect: True True
```

### Relationship to the existing branches

- `fix/free-leg-frame-mismatch` (`9fc64e8`) is the **partial** 2026-07-31 fix. It is
  already merged into `main`. Its diff against current `main` looks enormous only
  because it predates the tree consolidation — ignore that noise.
- `fix/fep-leg-frame-registration` **supersedes** it. Do not try to apply both.

---

## 3 · Clear the remaining blocker: the CUDA pin

This is the reason the 52-edge network produced nothing on 2026-07-24.

`cuda-version` must be **≤ the CUDA version of the driver on the compute node**, and
you must pin the **whole** toolchain, not just `cuda-version` — an unpinned
`cuda-nvcc` resolves to 12.9 and every OpenMM context creation then fails with
`PTX 222` while the job still exits 0.

```bash
# 1. Find the node's actual ceiling. Do this ON A COMPUTE NODE, not the login node.
srun --gres=gpu:1 --pty nvidia-smi        # read "CUDA Version:" in the header

# 2. Pin environment-fep.yml to that value (12.8 on this cluster as of 2026-07),
#    matching cry-loch-babel. Keep cuda-nvvm if the bound leg will use Loch GCMC.

# 3. Gate on it. THIS IS NOT OPTIONAL.
srun --gres=gpu:1 --pty scripts/preflight_fep.sh
```

`preflight_fep.sh` creates an OpenMM CUDA context (catches PTX 222 in ~2 s) and then
runs a tiny real SOMD2 leg. **Run it on the target node.** A GPU smoke on a box whose
driver is newer than the cluster's gives a false pass — this is driver skew, not
CPU-vs-GPU.

---

## 4 · Discard the invalidated inputs

Every existing `.bss` stream and `fep_preparation.complete.json` is invalid. Two
independent reasons: the free-leg geometry was wrong, and `implementation_signature`
hashes the pipeline sources so the markers no longer match anyway.

```bash
# Inspect before deleting. Confirm these are FEP setup dirs and nothing else.
find "$HOME" -name 'fep_preparation.complete.json' | head -20
find "$HOME" -path '*/setup/*_free.bss' | wc -l

# Then remove the setup directories so they regenerate.
# Do NOT delete endpoint runs, production trajectories, or density analyses.
```

**Never** pass `--force` to reuse a pre-fix marker. Regeneration is mapping plus one
tLEaP call per edge, about a minute each, and `fep_edge.slurm` does it inside every
array task anyway — so this costs you nothing you were not already spending.

**Do not touch:** `runs/ev71-density-series/` (the 192-run endpoint matrix),
`cry-loch-multi/`, `cry-loch-full/`, any `production/` or `density_analysis/`
directory. None of that is affected by this bug.

---

## 5 · Verify the fix on the cluster before submitting a network

Three checks, in order. Do all three.

**5a. Real script, null perturbation — the run path.** Uses any completed endpoint
preparation as both states:

```bash
PREP=<a completed endpoint>/preparation
python -u prepare_fep.py \
    --state-a-preparation "$PREP" --state-b-preparation "$PREP" \
    --output-dir /tmp/nullcheck --edge-id nullcheck --align-to-bound-pose
```

Expect on stdout:

```
[prepare_fep] --align-to-bound-pose is deprecated and ignored: ...
[prepare_fep] frame check: bound offset 0.000 A, free offset 0.000 A,
              bound-to-free separation 23.64 A [expected to be large: separate boxes]
```

Both offsets must be **~1e-14 Å**, i.e. machine precision. The bound-to-free
separation is *supposed* to be large — the legs are in different boxes.

**5b. One real edge — the geometry.** A null perturbation cannot validate frames
(§1). Prepare a real edge from the network and read its marker:

```bash
python -c "
import json,sys
d=json.load(open(sys.argv[1]))
fc=d['frame_check']
print('bound offset', fc['bound_centroid_offset_angstrom'])
print('free  offset', fc['free_centroid_offset_angstrom'])
print('tolerance   ', fc['tolerance_angstrom'])
assert fc['bound_centroid_offset_angstrom'] < 1e-6
assert fc['free_centroid_offset_angstrom']  < 1e-6
print('OK')
" <edge>/setup/fep_preparation.complete.json
```

**5c. Closest solute–solvent contact — the physics.** The check that actually
distinguishes a harmless rigid translation from a steric clash. For both legs of that
edge, the perturbable ligand's closest distance to a water oxygen must be **> 2.5 Å**
with **zero** atoms inside 2.0 Å. Notebook 06 §3b has working code for this
(`BSS.Stream.load`, then `_toRegularMolecule(is_lambda1=False)` — a perturbable
molecule returns `None` from `atom.coordinates()`).

Reference values from the fixed code: bound **3.30 Å**, free **3.16 Å**, zero clashes.

**If any check fails, stop.** Do not raise `--max-leg-centroid-offset` to make it
pass; that flag exists to be tightened, not loosened.

---

## 6 · Run the network

```bash
scripts/submit_fep_edges.sh \
    --manifest fep_manifest.tsv \
    --batch 12 \
    --rowan-edges fep_edges/rowan_xtal_edges_full.tsv
```

52 edges, connected, 21 independent cycles. `--batch N` is the concurrency knob and
the way to share the cluster: two users at `--batch 12` with separate `--run-root`
coexist on 24 GPUs. Chain is edge array → `fep_aggregate.slurm` → `fep_compare.slurm`.

Budget: 11 λ × 5 ns per leg, run **serially** on one GPU — roughly 2 h/leg at
~700 ns/day, plus 100 ps/window equilibration and sampler startup.

---

## 7 · Validity gates on the results — in this order

Do not skip to the correlation plot.

1. **One `energy_traj_*.parquet` per λ window, per leg.** `run_fep_leg.py` enforces
   this because a window worker can die while the top-level `somd2` exits 0.
2. **Non-vanishing adjacent-window overlap on both legs.** Near-zero overlap means
   the λ schedule is too coarse for that edge and the ΔΔG is noise *regardless of its
   printed uncertainty*. Never read an uncertainty without the overlap beside it.
3. **Cycle closure.** 21 independent cycles; each should close within its propagated
   uncertainty. `standardised_residual` in `fep_network_edges.csv` is the per-edge
   version.
4. **Then** `compare_to_rowan.py` against `rowan_results_per_edge_wide.csv` and the
   per-compound experimental CSV, at edge *and* ligand level.

A missing interior λ window **severs** a leg into disconnected segments and yields no
ΔΔG at all, even though every file present looks valid.

One failed array task blocks the whole network: `aggregate_fep_network.py` requires
every manifest edge. To analyse the rest, drop that edge from a **copy** of the
manifest — the network usually stays connected and loses only cycle redundancy.

---

## 8 · The one experiment worth doing beyond this

If any real-edge ΔΔG from **before** 2026-08-10 still exists on the cluster, pull it
out and keep it. Running `compare_to_rowan.py` on the old numbers and the new ones
against the published Rowan ΔΔG values would quantify what the frame bug actually
cost. That comparison could not be made on the workstation, because the only
completed pre-fix runs there were a 2-window smoke and a null perturbation.

```bash
find "$HOME" -name analysis.json -path '*fep*' 2>/dev/null
find "$HOME" -name 'fep_network_analysis.json' 2>/dev/null
```

---

## 9 · Standing rules in this project — violating these has cost real time

1. **Exit status 0 means nothing.** Verify by artefact: expected file present, non-zero
   frames, finite numbers.
2. **A stage completing is not evidence its output is correct.** The endpoint NPT once
   ran 2 ns and reported a perfect 300.29 K / 1.0185 g/mL trace while carrying waters
   with zero charge and zero LJ.
3. **Minimisation succeeding is not correctness.** A ligand in bulk solvent, and a
   non-interacting water, both minimise beautifully. **The loud failure is safer than
   the quiet one.**
4. **`profile=smoke` is plumbing validation only** — 3 frames. Never a short
   scientific run.
5. **SOMD2 appends to `log.txt` across runs.** Grepping for errors returns stale
   failures from days earlier. Check timestamps and cross-check `sacct` job IDs.
6. **`Overall performance: N ns day⁻¹` is not throughput** — it is one window's
   runtime over the whole-leg wall time. Use the per-λ `complete, speed = ...` line.
7. **Checkpoint extension is version-dependent**: `.s3` on older SOMD2, `.npz` on
   2026.1. Matching only one silently falls through to `--overwrite` and re-runs every
   completed window.
8. **Editing pipeline sources invalidates stage markers** (`implementation_signature`
   hashes them). Completed *analysis* outputs are unaffected. When retrying one failed
   array task after a code fix, **submit only that array index**.
9. **The free leg contains no receptor.** Chain duplication, pocket waters and
   binding-site geometry cannot explain a free-leg failure. But check `frame_check`
   first — it is one number and it is decisive.
10. **Weigh base rates before accepting a systemic hypothesis.** If a theory predicts
    every edge should fail and 51/52 succeeded, the theory is refuted.
11. **Never hardcode a system, path, or dataset.** A wrong default does not fail
    loudly — it simulates the wrong thing and reports success. Make the argument
    required instead.
12. **Never add a capability without an executed path to it.** Dependency + calling
    code + one run on real data is *one* unit of work, not three.

---

## 10 · Where things are

| What | Where |
|---|---|
| This fix | branch `fix/fep-leg-frame-registration`, `csbrt/src/csbrt/prepare_fep.py` |
| Partial predecessor (already in `main`) | `9fc64e8`, branch `fix/free-leg-frame-mismatch` |
| Nine executed handover notebooks | `handover/` in this repo |
| FEP operating guide | `handover/references/fep.md` |
| Pipeline/script map — read before editing any Python | `handover/references/pipeline-map.md` |
| Ludovic-parity contract | `handover/references/ludovic-parity.md` |
| Endpoint stability diagnostics | `handover/references/diagnostics.md` |
| Porting to a new receptor/library | `handover/references/porting.md` |

The 31 GB of trajectories and density analyses are **not** in this repo. They are on
the workstation under `intern_projects/project_1/{cry-loch-multi,equilibration,output}`
and `intern_projects/project_2/runs/`, and on the cluster under `$HOME/cry/`.

Start with `handover/README.md`, then notebook 00 (stdlib only, runs on a login node —
it parses the protocol constants out of the source and asserts them against the
documented totals, so it fails if anything has drifted).
