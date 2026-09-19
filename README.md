# csbrt-gcmc-fep
Workflow for automated water network GCMC and FEP

## Layout

| Directory | What it is |
| --- | --- |
| [`csbrt/`](csbrt/) | **Unified package.** The whole workflow behind one driver and one conda environment: OpenFold3 loop modelling -> graft -> protonation -> ligand prep -> Loch GCMC -> SOMD2 relative FEP -> analysis. Run end-to-end or stage by stage. Also holds the Slurm wrappers for cluster use and the Grand Canonical Integration (GCI) tooling. The former `loch_fep_pipeline/` tree was folded in here; its last state is preserved on the `legacy/loch-fep-pipeline` branch. |
| [`protein_prep/`](protein_prep/) | Receptor preparation for CRY1 (7DLI). The original **Boltz-2** loop-modelling + OpenFE MD workflow ([`README.md`](protein_prep/README.md)), plus an **OpenFold3** fork of the same steps ([`README-openfold3.md`](protein_prep/README-openfold3.md)) for the unified environment. |
| [`ev71_gcmc_validation_data/`](ev71_gcmc_validation_data/) | EV71 GCMC validation study: reports, figures, and the scripts that generated them. |

## Quick start (unified package)

```bash
cd csbrt
./install.sh                 # conda env + pip layer + the csbrt CLI
mamba activate csbrt

cp config.example.yaml run.yaml    # edit paths, ligand, loop, pH
csbrt --all --config run.yaml      # or: csbrt --from equilibrate --through gcmc
```

See [`csbrt/README.md`](csbrt/README.md) for stage contracts, the environment
constraints (why conda is mandatory, why the CUDA toolchain is pinned whole, why
the predictor is OpenFold3), and the diagnostics worth running.

## MACE ML/MM surrogate

The perturbable ligand can optionally be evaluated with a MACE foundation model
while the protein and bulk solvent stay classical, with uncertainty monitoring
and an automatic fallback to the classical Hamiltonian
([`csbrt/src/csbrt/mace_surrogate/`](csbrt/src/csbrt/mace_surrogate/)). It is off
by default.

- [`docs/mlff_delivery_report.md`](docs/mlff_delivery_report.md) — **start
  here.** What was built, what was verified and with what numbers, what could
  not be verified here, the small-scale test to run, and the open items.
- [`docs/mlff_decisions.md`](docs/mlff_decisions.md) — every judgement call:
  what the plan said, what was done instead, why, and how to reverse it.
- [`docs/mlff_throughput_expectations.md`](docs/mlff_throughput_expectations.md)
  — hybrid ML/MM is about an order of magnitude *slower* than the classical
  pipeline, not faster; what it buys instead, and where a 50% time reduction
  could actually come from.
- [`docs/mlff_active_learning_architecture.md`](docs/mlff_active_learning_architecture.md)
  — component-by-component specification, and the validation checklist before
  trusting a ΔΔG from it.

## Two structure-prediction tracks

`protein_prep/` keeps both, and neither replaces the other: Boltz-2 pins
`numpy<2.0`, which is incompatible with the numpy 2.x sire/somd2/loch stack, so the
Boltz track runs in its own environments while the OpenFold3 fork is what the
unified `csbrt` environment uses. See
[`protein_prep/README-openfold3.md`](protein_prep/README-openfold3.md) for the
step mapping and the caveats when switching.
