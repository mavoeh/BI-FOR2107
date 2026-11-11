# FOR2107 adaptation of Tozzi et al. (2024)

This repository implements a reproducible, pure-Python adaptation of
"Personalized brain circuit scores identify clinically distinct biotypes in
depression and anxiety" for the FOR2107 cohort described by Kircher et al.
(2018). Docker is used only as the recommended runtime for fMRIPrep.

The primary analysis is cross-sectional (`ses-01`) and multimodal. It clusters
participants with major depressive disorder and/or a configured anxiety
diagnosis. Healthy controls are used to construct normative circuit scores and
are not clustered. Other diagnoses and later sessions can be enabled without
changing the primary specification.

## Start here: the current analysis

The current analysis asks whether FOR2107 patients show reproducible
subgroups like Tozzi's six biotypes: k = 6 clusters are split-half tested
across many feature sets and compared with simulated data without subgroups
and with Tozzi's biotypes planted at the strength of his evidence (method:
`docs/METHODS.md`, "Split-half reproducibility across feature sets").

It starts from the extracted features in `features/` (see
[features/README.md](features/README.md)), so feature extraction, the raw
imaging data and the clinical database are not needed: a clone of this
repository runs it. The feature tables are participant-level data tracked
in Git, so share the repository only under the FOR2107 data use rules.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt -e .
.venv/bin/for2107-repro stability --config configs/stability_tozzi.yaml
# robustness of the planted strength to the assumed correlations
.venv/bin/for2107-repro planted-robustness --config configs/planted_robustness_tozzi.yaml
```

`requirements-lock.txt` pins the package versions that produced the current
results. The run takes about half an hour with 24 workers (lower
`analysis.n_jobs` in the config on smaller machines; results do not depend
on it) and writes `derivatives/for2107_tozzi_stability/`:

| Output | Content |
|---|---|
| `split_half_summary.tsv` | Per feature set and source (observed, null, planted): matched and unmatched cosine, separation, replicated clusters, ARI |
| `cross_set_summary.tsv` | Agreement between feature sets and between independent measurements |
| `tozzi_match_summary.tsv` | Similarity of the clusters to Tozzi's biotype profiles |
| `tozzi_checks_summary.tsv` | Tozzi's own split-half profile check and permutation test |
| `figures/` | `figure1_method`, `figure2_results` and detail figures, with `captions.md` |

The code is in `src/for2107_repro/stability.py` (figures:
`stability_figures.py` and `stability_overview.py`), the settings in
`configs/stability_tozzi.yaml`.
Tozzi's per-feature biotype means used for the planted reference are in
`configs/tozzi_biotype_feature_means.tsv` (digitized by
`scripts/digitize_tozzi_suppfig5.py`). The other analyses in this README
are earlier steps; they read `derivatives/` and need the feature extraction
outputs or the raw data.

## Important interpretation boundary

This is a methodological reproduction, not a numerical replication. FOR2107
does not contain Tozzi's Go-NoGo paradigm or identical emotion tasks. The
primary construct-aligned feature set therefore uses resting-state, face
matching, and affective priming. Encoding features are retained as an explicit
exploratory extension and are never relabeled as cognitive control.

The ten subjects in `data/raw_examples/` are for pipeline smoke testing only.
No cluster solution or inferential statistic from that subset should be
interpreted scientifically.

## Quick start

```bash
conda env create -f environment.yml
conda activate for2107-repro
for2107-repro preflight --config configs/smoke.yaml
for2107-repro inventory --config configs/smoke.yaml
for2107-repro run --config configs/smoke.yaml
```

All commands report timestamped progress to stderr. A full run shows the nine
pipeline stages, participant-level feature-extraction progress, and periodic
updates for the clustering null and stability iterations. The final JSON result
remains on stdout, so it can still be redirected or parsed independently. Add
`--quiet` to suppress progress messages:

```bash
for2107-repro run --config configs/external_rs.yaml --quiet
```

The isolated Phase 1 analysis retains the 14 published Tozzi resting-state
connections as separate features, evaluates candidate solutions from k=2 to
k=15, treats k=6 as confirmatory, and adds prespecified selection, repeated
split-half, leave-site-out, and nuisance-variable checks:

```bash
for2107-repro run --config configs/external_rs_phase1.yaml
```

It writes only to `derivatives/for2107_tozzi_external_rs_phase1/` and
`reports/external_rs_phase1/`; the earlier external resting-state results are
not overwritten.

For cleaned MNI-space resting-state inputs, all configured ROI time series are
extracted in one decompression pass per participant. Statistical null and
stability iterations use deterministic process-based parallelism controlled by
`analysis.n_jobs`; changing the worker count does not change seeded results.

## Independent Tozzi-method task-residual analysis

`run-tozzi` is a separate implementation path using the staged FSL FEAT AP/FM
outputs. It requires both task runs and their event files, fits task GLMs,
regresses task, six motion, and registered white-matter/CSF signals from ROI
time series, band-passes each run's residuals at 0.009–0.08 Hz, concatenates
the residual runs, and calculates the 14 published task-free edges. It also
adds clearly labeled FOR2107 adaptations: AP sad/neutral and happy/neutral,
FM faces/shapes activation contrasts, and specified PPI edge contrasts. No
Go/NoGo proxy is generated. See `docs/METHODS.md` for the method differences.

Run it independently with:

```bash
for2107-repro run-tozzi --config configs/tozzi_for2107.yaml
```

This writes to `derivatives/for2107_tozzi_method/`, separate from the existing
outputs. The initial configuration uses MDD + anxiety as the clinical cohort
and healthy controls as reference; set `selection.include_all_diagnoses: true`
to include all non-control diagnoses. The feature table, paired-run discovery,
participant audit, provenance, and HTML/PDF summary report are all generated by
this command.

When clustering is ready, `analyze-tozzi` also runs the existing paper-style
figure suite from `report.py`. SVG figures and a manifest are written under
`derivatives/for2107_tozzi_method/figures/paper/`; the complete clustering
Supplementary Figures 1–6 are written under `figures/supplementary/`.

The ten-subject raw example dataset is still only for smoke testing. The task
analysis uses the separately staged FSL FEAT task data, not fMRIPrep derivatives.

For preprocessing, install Docker Engine and obtain a free FreeSurfer license.
Current fMRIPrep requires the license even though the default ROI workflow uses
`--fs-no-reconall`. Save it as `license.txt` in the project root (the file is
ignored by Git). The pipeline prints the exact pinned command before execution:

```bash
for2107-repro fmriprep-command --config configs/smoke.yaml
```

Set `preprocessing.execute: true` only when Docker and the license are ready.
Raw inputs are mounted read-only and are never modified.

## Tozzi-faithful replication

To test as directly as FOR2107 allows whether Tozzi's results replicate,
re-extract construct-matched task features (face matching -> threat circuit,
affective priming -> sad and positive circuits) and run Tozzi's published
analysis steps against his own benchmark numbers:

```bash
for2107-repro run-tozzi --config configs/tozzi_for2107_construct.yaml
for2107-repro replicate-tozzi --config configs/replication_tozzi.yaml
```

Outputs in `derivatives/for2107_tozzi_replication/`: per analysis
(`rest_task_residual`, `rest_scan`, `construct_matched`) the structure tests,
a side-by-side table with Tozzi's values (`tozzi_benchmark.tsv`), cluster
profiles and clinical generalization; plus `reliability/` (agreement across
independent measurements) and `power/` (detection of Tozzi-strength biotypes).
Clinical comparisons use a Mann–Whitney test; Tozzi's coded one-sample
Wilcoxon test is reported alongside because it is biased for skewed scales
(see `docs/METHODS.md`). To redo only the clinical comparisons from the saved
circuit scores:

```bash
for2107-repro replicate-tozzi-clinical --config configs/replication_tozzi.yaml
```

## Features of all Tozzi networks and split-half stability

Every feature Tozzi's seven ROI networks allow with FOR2107 data (all
within-network edges as task-free and rest-scan connectivity, activation of
all 29 ROIs and PPI of all within-network pairs in each contrast), with a
catalog of how each column relates to Tozzi's 41 features:

```bash
for2107-repro run-tozzi --config configs/tozzi_for2107_all_networks.yaml
for2107-repro run --config configs/rest_all_networks.yaml
```

The feature tables, the eligible participants and the head motion of the
task runs are then exported to `features/`, which needs the clinical
database and the FEAT outputs (see [features/README.md](features/README.md)):

```bash
for2107-repro export-features --config configs/export_features.yaml
```

Split-half reproducibility of k = 6 clusters across many feature sets, set
against a cluster-free null and planted Tozzi-strength biotypes, with
cross-set, independent-measurement and Tozzi-profile comparisons, runs from
`features/` (`derivatives/for2107_tozzi_stability/`):

```bash
for2107-repro stability --config configs/stability_tozzi.yaml
```

The run also applies Tozzi's own split-half profile check and permutation
test to FOR2107 and both references (`tozzi_checks_summary.tsv`). To add
them to a finished run without repeating it:

```bash
for2107-repro tozzi-checks --config configs/stability_tozzi.yaml
```

The planted strength is recalibrated under Tozzi's fitted correlation
structure (from `calibrate-tozzi`, saved as
`configs/tozzi_calibrated_correlation.tsv`) to check that it does not depend
on FOR2107's correlations (`derivatives/for2107_tozzi_planted_robustness/`):

```bash
for2107-repro planted-robustness --config configs/planted_robustness_tozzi.yaml
```

## Normative deviation analysis

The clustering results do not support discrete biotypes, so `normative`
scores each participant's circuit features as deviations from healthy
controls instead. First re-extract the task features with activation and PPI
betas in percent signal change (connectivity features are unchanged; no
clustering is run):

```bash
for2107-repro run-tozzi --config configs/tozzi_for2107_psc.yaml
for2107-repro normative --config configs/normative_for2107.yaml
```

The normative step excludes high-motion participants (framewise displacement
from both task runs), harmonizes scanner epochs (site x body-coil,
gradient-coil and PreScan Normalize periods) with ComBat learned on controls
only, models each feature against age, sex and log mean FD in controls, and
writes out-of-fold control
and patient deviation scores, calibration and confound checks, and first
deviation summaries to `derivatives/for2107_normative/`. All non-control
diagnoses are scored; see `docs/METHODS.md`.

## Outputs

Generated files live under `derivatives/` and `reports/` (both ignored):

- deidentified inventory and eligibility tables;
- non-destructive BIDS staging metadata;
- feature, QC, harmonization, and circuit-score tables;
- clustering and clinical-validation results;
- self-contained HTML and PDF reports, plus standalone SVG result figures;
- configuration, software versions, checksums, and random seeds.

The exported features in `features/` are the one data folder tracked in Git.

See [docs/METHODS.md](docs/METHODS.md) for the reproduction contract and
[docs/INSTALL.md](docs/INSTALL.md) for fMRIPrep setup.
