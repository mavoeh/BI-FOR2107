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

The smoke dataset currently has no task `events.tsv` files or fMRIPrep
derivatives. Inventory and validation will report these limitations rather than
inventing task designs. Synthetic tests cover the unavailable task-analysis
path until event files arrive.

For preprocessing, install Docker Engine and obtain a free FreeSurfer license.
Current fMRIPrep requires the license even though the default ROI workflow uses
`--fs-no-reconall`. Save it as `license.txt` in the project root (the file is
ignored by Git). The pipeline prints the exact pinned command before execution:

```bash
for2107-repro fmriprep-command --config configs/smoke.yaml
```

Set `preprocessing.execute: true` only when Docker and the license are ready.
Raw inputs are mounted read-only and are never modified.

## Outputs

Generated files live under `derivatives/` and `reports/` (both ignored):

- deidentified inventory and eligibility tables;
- non-destructive BIDS staging metadata;
- feature, QC, harmonization, and circuit-score tables;
- clustering and clinical-validation results;
- self-contained HTML and PDF reports, plus standalone SVG result figures;
- configuration, software versions, checksums, and random seeds.

See [docs/METHODS.md](docs/METHODS.md) for the reproduction contract and
[docs/INSTALL.md](docs/INSTALL.md) for fMRIPrep setup.
