# Extracted features

This folder is the starting point of the split-half stability analysis. It
holds everything that analysis takes from feature extraction: participants,
head motion, and the task and rest-scan features. Everything after this
point runs from a clone of the repository, without the clinical database,
the FEAT task outputs or the preprocessed rest scans.

The tables are participant-level FOR2107 data (pseudonymized IDs, diagnosis,
age, sex) tracked in Git with the code, so share the repository only under
the FOR2107 data use rules. To check that the tables are unchanged:

```bash
cd features && sha256sum -c SHA256SUMS
```

## Files

| File | Rows | Content |
|---|---|---|
| `participants.tsv` | 2555 | Eligible participants with task or rest features |
| `task_motion.tsv` | 2555 | Head motion of the AP and FM task runs |
| `task_features.tsv` | 2295 | Task-run features (555 columns) |
| `task_feature_catalog.tsv` | 555 | One row per task feature column |
| `rest_features.tsv` | 2765 | Rest-scan connectivity (52 edges) |
| `provenance.json` | | Export config, software versions, checksums of the sources |
| `SHA256SUMS` | | Checksums of the five tables |

All tables are tab-separated with one row per participant (`subject_id`,
e.g. `sub-0001`), except the catalog.

### `participants.tsv`

Participants in the clinical database (`ses-01`) who pass its exclusion
flags (`DO_Klinisch_final`, `DO_Medikation_fMRT`) and have age, sex and a
known scanner epoch, restricted to those with task or rest features.

| Column | Values |
|---|---|
| `analysis_role` | `healthy_reference` (controls) or `clinical` |
| `diagnosis_group` | `HC` 1123, `MDD` 781, `MDD_anxiety` 247, `anxiety` 19, `BD` 176, `SZ` 113, `SZA` 62, `other` 34 |
| `site` | `Marburg`, `Münster` |
| `scanner_epoch` | Five epochs from site, body coil, gradient coil and PreScan Normalize |
| `age` | Years |
| `female` | 1 = female, 0 = male |

The stability analysis keeps controls and the groups in
`stability.clinical_groups` (MDD, MDD_anxiety, anxiety). Other diagnoses are
included so they can be added without a new export.

### `task_motion.tsv`

Framewise displacement (Power et al., 2012) from the FSL MCFLIRT realignment
parameters of each task run (`AP` affective priming, `FM` face matching):

| Column | Meaning |
|---|---|
| `mean_fd_<run>` | Mean framewise displacement (mm) |
| `high_motion_pct_<run>` | % of frames with framewise displacement above `fd_threshold_mm` (0.3 mm) |
| `n_frames_<run>` | Number of frames |
| `mean_fd` | Mean of the two runs |

The analysis sets a scan's features to missing when more than 25% of its
frames moved (`stability.motion.max_fraction`), pooled over both runs for
the task-free edges. Every participant with task features has both runs
here; the others (rest scan only) have NaN. A different threshold needs a
new export (below).

### `task_features.tsv` and `task_feature_catalog.tsv`

The task table holds every feature that Tozzi's 29 published ROIs allow
with the FOR2107 task runs. The ROIs form seven networks (Attention 7, Default mode 4, Salience 4,
Negative affect sad 5, Negative affect threat 3, Positive affect 3,
Cognitive control 3), and their within-network pairs give 52 edges:

- task-free connectivity of the 52 edges from the band-passed task-GLM
  residuals of AP + FM (Tozzi's definition), and of each run alone (suffix
  `__AP`, `__FM`);
- activation of all 29 ROIs in AP sad > neutral, AP happy > neutral and FM
  faces > shapes;
- PPI of the 52 edges in the same three contrasts, both directions.

`n_ap_volumes` and `n_fm_volumes` give the number of volumes of each run
(182 and 187). The catalog describes each feature column:

| Column | Meaning |
|---|---|
| `feature` | Column name in `task_features.tsv` |
| `network`, `rois` | ROI network and ROIs |
| `source_task` | `AP+FM` (concatenated residuals), `AP` or `FM` |
| `contrast`, `modality` | Task contrast; `connectivity`, `activation` or `PPI` |
| `physiological_roi` | PPI direction (seed ROI) |
| `equivalence` | Relation to Tozzi's 41 features: `published_edge`, `closest_task_analogue`, `tozzi_rois_other_contrast`, or an extension |
| `tozzi_circuit` | Tozzi circuit the feature stands in for, else empty |
| `duplicate_of` | The sad- and threat-circuit amygdala masks are the same file, so 15 columns repeat another; this names the copy that is kept |

Without duplicates, the analysis's `main` block has 438 columns (task-free
edges and AP and FM activations and PPIs), which become 285 features once
the two PPI directions are averaged. The edges of each run form their own
blocks.

### `rest_features.tsv`

Connectivity of the same 52 edges in the resting-state scan, from the
cleaned MNI-space rest data (`data/preproc/rs`). One edge repeats another
(the shared amygdala mask) and is dropped by the analysis. The motion
columns (`high_motion_fraction`, `qc_pass`) are empty because no motion
parameters were available for the cleaned rest data
(`motion_qc_available` is false), so the rest edges are not motion-masked.

## Running the analysis

`configs/stability_tozzi.yaml` reads this folder (`stability.feature_dir`):

```bash
for2107-repro stability --config configs/stability_tozzi.yaml
```

This scores the features (motion masking, imputation, ComBat on site,
z-scores to controls), runs the split-half analysis with the null and
planted references and Tozzi's own checks, and writes
`derivatives/for2107_tozzi_stability/` with its figures. It takes about
half an hour with 24 workers (`analysis.n_jobs`; results do not depend on
it), half of it scoring. Two related commands:

```bash
# Tozzi's split-half profile check and permutation test on a finished run
for2107-repro tozzi-checks --config configs/stability_tozzi.yaml
# Redraw the figures of a finished run
for2107-repro stability-figures --config configs/stability_tozzi.yaml
```

The planted-strength robustness check also reads this folder:

```bash
for2107-repro planted-robustness --config configs/planted_robustness_tozzi.yaml
```

It also uses the correlations of Tozzi's features fitted by `calibrate-tozzi`,
saved as `configs/tozzi_calibrated_correlation.tsv`. Refitting them needs
the circuit scores of the replication run
(`derivatives/for2107_tozzi_replication/`), which is not part of this folder.

## How the folder was made

The feature tables are byte-for-byte copies of the extraction outputs:

```bash
for2107-repro run-tozzi --config configs/tozzi_for2107_all_networks.yaml  # task features
for2107-repro run --config configs/rest_all_networks.yaml                 # rest features
for2107-repro export-features --config configs/export_features.yaml       # this folder
```

The export adds `participants.tsv` from the clinical database and
`task_motion.tsv` from the FEAT realignment parameters, and needs both
(`data/`). See `docs/METHODS.md` ("Features of all Tozzi networks",
"Split-half reproducibility across feature sets") for the methods.
