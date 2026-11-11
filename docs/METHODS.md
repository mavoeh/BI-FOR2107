# Reproduction contract

## Sources

- Tozzi et al. (2024), DOI `10.1038/s41591-024-03057-9`.
- Kircher et al. (2018/2019), DOI `10.1007/s00406-018-0943-x`.
- Tozzi analysis and ROI repository:
  `https://github.com/leotozzi88/cluster_study_2023`.

The upstream code revision and ROI checksums are recorded by `fetch-rois` in a
generated provenance manifest.

## Primary specification

1. Use baseline (`ses-01`) only.
2. Use clinical participants with MDD and/or configured anxiety diagnoses.
3. Use all eligible baseline healthy controls as the normative reference.
4. Apply modality-specific QC before feature construction.
5. Impute imaging features within scanner using random-forest predictive mean
   matching when the full sample supports it.
6. Harmonize scanner/site effects with ComBat while retaining configured
   biological covariates.
7. Express clinical imaging features in standard-deviation units relative to
   healthy controls.
8. Cluster clinical profiles using Pearson correlation distance and
   average-linkage hierarchical clustering.
9. Evaluate candidate solutions using elbow, silhouette null simulations,
   feature permutations, leave-one-out stability, leave-20%-out stability, and
   split-half replication.
10. Characterize clusters using measures that were not clustering inputs.

## Computational execution

Cleaned, same-grid resting-state inputs are decompressed once per participant;
all configured ROI time series are extracted during that pass and then grouped
into their respective circuits. Null simulations, feature permutations,
leave-one-out checks, and participant-subsample checks may run in parallel.
Random inputs are generated in the original deterministic order and worker
outputs are reassembled in that same order, so `analysis.n_jobs` changes runtime
but not seeded numerical results.

If a participant has no variable finite signal in a required ROI, that
participant is excluded from feature and clustering tables rather than stopping
the run. The subject identifier, exclusion reason, and failing ROI are retained
in `features/extraction_audit.tsv`, and the skipped count is recorded in
`features/extraction_status.json`.

## Feature tiers

`primary_multimodal` contains construct-aligned resting-state, face-matching,
and affective-priming features. `extended_multimodal` adds encoding features as
an exploratory memory construct. Every feature carries a source task, contrast,
ROI or ROI pair, modality, and equivalence rating.

The exact feature table cannot be finalized until FOR2107 task events and
contrast documentation are available. The pipeline rejects incomplete or
ambiguous mappings.

## Exact, adapted, and unavailable elements

Exact Python equivalents include correlation distance, average linkage,
candidate cluster cuts, within-cluster distance sums, silhouette statistics,
adjusted Rand indices, and profile-correlation assignment.

Adapted elements include task contrasts, clinical measures, scanner covariates,
and the Python implementation of random-forest predictive mean matching and
ComBat. The report labels each adaptation.

Unavailable in the local smoke data are real task designs, preprocessed BOLD
data, a sufficiently large healthy reference group, and a scientifically valid
cluster sample. These are validation gates, not values to infer.

## Independent task-residual implementation

The independent `run-tozzi` workflow has its own config, feature table, audit,
analysis directory, and report root. It does not call the existing resting-state
feature extractor and does not overwrite the existing `run` or `analyze`
outputs.

The task-free features are extracted as follows:

1. Require both baseline affective-priming (AP) and face-matching (FM) FEAT
   runs, along with each run's events file, six motion parameters, MNI
   registration, and matching anatomical tissue segmentation.
2. Sample the Tozzi MNI ROI masks into the task functional grid using the FEAT
   FLIRT transforms. Tissue probability maps are sampled from the corresponding
   anatomy into the same grid.
3. Fit a run-wise first-level model with a sampled canonical SPM HRF, event
   regressors, six motion regressors, and mean CSF/white-matter signal.
   FEAT's input series already has 100-second high-pass filtering and 5-mm
   smoothing; these steps are not repeated.
4. Apply a 0.009–0.08-Hz, second-order, zero-phase Butterworth filter separately
   to each run's model residuals, concatenate AP and FM residual time series,
   then calculate 14 published Tozzi task-free ROI-pair correlations and
   Fisher z-transform them.

This follows Tozzi's task-free sequence but is not yet exact: incoming FSL
preprocessing differs from the original, SciPy's Butterworth filter differs
from FSL's implementation, and this implementation fits a common model to raw
regional mean signals rather than storing whole-brain residual images. With a
common linear design, regional-mean GLM coefficients/residuals correspond to
averaging voxel-level coefficients/residuals; this equivalence does not remove
the preprocessing and filtering differences. These distinctions are retained
in the automated report.

The task-feature additions are separate and marked `_adapted`: AP uses sad >
neutral and happy > neutral; FM uses MatchFaces > MatchShapes. PPI estimates use
the first principal component as the seed signal and average reciprocal
seed/target estimates, but omit SPM's physiological deconvolution/reconvolution.
These contrasts are not identical to Tozzi's conscious/non-conscious facial
emotion conditions. FOR2107 does not have Go/NoGo, so no cognitive-control
proxy enters the feature matrix. The current configuration has 14 task-free
features and 17 task-specific adapted features; it is therefore not Tozzi's
original 41-feature matrix.

The corrected primary profile is available as
`configs/tozzi_for2107_rest_ap.yaml` and writes to
`derivatives/for2107_tozzi_rest_ap`. It keeps the 14 resting-state edge
features, adds only the closest affective-priming analogues (sad > neutral in
the negative-affect circuit and happy > neutral in the positive circuit), and
excludes face-matching task features because the available event files do not
identify face emotions. This profile contains 26 features (14 exact
resting-state edges plus 12 AP analogues). The AP features remain analogues,
not exact Tozzi task replications, because the FOR2107 paradigm is subliminal
affective priming rather than Tozzi's conscious/non-conscious threat and sad
face contrasts. The machine-readable rationale is saved as
`features/feature_selection.tsv`.

When selected diagnosis and control sample sizes meet the configured gates,
the command applies the repository's established harmonization, control
referencing, correlation-distance/average-linkage clustering, and validation
steps to this feature table. Such results remain an adapted analysis and must
not be presented as an exact replication.

## Percent-signal-change task features

`tozzi.signal_scaling: percent_signal_change` divides each ROI's activation
contrast and PPI interaction beta by that ROI's temporal mean signal in the
run. FEAT (FSL 6.0.7) grand-mean scales each run and re-adds the temporal mean
after high-pass filtering, so this mean is the regional baseline. Residuals
are left unscaled, so the 14 task-free connectivity features are identical to
the raw configuration. `configs/tozzi_for2107_psc.yaml` writes these features
to `derivatives/for2107_tozzi_psc` and skips clustering
(`analysis.run_clustering: false`).

## Normative deviation modelling

`for2107-repro normative` (`configs/normative_for2107.yaml`) is the
dimensional alternative to clustering:

1. Cohort: healthy controls as defined by the variable map; all diagnosis
   groups as patients. FOR2107 drop-outs (`DO_Klinisch_final`,
   `DO_Medikation_fMRT`) and participants without age, sex or scanner epoch
   are excluded and audited.
2. Scanner epochs: site x hardware/sequence period, assigned from the database
   dummies (Marburg body-coil exchange June 2016 and gradient-coil exchange
   August 2018; Münster PreScan Normalize from August 2016). Each participant
   must match exactly one configured rule.
3. Head motion: Power framewise displacement (FD; rotations on a 50-mm
   sphere) from the MCFLIRT parameters of both FEAT task runs (AP, FM).
   Participants are excluded if either run has mean FD > 0.5 mm or more than
   20% of volumes with FD > 0.5 mm (`cohort/motion.tsv`); log mean FD over
   both runs enters the models below as a covariate. Exclusion rates differ
   by group (about 5–8% of controls and MDD/anxiety, 26–29% of bipolar and
   psychotic disorders), which is a selection caveat for those groups.
4. Harmonization: ComBat (neuroHarmonize, empirical Bayes) with scanner epochs
   as batches, learned on controls only with age, age², sex and log mean FD
   protected, then applied unchanged to patients. The smallest epoch (Marburg
   after the 2018 gradient-coil exchange) keeps 19 controls after motion
   exclusion; the minimum is set to 15 per epoch.
5. Normative model per feature, fitted in controls: mean = cubic B-spline of
   age (quantile knots) + sex + log mean FD, by weighted least squares;
   log-variance = same design, by a gamma GLM on squared residuals
   (iterated); predictive variance includes coefficient uncertainty. A sinh-arcsinh distribution
   (Jones & Pewsey) fitted to the standardized residuals corrects skew and
   heavy tails, so deviations are normal quantiles of the control
   distribution.
6. Controls are scored out of fold (stratified 10-fold by scanner epoch, with
   ComBat re-learned per fold); patients are scored with models fitted on all
   controls. Calibration (`model/calibration.tsv`) and age/motion/epoch
   confound checks (`model/confound_checks.tsv`) use the out-of-fold control
   scores.
7. Summaries: extreme deviations (|z| > 2.6) per feature and direction versus
   controls (Fisher exact, FDR within group), per-participant deviation burden
   (number of extreme deviations, mean squared deviation), overlap of extreme
   deviations across individuals, and Spearman associations of burden and
   feature deviations with clinical measures in the MDD/anxiety group (FDR
   across all tests).

## Tozzi-faithful replication

`for2107-repro replicate-tozzi` (`configs/replication_tozzi.yaml`) follows
Tozzi's published R code (leotozzi88/cluster_study_2023, `analysis_main`)
step by step and compares FOR2107 with Tozzi's own numbers for the same
feature subset where those were published:

1. Features: `configs/tozzi_for2107_construct.yaml` maps each FOR2107 task to
   the Tozzi circuit with the closest construct, on Tozzi's ROIs. Face
   matching (fearful/angry faces; Kircher et al. 2018) feeds the conscious
   threat circuit (amygdala, dACC; 3 activations + 2 PPIs), subliminal
   affective-priming sad > neutral the sad circuit (5 + 4), and happy >
   neutral the positive circuit (3). With the 14 task-free edges these are 31
   of Tozzi's 41 features; nonconscious threat and Go/NoGo cognitive control
   are unavailable. Activation and PPI betas are in percent signal change; both
   PPI directions are kept (`ppi_directions: separate`), as are single-run
   task-free edges (`run_connectivity: true`).
2. Task-free connectivity is computed, as in Tozzi, from band-passed task-GLM
   residuals concatenated across tasks (here AP and FM). The dedicated
   resting-state scan is analysed separately as a robustness check; no
   realignment parameters are staged for it, so it has no motion exclusion.
3. Motion: a scan's features are set to missing when more than 25% of its
   frames have FD > 0.3 mm (Tozzi's `fd03` rule); for the task-free edges the
   fraction is pooled over both runs. Participants with no remaining features
   are dropped; others are imputed.
4. Random-forest predictive-mean-matching imputation within scanner (Tozzi:
   `miceRanger`, m = 1), ComBat on scanner (site) without covariates, and
   z-scores relative to all healthy controls. PPI directions are averaged
   after standardization.
5. Correlation-distance, average-linkage clustering of patients (MDD, MDD +
   anxiety, anxiety). Elbow, silhouette versus multivariate-normal
   simulations and feature permutations (10,000 each; Tozzi's p is the plain
   exceedance fraction), leave-one-out and leave-20%-out ARI (10,000).
   The rest-only analyses test k = 2 and 6, as Tozzi's rest-only analysis
   did; the construct-matched analysis tests k = 2–15.
6. Clusters are named Tozzi-style from circuit-level means of at least
   0.5 SD. Clinical differences compare each cluster with the other patients
   on the closest FOR2107 measures (see the config) using a Mann–Whitney U
   test (rank-biserial r), as described in Tozzi's methods. Tozzi's code
   instead runs a one-sample Wilcoxon test of the cluster against the other
   patients' median (r = z / √N); that test assesses symmetry about the
   median, so it rejects for skewed or zero-inflated scales without any group
   difference (in FOR2107, under shuffled labels: SHAPS 87%, HAMD suicide item
   100%). It is reported alongside (`clinical_k*_tozzi_wilcoxon.tsv`), with
   both tests' false-positive rates under shuffled labels
   (`clinical_test_calibration_k*.tsv`). `replicate-tozzi-clinical` redoes
   only these comparisons from saved circuit scores. A difference generalizes when it
   is significant in the whole sample and in both parts of a random split-half
   or of a leave-one-site-out split (derivation part in the same direction,
   held-out part one-sided), with held-out participants assigned to the
   derivation centroids by profile correlation, as in Tozzi's code.
7. Reliability: agreement of the 14 task-free edges between the resting-state
   scan and the task residuals, and between the AP and FM runs (patients
   without imputed values): per-edge r and ICC(A,1), within-person profile
   correlation, independent-clustering ARI, and test–retest kappa of
   nearest-centroid biotype assignment.
8. Power: Tozzi's six biotypes are planted with circuit-level means digitized
   from his Fig. 3 (±0.05 SD), his biotype sizes, and FOR2107 control feature
   correlations (unavailable circuits get the mean within-circuit task
   correlation). Because cluster means are inflated by selection, the effect
   is scaled so that Tozzi's design (41 features, n = 801, k = 6) reproduces
   his multinormal-null p = .016; the as-published scale is reported as an
   optimistic bound. Detection rate (p < .05 against the multinormal null of
   the population mixture) and recovery of true labels are then estimated for
   FOR2107's feature sets and sample sizes, and for Tozzi's rest-only design
   as a check against his published rest-only result.

Remaining differences: FSL FEAT rather than fMRIPrep preprocessing; PPI without
SPM deconvolution; FOR2107 contrasts against neutral faces (AP) or shapes (FM)
rather than Tozzi's threat/sad/happy versus neutral conscious faces; mostly
medicated patients; clinical measures are analogues except SHAPS.

## Features of all Tozzi networks

`configs/tozzi_for2107_all_networks.yaml` (task runs) and
`configs/rest_all_networks.yaml` (rest scan) extract every feature that
Tozzi's 29 published ROIs allow with FOR2107 data. The ROIs form seven
networks by file prefix (Attention 7, Default mode 4, Salience 4, Negative
affect sad 5, Negative affect threat 3, Positive affect 3, Cognitive control
3); every within-network pair gives 52 edges
(`configs/feature_map_all_networks.yaml`), of which Tozzi's 14 published
task-free edges keep their names.

- Task-free connectivity of the 52 pairs from the band-passed task-GLM
  residuals of AP + FM (Tozzi's definition) and of each run alone.
- Activation of all 29 ROIs in AP sad > neutral, AP happy > neutral and FM
  faces > shapes (percent signal change).
- PPI of the 52 pairs in the same three contrasts, both directions.
- Rest-scan connectivity of the 52 pairs.

`feature_selection.tsv` catalogs each task column: network, runs, contrast,
ROIs, PPI direction and its relation to Tozzi's 41 features (taken from his
`analysis_main/01_merge_data_qc.R`): `published_edge` and
`closest_task_analogue` (his feature, in the contrast configured in
`tozzi.construct_matches` to stand in for its circuit; `tozzi_circuit` names
the circuit), `tozzi_rois_other_contrast`, or an extension. The sad- and
threat-circuit amygdala masks are the same file, so 15 task columns (and one
rest edge) repeat another; `duplicate_of` names the copy that is kept, which
is the Tozzi feature where one of the two is (e.g. the threat-circuit
amygdala in face matching). Of Tozzi's 41 features, 31
are construct-matched; nonconscious threat (sgACC, not among the published
masks; no threat prime) and Go/NoGo cognitive control are unavailable.

## Split-half reproducibility across feature sets

`for2107-repro stability --config configs/stability_tozzi.yaml` asks how
reproducible k = 6 clusters are across random halves of the patients and
across feature choices, calibrated against what no clusters and Tozzi-strength
biotypes would give. It reads the exported feature folder `features/`
(`export-features`, `configs/export_features.yaml`; `features/README.md`):
the eligible participants from the clinical database, the head motion of each
task run, and the task and rest-scan feature tables above.

1. Scoring as in the replication (motion masking, random-forest imputation
   within site, ComBat, control z-scores, PPI directions averaged), separately
   per measurement block so that no imputation crosses blocks: `main` (AP + FM
   edges, activations, PPIs; duplicates dropped; 285 features), and the edges
   of the AP run, the FM run and the rest scan alone.
2. Feature sets: all features; Tozzi's 31 construct-matched features; his 14
   task-free edges; all edges; extensions only; leave one network, modality or
   contrast out; Tozzi's set without one circuit; and random subsets of the
   Tozzi and full pools with 10–75% of features left out.
3. Split-half: per set, 100 random 50/50 splits of the patients; each half is
   clustered (correlation distance, average linkage, k = 6), and the halves'
   centroids are compared by cosine after Hungarian matching. Rows are ordered
   by cluster size, so matched pairs form the diagonal. Summaries: mean matched
   and unmatched cosine and their difference (separation; also after removing
   each half's mean profile), clusters replicated at cosine >= 0.8 with at least
   10 members each, and the ARI between one half's clustering and assignment
   to the other half's centroids (Tozzi's split-half rule).
4. References: the same splits on data simulated with the observed size, (a)
   multivariate normal with the patients' mean and covariance (no clusters)
   and (b) Tozzi's six biotypes, each simulated patient drawn into one
   biotype in his proportions and given that biotype's mean on each of his
   features, with the controls' covariance as within-biotype noise. The
   per-feature means are measured from the bars of his Supplementary Figure 5
   (`scripts/digitize_tozzi_suppfig5.py` ->
   `configs/tozzi_biotype_feature_means.tsv`; about +-0.02 SD; their circuit
   means agree with his Fig. 3 to 0.024 SD on average). FOR2107 features are
   matched to his codes as in Supplementary Table 18 (`planted.feature_codes`);
   features outside his definitions get no biotype effect. Because cluster
   means are inflated by selection, the means are scaled so that his own
   design (41 features, n = 801, k = 6, his unavailable features filled with
   the mean within-circuit correlation) reproduces his multinormal-null
   p = .016 (`planted_calibration.tsv`). `planted.level: circuit` instead uses
   one Fig. 3 mean per circuit with the fixed scale 1.112.
   `position` places the observed median between (a) = 0 and (b) = 1; it is
   reported only where a single split tells (b) from (a) with AUC >= 0.75
   (`*_auc`), because otherwise the two references overlap.
   Matching and cutting a continuous distribution along its main axes both
   produce reproducible-looking centroids, so only the comparison with (a) and
   (b) is interpretable.
5. Cross-set: the same patients clustered with two feature sets; ARI of the
   labels and matched cosine of the centroids on the shared features
   (configured pairs and random pairs). Independent measurements of the same
   edges (AP vs FM run; AP + FM residuals, AP and FM each vs rest scan;
   patients without imputed values) are compared the same way. Each observed
   value is set against 100 null and 100 planted draws (one-sided p).
6. Tozzi profiles: per set with at least three of his circuits, the cosine
   between the six clusters' circuit-score profiles and his six biotype
   profiles after matching, against the same references.
7. Tozzi's own checks on his 31 features (`stability.tozzi_checks`;
   `for2107-repro tozzi-checks` adds them to a finished run with its saved
   scale). His split-half profile check (Supplementary Figure 6) clusters one
   random half, assigns the other half to its centroids by profile
   correlation and correlates each cluster's mean profile between the halves;
   it runs on 100 splits of the patients and on 100 draws of each reference of
   their size (`tozzi_split_half_r.tsv`). His permutation test of the k = 6
   silhouette shuffles each feature on its own (100 shuffles; p is his plain
   exceedance share); it runs on the patients and on 100 datasets of each
   reference at his n = 801 (`tozzi_permutation.tsv`). Median profile r is
   0.91 for FOR2107 and the null and 0.92 for planted biotypes (his 0.91);
   the permutation test rejects (p < .05) for FOR2107, 97 of 100 null and all
   planted datasets (`tozzi_checks_summary.tsv`). Neither check separates the
   references, so neither is evidence for subgroups.

The planted reference assumes that features outside Tozzi's definitions carry
no biotype signal.

The scale depends on the assumed correlations of his 41 features, which are
not published within the task-free and task blocks. `planted-robustness`
(`configs/planted_robustness_tozzi.yaml` ->
`derivatives/for2107_tozzi_planted_robustness/`) repeats the calibration on
one grid with the same random numbers for three structures: FOR2107 control
correlations as within-biotype noise (scale 1.05; the stability run's own
calibration gave 1.07), the correlations fitted to his null silhouette curve
(`calibrate-tozzi`; saved as `configs/tozzi_calibrated_correlation.tsv`) as
within-biotype noise (1.35), and the same correlations
as the total of all his patients, with the biotypes carved out of it (1.05;
the total then moves by at most 0.07). No structure calls for weaker planted
biotypes, so the planted reference is, if anything, conservative. The scale
is interpolated on log p at the first grid point whose median p reaches .016;
beyond it the p-value sits at its floor of 1 / (null draws + 1).

## Privacy

Source JSON and SPSS files may contain identifying fields. Generated tables use
BIDS IDs only. Patient name, birth date, acquisition date/time, free text, and
unmapped SPSS variables are excluded from reports and staged sidecars.
