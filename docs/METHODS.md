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

## Privacy

Source JSON and SPSS files may contain identifying fields. Generated tables use
BIDS IDs only. Patient name, birth date, acquisition date/time, free text, and
unmapped SPSS variables are excluded from reports and staged sidecars.
