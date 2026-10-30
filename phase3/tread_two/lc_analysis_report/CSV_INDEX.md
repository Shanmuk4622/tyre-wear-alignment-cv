# Derived CSV inventory

All tables below are new post-hoc derivatives of saved NB11 evidence. The original raw tables remain in their original fit namespace. Repeated draws and probes do not increase independent tyre count.

| Table | Rows | Columns |
|---|---:|---:|
| [prediction_audit.csv](prediction_audit.csv) | 53,342 | 52 |
| [cohort_metrics.csv](cohort_metrics.csv) | 256 | 15 |
| [class_metrics.csv](class_metrics.csv) | 582 | 11 |
| [confusion.csv](confusion.csv) | 1,746 | 7 |
| [selection_audit.csv](selection_audit.csv) | 60 | 10 |
| [endpoint_tyre_metrics.csv](endpoint_tyre_metrics.csv) | 28 | 16 |
| [probe_pairs.csv](probe_pairs.csv) | 1,236 | 20 |
| [probe_summary.csv](probe_summary.csv) | 12 | 7 |
| [query_exposure.csv](query_exposure.csv) | 386 | 11 |
| [sampling_groups.csv](sampling_groups.csv) | 11 | 6 |
| [source_registry.csv](source_registry.csv) | 570 | 13 |
| [gradient_parameters.csv](gradient_parameters.csv) | 22 | 5 |
| [optimizer_epochs.csv](optimizer_epochs.csv) | 60 | 10 |
| [pooled_reference_affinities.csv](pooled_reference_affinities.csv) | 25,956 | 13 |
| [reference_contributions.csv](reference_contributions.csv) | 77,868 | 13 |
| [logit_reconstruction.csv](logit_reconstruction.csv) | 4,326 | 10 |
| [endpoint_coefficients.csv](endpoint_coefficients.csv) | 108 | 8 |
| [bank_embedding_cosines.csv](bank_embedding_cosines.csv) | 648 | 6 |
| [patch_concentration.csv](patch_concentration.csv) | 1,442 | 13 |
| [endpoint_patch_evidence.csv](endpoint_patch_evidence.csv) | 138,432 | 11 |
| [endpoint_patch_support.csv](endpoint_patch_support.csv) | 46,144 | 11 |
| [nearest_reference.csv](nearest_reference.csv) | 1,442 | 13 |
| [reference_class_affinities.csv](reference_class_affinities.csv) | 84 | 8 |
| [occlusion_selection_check.csv](occlusion_selection_check.csv) | 206 | 22 |
| [dense_region_by_role.csv](dense_region_by_role.csv) | 6 | 6 |
| [dense_points_by_role.csv](dense_points_by_role.csv) | 7 | 6 |
| [endpoint_calibration.csv](endpoint_calibration.csv) | 20 | 5 |
| [confidence_risk.csv](confidence_risk.csv) | 42 | 6 |
| [paired_loss_audit.csv](paired_loss_audit.csv) | 23,520 | 17 |
| [paired_epoch_metrics.csv](paired_epoch_metrics.csv) | 60 | 8 |
| [paired_class_metrics.csv](paired_class_metrics.csv) | 180 | 8 |
| [paired_patch_metrics.csv](paired_patch_metrics.csv) | 32 | 5 |
| [appearance_factor_epochs.csv](appearance_factor_epochs.csv) | 120 | 11 |
| [reference_permission_summary.csv](reference_permission_summary.csv) | 33 | 8 |
| [query_draw_exposure.csv](query_draw_exposure.csv) | 386 | 12 |
| [source_group_draws.csv](source_group_draws.csv) | 11 | 8 |
| [resource_epochs.csv](resource_epochs.csv) | 60 | 7 |
| [v1_r2_comparison.csv](v1_r2_comparison.csv) | 14 | 20 |
| [input_inventory.csv](input_inventory.csv) | 696 | 3 |
| [table_dictionary.csv](table_dictionary.csv) | 441 | 5 |

[Field dictionary](table_dictionary.csv) · [Calculation methods](METHODS.md) · [Interactive image/probe explorer](ATLAS_Failure_Explorer.html).

All source hashes are retained in input_inventory.csv. Empty fields are unavailable/not applicable, especially absent mid-class target recall. Source masks/points are frozen predictions, not new physical wear measurements.
