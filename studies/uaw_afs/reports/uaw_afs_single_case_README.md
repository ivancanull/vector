# UAW-AFS Single-Case VF Reconstruction Experiment

This experiment records the single-case check for reconstructing a full
Touchstone response from a 20-point UAW-AFS HFSS run using vector fitting.

## Summary

The original full-wave `.s2p` response was compared against a vector-fitting
reconstruction fit from the 20 explicit AFS frequency rows. The reconstructed
response was evaluated on the original 401-point full-wave frequency grid.

| Field | Value |
| --- | --- |
| Job ID | `uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20` |
| Full-wave frequency rows | 401 |
| AFS sampled frequency rows | 20 |
| Reconstructed frequency rows | 401 |
| BCLF objective, original | -1.0923095497 |
| BCLF objective, reconstructed | -1.0904454013 |
| BCLF objective absolute error | 0.0018641484 |
| Maximum channel dB error | 0.0184603548 dB |

Conclusion: the vector-fitting reconstruction is visually indistinguishable
from the original full-wave response at this scale. The largest observed dB
error is about 0.0185 dB.

## Inputs And Artifacts

| Artifact | Path |
| --- | --- |
| Notebook | [`../../../notebooks/uaw_afs_frequency_points_contract.ipynb`](../../../notebooks/uaw_afs_frequency_points_contract.ipynb) |
| Source full-wave Touchstone | [`../../../data/bandpass_coupled_line_filter_export/ext/Bandpass_Coupled_Line_Filter_export_Bandpass_Coupled_Line_Filter_L_7d88_L1_8d0_L2_7d88_W_1d07_W1_0d9_W2_1d06_S1_0d14_S2_0d56.s2p`](../../../data/bandpass_coupled_line_filter_export/ext/Bandpass_Coupled_Line_Filter_export_Bandpass_Coupled_Line_Filter_L_7d88_L1_8d0_L2_7d88_W_1d07_W1_0d9_W2_1d06_S1_0d14_S2_0d56.s2p) |
| 20-point AFS Touchstone | [`../../../data/bandpass_coupled_line_filter_export/ext/afs/uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20.s2p`](../../../data/bandpass_coupled_line_filter_export/ext/afs/uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20.s2p) |
| AFS metadata | [`../../../data/bandpass_coupled_line_filter_export/ext/afs/uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20.metadata.json`](../../../data/bandpass_coupled_line_filter_export/ext/afs/uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20.metadata.json) |
| Reconstruction summary | [`uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20_reconstruction_summary.json`](../../../experiments/uaw_afs_single_case/uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20_reconstruction_summary.json) |
| Reconstruction metrics | [`uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20_reconstruction_metrics.csv`](../../../experiments/uaw_afs_single_case/uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20_reconstruction_metrics.csv) |
| VF fit status | [`uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20_vf_fit_status.csv`](../../../experiments/uaw_afs_single_case/uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20_vf_fit_status.csv) |
| Reconstructed Touchstone | [`uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20_vf_reconstructed.s2p`](../../../experiments/uaw_afs_single_case/uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20_vf_reconstructed.s2p) |

Sampled frequencies in GHz:

```text
3.578, 3.695, 3.877, 4.384, 4.839, 4.943, 5.034, 6.035, 6.061, 6.308,
6.399, 6.607, 6.659, 6.763, 6.789, 7.244, 7.621, 8.141, 8.362, 8.479
```

## Method

Each S-parameter channel was fit independently:

| Setting | Value |
| --- | --- |
| Channels | `S11`, `S21`, `S12`, `S22` |
| Real poles | 4 |
| Complex pole pairs | 8 |
| Maximum iterations | 30 |
| Tolerance | `1e-7` |
| Evaluation grid | Original full-wave frequency grid |

## Reconstruction Metrics

| Channel | dB MAE | dB RMSE | Max dB error | Complex MAE | Complex RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| S11 | 0.0007258594 | 0.0018582074 | 0.0184603548 | 9.6287788936e-05 | 0.0002123312 |
| S21 | 0.0028053439 | 0.0058117994 | 0.0175453263 | 1.3140669900e-05 | 2.5438614885e-05 |
| S12 | 0.0028051407 | 0.0058114852 | 0.0175446158 | 1.3140630570e-05 | 2.5438608680e-05 |
| S22 | 0.0005632836 | 0.0012043172 | 0.0039534653 | 9.2119962705e-05 | 0.0002033620 |

## Plots

![Original S11/S21 versus vector-fitting reconstruction](../figures/uaw_afs_single_case/plots/uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20_s11_s21_original_vs_vf.png)

![Absolute dB reconstruction error for all channels](../figures/uaw_afs_single_case/plots/uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20_all_channels_db_error.png)
