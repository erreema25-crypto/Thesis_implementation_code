"""Create the thesis notebook with concise cells and reproducible code."""

from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parent
METRICS = pd.read_csv(ROOT / "outputs/tables/final_model_metrics.csv").set_index("Model")
PER_CLASS = pd.read_csv(ROOT / "outputs/tables/per_class_metrics.csv")
SUBTYPE = pd.read_csv(ROOT / "outputs/tables/proposed_attack_subtype_recall.csv").set_index("Attack subtype")
ROBUSTNESS = pd.read_csv(ROOT / "outputs/tables/training_seed_robustness_summary.csv").set_index("Model")
PROPOSED = METRICS.loc["CNN-BiLSTM with EKF"]
ABLATION = METRICS.loc["CNN-BiLSTM without EKF"]
PROPOSED_CLASS = PER_CLASS[PER_CLASS["Model"] == "CNN-BiLSTM with EKF"].set_index("Class")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        f"""
# EKF Residual-Enhanced CNN–BiLSTM for Explainable GNSS Spoofing and Jamming Detection

**Thesis experiment notebook — reproducible, source-group-disjoint, and cost-sensitive**

## tl;dr

On the final held-out source groups, the proposed **CNN–BiLSTM with EKF residuals** achieved
**{PROPOSED['Accuracy']:.1%} accuracy**, **{PROPOSED['Macro F1']:.3f} macro-F1**, and
**{PROPOSED['Macro Recall']:.3f} macro recall**. Per-class recall was
**{PROPOSED_CLASS.loc['Normal', 'Recall']:.1%} Normal**,
**{PROPOSED_CLASS.loc['Spoofing', 'Recall']:.1%} Spoofing**, and
**{PROPOSED_CLASS.loc['Jamming', 'Recall']:.1%} Jamming**.

The controlled EKF ablation is substantial: removing EKF features reduced accuracy from
{PROPOSED['Accuracy']:.1%} to {ABLATION['Accuracy']:.1%} and macro-F1 from
{PROPOSED['Macro F1']:.3f} to {ABLATION['Macro F1']:.3f}. Random Forest remains the
strongest aggregate model in this Kaggle-conditioned semi-synthetic experiment ({METRICS.loc['Random Forest', 'Accuracy']:.1%}
accuracy), so the deep model is not presented as universally superior.

The principal unresolved weakness is stealthy gradual drift: proposed-model recall is
{SUBTYPE.loc['spoof_drift', 'Recall']:.1%}, compared with
{SUBTYPE.loc['spoof_jump', 'Recall']:.1%} for sudden spoofing jumps. Results describe
controlled injected attacks and do **not** establish field performance. Across five neural
training seeds, the proposed model achieved {ROBUSTNESS.loc['CNN-BiLSTM with EKF', 'Macro_F1_mean']:.3f}
mean macro-F1 (SD {ROBUSTNESS.loc['CNN-BiLSTM with EKF', 'Macro_F1_std']:.3f}).
"""
    ),
    md(
        """
## Context & methods

### Research question

Does adding EKF innovation/residual information improve balanced three-class detection
relative to an otherwise identical CNN–BiLSTM that receives only navigation/IMU features?

### Evaluation safeguards

- The unit of independence is a **Kaggle source segment**. Its matched Normal,
  Spoofing, and Jamming variants are kept together in exactly one split.
- Scaling is fit on training groups only. Early stopping and class-decision costs use validation groups only.
- The test set is held out until final evaluation; overlapping windows are never treated as independent for confidence intervals.
- The primary selection criteria are macro-F1, macro recall, and attack-class recall—not accuracy alone.
- Source groups are assigned approximately 70/15/15 to train/validation/test.
- The experiment uses five neural training seeds for the primary EKF ablation.

### Key assumptions

The supplied Kaggle rows are used directly as the source of ordered speed, IMU, gyro, and
altitude patterns. Because their consecutive latitude/longitude fixes are not physically
coherent, those patterns condition a local kinematic carrier rather than being treated as
an unmodified flight path. Synthetic attacks are then inserted into the GNSS channel of
each Kaggle-derived carrier. The result is explicitly **semi-synthetic**, not field attack data.
"""
    ),
    code(
        """
from pathlib import Path
import json, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch
from IPython.display import Image, display

import gnss_thesis_pipeline as gp

cfg = gp.ExperimentConfig()
gp.seed_everything(cfg.seed)
gp.configure_style()
paths = gp.ensure_output_dirs(cfg.output_dir)
device = gp.select_device()
print(f"Python {platform.python_version()} | PyTorch {torch.__version__} | device={device}")
print(f"Window: {cfg.window_s:.1f} s ({cfg.window_length} samples); stride: {cfg.stride_s:.1f} s")
"""
    ),
    md("## Data\n\n### 1. Audit the supplied source CSV"),
    code(
        """
source_df, source_audit = gp.audit_source_csv("uav_navigation_dataset.csv")
gp.save_table(source_audit, paths["tables"] / "source_data_quality_audit.csv")
display(source_audit.style.format({"value": "{:.3f}"}))
display(source_df.head(3))
"""
    ),
    md(
        """
**Interpretation.** The file is complete and timestamps are unique, but its median implied
horizontal GNSS speed is about 1,017 m/s while its recorded median speed is about 17.7 m/s;
99.3% of successive fixes imply more than 100 m/s. IMU, speed, and position therefore do
not form one plausible flight trajectory. Applying an EKF directly to these rows makes
normal innovations attack-like and removes the physical basis for the proposed method.

The defensible remedy is a transparent carrier transformation: consecutive 100-row Kaggle
segments supply speed and sensor variation, a bounded kinematic reconstruction restores
physical continuity, and attack effects are then inserted into GNSS only. Source row ranges
are retained for provenance; raw-coordinate performance is not claimed.
"""
    ),
    md("### 2. Overall methodology and model architecture"),
    code(
        """
gp.make_methodology_diagram(paths["figures"] / "methodology_pipeline.png")
gp.make_architecture_diagram(paths["figures"] / "cnn_bilstm_architecture.png")
display(Image(filename=str(paths["figures"] / "methodology_pipeline.png")))
display(Image(filename=str(paths["figures"] / "cnn_bilstm_architecture.png")))
"""
    ),
    md(
        """
### 3. Build Kaggle-conditioned carriers and inject attacks

Each non-overlapping Kaggle segment produces matched Normal, Spoofing, and Jamming
variants. Recorded speed and robust IMU/gyro patterns condition the carrier motion.
Spoofing comprises gradual coordinate dragging and sudden jumps.
Jamming comprises high-variance measurement corruption and signal freezing. Attack onset,
direction, magnitude, and noise are randomized per flight. An attack persists to flight end;
otherwise the abrupt return to clean GNSS would create an unlabeled recovery anomaly.
"""
    ),
    code(
        """
dataset, flight_manifest = gp.generate_dataset(source_df, cfg)
split_manifest = gp.grouped_scenario_split(flight_manifest, cfg)
dataset.to_csv(paths["data"] / "kaggle_conditioned_attack_dataset.csv.gz", index=False, compression="gzip")
gp.save_table(split_manifest, paths["tables"] / "flight_split_manifest.csv")

split_summary = []
for split in ["train", "validation", "test"]:
    ids = split_manifest.loc[split_manifest["split"] == split, "flight_id"]
    rows = dataset[dataset["flight_id"].isin(ids)]
    groups = split_manifest.loc[split_manifest["split"] == split, "source_segment_id"].nunique()
    split_summary.append({"Split": split, "Flights": ids.nunique(), "Source groups": groups,
                          "Source-group fraction": groups / flight_manifest.source_segment_id.nunique(),
                          "Rows": len(rows)})
display(pd.DataFrame(split_summary))
display(pd.crosstab(split_manifest["scenario"], split_manifest["split"]))
"""
    ),
    md(
        r"""
### 4. EKF innovation/residual generation

The implemented estimator is a genuine EKF. Its state is north/east/up position,
horizontal speed, heading, and vertical speed. The coordinated-motion prediction is
nonlinear because position depends on $v\cos\psi$ and $v\sin\psi$; covariance propagation
uses the analytical state Jacobian. GNSS position and reported speed form a four-dimensional
measurement. Innovation is recorded **before** the gated update. A finite-difference test
verifies the analytical process Jacobian to within $10^{-5}$.

This robust gate is part of the estimator design and is disclosed because it materially
affects separability. It must be re-calibrated for a real vehicle and receiver.
"""
    ),
    code(
        """
gp.plot_residual_concept(dataset, paths["figures"] / "ekf_residual_concept.png")
jacobian_validation = gp.validate_ekf_jacobian(cfg.seed, 1 / cfg.sample_rate_hz)
gp.save_table(jacobian_validation, paths["tables"] / "ekf_jacobian_validation.csv")
residual_summary = dataset.groupby("label")[["innovation_norm", "nis", "ekf_speed_error"]].median()
residual_summary.index = gp.CLASS_NAMES
display(residual_summary.style.format("{:.3f}"))
display(jacobian_validation)
display(Image(filename=str(paths["figures"] / "ekf_residual_concept.png")))
"""
    ),
    md("### 5. Causal windows and leakage-safe preprocessing"),
    code(
        """
full_features = gp.BASE_FEATURES + gp.EKF_FEATURES
raw_full = gp.make_windows(dataset, split_manifest, cfg, full_features)
raw_base = gp.make_windows(dataset, split_manifest, cfg, gp.BASE_FEATURES)

full_scaler = gp.fit_sequence_scaler(raw_full)
base_scaler = gp.fit_sequence_scaler(raw_base)
full = gp.apply_sequence_scaler(raw_full, full_scaler)
base = gp.apply_sequence_scaler(raw_base, base_scaler)

window_rows = []
for split in ["train", "validation", "test"]:
    counts = np.bincount(raw_full[split]["y"], minlength=3)
    window_rows.append({
        "Split": split, "Flights": split_manifest.query("split == @split").flight_id.nunique(),
        "Source groups": split_manifest.query("split == @split").source_segment_id.nunique(),
        "Windows": len(raw_full[split]["y"]),
        **{name: int(counts[i]) for i, name in enumerate(gp.CLASS_NAMES)}
    })
split_table = pd.DataFrame(window_rows)
gp.save_table(split_table, paths["tables"] / "split_summary.csv")
gp.plot_class_distribution(raw_full, paths["figures"] / "class_distribution.png")
display(split_table)
display(Image(filename=str(paths["figures"] / "class_distribution.png")))
"""
    ),
    md(
        """
## Results

### 6. Train baselines and the controlled EKF ablation

Random Forest and SVM use mean, standard deviation, minimum, maximum, last value, and
linear slope for every window feature. CNN-only, LSTM-only, and both CNN–BiLSTM variants
use full sequences. The two CNN–BiLSTM variants have the same architecture and training
procedure; only the EKF feature group differs. Deep models use focal loss, moderate
class weighting, early stopping on validation macro-F1, and no test feedback.
"""
    ),
    code(
        """
classical_models = gp.train_classical_models(full, cfg.seed)
neural_specs = {
    "CNN-only": (gp.CNNClassifier(len(full_features), cfg.dropout), full),
    "LSTM-only": (gp.LSTMClassifier(len(full_features), cfg.dropout), full),
    "CNN-BiLSTM without EKF": (gp.CNNBiLSTMClassifier(len(gp.BASE_FEATURES), cfg.dropout), base),
    "CNN-BiLSTM with EKF": (gp.CNNBiLSTMClassifier(len(full_features), cfg.dropout), full),
}
trained_models, history_frames = {}, []
for name, (model, windows) in neural_specs.items():
    model, history = gp.train_torch_model(model, windows, cfg, device, name)
    trained_models[name] = (model, windows)
    history_frames.append(history)
    print(f"{name}: {len(history)} epochs; best val macro-F1={history.val_macro_f1.max():.3f}")
training_history = pd.concat(history_frames, ignore_index=True)
gp.save_table(training_history, paths["tables"] / "training_history.csv")
"""
    ),
    code(
        """
gp.plot_training_curves(training_history, paths["figures"] / "training_validation_curves.png")
display(Image(filename=str(paths["figures"] / "training_validation_curves.png")))
"""
    ),
    md(
        """
### 7. Validation-only decision tuning and final test evaluation

Because missing attacks is costlier than a modest false-alarm increase, class-probability
biases are selected on validation groups using a predeclared composite of macro-F1, macro
recall, mean attack recall, and minimum class recall. This changes only the decision rule,
not model weights.
Final uncertainty uses a cluster bootstrap over complete Kaggle source groups; each
resampled group includes its matched Normal, Spoofing, and Jamming variants.
"""
    ),
    code(
        """
threshold_cfg = gp.tune_ekf_threshold(raw_full, full_features)
metrics, per_class, probabilities, predictions, class_biases = gp.evaluate_models(
    classical_models, trained_models, raw_full, full_features, threshold_cfg, device
)
test_y = raw_full["test"]["y"]
test_flights = raw_full["test"]["flight_id"]
test_source_groups = raw_full["test"]["source_segment_id"]

ci_rows = []
for name, pred in predictions.items():
    f1_lo, f1_hi = gp.flight_bootstrap_ci(test_y, pred, test_source_groups, "macro_f1", seed=cfg.seed)
    ac_lo, ac_hi = gp.flight_bootstrap_ci(test_y, pred, test_source_groups, "accuracy", seed=cfg.seed + 1)
    ci_rows.append({"Model": name, "Accuracy 95% CI low": ac_lo, "Accuracy 95% CI high": ac_hi,
                    "Macro F1 95% CI low": f1_lo, "Macro F1 95% CI high": f1_hi})
metrics = metrics.merge(pd.DataFrame(ci_rows), on="Model", how="left")
gp.save_table(metrics, paths["tables"] / "final_model_metrics.csv")
gp.save_table(per_class, paths["tables"] / "per_class_metrics.csv")
display(metrics.style.format(precision=3).hide(axis="index"))
"""
    ),
    code(
        """
proposed_per_class = per_class.query("Model == 'CNN-BiLSTM with EKF'").copy()
display(proposed_per_class.style.format(precision=3).hide(axis="index"))

scenario_lookup = split_manifest.set_index("flight_id")["scenario"]
test_scenarios = pd.Series(test_flights).map(scenario_lookup).to_numpy()
subtype_rows = []
for scenario in ["spoof_drift", "spoof_jump", "jam_noise", "jam_freeze"]:
    expected = 1 if scenario.startswith("spoof") else 2
    mask = (test_scenarios == scenario) & (test_y == expected)
    subtype_rows.append({"Attack subtype": scenario, "Support": int(mask.sum()),
                         "Recall": float((predictions['CNN-BiLSTM with EKF'][mask] == expected).mean())})
subtype_recall = pd.DataFrame(subtype_rows)
gp.save_table(subtype_recall, paths["tables"] / "proposed_attack_subtype_recall.csv")
gp.plot_attack_subtype_recall(subtype_recall, paths["figures"] / "attack_subtype_recall_hq.png")
display(subtype_recall.style.format({"Recall": "{:.1%}"}).hide(axis="index"))
display(Image(filename=str(paths["figures"] / "attack_subtype_recall_hq.png")))
"""
    ),
    md("### 8. Confusion matrix, model comparison, ablation, and ROC–AUC"),
    code(
        """
gp.plot_confusion(test_y, predictions["CNN-BiLSTM with EKF"], paths["figures"] / "confusion_matrix_proposed.png",
                  "Proposed CNN–BiLSTM: row-normalized test confusion matrix")
gp.plot_model_comparison(metrics, paths["figures"] / "model_comparison.png")
gp.plot_ablation(metrics, paths["figures"] / "ekf_ablation.png")
gp.plot_roc_curves(test_y, probabilities, paths["figures"] / "roc_auc_curves.png")
for filename in ["confusion_matrix_proposed.png", "model_comparison.png", "ekf_ablation.png", "roc_auc_curves.png"]:
    display(Image(filename=str(paths["figures"] / filename)))
"""
    ),
    md(
        """
### 9. Five-seed EKF ablation robustness

The full experiment runner repeats the matched with-EKF and without-EKF networks for seeds
42, 52, 62, 72, and 82 on the fixed source-group split. This isolates optimisation-seed
sensitivity; it does not substitute for evaluation over alternative data partitions.
"""
    ),
    code(
        """
seed_runs = pd.read_csv(paths["tables"] / "training_seed_robustness.csv")
seed_summary = pd.read_csv(paths["tables"] / "training_seed_robustness_summary.csv")
gp.plot_seed_robustness(seed_runs, paths["figures"] / "training_seed_robustness.png")
display(seed_summary.style.format(precision=3).hide(axis="index"))
display(Image(filename=str(paths["figures"] / "training_seed_robustness.png")))
"""
    ),
    md(
        """
ROC–AUC is reported as a ranking diagnostic, not the primary endpoint: the operational
question is multiclass assignment under unequal error costs, and class distributions are
not the only concern. Macro-F1, macro recall, the confusion matrix, and per-class recall
therefore carry more weight in interpretation.
"""
    ),
    md("### 10. Explainability and runtime"),
    code(
        """
importance = gp.sequence_permutation_importance(
    trained_models["CNN-BiLSTM with EKF"][0], full["validation"]["X"], full["validation"]["y"],
    full_features, device, repeats=5, seed=cfg.seed
)
shap_importance = gp.gradient_shap_importance(
    trained_models["CNN-BiLSTM with EKF"][0], full["train"]["X"], full["validation"]["X"],
    full_features, seed=cfg.seed
)
latency = gp.benchmark_latency(trained_models, device)
gp.save_table(importance, paths["tables"] / "sequence_feature_importance.csv")
gp.save_table(shap_importance, paths["tables"] / "gradient_shap_importance.csv")
gp.save_table(latency, paths["tables"] / "inference_latency.csv")
gp.plot_feature_importance(importance, paths["figures"] / "feature_importance.png")
gp.plot_shap_importance(shap_importance, paths["figures"] / "gradient_shap_importance.png")
display(importance.head(12).style.format(precision=4).hide(axis="index"))
display(Image(filename=str(paths["figures"] / "feature_importance.png")))
display(shap_importance.head(12).style.format(precision=4).hide(axis="index"))
display(Image(filename=str(paths["figures"] / "gradient_shap_importance.png")))
display(latency.style.format(precision=3).hide(axis="index"))
"""
    ),
    md(
        """
Two complementary validation-only explanations are reported. Sequence permutation measures
the change in macro-F1 after moving a complete feature trajectory between windows, whereas
GradientSHAP attributes model logits relative to a training-window background. Agreement
supports robustness; disagreement is expected for correlated residual channels. Neither is
a causal explanation. Latency is batch-one framework latency on the recorded device and excludes
sensor acquisition, window accumulation, preprocessing, and EKF computation.
"""
    ),
    md("## Takeaways"),
    md(
        f"""
1. **The Kaggle data now participate in model construction.** Non-overlapping source
   segments provide speed and sensor patterns, while an explicit kinematic transformation
   repairs coordinate continuity before controlled attack insertion.
2. **EKF residuals materially improve balanced performance.** Relative to the controlled
   CNN–BiLSTM ablation, the proposed model improves accuracy by
   {(PROPOSED['Accuracy'] - ABLATION['Accuracy']):.1%} points, macro-F1 by
   {(PROPOSED['Macro F1'] - ABLATION['Macro F1']):.3f}, and macro recall by
   {(PROPOSED['Macro Recall'] - ABLATION['Macro Recall']):.3f}.
3. **The proposed model is useful but not universally best.** Random Forest has the highest
   aggregate test score, while the proposed model provides a temporal architecture and a
   direct EKF ablation aligned with the thesis hypothesis.
4. **Spoofing remains the limiting class.** The aggregate spoofing recall of
   {PROPOSED_CLASS.loc['Spoofing', 'Recall']:.1%} hides a major subtype gap: gradual drift
   remains difficult at {SUBTYPE.loc['spoof_drift', 'Recall']:.1%}. Jamming is much more
   separable in this generator and should not be described as equally difficult.
5. **These are semi-synthetic results.** A source-group split, five training seeds, and
   source-group bootstrap intervals limit common overstatement, but only real receiver logs, unseen
   trajectories, and hardware-in-the-loop attacks can establish external validity.
"""
    ),
    md(
        """
## Limitations and threats to validity

- **Semi-synthetic-to-real gap:** injected attacks cannot reproduce every receiver, multipath,
  satellite geometry, RF front-end, or adversary adaptation.
- **Source transformation:** recorded sensor patterns are retained, but the carrier path is
  a bounded kinematic reconstruction rather than an unmodified recorded trajectory.
- **Estimator-assisted separability:** the robust innovation gate is an explicit design
  choice and can amplify attack residuals. It requires calibration and sensitivity analysis
  on real data.
- **Gradual spoofing:** slow, kinematically plausible coordinate dragging can remain close to
  the estimator trajectory. Additional receiver observables (C/N0, AGC, Doppler, satellite
  residuals, clock bias) are the most credible next features.
- **Uncertainty:** confidence intervals are wide because the independent sample size is the
  seven held-out source groups, not the number of overlapping windows.
- **No deployment claim:** latency excludes upstream sensing and preprocessing, and no
  safety assurance, adversarial robustness, or distribution-shift study was performed.
"""
    ),
    md("## Reproducibility and saved artifacts"),
    code(
        """
metadata = {
    "features": full_features, "class_names": gp.CLASS_NAMES,
    "window_length": cfg.window_length, "sample_rate_hz": cfg.sample_rate_hz,
    "scaler_mean": full_scaler.mean_.tolist(), "scaler_scale": full_scaler.scale_.tolist(),
}
gp.save_checkpoint(trained_models["CNN-BiLSTM with EKF"][0],
                   paths["models"] / "cnn_bilstm_with_ekf.pt", metadata)
gp.save_checkpoint(trained_models["CNN-BiLSTM without EKF"][0],
                   paths["models"] / "cnn_bilstm_without_ekf.pt",
                   {**metadata, "features": gp.BASE_FEATURES,
                    "scaler_mean": base_scaler.mean_.tolist(), "scaler_scale": base_scaler.scale_.tolist()})
gp.save_run_manifest(paths["root"] / "run_manifest.json", cfg, {
    "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
    "scikit_learn": sklearn.__version__, "threshold_baseline": threshold_cfg,
    "validation_tuned_class_biases": class_biases,
    "methodology": "Kaggle-conditioned carrier trajectories with synthetic GNSS attack injection",
    "ekf_state": "[north, east, up, horizontal speed, heading, vertical speed]",
    "ekf_nonlinearity": "trigonometric coordinated-motion process model with analytical Jacobian",
    "robustness_seeds": [42, 52, 62, 72, 82],
    "bootstrap_cluster": "source_segment_id",
    "test_flights": split_manifest.query("split == 'test'").flight_id.tolist(),
    "test_source_segments": split_manifest.query("split == 'test'").source_segment_id.unique().tolist(),
})

artifact_rows = []
for folder in [paths["figures"], paths["tables"], paths["models"]]:
    for file in sorted(folder.glob("*")):
        artifact_rows.append({"Type": folder.name, "File": str(file), "Size (KB)": file.stat().st_size / 1024})
display(pd.DataFrame(artifact_rows).style.format({"Size (KB)": "{:.1f}"}).hide(axis="index"))
"""
    ),
]


nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": platform.python_version() if False else "3.10+"},
    },
)
nbf.write(nb, ROOT / "Improved_EKF_CNN_BiLSTM_Thesis.ipynb")

