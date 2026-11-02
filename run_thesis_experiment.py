"""Execute the thesis experiment and save all reproducible artifacts."""

from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

import gnss_thesis_pipeline as gp


def main() -> None:
    cfg = gp.ExperimentConfig()
    gp.seed_everything(cfg.seed)
    gp.configure_style()
    paths = gp.ensure_output_dirs(cfg.output_dir)
    device = gp.select_device()
    print(f"Device: {device} | PyTorch {torch.__version__} | Python {platform.python_version()}")

    source, audit = gp.audit_source_csv("uav_navigation_dataset.csv")
    gp.save_table(audit, paths["tables"] / "source_data_quality_audit.csv")
    jacobian_validation = gp.validate_ekf_jacobian(cfg.seed, 1.0 / cfg.sample_rate_hz)
    gp.save_table(jacobian_validation, paths["tables"] / "ekf_jacobian_validation.csv")
    if not jacobian_validation["passed"].all():
        raise RuntimeError("Analytical EKF Jacobian failed finite-difference validation")
    print(audit.to_string(index=False))

    dataset, manifest = gp.generate_dataset(source, cfg)
    split_manifest = gp.grouped_scenario_split(manifest, cfg)
    dataset.to_csv(
        paths["data"] / "kaggle_conditioned_attack_dataset.csv.gz",
        index=False,
        compression="gzip",
    )
    gp.save_table(split_manifest, paths["tables"] / "flight_split_manifest.csv")

    full_features = gp.BASE_FEATURES + gp.EKF_FEATURES
    raw_full = gp.make_windows(dataset, split_manifest, cfg, full_features)
    raw_base = gp.make_windows(dataset, split_manifest, cfg, gp.BASE_FEATURES)
    full_scaler = gp.fit_sequence_scaler(raw_full)
    base_scaler = gp.fit_sequence_scaler(raw_base)
    full = gp.apply_sequence_scaler(raw_full, full_scaler)
    base = gp.apply_sequence_scaler(raw_base, base_scaler)

    split_rows = []
    for split in ["train", "validation", "test"]:
        counts = np.bincount(raw_full[split]["y"], minlength=3)
        split_rows.append(
            {
                "Split": split,
                "Flights": split_manifest[split_manifest["split"] == split]["flight_id"].nunique(),
                "Source groups": split_manifest.loc[
                    split_manifest["split"] == split, "source_segment_id"
                ].nunique(),
                "Source-group fraction": split_manifest.loc[
                    split_manifest["split"] == split, "source_segment_id"
                ].nunique()
                / split_manifest["source_segment_id"].nunique(),
                "Windows": len(raw_full[split]["y"]),
                **{name: int(counts[i]) for i, name in enumerate(gp.CLASS_NAMES)},
            }
        )
    gp.save_table(pd.DataFrame(split_rows), paths["tables"] / "split_summary.csv")
    print(pd.DataFrame(split_rows).to_string(index=False))

    classical = gp.train_classical_models(full, cfg.seed)
    neural_specs = {
        "CNN-only": (gp.CNNClassifier(len(full_features), cfg.dropout), full),
        "LSTM-only": (gp.LSTMClassifier(len(full_features), cfg.dropout), full),
        "CNN-BiLSTM without EKF": (gp.CNNBiLSTMClassifier(len(gp.BASE_FEATURES), cfg.dropout), base),
        "CNN-BiLSTM with EKF": (gp.CNNBiLSTMClassifier(len(full_features), cfg.dropout), full),
    }
    trained = {}
    histories = []
    for name, (model, windows) in neural_specs.items():
        print(f"\nTraining {name}")
        model, history = gp.train_torch_model(model, windows, cfg, device, name)
        trained[name] = (model, windows)
        histories.append(history)
        best = history.loc[history["val_macro_f1"].idxmax()]
        print(f"  epochs={len(history)} best validation macro-F1={best['val_macro_f1']:.4f}")
    history = pd.concat(histories, ignore_index=True)
    gp.save_table(history, paths["tables"] / "training_history.csv")

    threshold_cfg = gp.tune_ekf_threshold(raw_full, full_features)
    metrics, per_class, probabilities, predictions, class_biases = gp.evaluate_models(
        classical, trained, raw_full, full_features, threshold_cfg, device
    )
    test_y = raw_full["test"]["y"]
    test_flights = raw_full["test"]["flight_id"]
    test_source_groups = raw_full["test"]["source_segment_id"]
    ci_rows = []
    for name, prediction in predictions.items():
        f1_low, f1_high = gp.flight_bootstrap_ci(
            test_y, prediction, test_source_groups, metric="macro_f1", seed=cfg.seed
        )
        acc_low, acc_high = gp.flight_bootstrap_ci(
            test_y, prediction, test_source_groups, metric="accuracy", seed=cfg.seed + 1
        )
        ci_rows.append(
            {
                "Model": name,
                "Accuracy 95% CI low": acc_low,
                "Accuracy 95% CI high": acc_high,
                "Macro F1 95% CI low": f1_low,
                "Macro F1 95% CI high": f1_high,
            }
        )
    metrics = metrics.merge(pd.DataFrame(ci_rows), on="Model", how="left")
    gp.save_table(metrics, paths["tables"] / "final_model_metrics.csv")
    gp.save_table(per_class, paths["tables"] / "per_class_metrics.csv")
    scenario_lookup = split_manifest.set_index("flight_id")["scenario"]
    test_scenarios = pd.Series(test_flights).map(scenario_lookup).to_numpy()
    subtype_rows = []
    proposed_prediction = predictions["CNN-BiLSTM with EKF"]
    for scenario in ["spoof_drift", "spoof_jump", "jam_noise", "jam_freeze"]:
        expected_class = 1 if scenario.startswith("spoof") else 2
        mask = (test_scenarios == scenario) & (test_y == expected_class)
        subtype_rows.append(
            {
                "Attack subtype": scenario,
                "Support": int(mask.sum()),
                "Recall": float((proposed_prediction[mask] == expected_class).mean()),
            }
        )
    subtype_table = pd.DataFrame(subtype_rows)
    gp.save_table(subtype_table, paths["tables"] / "proposed_attack_subtype_recall.csv")
    drift_mask = (test_scenarios == "spoof_drift") & (test_y == 1)
    drift_strength = raw_full["test"]["attack_strength"][drift_mask]
    drift_correct = proposed_prediction[drift_mask] == 1
    severity_rows = []
    for label, low, high in [
        ("< 5 m", 0.0, 5.0),
        ("5–15 m", 5.0, 15.0),
        ("15–30 m", 15.0, 30.0),
        (">= 30 m", 30.0, np.inf),
    ]:
        mask = (drift_strength >= low) & (drift_strength < high)
        severity_rows.append(
            {
                "Gradual-drift offset": label,
                "Support": int(mask.sum()),
                "Recall": float(drift_correct[mask].mean()) if mask.any() else np.nan,
            }
        )
    gp.save_table(
        pd.DataFrame(severity_rows),
        paths["tables"] / "gradual_drift_recall_by_offset.csv",
    )

    latency_rows = []
    test_times = raw_full["test"]["end_time_s"]
    for flight_id in np.unique(test_flights[test_scenarios == "spoof_drift"]):
        indices = np.flatnonzero((test_flights == flight_id) & (test_y == 1))
        correct = (proposed_prediction[indices] == 1).astype(np.int64)
        sustained = np.convolve(correct, np.ones(3, dtype=np.int64), mode="valid") >= 3
        detected = bool(sustained.any())
        first_index = int(np.flatnonzero(sustained)[0] + 2) if detected else -1
        onset = float(test_times[indices[0]])
        detection_time = float(test_times[indices[first_index]]) if detected else np.nan
        latency_rows.append(
            {
                "Flight": int(flight_id),
                "Detected": detected,
                "Attack-labelled first window (s)": onset,
                "Sustained detection time (s)": detection_time,
                "Detection latency (s)": detection_time - onset if detected else np.nan,
            }
        )
    gp.save_table(
        pd.DataFrame(latency_rows),
        paths["tables"] / "gradual_drift_detection_latency.csv",
    )
    print("\nHeld-out test metrics")
    print(metrics.to_string(index=False))

    latency = gp.benchmark_latency(trained, device)
    gp.save_table(latency, paths["tables"] / "inference_latency.csv")
    importance = gp.sequence_permutation_importance(
        trained["CNN-BiLSTM with EKF"][0],
        full["validation"]["X"],
        full["validation"]["y"],
        full_features,
        device,
    )
    gp.save_table(importance, paths["tables"] / "sequence_feature_importance.csv")
    shap_importance = gp.gradient_shap_importance(
        trained["CNN-BiLSTM with EKF"][0],
        full["train"]["X"],
        full["validation"]["X"],
        full_features,
        seed=cfg.seed,
    )
    gp.save_table(shap_importance, paths["tables"] / "gradient_shap_importance.csv")

    robustness_rows = []
    robustness_seeds = [42, 52, 62, 72, 82]
    for robustness_seed in robustness_seeds:
        for name, windows, n_features in [
            ("CNN-BiLSTM without EKF", base, len(gp.BASE_FEATURES)),
            ("CNN-BiLSTM with EKF", full, len(full_features)),
        ]:
            if robustness_seed == cfg.seed:
                probability = probabilities[name]
            else:
                gp.seed_everything(robustness_seed)
                seed_model = gp.CNNBiLSTMClassifier(n_features, cfg.dropout)
                seed_model, _ = gp.train_torch_model(
                    seed_model,
                    windows,
                    cfg,
                    device,
                    f"{name} seed {robustness_seed}",
                )
                validation_probability = gp.predict_torch(
                    seed_model, windows["validation"]["X"], device
                )
                bias = gp.tune_class_bias(
                    raw_full["validation"]["y"], validation_probability
                )
                probability = gp.apply_class_bias(
                    gp.predict_torch(seed_model, windows["test"]["X"], device), bias
                )
            seed_metrics, _ = gp._classification_metrics(test_y, probability)
            robustness_rows.append({"Seed": robustness_seed, "Model": name, **seed_metrics})
    robustness = pd.DataFrame(robustness_rows)
    gp.save_table(robustness, paths["tables"] / "training_seed_robustness.csv")
    robustness_summary = robustness.groupby("Model").agg(
        Seeds=("Seed", "nunique"),
        Accuracy_mean=("Accuracy", "mean"),
        Accuracy_std=("Accuracy", "std"),
        Macro_recall_mean=("Macro Recall", "mean"),
        Macro_recall_std=("Macro Recall", "std"),
        Macro_F1_mean=("Macro F1", "mean"),
        Macro_F1_std=("Macro F1", "std"),
    ).reset_index()
    gp.save_table(robustness_summary, paths["tables"] / "training_seed_robustness_summary.csv")

    gp.plot_class_distribution(raw_full, paths["figures"] / "class_distribution.png")
    gp.plot_residual_concept(dataset, paths["figures"] / "ekf_residual_concept.png")
    gp.plot_training_curves(history, paths["figures"] / "training_validation_curves.png")
    gp.plot_confusion(
        test_y,
        predictions["CNN-BiLSTM with EKF"],
        paths["figures"] / "confusion_matrix_proposed.png",
        "Proposed CNN–BiLSTM: row-normalized test confusion matrix",
    )
    gp.plot_model_comparison(metrics, paths["figures"] / "model_comparison.png")
    gp.plot_ablation(metrics, paths["figures"] / "ekf_ablation.png")
    gp.plot_roc_curves(test_y, probabilities, paths["figures"] / "roc_auc_curves.png")
    gp.plot_feature_importance(importance, paths["figures"] / "feature_importance.png")
    gp.plot_shap_importance(shap_importance, paths["figures"] / "gradient_shap_importance.png")
    gp.plot_attack_subtype_recall(
        subtype_table, paths["figures"] / "attack_subtype_recall_hq.png"
    )
    gp.plot_seed_robustness(robustness, paths["figures"] / "training_seed_robustness.png")
    gp.make_methodology_diagram(paths["figures"] / "methodology_pipeline.png")
    gp.make_architecture_diagram(paths["figures"] / "cnn_bilstm_architecture.png")

    metadata = {
        "features": full_features,
        "class_names": gp.CLASS_NAMES,
        "window_length": cfg.window_length,
        "sample_rate_hz": cfg.sample_rate_hz,
        "scaler_mean": full_scaler.mean_.tolist(),
        "scaler_scale": full_scaler.scale_.tolist(),
    }
    gp.save_checkpoint(
        trained["CNN-BiLSTM with EKF"][0],
        paths["models"] / "cnn_bilstm_with_ekf.pt",
        metadata,
    )
    gp.save_checkpoint(
        trained["CNN-BiLSTM without EKF"][0],
        paths["models"] / "cnn_bilstm_without_ekf.pt",
        {**metadata, "features": gp.BASE_FEATURES, "scaler_mean": base_scaler.mean_.tolist(), "scaler_scale": base_scaler.scale_.tolist()},
    )
    gp.save_run_manifest(
        paths["root"] / "run_manifest.json",
        cfg,
        {
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "threshold_baseline": threshold_cfg,
            "validation_tuned_class_biases": class_biases,
            "methodology": "Kaggle-conditioned carrier trajectories with synthetic GNSS attack injection",
            "ekf_state": "[north, east, up, horizontal speed, heading, vertical speed]",
            "ekf_nonlinearity": "trigonometric coordinated-motion process model with analytical Jacobian",
            "robustness_seeds": robustness_seeds,
            "bootstrap_cluster": "source_segment_id",
            "test_flights": split_manifest.loc[split_manifest["split"] == "test", "flight_id"].tolist(),
            "test_source_segments": split_manifest.loc[
                split_manifest["split"] == "test", "source_segment_id"
            ].unique().tolist(),
        },
    )


if __name__ == "__main__":
    main()
