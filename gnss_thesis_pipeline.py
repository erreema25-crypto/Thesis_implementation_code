"""Reproducible EKF-residual GNSS attack experiments.

The module is intentionally notebook-friendly: every public function is small
enough to inspect, all random state is explicit, and all learned transforms are
fit on training flights only.  It supports CPU, CUDA workstations, Google
Colab, and NVIDIA Jetson/Thor PyTorch environments.
"""

from __future__ import annotations

import json
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = ["Normal", "Spoofing", "Jamming"]
CLASS_COLORS = {"Normal": "#2F6690", "Spoofing": "#E09F3E", "Jamming": "#9E2A2B"}
MODEL_COLORS = {
    "Random Forest": "#6C757D",
    "SVM": "#495057",
    "CNN-only": "#7A9E9F",
    "LSTM-only": "#B56576",
    "CNN-BiLSTM without EKF": "#E09F3E",
    "CNN-BiLSTM with EKF": "#2F6690",
    "EKF threshold": "#8A817C",
}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    sample_rate_hz: int = 10
    flight_duration_s: int = 100
    flights_per_class: int = 48
    source_segment_rows: int = 100
    window_s: float = 4.0
    stride_s: float = 1.0
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    batch_size: int = 256
    max_epochs: int = 35
    patience: int = 7
    learning_rate: float = 8e-4
    weight_decay: float = 2e-4
    dropout: float = 0.25
    min_delta: float = 1e-4
    num_workers: int = 0
    output_dir: str = "outputs"

    @property
    def n_steps(self) -> int:
        return self.sample_rate_hz * self.flight_duration_s

    @property
    def window_length(self) -> int:
        return int(round(self.sample_rate_hz * self.window_s))

    @property
    def stride(self) -> int:
        return int(round(self.sample_rate_hz * self.stride_s))


BASE_FEATURES = [
    "acc_n",
    "acc_e",
    "acc_u",
    "gyro_z",
    "speed",
    "gnss_delta_n",
    "gnss_delta_e",
    "gnss_delta_u",
    "gnss_step",
]

EKF_FEATURES = [
    "innovation_n",
    "innovation_e",
    "innovation_u",
    "innovation_norm",
    "nis",
    "innovation_ewma",
    "innovation_bias_norm",
    "ekf_speed_error",
]


def seed_everything(seed: int) -> None:
    """Make Python, NumPy and PyTorch runs as deterministic as practical."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
        }
    )


def ensure_output_dirs(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    paths = {
        "root": root,
        "figures": root / "figures",
        "tables": root / "tables",
        "models": root / "models",
        "data": root / "data",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def audit_source_csv(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile the supplied CSV and quantify its temporal physical consistency."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    dt = df["timestamp"].diff().dt.total_seconds()
    lat0 = np.deg2rad(df["latitude"].median())
    north_step = df["latitude"].diff() * 111_320.0
    east_step = df["longitude"].diff() * 111_320.0 * np.cos(lat0)
    horizontal_step = np.hypot(north_step, east_step)
    implied_speed = horizontal_step / dt.replace(0, np.nan)
    stated_speed = df["speed"]
    audit = pd.DataFrame(
        {
            "check": [
                "rows",
                "columns",
                "missing cells",
                "duplicate timestamps",
                "median sampling interval (s)",
                "median implied GNSS speed (m/s)",
                "median stated speed (m/s)",
                "implied/stated median speed ratio",
                "fraction implied speed > 100 m/s",
            ],
            "value": [
                len(df),
                df.shape[1],
                int(df.isna().sum().sum()),
                int(df["timestamp"].duplicated().sum()),
                float(dt.median()),
                float(implied_speed.median()),
                float(stated_speed.median()),
                float(implied_speed.median() / stated_speed.median()),
                float((implied_speed > 100).mean()),
            ],
        }
    )
    return df, audit


def _robust_pattern(values: np.ndarray) -> np.ndarray:
    """Return a clipped, robustly standardised source-data pattern."""
    values = np.asarray(values, dtype=np.float64)
    centre = np.nanmedian(values)
    q25, q75 = np.nanquantile(values, [0.25, 0.75])
    scale = max((q75 - q25) / 1.349, 1e-6)
    return np.clip((values - centre) / scale, -3.0, 3.0)


def _interp_source(values: np.ndarray, n_steps: int) -> np.ndarray:
    source_axis = np.linspace(0.0, 1.0, len(values))
    target_axis = np.linspace(0.0, 1.0, n_steps)
    return np.interp(target_axis, source_axis, np.asarray(values, dtype=np.float64))


def _build_kaggle_carrier(
    source_segment: pd.DataFrame,
    rng: np.random.Generator,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Construct a continuous carrier flight from one ordered Kaggle segment.

    The source coordinates are not integrated directly because their consecutive
    jumps fail the physical-consistency audit. Recorded speed and robust temporal
    patterns from the IMU/gyro channels instead condition a coherent local
    trajectory. Synthetic attacks are injected only after this transformation.
    """
    required = {
        "latitude",
        "longitude",
        "altitude",
        "imu_acc_x",
        "imu_acc_y",
        "imu_acc_z",
        "imu_gyro_z",
        "speed",
    }
    missing = required.difference(source_segment.columns)
    if missing:
        raise ValueError(f"Kaggle source is missing required columns: {sorted(missing)}")

    n = cfg.n_steps
    dt = 1.0 / cfg.sample_rate_hz
    time_s = np.arange(n, dtype=np.float64) * dt

    def smooth(values: Iterable[float], width: int) -> np.ndarray:
        return (
            pd.Series(np.asarray(values, dtype=np.float64))
            .rolling(width, center=True, min_periods=1)
            .median()
            .to_numpy()
        )

    source_speed = np.clip(smooth(source_segment["speed"], 11), 5.0, 24.0)
    speed_true = smooth(_interp_source(source_speed, n), 21)

    gyro_source = smooth(source_segment["imu_gyro_z"], 9)
    gyro_pattern = _interp_source(_robust_pattern(gyro_source), n)
    yaw_rate = np.clip(0.022 * gyro_pattern, -0.065, 0.065)

    raw_lat = source_segment["latitude"].to_numpy(dtype=np.float64)
    raw_lon = source_segment["longitude"].to_numpy(dtype=np.float64)
    lat_delta = np.nanmedian(raw_lat[-5:]) - np.nanmedian(raw_lat[:5])
    lon_delta = np.nanmedian(raw_lon[-5:]) - np.nanmedian(raw_lon[:5])
    heading0 = float(
        np.arctan2(
            lon_delta * np.cos(np.deg2rad(np.nanmedian(raw_lat))),
            lat_delta,
        )
    )
    if not np.isfinite(heading0) or abs(lat_delta) + abs(lon_delta) < 1e-12:
        heading0 = float(rng.uniform(-np.pi, np.pi))
    heading = heading0 + np.cumsum(yaw_rate) * dt

    vel_n = speed_true * np.cos(heading)
    vel_e = speed_true * np.sin(heading)
    vertical_source = smooth(source_segment["imu_acc_z"], 9)
    vertical_pattern = _interp_source(_robust_pattern(vertical_source), n)
    acc_u_true = 0.055 * vertical_pattern
    vel_u = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        vel_u[i] = np.clip(0.998 * vel_u[i - 1] + acc_u_true[i - 1] * dt, -1.2, 1.2)

    position = np.zeros((n, 3), dtype=np.float64)
    position[0, 2] = float(np.clip(np.nanmedian(source_segment["altitude"]), 20.0, 500.0))
    for i in range(1, n):
        position[i, 0] = position[i - 1, 0] + 0.5 * (vel_n[i - 1] + vel_n[i]) * dt
        position[i, 1] = position[i - 1, 1] + 0.5 * (vel_e[i - 1] + vel_e[i]) * dt
        position[i, 2] = position[i - 1, 2] + 0.5 * (vel_u[i - 1] + vel_u[i]) * dt

    acc_n_true = np.gradient(vel_n, dt)
    acc_e_true = np.gradient(vel_e, dt)
    imu_n_pattern = _interp_source(
        _robust_pattern(smooth(source_segment["imu_acc_x"], 7)), n
    )
    imu_e_pattern = _interp_source(
        _robust_pattern(smooth(source_segment["imu_acc_y"], 7)), n
    )
    imu_bias = rng.normal(0.0, [0.025, 0.025, 0.015])
    imu = np.column_stack(
        [
            acc_n_true + 0.045 * imu_n_pattern,
            acc_e_true + 0.045 * imu_e_pattern,
            acc_u_true,
        ]
    )
    imu += imu_bias + rng.normal(0.0, [0.055, 0.055, 0.035], (n, 3))
    gyro_z = yaw_rate + rng.normal(0.0, 0.006, n)
    gnss_clean = position + rng.normal(0.0, [1.5, 1.5, 2.2], (n, 3))
    speed_measured = speed_true + rng.normal(0.0, 0.25, n)

    return pd.DataFrame(
        {
            "time_s": time_s,
            "true_n": position[:, 0],
            "true_e": position[:, 1],
            "true_u": position[:, 2],
            "vel_n": vel_n,
            "vel_e": vel_e,
            "vel_u": vel_u,
            "acc_n": imu[:, 0],
            "acc_e": imu[:, 1],
            "acc_u": imu[:, 2],
            "gyro_z": gyro_z,
            "speed": speed_measured,
            "gnss_n_clean": gnss_clean[:, 0],
            "gnss_e_clean": gnss_clean[:, 1],
            "gnss_u_clean": gnss_clean[:, 2],
        }
    )


def _inject_attack(
    clean: pd.DataFrame,
    rng: np.random.Generator,
    class_id: int,
    scenario: str,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Inject diverse spoofing or jamming into GNSS measurements only."""
    frame = clean.copy()
    gnss = frame[["gnss_n_clean", "gnss_e_clean", "gnss_u_clean"]].to_numpy(copy=True)
    label = np.zeros(len(frame), dtype=np.int64)
    attack_strength = np.zeros(len(frame), dtype=np.float64)
    gnss_update = np.ones(len(frame), dtype=np.int64)

    if class_id != 0:
        onset_s = rng.uniform(32, 52)
        start = int(onset_s * cfg.sample_rate_hz)
        # The attack persists to flight end.  Ending it abruptly would create a
        # second, unlabeled measurement jump and contaminate the Normal class
        # with EKF recovery transients.
        end = len(frame)
        idx = np.arange(start, end)
        progress = np.linspace(0.0, 1.0, len(idx))
        label[idx] = class_id

        if class_id == 1:
            direction = rng.normal(size=3)
            direction[2] *= 0.35
            direction /= np.linalg.norm(direction)
            if scenario == "spoof_drift":
                # Continuous carry-off with no discrete jump. The range is
                # deliberately broad and includes low-offset onset windows;
                # severity-stratified recall is reported separately.
                terminal_offset = rng.uniform(50, 100)
                curve = progress ** rng.uniform(0.8, 1.2)
                offset = terminal_offset * curve[:, None] * direction
            elif scenario == "spoof_jump":
                jump = rng.uniform(22, 65)
                drift = rng.uniform(5, 25) * progress
                offset = (jump + drift)[:, None] * direction
            else:
                raise ValueError(f"Unknown spoofing scenario: {scenario}")
            gnss[idx] += offset
            attack_strength[idx] = np.linalg.norm(offset, axis=1)

        elif class_id == 2:
            if scenario == "jam_noise":
                sigma = rng.uniform(9, 18) + progress * rng.uniform(12, 28)
                noise = rng.normal(size=(len(idx), 3)) * sigma[:, None]
                noise[:, 2] *= 1.25
                gnss[idx] += noise
                attack_strength[idx] = sigma
            elif scenario == "jam_freeze":
                hold = gnss[start - 1].copy()
                rare_update_probability = rng.uniform(0.005, 0.025)
                for local_i, global_i in enumerate(idx):
                    if rng.random() < rare_update_probability:
                        hold = gnss[global_i] + rng.normal(0, [4, 4, 6])
                        gnss_update[global_i] = 1
                    else:
                        gnss_update[global_i] = 0
                    gnss[global_i] = hold
                attack_strength[idx] = np.linalg.norm(
                    gnss[idx] - frame.loc[idx, ["true_n", "true_e", "true_u"]].to_numpy(), axis=1
                )
            else:
                raise ValueError(f"Unknown jamming scenario: {scenario}")

    frame[["gnss_n", "gnss_e", "gnss_u"]] = gnss
    frame["label"] = label
    frame["attack_strength"] = attack_strength
    frame["gnss_update"] = gnss_update
    return frame


def _wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def ekf_process_model(x: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    """Nonlinear coordinated-motion process model.

    State order is north, east, up, horizontal speed, heading, vertical speed;
    control order is north/east/up acceleration and yaw rate.
    """
    n, e, u, speed, heading, vertical_speed = x
    acc_n, acc_e, acc_u, yaw_rate = control
    forward_acc = acc_n * np.cos(heading) + acc_e * np.sin(heading)
    return np.array(
        [
            n + speed * np.cos(heading) * dt + 0.5 * acc_n * dt**2,
            e + speed * np.sin(heading) * dt + 0.5 * acc_e * dt**2,
            u + vertical_speed * dt + 0.5 * acc_u * dt**2,
            max(0.1, speed + forward_acc * dt),
            _wrap_angle(heading + yaw_rate * dt),
            vertical_speed + acc_u * dt,
        ],
        dtype=np.float64,
    )


def ekf_process_jacobian(x: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    """Analytical Jacobian of :func:`ekf_process_model` with respect to state."""
    _, _, _, speed, heading, _ = x
    acc_n, acc_e, _, _ = control
    jacobian = np.eye(6, dtype=np.float64)
    jacobian[0, 3] = np.cos(heading) * dt
    jacobian[0, 4] = -speed * np.sin(heading) * dt
    jacobian[1, 3] = np.sin(heading) * dt
    jacobian[1, 4] = speed * np.cos(heading) * dt
    jacobian[2, 5] = dt
    jacobian[3, 4] = (-acc_n * np.sin(heading) + acc_e * np.cos(heading)) * dt
    return jacobian


def validate_ekf_jacobian(seed: int = 42, dt: float = 0.1) -> pd.DataFrame:
    """Compare the analytical process Jacobian with central finite differences."""
    rng = np.random.default_rng(seed)
    x = np.array([10.0, -5.0, 120.0, 14.0, 0.7, 0.2]) + rng.normal(0, 0.02, 6)
    control = np.array([0.3, -0.2, 0.05, 0.01])
    analytical = ekf_process_jacobian(x, control, dt)
    numerical = np.zeros_like(analytical)
    epsilon = 1e-6
    for column in range(len(x)):
        plus, minus = x.copy(), x.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        numerical[:, column] = (
            ekf_process_model(plus, control, dt) - ekf_process_model(minus, control, dt)
        ) / (2.0 * epsilon)
    difference = np.abs(analytical - numerical)
    return pd.DataFrame(
        {
            "check": ["maximum absolute Jacobian error", "mean absolute Jacobian error"],
            "value": [float(difference.max()), float(difference.mean())],
            "tolerance": [1e-5, 1e-6],
            "passed": [bool(difference.max() < 1e-5), bool(difference.mean() < 1e-6)],
        }
    )


def run_ekf(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Run a genuine nonlinear EKF and log pre-update innovations."""
    dt = 1.0 / cfg.sample_rate_hz
    measurement_jacobian = np.zeros((4, 6), dtype=np.float64)
    measurement_jacobian[0, 0] = measurement_jacobian[1, 1] = 1.0
    measurement_jacobian[2, 2] = measurement_jacobian[3, 3] = 1.0
    process_covariance = np.diag([0.08, 0.08, 0.12, 0.30, 0.018, 0.16]) ** 2
    measurement_covariance = np.diag([4.0, 4.0, 6.0, 1.0]) ** 2
    nis_gate = 18.47  # chi-square(4), 99.9%
    position_innovation_gate_m = 20.0
    identity = np.eye(6)

    positions = frame[["gnss_n", "gnss_e", "gnss_u"]].to_numpy(dtype=np.float64)
    measured_speed = frame["speed"].to_numpy(dtype=np.float64)
    controls = frame[["acc_n", "acc_e", "acc_u", "gyro_z"]].to_numpy(dtype=np.float64)
    warmup = min(20, len(frame) - 1)
    warmup_time = np.arange(warmup + 1) * dt
    initial_velocity = np.array(
        [np.polyfit(warmup_time, positions[: warmup + 1, axis], 1)[0] for axis in range(3)]
    )
    initial_heading = float(np.arctan2(initial_velocity[1], initial_velocity[0]))
    x = np.array(
        [
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            max(0.1, np.linalg.norm(initial_velocity[:2])),
            initial_heading,
            initial_velocity[2],
        ]
    )
    covariance = np.diag([4.0, 4.0, 6.0, 2.0, 0.20, 1.0]) ** 2
    innovations = np.zeros((len(frame), 4), dtype=np.float64)
    nis = np.zeros(len(frame), dtype=np.float64)
    filtered = np.zeros((len(frame), 3), dtype=np.float64)
    filtered_velocity = np.zeros((len(frame), 3), dtype=np.float64)
    accepted = np.zeros(len(frame), dtype=np.int64)

    for index, (position, speed, control) in enumerate(zip(positions, measured_speed, controls)):
        if index > 0:
            transition_jacobian = ekf_process_jacobian(x, control, dt)
            x = ekf_process_model(x, control, dt)
            covariance = (
                transition_jacobian @ covariance @ transition_jacobian.T + process_covariance
            )

        measurement = np.r_[position, speed]
        predicted_measurement = x[[0, 1, 2, 3]]
        innovation = measurement - predicted_measurement
        innovation_covariance = (
            measurement_jacobian @ covariance @ measurement_jacobian.T
            + measurement_covariance
        )
        nis[index] = float(
            innovation @ np.linalg.solve(innovation_covariance, innovation)
        )
        if (
            nis[index] <= nis_gate
            and np.linalg.norm(innovation[:3]) <= position_innovation_gate_m
        ):
            kalman_gain = np.linalg.solve(
                innovation_covariance,
                measurement_jacobian @ covariance,
            ).T
            x = x + kalman_gain @ innovation
            x[4] = _wrap_angle(x[4])
            residual_operator = identity - kalman_gain @ measurement_jacobian
            covariance = (
                residual_operator @ covariance @ residual_operator.T
                + kalman_gain @ measurement_covariance @ kalman_gain.T
            )
            accepted[index] = 1

        innovations[index] = innovation
        filtered[index] = x[:3]
        filtered_velocity[index] = [
            x[3] * np.cos(x[4]),
            x[3] * np.sin(x[4]),
            x[5],
        ]

    result = frame.copy()
    result[["innovation_n", "innovation_e", "innovation_u"]] = innovations[:, :3]
    result["innovation_speed"] = innovations[:, 3]
    result["innovation_norm"] = np.linalg.norm(innovations[:, :3], axis=1)
    result["nis"] = np.clip(nis, 0, 1e5)
    result["innovation_ewma"] = result["innovation_norm"].ewm(span=20, adjust=False).mean()
    signed_bias = result[["innovation_n", "innovation_e", "innovation_u"]].ewm(
        span=40, adjust=False
    ).mean()
    result["innovation_bias_norm"] = np.linalg.norm(signed_bias.to_numpy(), axis=1)
    result[["ekf_n", "ekf_e", "ekf_u"]] = filtered
    result[["ekf_vn", "ekf_ve", "ekf_vu"]] = filtered_velocity
    result["ekf_speed_error"] = np.abs(innovations[:, 3])
    result["ekf_update_accepted"] = accepted
    deltas = result[["gnss_n", "gnss_e", "gnss_u"]].diff().fillna(0).to_numpy()
    result[["gnss_delta_n", "gnss_delta_e", "gnss_delta_u"]] = deltas
    result["gnss_step"] = np.linalg.norm(deltas, axis=1)
    return result


def generate_dataset(
    source: pd.DataFrame,
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Kaggle-conditioned carriers and inject controlled attacks.

    Every source segment produces matched Normal, Spoofing, and Jamming
    variants. The three variants always remain in the same partition so that a
    carrier trajectory cannot leak across train/validation/test boundaries.
    """
    seed_everything(cfg.seed)
    required_rows = cfg.flights_per_class * cfg.source_segment_rows
    if len(source) < required_rows:
        raise ValueError(
            f"Need at least {required_rows} Kaggle rows for "
            f"{cfg.flights_per_class} non-overlapping carriers; received {len(source)}."
        )
    ordered = source.sort_values("timestamp").reset_index(drop=False).rename(
        columns={"index": "source_row_index"}
    )
    master_rng = np.random.default_rng(cfg.seed)
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []
    flight_id = 0

    for source_segment_id in range(cfg.flights_per_class):
        row_start = source_segment_id * cfg.source_segment_rows
        row_end = row_start + cfg.source_segment_rows
        segment = ordered.iloc[row_start:row_end].copy()
        carrier_seed = int(master_rng.integers(0, 2**31 - 1))
        carrier = _build_kaggle_carrier(segment, np.random.default_rng(carrier_seed), cfg)
        carrier["source_segment_id"] = source_segment_id
        carrier["source_row_start"] = int(segment["source_row_index"].min())
        carrier["source_row_end"] = int(segment["source_row_index"].max())
        carrier["source_timestamp_start"] = str(segment["timestamp"].iloc[0])
        carrier["source_timestamp_end"] = str(segment["timestamp"].iloc[-1])
        carrier["source_dataset"] = "Kaggle UAV Autonomous Navigation Dataset"

        scenario_plan = [
            (0, "normal"),
            (1, "spoof_drift" if source_segment_id % 2 == 0 else "spoof_jump"),
            (2, "jam_noise" if source_segment_id % 2 == 0 else "jam_freeze"),
        ]
        for class_id, scenario in scenario_plan:
            attack_seed = int(master_rng.integers(0, 2**31 - 1))
            attacked = _inject_attack(
                carrier,
                np.random.default_rng(attack_seed),
                class_id,
                scenario,
                cfg,
            )
            attacked = run_ekf(attacked, cfg)
            attacked.insert(0, "flight_id", flight_id)
            attacked.insert(1, "scenario", scenario)
            frames.append(attacked)
            manifest_rows.append(
                {
                    "flight_id": flight_id,
                    "source_segment_id": source_segment_id,
                    "source_row_start": int(segment["source_row_index"].min()),
                    "source_row_end": int(segment["source_row_index"].max()),
                    "source_timestamp_start": str(segment["timestamp"].iloc[0]),
                    "source_timestamp_end": str(segment["timestamp"].iloc[-1]),
                    "scenario": scenario,
                    "flight_class": class_id,
                    "flight_class_name": CLASS_NAMES[class_id],
                    "carrier_seed": carrier_seed,
                    "attack_seed": attack_seed,
                }
            )
            flight_id += 1
    return pd.concat(frames, ignore_index=True), pd.DataFrame(manifest_rows)


def grouped_scenario_split(manifest: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Split source carriers 70/15/15 and keep all matched variants together."""
    rng = np.random.default_rng(cfg.seed + 101)
    split_manifest = manifest.copy()
    split_manifest["split"] = ""
    source_ids = np.sort(manifest["source_segment_id"].unique())
    even_ids = source_ids[source_ids % 2 == 0].copy()
    odd_ids = source_ids[source_ids % 2 == 1].copy()
    rng.shuffle(even_ids)
    rng.shuffle(odd_ids)
    interleaved = np.ravel(np.column_stack([even_ids, odd_ids]))
    n = len(interleaved)
    n_train = max(1, int(round(n * cfg.train_fraction)))
    n_val = max(1, int(round(n * cfg.val_fraction)))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    assignments = {
        "train": interleaved[:n_train],
        "validation": interleaved[n_train : n_train + n_val],
        "test": interleaved[n_train + n_val :],
    }
    for split, group_ids in assignments.items():
        split_manifest.loc[
            split_manifest["source_segment_id"].isin(group_ids), "split"
        ] = split
    assert (split_manifest["split"] != "").all()
    assert split_manifest.groupby("source_segment_id")["split"].nunique().max() == 1
    return split_manifest


def make_windows(
    dataset: pd.DataFrame,
    split_manifest: pd.DataFrame,
    cfg: ExperimentConfig,
    features: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Build causal windows; the class at the final sample is the target."""
    merged = dataset.merge(
        split_manifest[["flight_id", "split", "source_segment_id"]],
        on="flight_id",
        how="left",
        suffixes=("", "_manifest"),
    )
    output: dict[str, dict[str, list]] = {
        split: {
            "X": [], "y": [], "flight_id": [], "source_segment_id": [],
            "end_time_s": [], "attack_strength": []
        }
        for split in ["train", "validation", "test"]
    }
    for flight_id, frame in merged.groupby("flight_id", sort=True):
        frame = frame.sort_values("time_s")
        split = frame["split"].iloc[0]
        values = frame[features].to_numpy(dtype=np.float32)
        labels = frame["label"].to_numpy(dtype=np.int64)
        times = frame["time_s"].to_numpy(dtype=np.float32)
        strengths = frame["attack_strength"].to_numpy(dtype=np.float32)
        for end in range(cfg.window_length, len(frame) + 1, cfg.stride):
            start = end - cfg.window_length
            output[split]["X"].append(values[start:end])
            output[split]["y"].append(labels[end - 1])
            output[split]["flight_id"].append(flight_id)
            output[split]["source_segment_id"].append(int(frame["source_segment_id"].iloc[0]))
            output[split]["end_time_s"].append(times[end - 1])
            output[split]["attack_strength"].append(strengths[end - 1])
    packed: dict[str, dict[str, np.ndarray]] = {}
    for split, values in output.items():
        packed[split] = {
            "X": np.stack(values["X"]).astype(np.float32),
            "y": np.asarray(values["y"], dtype=np.int64),
            "flight_id": np.asarray(values["flight_id"], dtype=np.int64),
            "source_segment_id": np.asarray(values["source_segment_id"], dtype=np.int64),
            "end_time_s": np.asarray(values["end_time_s"], dtype=np.float32),
            "attack_strength": np.asarray(values["attack_strength"], dtype=np.float32),
        }
    return packed


def fit_sequence_scaler(windows: dict[str, dict[str, np.ndarray]]) -> StandardScaler:
    scaler = StandardScaler()
    train = windows["train"]["X"]
    scaler.fit(train.reshape(-1, train.shape[-1]))
    return scaler


def apply_sequence_scaler(
    windows: dict[str, dict[str, np.ndarray]], scaler: StandardScaler
) -> dict[str, dict[str, np.ndarray]]:
    scaled = deepcopy(windows)
    for split in scaled:
        X = scaled[split]["X"]
        scaled[split]["X"] = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
    return scaled


def summarize_windows(X: np.ndarray) -> np.ndarray:
    """Convert each sequence into transparent distribution/dynamics summaries."""
    time_axis = np.linspace(-1, 1, X.shape[1], dtype=np.float32)
    denom = np.sum(time_axis**2)
    centered = X - X.mean(axis=1, keepdims=True)
    slope = np.einsum("ntf,t->nf", centered, time_axis) / denom
    return np.concatenate(
        [X.mean(1), X.std(1), X.min(1), X.max(1), X[:, -1, :], slope], axis=1
    ).astype(np.float32)


class CNNClassifier(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_features, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.transpose(1, 2)))


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 64, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Dropout(dropout), nn.Linear(64, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1])


class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(64, 48, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(96, 48), nn.Tanh(), nn.Linear(48, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(96), nn.Dropout(dropout), nn.Linear(96, 48), nn.ReLU(), nn.Dropout(dropout), nn.Linear(48, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.bilstm(encoded)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.head(pooled)


class FocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 1.5, label_smoothing: float = 0.02):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        return ((1.0 - pt).pow(self.gamma) * ce).mean()


def _loader(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ExperimentConfig,
    training: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    if training:
        return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    return DataLoader(dataset, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)


@torch.no_grad()
def predict_torch(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    probabilities = []
    for (features,) in loader:
        probabilities.append(torch.softmax(model(features.to(device)), dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def train_torch_model(
    model: nn.Module,
    windows: dict[str, dict[str, np.ndarray]],
    cfg: ExperimentConfig,
    device: torch.device,
    model_name: str,
) -> tuple[nn.Module, pd.DataFrame]:
    """Train with balanced sampling and select strictly on validation macro-F1."""
    train_X, train_y = windows["train"]["X"], windows["train"]["y"]
    val_X, val_y = windows["validation"]["X"], windows["validation"]["y"]
    class_counts = np.bincount(train_y, minlength=3)
    class_weights = np.sqrt(class_counts.sum() / (3 * np.maximum(class_counts, 1)))
    class_weights /= class_weights.mean()
    loss_fn = FocalLoss(torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    train_loader = _loader(train_X, train_y, cfg, training=True)
    model = model.to(device)
    best_state = deepcopy(model.state_dict())
    best_score = -np.inf
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        correct = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(targets)
            correct += int((logits.argmax(1) == targets).sum().item())
            seen += len(targets)
        val_prob = predict_torch(model, val_X, device)
        val_pred = val_prob.argmax(1)
        val_f1 = f1_score(val_y, val_pred, average="macro", zero_division=0)
        val_recall = recall_score(val_y, val_pred, average="macro", zero_division=0)
        scheduler.step(val_f1)
        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": running_loss / seen,
                "train_accuracy": correct / seen,
                "val_macro_f1": val_f1,
                "val_macro_recall": val_recall,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if val_f1 > best_score + cfg.min_delta:
            best_score = val_f1
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def train_classical_models(
    windows: dict[str, dict[str, np.ndarray]], seed: int
) -> dict[str, object]:
    X_train = summarize_windows(windows["train"]["X"])
    y_train = windows["train"]["y"]
    models: dict[str, object] = {
        "Random Forest": RandomForestClassifier(
            n_estimators=450,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "SVM": Pipeline(
            [
                ("scale", StandardScaler()),
                ("svc", SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
            ]
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def _classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> tuple[dict, pd.DataFrame]:
    prediction = probability.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, labels=np.arange(3), zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, prediction, average="weighted", zero_division=0
    )
    try:
        macro_roc_auc = roc_auc_score(
            label_binarize(y_true, classes=np.arange(3)), probability, average="macro", multi_class="ovr"
        )
    except ValueError:
        macro_roc_auc = np.nan
    metrics = {
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": weighted_precision,
        "Recall": weighted_recall,
        "F1": weighted_f1,
        "Macro Precision": precision.mean(),
        "Macro F1": f1.mean(),
        "Macro Recall": recall.mean(),
        "Macro ROC-AUC": macro_roc_auc,
    }
    per_class = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support,
        }
    )
    return metrics, per_class


def tune_ekf_threshold(
    raw_windows: dict[str, dict[str, np.ndarray]], full_features: list[str]
) -> dict[str, float]:
    """Tune a transparent two-threshold EKF baseline on validation data only."""
    nis_idx = full_features.index("nis")
    innov_idx = full_features.index("innovation_norm")
    step_idx = full_features.index("gnss_step")
    train_X, train_y = raw_windows["train"]["X"], raw_windows["train"]["y"]
    val_X, val_y = raw_windows["validation"]["X"], raw_windows["validation"]["y"]
    normal_nis = train_X[train_y == 0, :, nis_idx].mean(axis=1)
    best = {"score": -1.0}
    for q in [0.90, 0.925, 0.95, 0.975, 0.99]:
        anomaly_threshold = float(np.quantile(normal_nis, q))
        for variability_threshold in np.quantile(
            val_X[:, :, innov_idx].std(axis=1), [0.35, 0.5, 0.65, 0.8]
        ):
            pred = ekf_threshold_predict(
                val_X,
                full_features,
                anomaly_threshold,
                float(variability_threshold),
            )
            score = f1_score(val_y, pred, average="macro", zero_division=0)
            if score > best["score"]:
                best = {
                    "score": float(score),
                    "anomaly_threshold": anomaly_threshold,
                    "variability_threshold": float(variability_threshold),
                }
    return best


def ekf_threshold_predict(
    X: np.ndarray,
    features: list[str],
    anomaly_threshold: float,
    variability_threshold: float,
) -> np.ndarray:
    nis = X[:, :, features.index("nis")].mean(axis=1)
    variability = X[:, :, features.index("innovation_norm")].std(axis=1)
    pred = np.zeros(len(X), dtype=np.int64)
    anomalous = nis > anomaly_threshold
    pred[anomalous & (variability <= variability_threshold)] = 1
    pred[anomalous & (variability > variability_threshold)] = 2
    return pred


def hard_probabilities(prediction: np.ndarray, confidence: float = 0.98) -> np.ndarray:
    probability = np.full((len(prediction), 3), (1 - confidence) / 2, dtype=np.float64)
    probability[np.arange(len(prediction)), prediction] = confidence
    return probability


def tune_class_bias(y_true: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Tune decision costs on validation macro-F1, macro recall and attack recall."""
    best_score = -np.inf
    best_bias = np.ones(3, dtype=np.float64)
    for normal_bias in [0.75, 0.85, 1.0, 1.15]:
        for spoof_bias in [0.9, 1.1, 1.3, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5]:
            for jam_bias in [0.85, 1.0, 1.15, 1.3]:
                bias = np.array([normal_bias, spoof_bias, jam_bias])
                pred = (probability * bias).argmax(axis=1)
                macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
                macro_recall = recall_score(y_true, pred, average="macro", zero_division=0)
                per_class_recall = recall_score(
                    y_true, pred, labels=np.arange(3), average=None, zero_division=0
                )
                attack_recall = per_class_recall[1:].mean()
                minimum_recall = per_class_recall.min()
                score = (
                    0.35 * macro_f1
                    + 0.25 * macro_recall
                    + 0.25 * attack_recall
                    + 0.15 * minimum_recall
                )
                if score > best_score:
                    best_score = score
                    best_bias = bias
    return best_bias


def apply_class_bias(probability: np.ndarray, bias: np.ndarray) -> np.ndarray:
    adjusted = probability * bias
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def evaluate_models(
    classical_models: dict[str, object],
    neural_models: dict[str, tuple[nn.Module, dict[str, dict[str, np.ndarray]]]],
    raw_full_windows: dict[str, dict[str, np.ndarray]],
    full_features: list[str],
    threshold_config: dict[str, float],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[float]]]:
    y_true = raw_full_windows["test"]["y"]
    probabilities: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    class_biases: dict[str, list[float]] = {}
    reference_windows = neural_models["CNN-BiLSTM with EKF"][1]
    classical_X = summarize_windows(reference_windows["test"]["X"])
    classical_val_X = summarize_windows(reference_windows["validation"]["X"])
    y_val = raw_full_windows["validation"]["y"]
    for name, model in classical_models.items():
        validation_probability = model.predict_proba(classical_val_X)
        bias = tune_class_bias(y_val, validation_probability)
        probabilities[name] = apply_class_bias(model.predict_proba(classical_X), bias)
        class_biases[name] = bias.tolist()
    for name, (model, windows) in neural_models.items():
        validation_probability = predict_torch(model, windows["validation"]["X"], device)
        bias = tune_class_bias(y_val, validation_probability)
        probabilities[name] = apply_class_bias(predict_torch(model, windows["test"]["X"], device), bias)
        class_biases[name] = bias.tolist()
    threshold_pred = ekf_threshold_predict(
        raw_full_windows["test"]["X"],
        full_features,
        threshold_config["anomaly_threshold"],
        threshold_config["variability_threshold"],
    )
    probabilities["EKF threshold"] = hard_probabilities(threshold_pred)
    class_biases["EKF threshold"] = [1.0, 1.0, 1.0]

    metric_rows = []
    class_rows = []
    for name, probability in probabilities.items():
        metrics, per_class = _classification_metrics(y_true, probability)
        metric_rows.append({"Model": name, **metrics})
        per_class.insert(0, "Model", name)
        class_rows.append(per_class)
        predictions[name] = probability.argmax(1)
    metrics_df = pd.DataFrame(metric_rows).sort_values(["Macro F1", "Macro Recall"], ascending=False)
    per_class_df = pd.concat(class_rows, ignore_index=True)
    return metrics_df, per_class_df, probabilities, predictions, class_biases


def flight_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cluster_ids: np.ndarray,
    metric: str = "macro_f1",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Cluster bootstrap over independent carrier/source identifiers.

    The retained function name is for backward compatibility. In the final
    experiment ``cluster_ids`` are Kaggle source-segment identifiers, so all
    matched scenarios and overlapping windows are resampled together.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(cluster_ids)
    scores = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(cluster_ids == cid) for cid in sampled])
        if metric == "macro_f1":
            score = f1_score(y_true[indices], y_pred[indices], average="macro", zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(y_true[indices], y_pred[indices])
        else:
            raise ValueError(metric)
        scores.append(score)
    return tuple(np.quantile(scores, [0.025, 0.975]))


def benchmark_latency(
    neural_models: dict[str, tuple[nn.Module, dict[str, dict[str, np.ndarray]]]],
    device: torch.device,
    warmups: int = 20,
    runs: int = 200,
) -> pd.DataFrame:
    rows = []
    for name, (model, windows) in neural_models.items():
        model.eval()
        sample = torch.from_numpy(windows["test"]["X"][:1]).to(device)
        with torch.no_grad():
            for _ in range(warmups):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(runs):
                start = time.perf_counter()
                model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        rows.append(
            {
                "Model": name,
                "Device": str(device),
                "Mean latency (ms/window)": np.mean(times),
                "P95 latency (ms/window)": np.quantile(times, 0.95),
                "Throughput (windows/s)": 1000 / np.mean(times),
                "Parameters": sum(p.numel() for p in model.parameters()),
            }
        )
    return pd.DataFrame(rows)


def sequence_permutation_importance(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    device: torch.device,
    repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure macro-F1 loss when one entire feature trajectory is permuted."""
    rng = np.random.default_rng(seed)
    baseline = f1_score(y, predict_torch(model, X, device).argmax(1), average="macro", zero_division=0)
    rows = []
    for feature_index, feature in enumerate(feature_names):
        drops = []
        for _ in range(repeats):
            permuted = X.copy()
            order = rng.permutation(len(X))
            permuted[:, :, feature_index] = permuted[order, :, feature_index]
            score = f1_score(y, predict_torch(model, permuted, device).argmax(1), average="macro", zero_division=0)
            drops.append(baseline - score)
        rows.append(
            {
                "Feature": feature,
                "Macro-F1 decrease": np.mean(drops),
                "Std": np.std(drops),
                "Feature group": "EKF residual" if feature in EKF_FEATURES else "Navigation/IMU",
            }
        )
    return pd.DataFrame(rows).sort_values("Macro-F1 decrease", ascending=False)


def gradient_shap_importance(
    model: nn.Module,
    background_X: np.ndarray,
    explanation_X: np.ndarray,
    feature_names: list[str],
    background_size: int = 64,
    explanation_size: int = 96,
    nsamples: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute global GradientSHAP importance on complete standardised windows.

    Attribution is calculated on CPU for reproducibility and then aggregated as
    mean absolute logit contribution over explained windows, time, and classes.
    """
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required for gradient_shap_importance") from exc

    rng = np.random.default_rng(seed)
    background_indices = rng.choice(
        len(background_X), min(background_size, len(background_X)), replace=False
    )
    explanation_indices = rng.choice(
        len(explanation_X), min(explanation_size, len(explanation_X)), replace=False
    )
    cpu_model = deepcopy(model).cpu().eval()
    background = torch.from_numpy(background_X[background_indices]).float()
    examples = torch.from_numpy(explanation_X[explanation_indices]).float()
    explainer = shap.GradientExplainer(cpu_model, background)
    shap_values = explainer.shap_values(examples, nsamples=nsamples, rseed=seed)
    if isinstance(shap_values, list):
        values = np.stack(shap_values, axis=-1)
    else:
        values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[..., None]
    if values.shape[2] != len(feature_names) and values.shape[-1] == len(feature_names):
        values = np.moveaxis(values, -1, 2)
    if values.shape[2] != len(feature_names):
        raise ValueError(f"Unexpected SHAP output shape {values.shape}")
    importance = np.mean(np.abs(values), axis=tuple(i for i in range(values.ndim) if i != 2))
    total = float(importance.sum())
    rows = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean absolute SHAP value": importance,
            "Normalised importance": importance / total if total > 0 else importance,
            "Feature group": [
                "EKF residual" if feature in EKF_FEATURES else "Navigation/IMU"
                for feature in feature_names
            ],
            "Explained windows": len(explanation_indices),
            "Background windows": len(background_indices),
        }
    )
    return rows.sort_values("Mean absolute SHAP value", ascending=False).reset_index(drop=True)


def save_table(df: pd.DataFrame, path: str | Path, float_format: str = "%.6f") -> None:
    df.to_csv(path, index=False, float_format=float_format)


def save_checkpoint(model: nn.Module, path: str | Path, metadata: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def save_run_manifest(path: str | Path, cfg: ExperimentConfig, extra: dict) -> None:
    payload = {"config": asdict(cfg), **extra}
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def plot_class_distribution(windows: dict[str, dict[str, np.ndarray]], path: str | Path) -> None:
    rows = []
    for split, values in windows.items():
        counts = np.bincount(values["y"], minlength=3)
        for i, count in enumerate(counts):
            rows.append({"Split": split.title(), "Class": CLASS_NAMES[i], "Windows": count})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = table.pivot(index="Split", columns="Class", values="Windows").reindex(
        index=["Train", "Validation", "Test"], columns=CLASS_NAMES
    )
    pivot.plot(kind="bar", ax=ax, color=[CLASS_COLORS[c] for c in CLASS_NAMES], width=0.75)
    ax.set(title="Window class distribution by source-group-disjoint split", xlabel="", ylabel="Number of windows")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_residual_concept(dataset: pd.DataFrame, path: str | Path) -> None:
    example = dataset.query("scenario == 'spoof_drift'").groupby("flight_id").head(10)
    flight_id = int(example["flight_id"].iloc[0])
    frame = dataset[dataset["flight_id"] == flight_id]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    axes[0].plot(frame["time_s"], frame["true_n"], color="#6C757D", label="Reference north")
    axes[0].plot(frame["time_s"], frame["gnss_n"], color=CLASS_COLORS["Spoofing"], alpha=0.8, label="Observed GNSS north")
    axes[0].set(ylabel="North position (m)", title="EKF residual generation: prediction–measurement inconsistency")
    axes[0].legend(ncol=2)
    axes[1].plot(frame["time_s"], frame["innovation_norm"], color="#2F6690", label="Innovation norm")
    axes[1].fill_between(frame["time_s"], 0, frame["innovation_norm"], where=frame["label"].eq(1), color=CLASS_COLORS["Spoofing"], alpha=0.2, label="Spoofing interval")
    axes[1].set(xlabel="Time (s)", ylabel="Innovation norm (m)")
    axes[1].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: pd.DataFrame, path: str | Path) -> None:
    models = history["model"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    palette = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for color, model_name in zip(palette, models):
        frame = history[history["model"] == model_name]
        axes[0].plot(frame["epoch"], frame["train_loss"], color=color, label=model_name)
        axes[1].plot(frame["epoch"], frame["val_macro_f1"], color=color, label=model_name)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Focal loss")
    axes[1].set(title="Validation macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1.02))
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(3), normalize="true")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, f"{matrix[row, col]:.1%}", ha="center", va="center", color="white" if matrix[row, col] > 0.55 else "#212529")
    ax.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(3),
        yticks=np.arange(3),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized proportion")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame, path: str | Path) -> None:
    ordered = metrics.sort_values("Macro F1")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    width = 0.24
    for offset, metric, color in [(-width, "Accuracy", "#A7C7E7"), (0, "Macro Recall", "#E09F3E"), (width, "Macro F1", "#2F6690")]:
        bars = ax.barh(y + offset, ordered[metric], height=width, label=metric, color=color)
        for bar, value in zip(bars, ordered[metric]):
            ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    ax.set(
        yticks=y,
        yticklabels=ordered["Model"],
        xlim=(0, 1.08),
        xlabel="Score",
        title="Held-out source-group test performance",
    )
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ablation(metrics: pd.DataFrame, path: str | Path) -> None:
    names = ["CNN-BiLSTM without EKF", "CNN-BiLSTM with EKF"]
    frame = metrics.set_index("Model").loc[names]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, name in enumerate(names):
        values = frame.loc[name, ["Accuracy", "Macro Recall", "Macro F1"]].to_numpy(dtype=float)
        ax.bar(x + (i - 0.5) * 0.34, values, width=0.34, color=["#E09F3E", "#2F6690"][i], label=name)
    ax.set(xticks=x, xticklabels=["Accuracy", "Macro recall", "Macro F1"], ylim=(0, 1.05), ylabel="Score", title="Controlled EKF residual ablation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, probabilities: dict[str, np.ndarray], path: str | Path) -> None:
    selected = ["Random Forest", "CNN-BiLSTM without EKF", "CNN-BiLSTM with EKF"]
    y_bin = label_binarize(y_true, classes=np.arange(3))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    for class_index, ax in enumerate(axes):
        for name in selected:
            fpr, tpr, _ = roc_curve(y_bin[:, class_index], probabilities[name][:, class_index])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", color=MODEL_COLORS[name])
        ax.plot([0, 1], [0, 1], "--", color="#ADB5BD", linewidth=1)
        ax.set(title=CLASS_NAMES[class_index], xlabel="False-positive rate", ylabel="True-positive rate")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("One-vs-rest ROC curves on held-out flights", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, path: str | Path, top_n: int = 12) -> None:
    frame = importance.head(top_n).sort_values("Macro-F1 decrease")
    colors = frame["Feature group"].map({"EKF residual": "#2F6690", "Navigation/IMU": "#B8B8B8"})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(frame["Feature"], frame["Macro-F1 decrease"], xerr=frame["Std"], color=colors, capsize=3)
    ax.axvline(0, color="#212529", linewidth=0.8)
    ax.set(title="Sequence permutation importance for the proposed model", xlabel="Decrease in validation macro-F1", ylabel="")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_shap_importance(importance: pd.DataFrame, path: str | Path, top_n: int = 12) -> None:
    frame = importance.head(top_n).sort_values("Mean absolute SHAP value")
    colors = frame["Feature group"].map(
        {"EKF residual": "#2F6690", "Navigation/IMU": "#B8B8B8"}
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    bars = ax.barh(frame["Feature"], frame["Mean absolute SHAP value"], color=colors)
    for bar, value in zip(bars, frame["Mean absolute SHAP value"]):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f"  {value:.4f}",
            va="center",
            fontsize=8.5,
        )
    ax.set(
        title="GradientSHAP global feature importance (validation windows)",
        xlabel="Mean absolute SHAP value (model-logit units)",
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_attack_subtype_recall(subtype: pd.DataFrame, path: str | Path) -> None:
    """Publication-quality subtype recall chart with exact values and support."""
    label_map = {
        "spoof_drift": "Gradual spoofing\ndrift",
        "spoof_jump": "Spoofing\njump",
        "jam_noise": "Noise\njamming",
        "jam_freeze": "GNSS\nfreeze",
    }
    order = list(label_map)
    frame = subtype.set_index("Attack subtype").loc[order].reset_index()
    labels = [label_map[value] for value in frame["Attack subtype"]]
    colors = ["#D58A28", "#E09F3E", "#2F6690", "#6A4C93"]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    bars = ax.bar(labels, frame["Recall"], color=colors, width=0.64, edgecolor="#243447", linewidth=0.7)
    for bar, recall, support in zip(bars, frame["Recall"], frame["Support"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            recall + 0.025,
            f"{recall:.1%}\n$n={int(support):,}$",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
        )
    ax.set(ylabel="Recall on held-out windows", xlabel="", ylim=(0, 1.13))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.suptitle(
        "Detection recall by attack subtype",
        x=0.125,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.125,
        0.935,
        "Proposed CNN–BiLSTM with EKF; grouped 70/15/15 source-carrier split",
        ha="left",
        fontsize=9.5,
        color="#52616B",
    )
    ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_seed_robustness(results: pd.DataFrame, path: str | Path) -> None:
    summary = (
        results.groupby("Model")[["Accuracy", "Macro Recall", "Macro F1"]]
        .agg(["mean", "std"])
        .reindex(["CNN-BiLSTM without EKF", "CNN-BiLSTM with EKF"])
    )
    metrics = ["Accuracy", "Macro Recall", "Macro F1"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for index, (name, color) in enumerate(
        [("CNN-BiLSTM without EKF", "#E09F3E"), ("CNN-BiLSTM with EKF", "#2F6690")]
    ):
        means = np.array([summary.loc[name, (metric, "mean")] for metric in metrics])
        stds = np.array([summary.loc[name, (metric, "std")] for metric in metrics])
        bars = ax.bar(
            x + (index - 0.5) * 0.34,
            means,
            width=0.34,
            yerr=stds,
            capsize=4,
            color=color,
            label=name,
        )
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.025, f"{mean:.3f}", ha="center", fontsize=8)
    ax.set(
        xticks=x,
        xticklabels=["Accuracy", "Macro recall", "Macro F1"],
        ylim=(0, 1.08),
        ylabel="Mean held-out score (error bars: ±1 SD)",
        title=f"Training-seed robustness ({results['Seed'].nunique()} seeds; fixed grouped split)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _diagram(path: str | Path, title: str, nodes: list[tuple[float, float, str, str]], arrows: list[tuple[int, int]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12.35)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=18)
    boxes = []
    for x, y, text, color in nodes:
        box = FancyBboxPatch((x, y), 2.1, 0.9, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=color, edgecolor="#343A40", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + 1.05, y + 0.45, text, ha="center", va="center", fontsize=9, color="#111827", wrap=True)
        boxes.append((x, y))
    for source, target in arrows:
        sx, sy = boxes[source]
        tx, ty = boxes[target]
        arrow = FancyArrowPatch((sx + 2.1, sy + 0.45), (tx, ty + 0.45), arrowstyle="-|>", mutation_scale=13, color="#495057", linewidth=1.25, connectionstyle="arc3,rad=0.0")
        ax.add_patch(arrow)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_methodology_diagram(path: str | Path) -> None:
    nodes = [
        (0.2, 3.2, "Kaggle telemetry\nsegments", "#E9ECEF"),
        (2.7, 3.2, "Carrier construction +\nattack injection", "#FCE8D5"),
        (5.2, 3.2, "Nonlinear EKF +\ninnovation residuals", "#DDEAF3"),
        (7.7, 3.2, "Causal sequence\nwindows", "#E8E5F2"),
        (10.0, 3.2, "Models, ablation\n& held-out test", "#E2F0D9"),
    ]
    _diagram(path, "Overall methodology pipeline", nodes, [(0, 1), (1, 2), (2, 3), (3, 4)])


def make_architecture_diagram(path: str | Path) -> None:
    nodes = [
        (0.2, 3.2, "40 × 17\nfeature window", "#E9ECEF"),
        (2.7, 3.2, "1-D convolutions\n48 → 64 channels", "#DDEAF3"),
        (5.2, 3.2, "Bidirectional LSTM\n48 + 48 units", "#E8E5F2"),
        (7.7, 3.2, "Attention pooling +\ndense 96 → 48", "#FCE8D5"),
        (10.0, 3.2, "Softmax output\nNormal / Spoof / Jam", "#E2F0D9"),
    ]
    _diagram(path, "Proposed EKF residual-enhanced CNN–BiLSTM", nodes, [(0, 1), (1, 2), (2, 3), (3, 4)])


def dataframe_markdown(df: pd.DataFrame, decimals: int = 3) -> str:
    return df.to_markdown(index=False, floatfmt=f".{decimals}f")
