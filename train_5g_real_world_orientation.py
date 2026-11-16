#!/usr/bin/env python3

"""
Training script for channel charting for position and orientation in real-world coordinates
with fixed or Optuna optimized parameters.

This script implements channel charting for position and orientation [1] 
using channel-state information (CSI) from the 5G physical uplink shared channel
(PUSCH). It uses triplet, bilateration, and bounding-box losses for position [2, 3] and
extends this by a novel orientation triplet [1] and orientation alignment [1] loss.

This code is based on https://github.com/IIP-Group/ChannelCharting_RealWorld used in [2, 3],
uses the CAEZ-5G-OUTDOOR dataset in [3] and is adapted for the joint estimation of 
position and orientation in [1].

Set :data:`METHOD` and :data:`USE_OPTUNA` below to choose the workflow:

* ``METHOD = "channel_charting"``: self-supervised method for position and orientation estimation.
* ``METHOD = "neural_positioning"``: supervised baseline for position and orientation estimation.
* ``USE_OPTUNA = False`` uses the method's fixed paper parameters.
* ``USE_OPTUNA = True`` runs/resumes a method-specific multi-objective search.

@author: Sueda Taner, modified by Reinhard Wiesmayr and Daniel Richner

References
----------
[1] D. Richner, R. Wiesmayr, F. Zumegen, and C. Studer, 
    "Channel Charting for Position and Orientation," arXiv:2606.18151, Jun. 2026.
[2] S. Taner, V. Palhares, and C. Studer, "Channel charting in real-world
    coordinates with distributed MIMO," IEEE Trans. Wireless Commun., 2025.
[3] R. Wiesmayr et al., "CSI-based user positioning, channel charting, and
    device classification with an NVIDIA 5G testbed," Asilomar, 2025.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.cc_helpers import FeatureExtractor, angular_metrics, evaluate_cc, plot_chart
from utils.nn_models import SupervisedModel, TripletModel, predict
from utils.parameter import Parameter
from utils.segment_processing import find_bounding_box


# =============================================================================
# User configuration
# =============================================================================

# These independent switches provide four supported configurations.
METHOD = "channel_charting" # "channel_charting"  or "neural_positioning".
USE_OPTUNA = False  # Change to True to run or resume parameter optimization.
SUPPORTED_METHODS = ("channel_charting", "neural_positioning")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR.parent / "datasets").resolve()
RESULTS_DIR = SCRIPT_DIR / "results"

PRIMARY_DATASET = DATA_DIR / (
    "robot_measurement_outdoor_2025_10_11_auto_correlation_no_norm_"
    "truncated_25_sorted_4_rx_ant_wo_last_10000.npz"
)
HELD_OUT_DATASET = DATA_DIR / (
    "robot_measurement_outdoor_2025_10_11_auto_correlation_no_norm_"
    "truncated_25_sorted_4_rx_ant_only_last_10000.npz"
)

RUN_NAME = "caez_5g_outdoor"
RANDOM_SEED = 10
CUDA_DEVICE_INDEX = 0
VALIDATION_RATIO = 0.20

SAVE_RESULTS = True
EVALUATE_TRAINING_SET = False
SAVE_OPTUNA_TRIAL_ARTIFACTS = True

OPTUNA_TRIALS = 100

# Plotting configuration used by the paper-style figures.
AXIS_LIMITS = (-6.0, 6.0)
AXIS_TICKS = (-6, -4, -2, 0, 2, 4, 6)
ARROW_START_INDEX = 9000
ARROW_STEP = 20
ARROW_LENGTH = 0.5

# Dataset settings that are shared by both learning methods.
FEATURE_METHOD = "none"
NUM_DELAY_TAPS = 50
POWER_THRESHOLD_DB = -np.inf
POWER_SOURCE = "csi"  # Either "csi" or "distance".
BOUNDING_BOX_SOURCE = "ap"  # Either "ap" or "genie".
BOUNDING_BOX_MODE = 1


@dataclass(frozen=True)
class HyperParameters:
    """Parameters that define one training run."""

    lambda_b: float
    lambda_box: float
    lambda_rotation: float
    lambda_align: float
    M_t: float
    M_r: float
    M_b: float
    M_p: float
    Tc: float
    Tf: float
    learning_rate: float
    scheduler_step: int
    triplet_start_epoch: int
    hidden_features: tuple[int, ...]
    num_epochs: int
    batch_size: int
    num_triplets_per_anchor: int
    use_scheduler: bool
    num_reference_points: int


# Fixed configuration selected by the channel-charting Optuna study.
CHANNEL_CHARTING_FIXED_HYPERPARAMETERS = HyperParameters(
    lambda_b=1.38787620584494,
    lambda_box=0.388712221671457,
    lambda_rotation=0.0682341444208285,
    lambda_align=0.411325495545273,
    M_t=0.900165423923795,
    M_r=3.14209619695774,
    M_b=3.94142500609904,
    M_p=13.04573355,
    Tc=3.23441518569091,
    Tf=63.3156398134664,
    learning_rate=0.00438910713545695,
    scheduler_step=35,
    triplet_start_epoch=5,
    hidden_features=(192, 96, 48, 48, 48),
    num_epochs=200,
    batch_size=2048,
    num_triplets_per_anchor=2,
    use_scheduler=True,
    num_reference_points=0,
)

#  Fixed configuration for neural positioning.
NEURAL_POSITIONING_FIXED_HYPERPARAMETERS = HyperParameters(
    lambda_b=10.0,
    lambda_box=1.0,
    lambda_rotation=0.5,
    lambda_align=0.0,
    M_t=0.3,
    M_r=1.0,
    M_b=1.0,
    M_p=13.0,
    Tc=1.5,
    Tf=20.0,
    learning_rate=0.004,
    scheduler_step=50,
    triplet_start_epoch=0,
    hidden_features=(192, 96, 48, 48, 48),
    num_epochs=200,
    batch_size=2048,
    num_triplets_per_anchor=2,
    use_scheduler=False,
    num_reference_points=50,
)
OBJECTIVE_NAMES = ("KS", "1-TW", "1-CT", "MDE", "cos_dist", "ang_err")


@dataclass
class ExperimentData:
    """Prepared arrays and metadata shared by all training trials."""

    base_parameters: Parameter
    feature_extractor: FeatureExtractor
    access_point_positions: np.ndarray
    bounding_boxes: np.ndarray
    box_labels_train: np.ndarray | None
    features_train: torch.Tensor
    features_validation: torch.Tensor | None
    timestamps_train: torch.Tensor
    power_train_db: torch.Tensor | None
    positions_train: np.ndarray
    positions_validation: np.ndarray
    positions_held_out: np.ndarray
    rotations_train: np.ndarray
    rotations_validation: np.ndarray
    rotations_held_out: np.ndarray
    timestamps_validation: np.ndarray
    timestamps_held_out: np.ndarray
    channels_validation: np.ndarray
    channels_held_out: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    segment_start_indices: list[int]

    @property
    def num_access_points(self) -> int:
        return self.access_point_positions.shape[0]


@dataclass
class EvaluationResult:
    """Predictions and scalar metrics; rotations use (sin(phi), cos(phi))."""

    positions: np.ndarray
    rotations: np.ndarray
    metrics: dict[str, float]


@dataclass
class TrainingHistory:
    """Loss curves shared by channel charting and neural positioning."""

    total: np.ndarray
    validation: np.ndarray | None = None
    position: np.ndarray | None = None
    rotation: np.ndarray | None = None


def validate_method(method: str) -> None:
    if method not in SUPPORTED_METHODS:
        choices = ", ".join(f'"{choice}"' for choice in SUPPORTED_METHODS)
        raise ValueError(f"METHOD must be one of: {choices}.")


def result_directory(method: str, use_optuna: bool) -> Path:
    """Return the isolated output directory for one method and parameter mode."""
    validate_method(method)
    parameter_directory = "optuna_search" if use_optuna else "fixed_hyperparameters"
    return RESULTS_DIR / method / parameter_directory


def fixed_hyperparameters(method: str) -> HyperParameters:
    validate_method(method)
    if method == "neural_positioning":
        return NEURAL_POSITIONING_FIXED_HYPERPARAMETERS
    return CHANNEL_CHARTING_FIXED_HYPERPARAMETERS


def active_hyperparameters(hp: HyperParameters, method: str) -> dict[str, Any]:
    """Return only parameters that affect the selected method's loss."""
    validate_method(method)
    values = asdict(hp)
    if method == "neural_positioning":
        active_names = (
            "lambda_rotation",
            "learning_rate",
            "hidden_features",
            "num_epochs",
            "batch_size",
            "use_scheduler",
            "scheduler_step",
        )
        return {name: values[name] for name in active_names}
    return values


def optuna_study_name(method: str) -> str:
    validate_method(method)
    return f"{RUN_NAME}_{method}_multiobjective"


def select_device() -> torch.device:
    """Use the requested CUDA device when available, otherwise fall back safely."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if CUDA_DEVICE_INDEX < torch.cuda.device_count():
        return torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
    print(f"CUDA device {CUDA_DEVICE_INDEX} is unavailable; using cuda:0 instead.")
    return torch.device("cuda:0")


def seed_everything(seed: int) -> None:
    """Seed NumPy and PyTorch for repeatable splitting and training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_dataset(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Update DATA_DIR or the dataset filename in the configuration section."
        )


def reshape_channels(channels: np.ndarray) -> np.ndarray:
    """Merge the access-point and receive-antenna axes."""
    if channels.ndim != 4:
        raise ValueError(f"Expected a 4-D channel array, received {channels.shape}.")
    users, access_points, antennas, taps = channels.shape
    return channels.reshape(users, access_points * antennas, taps)


def load_npz_fields(path: Path, fields: tuple[str, ...]) -> dict[str, Any]:
    """Load requested fields and close the NPZ archive immediately."""
    with np.load(path, allow_pickle=True) as archive:
        missing = sorted(set(fields) - set(archive.files))
        if missing:
            raise KeyError(f"{path.name} is missing fields: {', '.join(missing)}")
        return {field: archive[field] for field in fields}


def prepare_data(method: str) -> ExperimentData:
    """Load, split, and transform the primary and held-out datasets."""
    validate_method(method)
    require_dataset(PRIMARY_DATASET)
    require_dataset(HELD_OUT_DATASET)
    primary = load_npz_fields(
        PRIMARY_DATASET,
        ("oru_pos_dict", "H", "pos", "rot", "timestamps", "bounding_box"),
    )
    held_out = load_npz_fields(HELD_OUT_DATASET, ("H", "pos", "rot", "timestamps"))

    oru_positions = primary["oru_pos_dict"].item()
    oru_keys = (
        "position_data_oru7_cell41",
        "position_data_oru8_cell42",
        "position_data_oru5_cell43",
        "position_data_oru6_cell51",
    )
    missing_orus = [key for key in oru_keys if key not in oru_positions]
    if missing_orus:
        raise KeyError(f"Missing ORU positions: {', '.join(missing_orus)}")
    access_point_positions = np.asarray(
        [oru_positions[key] for key in oru_keys], dtype=np.float32
    )[:, [0, 2]]
    base_parameters = Parameter(access_point_positions)

    channels = reshape_channels(np.asarray(primary["H"], dtype=np.float32))
    channels_held_out = reshape_channels(np.asarray(held_out["H"], dtype=np.float32))
    positions = np.asarray(primary["pos"], dtype=np.float32)
    positions_held_out = np.asarray(held_out["pos"], dtype=np.float32)
    rotations = np.asarray(primary["rot"], dtype=np.float32)
    rotations_held_out = np.asarray(held_out["rot"], dtype=np.float32)
    timestamps = (primary["timestamps"] - primary["timestamps"][0]).astype(np.float32)
    timestamps_held_out = held_out["timestamps"].astype(np.float32)
    bounding_boxes = primary["bounding_box"]

    base_parameters.set_UE_info(positions)
    num_validation = round(base_parameters.U * VALIDATION_RATIO)
    split_rng = np.random.RandomState(RANDOM_SEED)
    validation_indices = split_rng.choice(base_parameters.U, num_validation, replace=False)
    all_indices = np.arange(base_parameters.U)
    train_indices = all_indices[~np.isin(all_indices, validation_indices)]

    channels_train = channels[train_indices]
    positions_train = positions[train_indices]
    rotations_train = rotations[train_indices]
    timestamps_train_np = timestamps[train_indices]

    num_access_points = access_point_positions.shape[0]
    if channels_train.shape[1] % num_access_points:
        raise ValueError("The channel antenna axis is not divisible by the AP count.")
    antennas_per_ap = channels_train.shape[1] // num_access_points

    feature_extractor = FeatureExtractor(FEATURE_METHOD)
    features_train = feature_extractor.feature_extract(
        torch.from_numpy(channels_train), num_access_points
    )
    if not torch.isfinite(features_train).all():
        raise ValueError("Feature extraction produced NaN or infinity.")

    candidate_starts = (100, features_train.shape[0] - 200)
    segment_start_indices = sorted(
        {index for index in candidate_starts if 0 <= index < features_train.shape[0]}
    )
    if not segment_start_indices:
        segment_start_indices = [0]

    power_train_db = None
    box_labels_train = None
    features_validation = None
    if method == "channel_charting":
        if POWER_SOURCE == "csi":
            zero_delay = channels_train.reshape(
                channels_train.shape[0], num_access_points, antennas_per_ap, -1
            )[..., 0]
            power_train = 10.0 * np.log10(
                np.linalg.norm(zero_delay, ord=2, axis=-1)
            )
            power_train -= np.mean(power_train, axis=0, keepdims=True)
        elif POWER_SOURCE == "distance":
            displacement = (
                positions_train[:, None, :2]
                - access_point_positions[None, :, :2]
            )
            distances = np.linalg.norm(displacement, axis=-1)
            power_train = -20.0 * np.log10(np.maximum(distances, 1e-12))
        else:
            raise ValueError('POWER_SOURCE must be either "csi" or "distance".')
        if not np.isfinite(power_train).all():
            raise ValueError("Receive-power calculation produced NaN or infinity.")
        power_train_db = torch.from_numpy(power_train.astype(np.float32))

        if BOUNDING_BOX_SOURCE == "ap":
            box_labels_train = np.argmax(power_train, axis=1)
        elif BOUNDING_BOX_SOURCE == "genie":
            all_box_labels = find_bounding_box(positions, bounding_boxes)
            box_labels_train = all_box_labels[train_indices]
            if np.any(box_labels_train < 0):
                raise ValueError("At least one training position lies outside every box.")
        else:
            raise ValueError('BOUNDING_BOX_SOURCE must be either "ap" or "genie".')
    else:
        features_validation = feature_extractor.feature_extract(
            torch.from_numpy(channels[validation_indices]), num_access_points
        )

    return ExperimentData(
        base_parameters=base_parameters,
        feature_extractor=feature_extractor,
        access_point_positions=access_point_positions,
        bounding_boxes=bounding_boxes,
        box_labels_train=box_labels_train,
        features_train=features_train,
        features_validation=features_validation,
        timestamps_train=torch.from_numpy(timestamps_train_np),
        power_train_db=power_train_db,
        positions_train=positions_train,
        positions_validation=positions[validation_indices],
        positions_held_out=positions_held_out,
        rotations_train=rotations_train,
        rotations_validation=rotations[validation_indices],
        rotations_held_out=rotations_held_out,
        timestamps_validation=primary["timestamps"][validation_indices],
        timestamps_held_out=timestamps_held_out,
        channels_validation=channels[validation_indices],
        channels_held_out=channels_held_out,
        train_indices=train_indices,
        validation_indices=validation_indices,
        segment_start_indices=segment_start_indices,
    )


def build_training_parameters(
    data: ExperimentData, hp: HyperParameters, method: str
) -> Parameter:
    """Build a fresh Parameter object for one independent training run."""
    validate_method(method)
    parameters = copy.deepcopy(data.base_parameters)
    parameters.set_UE_info(
        data.positions_train,
        data.base_parameters.color_map[data.train_indices],
        data.timestamps_train,
        data.train_indices,
    )
    parameters.fe = data.feature_extractor
    parameters.set_training_params(
        learning_type=(
            "Supervised" if method == "neural_positioning" else "Triplet"
        ),
        W=NUM_DELAY_TAPS,
        in_features=data.features_train.shape[1],
        hidden_features=hp.hidden_features,
        out_features=4,  # Two position outputs and two rotation components.
        Tc=hp.Tc,
        Tf=hp.Tf,
        M_t=hp.M_t,
        segment_start_idcs=data.segment_start_indices,
        P_thresh=POWER_THRESHOLD_DB,
        M_p=hp.M_p,
        M_b=hp.M_b,
        lambda_b=hp.lambda_b,
        lambda_box=hp.lambda_box,
        bb_src=BOUNDING_BOX_SOURCE,
        box_mode=BOUNDING_BOX_MODE,
        lr=hp.learning_rate,
        use_scheduler=int(hp.use_scheduler),
        scheduler_param=hp.scheduler_step,
        num_epochs=hp.num_epochs,
        batch_size=hp.batch_size,
        num_triplets_per_anchor=hp.num_triplets_per_anchor,
        n_ref=hp.num_reference_points,
        epoch_L1_on=hp.triplet_start_epoch,
        lambda_align=hp.lambda_align,
        M_r=hp.M_r,
        lambda_rotation=hp.lambda_rotation,
    )
    return parameters


def train_model(
    data: ExperimentData,
    hp: HyperParameters,
    device: torch.device,
    method: str,
) -> tuple[TripletModel | SupervisedModel, Parameter, TrainingHistory]:
    """Train one four-output model using the selected learning method."""
    seed_everything(RANDOM_SEED)
    parameters = build_training_parameters(data, hp, method)
    if parameters.out_features != 4:
        raise AssertionError("The network must emit two position and two rotation outputs.")

    if method == "channel_charting":
        if data.power_train_db is None or data.box_labels_train is None:
            raise RuntimeError("Channel-charting supervision data was not prepared.")
        training_set = torch.utils.data.TensorDataset(
            data.features_train,
            data.timestamps_train,
            data.power_train_db,
            torch.from_numpy(data.positions_train),
        )
        model = TripletModel(device, parameters)
        total_loss = model.train(
            training_set,
            torch.from_numpy(data.bounding_boxes),
            data.box_labels_train,
        )
        history = TrainingHistory(total=total_loss)
    else:
        if data.features_validation is None:
            raise RuntimeError("Neural-positioning validation features were not prepared.")
        training_set = torch.utils.data.TensorDataset(
            data.features_train,
            torch.from_numpy(data.positions_train),
            torch.from_numpy(data.rotations_train),
        )
        validation_set = torch.utils.data.TensorDataset(
            data.features_validation,
            torch.from_numpy(data.positions_validation),
            torch.from_numpy(data.rotations_validation),
        )
        model = SupervisedModel(device, parameters)
        total, position, rotation, validation = model.train(
            training_set, validation_set
        )
        history = TrainingHistory(
            total=total,
            validation=validation,
            position=position,
            rotation=rotation,
        )
    return model, parameters, history


def error_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def evaluate_model(
    model: TripletModel | SupervisedModel,
    data: ExperimentData,
    channels: np.ndarray,
    positions_true: np.ndarray,
    rotations_true: np.ndarray,
) -> EvaluationResult:
    """Predict one split and calculate topology, position, and angle metrics."""
    features = data.feature_extractor.feature_extract(
        torch.from_numpy(channels), data.num_access_points
    )
    positions_predicted, rotations_predicted = predict(model, features)
    positions_np = positions_predicted.numpy()
    rotations_np = rotations_predicted.numpy()
    if positions_np.shape[1] != 2 or rotations_np.shape[1] != 2:
        raise RuntimeError(
            "Expected two position and two rotation outputs; received "
            f"{positions_np.shape[1]} and {rotations_np.shape[1]}."
        )

    # To reproduce the original results, the supervised model's rotation outputs are swapped
    # during training.
    if isinstance(model, SupervisedModel):
        rotations_np = rotations_np[:, [1, 0]]

    ks = float(evaluate_cc(positions_true, positions_np, "KS"))
    tw, ct = evaluate_cc(positions_true, positions_np, "TW-CT")
    position_error = np.linalg.norm(positions_np - positions_true, axis=-1)
    predicted_angles = np.arctan2(rotations_np[:, 0], rotations_np[:, 1])
    angular_error = np.abs(
        np.angle(np.exp(1j * (predicted_angles - rotations_true.squeeze())))
    )
    cosine_distance, mean_angular_error = angular_metrics(rotations_np, rotations_true)
    true_boxes = find_bounding_box(positions_true, data.bounding_boxes)
    predicted_boxes = find_bounding_box(positions_np, data.bounding_boxes)
    valid_boxes = true_boxes >= 0
    box_accuracy = (
        float(np.mean(predicted_boxes[valid_boxes] == true_boxes[valid_boxes]))
        if np.any(valid_boxes)
        else float("nan")
    )
    metrics: dict[str, float] = {
        "KS": ks,
        "TW": float(tw),
        "CT": float(ct),
        "MDE": float(np.mean(position_error)),
        "cos_dist": cosine_distance,
        "ang_err": mean_angular_error,
        "box_accuracy": box_accuracy,
        "topology_sample_count": float(len(positions_true)),
    }
    metrics.update(
        {f"position_{key}": value for key, value in error_statistics(position_error).items()}
    )
    metrics.update(
        {f"angle_{key}": value for key, value in error_statistics(angular_error).items()}
    )
    return EvaluationResult(positions_np, rotations_np, metrics)


def print_metrics(split_name: str, metrics: dict[str, float]) -> None:
    print(f"\n{split_name}")
    print("-" * len(split_name))
    print(f"KS={metrics['KS']:.4f}  TW={metrics['TW']:.4f}  CT={metrics['CT']:.4f}")
    print(f"Topology-metric samples: {int(metrics['topology_sample_count'])}")
    print(
        "Position error [m]: "
        f"mean={metrics['position_mean']:.3f}  median={metrics['position_median']:.3f}  "
        f"95%={metrics['position_p95']:.3f}  max={metrics['position_max']:.3f}"
    )
    print(
        "Absolute angular error [rad]: "
        f"mean={metrics['angle_mean']:.3f}  median={metrics['angle_median']:.3f}  "
        f"95%={metrics['angle_p95']:.3f}  max={metrics['angle_max']:.3f}"
    )


def sample_arrow_indices(num_samples: int) -> np.ndarray:
    return np.arange(ARROW_START_INDEX, num_samples, ARROW_STEP)


def style_axis(axis: plt.Axes) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(*AXIS_LIMITS)
    axis.set_ylim(*AXIS_LIMITS)
    axis.set_xbound(*AXIS_LIMITS)
    axis.set_ybound(*AXIS_LIMITS)
    axis.set_xticks(AXIS_TICKS)
    axis.set_yticks(AXIS_TICKS)
    axis.set_xticks([], minor=True)
    axis.set_yticks([], minor=True)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.autoscale(enable=False)


def plot_heading_arrows(
    axis: plt.Axes, positions: np.ndarray, rotations_sin_cos: np.ndarray
) -> None:
    """Draw a time-ordered white-to-black heading overlay."""
    positions = np.asarray(positions)
    rotations_sin_cos = np.asarray(rotations_sin_cos)
    num_samples = min(len(positions), len(rotations_sin_cos))
    indices = sample_arrow_indices(num_samples)
    if indices.size == 0:
        return
    yaw = np.arctan2(
        rotations_sin_cos[indices, 0],
        rotations_sin_cos[indices, 1],
    )
    colors = plt.cm.binary(np.linspace(0.0, 1.0, indices.size))
    axis.quiver(
        positions[indices, 0],
        positions[indices, 1],
        np.sin(yaw),
        np.cos(yaw),
        color=colors,
        pivot="middle",
        angles="xy",
        scale_units="xy",
        scale=1.0 / ARROW_LENGTH,
        width=0.005,
        headwidth=1 / 0.005 / 2,
        headlength=1 / 0.005,
        headaxislength=1 / 0.005,
    )
    style_axis(axis)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def save_loss_plot(history: TrainingHistory, path: Path) -> None:
    figure, axis = plt.subplots()
    axis.plot(history.total, label="Training total")
    if history.validation is not None:
        axis.plot(history.validation, label="Validation total")
    if history.position is not None:
        axis.plot(history.position, label="Training position", alpha=0.75)
    if history.rotation is not None:
        axis.plot(history.rotation, label="Training rotation", alpha=0.75)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean loss")
    axis.legend()
    save_figure(figure, path)


def save_test_channel_chart(
    result: EvaluationResult,
    colors: np.ndarray,
    overlay: EvaluationResult,
    path: Path,
) -> None:
    figure = plot_chart(
        result.positions,
        colors,
        False,
        anchors=None,
        is_meters=False,
        title=None,
        lims=np.asarray([AXIS_LIMITS, AXIS_LIMITS]),
    )
    axis = figure.axes[0]
    style_axis(axis)
    plot_heading_arrows(axis, overlay.positions, overlay.rotations)
    save_figure(figure, path)


def save_ground_truth_plot(data: ExperimentData, path: Path) -> None:
    """Save the primary scenario with true held-out headings overlaid."""
    figure, axis = plt.subplots()
    axis.scatter(
        data.base_parameters.UE_pos[:, 0],
        data.base_parameters.UE_pos[:, 1],
        s=5,
        c=data.base_parameters.color_map,
    )
    held_out_sin_cos = np.column_stack(
        (
            np.sin(data.rotations_held_out.squeeze()),
            np.cos(data.rotations_held_out.squeeze()),
        )
    )
    plot_heading_arrows(axis, data.positions_held_out, held_out_sin_cos)
    style_axis(axis)
    save_figure(figure, path)


def json_compatible(value: Any) -> Any:
    """Convert NumPy/dataclass values and non-finite floats for strict JSON."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path: Path, contents: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_compatible(contents), handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_rotation_csv(
    path: Path,
    result: EvaluationResult,
    rotations_true: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    angles = np.arctan2(result.rotations[:, 0], result.rotations[:, 1])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("predicted_phi", "ground_truth_phi", "timestamp"))
        writer.writerows(
            zip(angles, rotations_true.squeeze(), timestamps.squeeze())
        )


def save_trial_artifacts(
    trial_number: int,
    hp: HyperParameters,
    method: str,
    validation: EvaluationResult,
    held_out: EvaluationResult,
    history: TrainingHistory,
    data: ExperimentData,
    study_directory: Path,
) -> None:
    trial_directory = (
        study_directory / "trial_artifacts" / f"trial_{trial_number:03d}"
    )
    trial_directory.mkdir(parents=True, exist_ok=True)
    save_loss_plot(history, trial_directory / "training_loss_curves.png")
    save_test_channel_chart(
        validation,
        data.base_parameters.color_map[data.validation_indices],
        held_out,
        trial_directory / "test_and_validation_chart.png",
    )
    write_json(
        trial_directory / "trial_summary.json",
        {
            "trial": trial_number,
            "method": method,
            "hyperparameters": active_hyperparameters(hp, method),
            "validation_metrics": validation.metrics,
        },
    )


def suggest_hyperparameters(trial: Any, method: str) -> HyperParameters:
    """Sample only the parameters used by the selected method."""
    base = fixed_hyperparameters(method)
    shared = {
        "lambda_rotation": trial.suggest_float(
            "lambda_rotation", 0.01, 5.0, log=True
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        ),
    }
    if base.use_scheduler:
        shared["scheduler_step"] = trial.suggest_int("scheduler_step", 25, 70)
    if method == "neural_positioning":
        return replace(base, **shared)
    return replace(
        base,
        **shared,
        lambda_b=trial.suggest_float("lambda_b", 1, 30.0, log=True),
        lambda_box=trial.suggest_float("lambda_box", 0.05, 10.0, log=True),
        lambda_align=trial.suggest_float("lambda_align", 0.01, 10.0, log=True),
        M_t=trial.suggest_float("M_t", 0.05, 1.0),
        M_r=trial.suggest_float("M_r", 0.1, 5.0),
        M_b=trial.suggest_float("M_b", 0.1, 5.0),
        M_p=trial.suggest_float("M_p", 3.0, 20.0),
        Tc=trial.suggest_float("Tc", 0.1, 5.0),
        Tf=trial.suggest_float("Tf", 6.0, 70.0),
        triplet_start_epoch=trial.suggest_int("triplet_start_epoch", 0, 5),
    )


def export_optuna_trials(study: Any, path: Path) -> None:
    """Write a dependency-free CSV containing objectives, parameters, and attrs."""
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {"number": trial.number, "state": trial.state.name}
        if trial.values is not None:
            row.update({name: value for name, value in zip(OBJECTIVE_NAMES, trial.values)})
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update({f"attr_{key}": value for key, value in trial.user_attrs.items()})
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_optuna(
    data: ExperimentData,
    device: torch.device,
    method: str,
    output_directory: Path,
) -> tuple[HyperParameters, int]:
    """Run/resume the search and return selected Pareto hyperparameters."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna mode requires the 'optuna' dependency. Install requirements.txt."
        ) from exc

    study_name = optuna_study_name(method)
    study_directory = output_directory
    study_directory.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(study_directory / 'optuna_study.sqlite3').as_posix()}"

    def objective(trial: Any) -> tuple[float, ...]:
        hp = suggest_hyperparameters(trial, method)
        model, _, history = train_model(data, hp, device, method)
        validation = evaluate_model(
            model,
            data,
            data.channels_validation,
            data.positions_validation,
            data.rotations_validation,
        )
        for key, value in validation.metrics.items():
            trial.set_user_attr(f"validation_{key}", value)
        if SAVE_OPTUNA_TRIAL_ARTIFACTS:
            held_out = evaluate_model(
                model,
                data,
                data.channels_held_out,
                data.positions_held_out,
                data.rotations_held_out,
            )
            save_trial_artifacts(
                trial.number,
                hp,
                method,
                validation,
                held_out,
                history,
                data,
                study_directory,
            )
        metrics = validation.metrics
        print(
            f"Trial {trial.number:03d}: MDE={metrics['MDE']:.3f}, "
            f"KS={metrics['KS']:.3f}, TW={metrics['TW']:.3f}, "
            f"CT={metrics['CT']:.3f}, angle={metrics['ang_err']:.3f}"
        )
        return (
            metrics["KS"],
            1.0 - metrics["TW"],
            1.0 - metrics["CT"],
            metrics["MDE"],
            metrics["cos_dist"],
            metrics["ang_err"],
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        directions=["minimize"] * len(OBJECTIVE_NAMES),
        sampler=optuna.samplers.NSGAIISampler(seed=RANDOM_SEED),
    )
    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, OPTUNA_TRIALS - completed)
    print(
        f"Optuna study '{study_name}': "
        f"{completed} completed, {remaining} remaining."
    )
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    export_optuna_trials(study, study_directory / "optuna_trials.csv")
    pareto_trials = study.best_trials
    if not pareto_trials:
        raise RuntimeError("The Optuna study has no completed Pareto trials.")
    selected = min(pareto_trials, key=lambda trial: trial.values[3])
    hp = replace(fixed_hyperparameters(method), **selected.params)
    write_json(
        study_directory / "selected_optuna_trial.json",
        {
            "trial": selected.number,
            "objectives": dict(zip(OBJECTIVE_NAMES, selected.values)),
            "method": method,
            "hyperparameters": active_hyperparameters(hp, method),
            "pareto_front_size": len(pareto_trials),
        },
    )
    print(
        f"Selected trial {selected.number} from a Pareto front of "
        f"{len(pareto_trials)} trials."
    )
    return hp, selected.number


def save_final_results(
    run_directory: Path,
    method: str,
    mode: str,
    selected_trial: int | None,
    model: TripletModel | SupervisedModel,
    parameters: Parameter,
    hp: HyperParameters,
    history: TrainingHistory,
    data: ExperimentData,
    training: EvaluationResult | None,
    validation: EvaluationResult,
    held_out: EvaluationResult,
    device: torch.device,
) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    state_dict_cpu = {
        key: value.detach().cpu() for key, value in model.network.state_dict().items()
    }
    torch.save(state_dict_cpu, run_directory / "trained_model_state_dict.pth")
    with (run_directory / "training_configuration.pkl").open("wb") as handle:
        pickle.dump(parameters, handle)
    metrics: dict[str, Any] = {
        "method": method,
        "mode": mode,
        "selected_optuna_trial": selected_trial,
        "device": str(device),
        "seed": RANDOM_SEED,
        "experiment_configuration": {
            "primary_dataset": str(PRIMARY_DATASET),
            "held_out_dataset": str(HELD_OUT_DATASET),
            "validation_ratio": VALIDATION_RATIO,
            "feature_method": FEATURE_METHOD,
            "num_delay_taps": NUM_DELAY_TAPS,
            "hidden_features": hp.hidden_features,
            "output_features": 4,
            "num_epochs": hp.num_epochs,
            "batch_size": hp.batch_size,
            "num_triplets_per_anchor": hp.num_triplets_per_anchor,
            "use_scheduler": hp.use_scheduler,
            "scheduler_step": hp.scheduler_step,
            "power_threshold_db": POWER_THRESHOLD_DB,
            "power_source": POWER_SOURCE,
            "bounding_box_source": BOUNDING_BOX_SOURCE,
            "bounding_box_mode": BOUNDING_BOX_MODE,
            "topology_metric_samples": "all",
        },
        "hyperparameters": active_hyperparameters(hp, method),
        "validation": validation.metrics,
        "held_out_test": held_out.metrics,
    }
    if training is not None:
        metrics["training"] = training.metrics
    write_json(run_directory / "evaluation_metrics.json", metrics)

    save_loss_plot(history, run_directory / "training_loss_curves.png")
    save_ground_truth_plot(data, run_directory / "ground_truth_chart.png")
    save_test_channel_chart(
        validation,
        data.base_parameters.color_map[data.validation_indices],
        held_out,
        run_directory / "test_and_validation_chart.png",
    )
    write_rotation_csv(
        run_directory / "held_out_orientation_predictions.csv",
        held_out,
        data.rotations_held_out,
        data.timestamps_held_out,
    )
    write_rotation_csv(
        run_directory / "validation_orientation_predictions.csv",
        validation,
        data.rotations_validation,
        data.timestamps_validation,
    )


def main() -> None:
    validate_method(METHOD)
    seed_everything(RANDOM_SEED)
    device = select_device()
    mode = "optuna" if USE_OPTUNA else "fixed"
    print(f"Method: {METHOD}")
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Primary dataset: {PRIMARY_DATASET}")
    print(f"Held-out dataset: {HELD_OUT_DATASET}")
    run_directory = result_directory(METHOD, USE_OPTUNA)
    data = prepare_data(METHOD)

    selected_trial: int | None = None
    if USE_OPTUNA:
        # For the results presented in the paper, a trial was manually selected from the Pareto front.
        hp, selected_trial = run_optuna(data, device, METHOD, run_directory)
    else:
        hp = fixed_hyperparameters(METHOD)

    print("Training final model...")
    model, parameters, history = train_model(data, hp, device, METHOD)
    training = None
    if EVALUATE_TRAINING_SET:
        with np.load(PRIMARY_DATASET, allow_pickle=True) as archive:
            channels_train = reshape_channels(archive["H"].astype(np.float32))[
                data.train_indices
            ]
        training = evaluate_model(
            model,
            data,
            channels_train,
            data.positions_train,
            data.rotations_train,
        )
        print_metrics("Training split", training.metrics)

    validation = evaluate_model(
        model,
        data,
        data.channels_validation,
        data.positions_validation,
        data.rotations_validation,
    )
    held_out = evaluate_model(
        model,
        data,
        data.channels_held_out,
        data.positions_held_out,
        data.rotations_held_out,
    )
    print_metrics("Validation split", validation.metrics)
    print_metrics("Held-out test set", held_out.metrics)

    if SAVE_RESULTS:
        save_final_results(
            run_directory,
            METHOD,
            mode,
            selected_trial,
            model,
            parameters,
            hp,
            history,
            data,
            training,
            validation,
            held_out,
            device,
        )
        print(f"Saved results to {run_directory}")


if __name__ == "__main__":
    main()
