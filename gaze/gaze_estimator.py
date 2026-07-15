"""L2CS-Net gaze estimation with resilient checkpoint loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torchvision import models, transforms

from config import (
    GAZE_BIN_COUNT,
    GAZE_BIN_OFFSET_DEGREES,
    GAZE_BIN_SIZE_DEGREES,
    GAZE_INPUT_SIZE,
    GAZE_MODEL_PATH,
    GAZE_NORMALIZATION_MEAN,
    GAZE_NORMALIZATION_STD,
    GAZE_USE_GPU,
    SCREEN_PITCH_MAX,
    SCREEN_PITCH_MIN,
    SCREEN_YAW_MAX,
    SCREEN_YAW_MIN,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GazeEstimate:
    """The gaze angles and screen-attention classification for one face."""

    yaw: float = 0.0
    pitch: float = 0.0
    direction: str = "Unknown"
    looking_at_screen: bool = False


class _L2CSNet(nn.Module):
    """L2CS-Net head on a checkpoint-compatible torchvision ResNet backbone."""

    def __init__(self, backbone: nn.Module, feature_dim: int, bin_count: int) -> None:
        super().__init__()
        # Register the backbone layers directly so their parameter names match
        # official L2CS-Net checkpoints (for example, ``layer3.4.conv2.weight``).
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc_yaw_gaze = nn.Linear(feature_dim, bin_count)
        self.fc_pitch_gaze = nn.Linear(feature_dim, bin_count)
        self.fc_finetune = nn.Linear(feature_dim + 3, 3)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        features = self.conv1(image)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = torch.flatten(self.avgpool(features), 1)
        return self.fc_yaw_gaze(features), self.fc_pitch_gaze(features)


class GazeEstimator:
    """Run L2CS-Net once per valid BGR face crop.

    Model-loading errors intentionally leave the estimator unavailable.  This lets
    the camera pipeline keep running and reports an ``Unknown`` gaze instead of
    turning a model deployment problem into a surveillance outage.
    """

    def __init__(self) -> None:
        self._device = self._select_device()
        self._model: _L2CSNet | None = None
        self._bin_centers: Tensor | None = None
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(GAZE_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=GAZE_NORMALIZATION_MEAN,
                    std=GAZE_NORMALIZATION_STD,
                ),
            ]
        )
        self._load_model()

    @property
    def is_available(self) -> bool:
        """Whether a valid model is ready for inference."""
        return self._model is not None

    def estimate(self, face_bgr: np.ndarray) -> GazeEstimate:
        """Estimate gaze from a non-empty BGR face crop.

        Invalid inputs and inference errors return ``Unknown``.  The exception is
        logged so that production monitoring can surface a persistent issue.
        """
        if self._model is None or not self._is_valid_face(face_bgr):
            return GazeEstimate()

        try:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            image = self._transform(face_rgb).unsqueeze(0).to(
                self._device, non_blocking=self._device.type == "cuda"
            )
            with torch.no_grad():
                yaw_logits, pitch_logits = self._model(image)
                yaw = self._decode_angle(yaw_logits)
                pitch = self._decode_angle(pitch_logits)

            looking_at_screen = self._is_looking_at_screen(yaw, pitch)
            return GazeEstimate(
                yaw=yaw,
                pitch=pitch,
                direction=self._classify_direction(yaw, pitch, looking_at_screen),
                looking_at_screen=looking_at_screen,
            )
        except (cv2.error, RuntimeError, TypeError, ValueError) as error:
            LOGGER.warning("Gaze inference failed: %s", error)
            return GazeEstimate()

    def _select_device(self) -> torch.device:
        if GAZE_USE_GPU and torch.cuda.is_available():
            LOGGER.info("L2CS-Net gaze inference using CUDA")
            return torch.device("cuda")
        if GAZE_USE_GPU:
            LOGGER.info("CUDA requested for L2CS-Net but unavailable; using CPU")
        return torch.device("cpu")

    def _load_model(self) -> None:
        model_path = Path(GAZE_MODEL_PATH)
        if not model_path.is_file():
            LOGGER.error("L2CS-Net checkpoint does not exist: %s", model_path)
            return

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = self._extract_state_dict(checkpoint)
            model = self._build_model(state_dict)
            incompatible = model.load_state_dict(state_dict, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError("checkpoint keys do not match the L2CS-Net model")

            self._model = model.to(self._device).eval()
            self._bin_centers = (
                torch.arange(GAZE_BIN_COUNT, dtype=torch.float32, device=self._device)
                * GAZE_BIN_SIZE_DEGREES
                + GAZE_BIN_OFFSET_DEGREES
            )
            LOGGER.info("Loaded L2CS-Net checkpoint from %s", model_path)
        except Exception as error:
            LOGGER.exception("Unable to load L2CS-Net checkpoint %s: %s", model_path, error)
            self._model = None
            self._bin_centers = None

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
        if isinstance(checkpoint, Mapping):
            for key in ("state_dict", "model_state_dict", "model"):
                candidate = checkpoint.get(key)
                if isinstance(candidate, Mapping):
                    checkpoint = candidate
                    break
        if not isinstance(checkpoint, Mapping):
            raise TypeError("checkpoint is not a state dictionary")

        state_dict = {
            str(key).removeprefix("module."): value
            for key, value in checkpoint.items()
            if isinstance(value, Tensor)
        }
        if not state_dict:
            raise ValueError("checkpoint contains no tensor parameters")
        return state_dict

    @staticmethod
    def _build_model(state_dict: Mapping[str, Tensor]) -> _L2CSNet:
        required_heads = ("fc_yaw_gaze.weight", "fc_pitch_gaze.weight", "fc_finetune.weight")
        if any(key not in state_dict for key in required_heads):
            raise KeyError("checkpoint is missing one or more L2CS-Net gaze heads")

        yaw_shape = tuple(state_dict["fc_yaw_gaze.weight"].shape)
        pitch_shape = tuple(state_dict["fc_pitch_gaze.weight"].shape)
        finetune_shape = tuple(state_dict["fc_finetune.weight"].shape)
        if len(yaw_shape) != 2 or yaw_shape != pitch_shape:
            raise ValueError("yaw and pitch heads have incompatible shapes")

        bin_count, feature_dim = yaw_shape
        stage_blocks = tuple(
            len({key.split(".")[1] for key in state_dict if key.startswith(f"layer{stage}.")})
            for stage in range(1, 5)
        )
        if any(block_count == 0 for block_count in stage_blocks):
            raise ValueError(
                "checkpoint does not define all four ResNet stages; found "
                f"{stage_blocks}"
            )
        if bin_count != GAZE_BIN_COUNT:
            raise ValueError(
                f"checkpoint uses {bin_count} gaze bins but GAZE_BIN_COUNT is {GAZE_BIN_COUNT}"
            )
        if finetune_shape != (3, feature_dim + 3):
            raise ValueError("checkpoint has an invalid L2CS-Net fine-tuning head")

        block = (
            models.resnet.Bottleneck
            if "layer1.0.conv3.weight" in state_dict
            else models.resnet.BasicBlock
        )
        backbone = models.resnet.ResNet(block, list(stage_blocks))
        if backbone.fc.in_features != feature_dim:
            raise ValueError(
                "checkpoint feature dimension does not match its inferred ResNet "
                f"backbone ({feature_dim} != {backbone.fc.in_features})"
            )
        backbone.fc = nn.Identity()
        return _L2CSNet(backbone, feature_dim, bin_count)

    def _decode_angle(self, logits: Tensor) -> float:
        if self._bin_centers is None:
            raise RuntimeError("gaze-bin centers are not initialized")
        probabilities = torch.softmax(logits, dim=1)
        return float(torch.sum(probabilities[0] * self._bin_centers).item())

    @staticmethod
    def _is_valid_face(face: object) -> bool:
        return (
            isinstance(face, np.ndarray)
            and face.ndim == 3
            and face.shape[0] > 0
            and face.shape[1] > 0
            and face.shape[2] == 3
            and face.dtype == np.uint8
        )

    @staticmethod
    def _is_looking_at_screen(yaw: float, pitch: float) -> bool:
        return (
            SCREEN_YAW_MIN <= yaw <= SCREEN_YAW_MAX
            and SCREEN_PITCH_MIN <= pitch <= SCREEN_PITCH_MAX
        )

    @staticmethod
    def _classify_direction(yaw: float, pitch: float, looking_at_screen: bool) -> str:
        if looking_at_screen:
            return "Looking Screen"
        if yaw < SCREEN_YAW_MIN:
            return "Looking Left"
        if yaw > SCREEN_YAW_MAX:
            return "Looking Right"
        if pitch > SCREEN_PITCH_MAX:
            return "Looking Up"
        if pitch < SCREEN_PITCH_MIN:
            return "Looking Down"
        return "Unknown"
