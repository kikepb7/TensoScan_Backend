"""Constants of the model I/O contract (`specs/contracts/digit-detector-model.md`, version 1.0.0).

They are shared by the synthetic generator, the training code, the exporter, the verifier and the evaluation harness so
that no module re-declares a class order or a threshold.
"""

from __future__ import annotations

from typing import Final

CONTRACT_VERSION: Final = "1.0.0"
MODEL_NAME: Final = "digit-detector"

INPUT_SIZE: Final = 320
INPUT_CHANNELS: Final = 3

#: Grey of the letterbox border (`114 / 255`), identical in the app (`LetterboxPreprocessor.PAD_VALUE`).
PAD_GREY: Final = 114
PAD_VALUE: Final = PAD_GREY / 255.0

#: Class order of the output columns (contract section 4.1).
CLASSES: Final[tuple[str, ...]] = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "colon", "dash", "E", "r")
NUM_CLASSES: Final = len(CLASSES)
BOX_VALUES: Final = 4
VALUES_PER_CANDIDATE: Final = BOX_VALUES + NUM_CLASSES
CLASS_INDEX: Final[dict[str, int]] = {name: index for index, name in enumerate(CLASSES)}

#: Strides of the three detection levels. 40*40 + 20*20 + 10*10 = 2100 candidates at 320 px.
STRIDES: Final[tuple[int, ...]] = (8, 16, 32)

#: Recommended post-processing (contract section 6); the sidecar repeats it.
SCORE_THRESHOLD: Final = 0.40
NMS_IOU_THRESHOLD: Final = 0.45
MAX_DETECTIONS: Final = 32

#: Budget of contract section 7.
MAX_MODEL_BYTES: Final = 6 * 1024 * 1024
MAX_LATENCY_P50_MS: Final = 25.0
MAX_LATENCY_P95_MS: Final = 45.0

#: Names of the model tensors.
INPUT_TENSOR_NAME: Final = "input"
OUTPUT_TENSOR_NAME: Final = "output"


def num_candidates(input_size: int = INPUT_SIZE, strides: tuple[int, ...] = STRIDES) -> int:
    """Number of anchors of a model with square input `input_size` and the given strides."""
    return sum((input_size // stride) ** 2 for stride in strides)
