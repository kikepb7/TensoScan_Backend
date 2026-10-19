"""Canonical payload of a measurement and its hash: the idempotency key of ``POST /v1/measurements`` (ADR-B002-01)."""

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC
from typing import Any

from app.domain.measurements.model import Measurement

CONFIDENCE_DECIMALS = 4


def canonical_payload(measurement: Measurement) -> dict[str, Any]:
    """What the client sent, normalised: UTC time with seconds, rounded confidence, sorted ``edited_fields``.

    ``id``, ``user_id`` and ``created_at`` are server-side and never part of the payload.
    """
    recognition = measurement.recognition
    return {
        "client_id": str(measurement.client_id),
        "systolic": measurement.reading.systolic,
        "diastolic": measurement.reading.diastolic,
        "pulse": measurement.reading.pulse,
        "measured_at": measurement.measured_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": measurement.source,
        "recognition": None
        if recognition is None
        else {
            "confidence": round(recognition.confidence, CONFIDENCE_DECIMALS),
            "model_version": recognition.model_version,
            "edited_fields": sorted(recognition.edited_fields),
        },
    }


def hash_payload(payload: Mapping[str, Any]) -> str:
    """sha256 of the JSON with sorted keys, so the key order of the input never matters."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def payload_hash(measurement: Measurement) -> str:
    return hash_payload(canonical_payload(measurement))
