#!/usr/bin/env python3
"""Phase 1 smoke test: protocol layer import and serialize/deserialize roundtrip.

Verifies that all message types in src/protocol/ can be imported, serialized to
dict, and deserialized back to the correct dataclass without any server or GPU
dependency.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.protocol import (
    AudioChunkSpec,
    AudioDType,
    HelloMessage,
    HelloAckMessage,
    HealthMessage,
    HealthResultMessage,
    InferChunkMessage,
    InferChunkResultMessage,
    InferenceSettings,
    ListModelsMessage,
    ListModelsResultMessage,
    LoadModelMessage,
    LoadModelResultMessage,
    ModelInfo,
    UpdateParamsMessage,
    UpdateParamsResultMessage,
    ErrorMessage,
    ErrorCode,
    MessageType,
    deserialize_message,
    serialize_message,
)


def _roundtrip(msg, label: str) -> None:
    """serialize → JSON string → deserialize して型が保たれることを確認する。"""
    serialized = serialize_message(msg)
    json_str = json.dumps(serialized)
    restored = deserialize_message(json.loads(json_str))
    assert type(restored) is type(msg), f"{label}: type mismatch {type(restored)} != {type(msg)}"
    print(f"[ok] {label}")


def main() -> None:
    print("=== Phase 1 smoke test: protocol layer ===\n")

    # --- HelloMessage / HelloAckMessage ---
    _roundtrip(HelloMessage(client="smoke-test", protocol_version=1), "hello")
    _roundtrip(
        HelloAckMessage(server="wsl-inference", protocol_version=1),
        "hello_ack",
    )

    # --- HealthMessage / HealthResultMessage ---
    _roundtrip(HealthMessage(), "health")
    _roundtrip(
        HealthResultMessage(ok=True, device="cuda:0", model_loaded=True, active_model="test-model"),
        "health_result",
    )

    # --- ListModelsMessage / ListModelsResultMessage ---
    _roundtrip(ListModelsMessage(), "list_models")
    _roundtrip(
        ListModelsResultMessage(models=[ModelInfo(name="test-model", has_index=False, version="v2")]),
        "list_models_result",
    )

    # --- LoadModelMessage / LoadModelResultMessage ---
    settings = InferenceSettings(
        model_name="test-model",
        pitch_shift=12,
        f0_method="rmvpe",
        index_rate=0.75,
        protect=0.33,
        filter_radius=3,
        rms_mix_rate=1.0,
        backend="rvc-python",
    )
    _roundtrip(LoadModelMessage(model_name="test-model", params=settings), "load_model")
    _roundtrip(
        LoadModelResultMessage(ok=True, active_model="test-model", device="cuda:0"),
        "load_model_result",
    )

    # --- UpdateParamsMessage / UpdateParamsResultMessage ---
    _roundtrip(UpdateParamsMessage(params=settings), "update_params")
    _roundtrip(UpdateParamsResultMessage(ok=True), "update_params_result")

    # --- InferChunkMessage / InferChunkResultMessage ---
    chunk_spec = AudioChunkSpec(sample_rate=48000, channels=1, dtype=AudioDType.FLOAT32, frame_count=960)
    infer_msg = InferChunkMessage(sequence=1, audio=chunk_spec)
    _roundtrip(infer_msg, "infer_chunk")
    _roundtrip(
        InferChunkResultMessage(
            sequence=1,
            audio=chunk_spec,
            processing_ms=42.0,
            fallback=False,
        ),
        "infer_chunk_result",
    )

    # --- ErrorMessage ---
    _roundtrip(
        ErrorMessage(code=ErrorCode.MODEL_NOT_FOUND, message="model not found", sequence=1),
        "error",
    )

    print(f"\n=== All 13 message types passed roundtrip check ===")


if __name__ == "__main__":
    main()
