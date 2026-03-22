"""WSL-side inference engine backed by rvc-python."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

from src.protocol import (
    AudioChunkSpec,
    AudioDType,
    HealthResultMessage,
    InferChunkResultMessage,
    InferenceSettings,
    LoadModelResultMessage,
    ModelInfo,
    UpdateParamsResultMessage,
)

from .model_registry import ModelRegistry

try:
    from rvc_python.infer import RVCInference

    RVC_PYTHON_AVAILABLE = True
except Exception:
    RVCInference = None
    RVC_PYTHON_AVAILABLE = False


class InferenceEngine:
    """Stateful model manager and single-shot chunk inference executor."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.registry = ModelRegistry(models_dir=models_dir)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
        self.current_model_name = ""
        self.current_settings = InferenceSettings()
        self._inference: Optional[RVCInference] = None
        self._lock = threading.Lock()

    @contextmanager
    def _weights_only_compat(self):
        old_env = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            yield
        finally:
            if old_env is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = old_env

    def _ensure_backend(self) -> RVCInference:
        if not RVC_PYTHON_AVAILABLE:
            raise RuntimeError("rvc-python is not available in current environment")
        if self._inference is None:
            self._inference = RVCInference(device=self.device)
        return self._inference

    def list_models(self) -> list[ModelInfo]:
        return self.registry.list_models()

    def health(self) -> HealthResultMessage:
        return HealthResultMessage(
            ok=True,
            device=self.device,
            model_loaded=bool(self.current_model_name),
            active_model=self.current_model_name,
        )

    def load_model(self, model_name: str, settings: InferenceSettings) -> LoadModelResultMessage:
        model_info = self.registry.get_model_paths(model_name)
        if model_info is None:
            raise FileNotFoundError(f"Model not found: {model_name}")

        resolved_settings = settings
        if settings.model_name != model_name:
            resolved_settings = InferenceSettings.from_dict(
                {
                    **settings.to_dict(),
                    "model_name": model_name,
                }
            )

        with self._lock, self._weights_only_compat():
            infer = self._ensure_backend()
            infer.load_model(
                model_info["pth"],
                version=model_info["version"],
                index_path=model_info["index"] or "",
            )
            infer.set_params(
                f0up_key=resolved_settings.pitch_shift,
                f0method=resolved_settings.f0_method,
                index_rate=resolved_settings.index_rate,
                filter_radius=resolved_settings.filter_radius,
                rms_mix_rate=resolved_settings.rms_mix_rate,
                protect=resolved_settings.protect,
            )
            self.current_model_name = model_name
            self.current_settings = resolved_settings

        return LoadModelResultMessage(ok=True, active_model=model_name, device=self.device)

    def update_params(self, settings: InferenceSettings) -> UpdateParamsResultMessage:
        if not self.current_model_name:
            raise RuntimeError("No active model")

        resolved_settings = settings
        if not settings.model_name:
            resolved_settings = InferenceSettings.from_dict(
                {
                    **settings.to_dict(),
                    "model_name": self.current_model_name,
                }
            )

        with self._lock:
            infer = self._ensure_backend()
            infer.set_params(
                f0up_key=resolved_settings.pitch_shift,
                f0method=resolved_settings.f0_method,
                index_rate=resolved_settings.index_rate,
                filter_radius=resolved_settings.filter_radius,
                rms_mix_rate=resolved_settings.rms_mix_rate,
                protect=resolved_settings.protect,
            )
            self.current_settings = resolved_settings

        return UpdateParamsResultMessage(ok=True)

    def infer_chunk(self, sequence: int, spec: AudioChunkSpec, payload: bytes) -> Tuple[InferChunkResultMessage, bytes]:
        if not self.current_model_name:
            raise RuntimeError("No active model")

        spec.validate()
        if len(payload) != spec.payload_nbytes:
            raise ValueError(
                f"Invalid payload size: expected {spec.payload_nbytes}, got {len(payload)}"
            )

        mono_audio = self._decode_payload(payload, spec)
        start_time = time.perf_counter()

        with self._lock, tempfile.TemporaryDirectory(prefix="voicechange_rpc_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            input_path = temp_dir_path / "input.wav"
            output_path = temp_dir_path / "output.wav"
            sf.write(input_path, mono_audio, spec.sample_rate)

            with self._weights_only_compat():
                infer = self._ensure_backend()
                infer.infer_file(str(input_path), str(output_path))

            converted_audio, converted_sr = sf.read(output_path, dtype="float32")

        processed = self._normalize_output(converted_audio, converted_sr, spec)
        output_bytes = self._encode_payload(processed, spec)
        processing_ms = (time.perf_counter() - start_time) * 1000.0

        return (
            InferChunkResultMessage(
                sequence=sequence,
                audio=spec,
                processing_ms=processing_ms,
                fallback=False,
            ),
            output_bytes,
        )

    def _decode_payload(self, payload: bytes, spec: AudioChunkSpec) -> np.ndarray:
        if spec.dtype is AudioDType.FLOAT32:
            audio = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=False)
        else:
            audio = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0

        audio = audio.reshape(spec.frame_count, spec.channels)
        if spec.channels > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        else:
            audio = audio[:, 0]
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _normalize_output(self, audio: np.ndarray, sample_rate: int, spec: AudioChunkSpec) -> np.ndarray:
        output = np.asarray(audio, dtype=np.float32)
        if output.ndim > 1:
            output = np.mean(output, axis=1, dtype=np.float32)

        if sample_rate != spec.sample_rate:
            output = librosa.resample(output, orig_sr=sample_rate, target_sr=spec.sample_rate)

        if len(output) < spec.frame_count:
            output = np.pad(output, (0, spec.frame_count - len(output)))
        elif len(output) > spec.frame_count:
            output = output[: spec.frame_count]

        if spec.channels > 1:
            output = np.repeat(output[:, None], spec.channels, axis=1).reshape(-1)

        return np.ascontiguousarray(output, dtype=np.float32)

    def _encode_payload(self, audio: np.ndarray, spec: AudioChunkSpec) -> bytes:
        if spec.dtype is AudioDType.FLOAT32:
            return np.asarray(audio, dtype="<f4").tobytes()

        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767.0).astype("<i2").tobytes()