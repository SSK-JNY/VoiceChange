"""WSL-side inference engine backed by rvc-python."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
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
    from rvc_python.modules.vc import modules as rvc_vc_modules

    RVC_PYTHON_AVAILABLE = True
except Exception:
    RVCInference = None
    rvc_vc_modules = None
    RVC_PYTHON_AVAILABLE = False


logger = logging.getLogger(__name__)


class _InMemoryAudioInput:
    """rvc_python.lib.audio.load_audio 向けのメモリ入力アダプタ。"""

    def __init__(self, sample_rate: int, pcm_i16: np.ndarray):
        self._sample_rate = int(sample_rate)
        self._pcm_i16 = np.asarray(pcm_i16, dtype=np.int16)

    def __getitem__(self, idx):
        if idx == 0:
            return self._sample_rate
        if idx == 1:
            return self._pcm_i16
        raise IndexError(idx)


class InferenceEngine:
    """Stateful model manager and single-shot chunk inference executor."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.registry = ModelRegistry(models_dir=models_dir)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
        self.current_model_name = ""
        self.current_settings = InferenceSettings()
        self._inference: Optional[RVCInference] = None
        self._lock = threading.Lock()
        # 短チャンクでの空出力回避のため、直近履歴を使って最小推論長を確保する
        self._history_audio = np.zeros(0, dtype=np.float32)
        self._history_sample_rate = 0
        self._min_infer_seconds = 0.35
        
        # サーバ側処理時間計測
        self._inference_times_ms = []
        self._max_timing_samples = 100
        self._last_stats_report_ts = time.time()
        self._inmemory_audio_patch_applied = False

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
        self._ensure_inmemory_audio_patch()
        return self._inference

    def _ensure_inmemory_audio_patch(self) -> None:
        """rvc_python の load_audio にメモリ入力互換を追加する。"""
        if self._inmemory_audio_patch_applied:
            return
        if rvc_vc_modules is None:
            return

        original_load_audio = rvc_vc_modules.load_audio

        def load_audio_compat(file, sr):
            if isinstance(file, _InMemoryAudioInput):
                audio = file[1].astype(np.float32) / 32768.0
                if audio.ndim == 2:
                    audio = np.mean(audio, axis=-1)
                return librosa.resample(audio, orig_sr=file[0], target_sr=16000)
            return original_load_audio(file, sr)

        rvc_vc_modules.load_audio = load_audio_compat
        self._inmemory_audio_patch_applied = True
        logger.info("Applied in-memory audio patch for rvc_python load_audio")

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

        start_time = time.perf_counter()
        mono_audio = self._decode_payload(payload, spec)

        # 無音に近いチャンクは推論をスキップしてパススルーする。
        # rvc-python 側で空配列扱いになるケースを減らし、リアルタイム性を優先する。
        if len(mono_audio) == 0 or float(np.max(np.abs(mono_audio))) < 1e-6:
            return self._passthrough_result(sequence, spec, mono_audio, start_time)

        # 直近履歴と結合して推論入力を安定化する
        t_build_input = time.perf_counter()
        infer_audio = self._build_stable_infer_input(mono_audio, spec.sample_rate)
        build_input_ms = (time.perf_counter() - t_build_input) * 1000.0
        
        infer_spec = AudioChunkSpec(
            sample_rate=spec.sample_rate,
            channels=1,
            dtype=AudioDType.FLOAT32,
            frame_count=len(infer_audio),
        )

        try:
            with self._lock:
                # rvc_python は int16 相当の配列を受けるため、メモリ入力に変換する
                t_fileio_in = time.perf_counter()
                pcm_i16 = (np.clip(infer_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                mem_input = _InMemoryAudioInput(infer_spec.sample_rate, pcm_i16)
                fileio_in_ms = (time.perf_counter() - t_fileio_in) * 1000.0

                t_infer = time.perf_counter()
                with self._weights_only_compat():
                    infer = self._ensure_backend()
                    if not infer.current_model:
                        raise RuntimeError("No active model")

                    model_info = infer.models[infer.current_model]
                    file_index = model_info.get("index", "")
                    converted_audio = infer.vc.vc_single(
                        sid=0,
                        input_audio_path=mem_input,
                        f0_up_key=infer.f0up_key,
                        f0_method=infer.f0method,
                        file_index=file_index,
                        index_rate=infer.index_rate,
                        filter_radius=infer.filter_radius,
                        resample_sr=infer.resample_sr,
                        rms_mix_rate=infer.rms_mix_rate,
                        protect=infer.protect,
                        f0_file="",
                        file_index2="",
                    )
                    converted_sr = int(getattr(infer.vc, "tgt_sr", infer_spec.sample_rate))
                infer_ms = (time.perf_counter() - t_infer) * 1000.0

                # 入出力ファイルI/Oは使用しない
                fileio_out_ms = 0.0

                if isinstance(converted_audio, tuple):
                    raise ValueError(f"invalid infer output type: {type(converted_audio)}")
                converted_audio = np.asarray(converted_audio, dtype=np.float32)
                if np.size(converted_audio) == 0:
                    raise ValueError("empty infer output")
        except Exception as exc:
            logger.warning("infer_chunk fallback to passthrough: %s", exc)
            return self._passthrough_result(sequence, spec, mono_audio, start_time)

        t_postproc = time.perf_counter()
        processed_full = self._normalize_output(converted_audio, converted_sr, infer_spec)
        processed = self._select_output_segment(processed_full, spec.frame_count)
        output_bytes = self._encode_payload(processed, spec)
        postproc_ms = (time.perf_counter() - t_postproc) * 1000.0
        
        processing_ms = (time.perf_counter() - start_time) * 1000.0
        
        # 処理時間を記録（ボトルネック検出用）
        self._inference_times_ms.append(infer_ms)
        if len(self._inference_times_ms) > self._max_timing_samples:
            self._inference_times_ms.pop(0)
        
        # 定期的に統計情報をログ出力
        now = time.time()
        if now - self._last_stats_report_ts > 5.0:
            avg_infer = np.mean(self._inference_times_ms) if self._inference_times_ms else 0.0
            max_infer = np.max(self._inference_times_ms) if self._inference_times_ms else 0.0
            logger.info(
                "Server inference stats: infer_avg=%.2fms/max=%.2fms, "
                "build_input=%.2fms, fileio_in=%.2fms, fileio_out=%.2fms, postproc=%.2fms",
                avg_infer, max_infer,
                build_input_ms, fileio_in_ms, fileio_out_ms, postproc_ms
            )
            self._last_stats_report_ts = now

        return (
            InferChunkResultMessage(
                sequence=sequence,
                audio=spec,
                processing_ms=processing_ms,
                fallback=False,
            ),
            output_bytes,
        )

    def _build_stable_infer_input(self, mono_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """直近履歴を前置して最小推論長を満たす入力を作る。"""
        min_frames = max(1, int(sample_rate * self._min_infer_seconds))

        if self._history_sample_rate != sample_rate:
            self._history_audio = np.zeros(0, dtype=np.float32)
            self._history_sample_rate = sample_rate

        working = np.concatenate([self._history_audio, mono_audio]).astype(np.float32, copy=False)
        if len(working) < min_frames:
            pad = min_frames - len(working)
            if len(working) > 0:
                working = np.pad(working, (0, pad), mode="edge")
            else:
                working = np.zeros(min_frames, dtype=np.float32)

        max_history = max(min_frames * 2, len(mono_audio))
        source_for_history = np.concatenate([self._history_audio, mono_audio]).astype(np.float32, copy=False)
        if len(source_for_history) > max_history:
            self._history_audio = source_for_history[-max_history:]
        else:
            self._history_audio = source_for_history

        return np.ascontiguousarray(working, dtype=np.float32)

    def _select_output_segment(self, processed_full: np.ndarray, frame_count: int) -> np.ndarray:
        """推論結果から要求フレーム数ぶんを末尾から切り出す。"""
        if len(processed_full) >= frame_count:
            return processed_full[-frame_count:]
        return np.pad(processed_full, (0, frame_count - len(processed_full)))

    def _passthrough_result(
        self,
        sequence: int,
        spec: AudioChunkSpec,
        mono_audio: np.ndarray,
        start_time: float,
    ) -> Tuple[InferChunkResultMessage, bytes]:
        """推論失敗時のパススルー結果を返す。"""
        processed = self._normalize_output(mono_audio, spec.sample_rate, spec)
        output_bytes = self._encode_payload(processed, spec)
        processing_ms = (time.perf_counter() - start_time) * 1000.0
        return (
            InferChunkResultMessage(
                sequence=sequence,
                audio=spec,
                processing_ms=processing_ms,
                fallback=True,
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