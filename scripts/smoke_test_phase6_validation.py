#!/usr/bin/env python3
"""Phase 6 validation: end-to-end checks for runtime stability.

Checks:
1) Server health endpoint
2) Single-shot file inference
3) 100 consecutive chunk inferences
4) Timeout bypass behavior in AudioModel.process_audio
5) Model switch / reload stability with memory-growth observation
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.client import InferenceClient
from src.models import AudioModel
from src.protocol import InferenceSettings


HOST = "127.0.0.1"
PORT = 18767
WS_URL = f"ws://{HOST}:{PORT}/ws"
HTTP_HEALTH_URL = f"http://{HOST}:{PORT}/health"
DEFAULT_MODEL_NAME = "05 つくよみちゃん公式RVCモデル 弱"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


ITERATIONS = _env_int("PHASE6_ITERATIONS", 100)
CONNECT_TIMEOUT_SEC = _env_float("PHASE6_CONNECT_TIMEOUT_SEC", 10.0)
LOAD_TIMEOUT_SEC = _env_float("PHASE6_LOAD_TIMEOUT_SEC", 120.0)
INFER_TIMEOUT_SEC = _env_float("PHASE6_INFER_TIMEOUT_SEC", 12.0)
FORCED_TIMEOUT_SEC = _env_float("PHASE6_FORCED_TIMEOUT_SEC", 0.02)
MEMORY_DELTA_LIMIT_MB = _env_float("PHASE6_MEMORY_DELTA_LIMIT_MB", 640.0)


def _check(label: str, condition: bool, detail: str = "") -> None:
    status = "[ok]" if condition else "[FAIL]"
    line = f"{status} {label}"
    if detail:
        line += f" - {detail}"
    print(line)
    if not condition:
        raise SystemExit(f"failed: {label}")


def _start_server_process() -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "src.server.inference_server",
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]
    proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent.parent))

    deadline = time.time() + 30.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early with code {proc.returncode}")
        try:
            with urlopen(HTTP_HEALTH_URL, timeout=2.0) as res:
                if res.status == 200:
                    return proc
        except Exception:
            pass
        time.sleep(0.5)

    proc.terminate()
    raise SystemExit("server startup timed out")


def _stop_server_process(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _read_rss_kb(pid: int) -> int:
    """Return VmRSS (kB) from /proc/<pid>/status. 0 if unavailable."""
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return 0
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        return 0
    return 0


def _prepare_audio_payload() -> tuple[bytes, int, int]:
    wav_path = Path("test/test.wav")
    if not wav_path.exists():
        raise SystemExit(f"Input file not found: {wav_path}")

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)

    target_sr = 48000
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    frame_count = int(sr)
    audio = audio[:frame_count]
    if len(audio) < frame_count:
        audio = np.pad(audio, (0, frame_count - len(audio)))

    payload = np.asarray(audio, dtype="<f4").tobytes()
    return payload, sr, frame_count


def main() -> None:
    print("=== Phase 6 validation ===\n")
    print(
        "thresholds: "
        f"iter={ITERATIONS}, connect={CONNECT_TIMEOUT_SEC:.2f}s, "
        f"load={LOAD_TIMEOUT_SEC:.2f}s, infer={INFER_TIMEOUT_SEC:.2f}s, "
        f"forced_timeout={FORCED_TIMEOUT_SEC:.3f}s, memory_limit={MEMORY_DELTA_LIMIT_MB:.1f}MB"
    )
    proc = _start_server_process()

    client = InferenceClient(WS_URL)
    payload, sr, frame_count = _prepare_audio_payload()

    try:
        # 1) サーバ単体ヘルスチェック
        with urlopen(HTTP_HEALTH_URL, timeout=3.0) as res:
            body = json.loads(res.read().decode("utf-8"))
        _check("server health endpoint", res.status == 200 and body.get("ok") is True, str(body))

        _check("client connect", client.connect(timeout=CONNECT_TIMEOUT_SEC))
        health = client.health()
        _check("client health RPC", health is not None and health.get("ok") is True, str(health))

        models = client.list_models()
        _check("list_models not empty", len(models) > 0, str(models))

        target_model = DEFAULT_MODEL_NAME if DEFAULT_MODEL_NAME in models else models[0]
        settings = InferenceSettings(
            model_name=target_model,
            pitch_shift=12,
            f0_method="rmvpe",
            index_rate=0.75,
            protect=0.33,
            filter_radius=3,
            rms_mix_rate=1.0,
            backend="rvc-python",
        )
        _check("load_model", client.load_model(target_model, settings, timeout=LOAD_TIMEOUT_SEC), target_model)

        # 2) 単発ファイル推論
        single = client.infer_chunk(
            payload,
            sample_rate=sr,
            frame_count=frame_count,
            sequence=1,
            timeout=INFER_TIMEOUT_SEC,
        )
        _check("single-shot infer_chunk", single is not None and len(single) > 0, f"bytes={0 if single is None else len(single)}")

        # 3) 音声チャンク100回連続推論
        start = time.time()
        ok_count = 0
        for i in range(ITERATIONS):
            out = client.infer_chunk(
                payload,
                sample_rate=sr,
                frame_count=frame_count,
                sequence=10 + i,
                timeout=INFER_TIMEOUT_SEC,
            )
            if out is not None and len(out) > 0:
                ok_count += 1
        elapsed = time.time() - start
        _check(
            f"{ITERATIONS} consecutive inferences",
            ok_count == ITERATIONS,
            f"ok={ok_count}/{ITERATIONS}, elapsed={elapsed:.1f}s",
        )

        # 4) タイムアウト時のバイパス確認（AudioModelのフォールバック）
        model = AudioModel()
        model.samplerate = sr
        model.blocksize = frame_count
        model.set_inference_client(client)
        model.set_rvc_model(target_model)
        model.set_rvc_pitch_shift(12)
        model.enable_rvc(True)
        model.rvc_processing_timeout = FORCED_TIMEOUT_SEC  # 故意にタイムアウトさせる

        in_audio = np.frombuffer(payload, dtype="<f4").reshape(-1, 1)
        out_audio = np.zeros_like(in_audio)
        model.process_audio(
            indata=in_audio,
            outdata=out_audio,
            frames=frame_count,
            time_info=None,
            status=None,
            mode="normal",
        )
        _check("timeout bypass fallback", float(np.max(np.abs(out_audio))) > 0.0)

        # 5) モデル切替時のメモリ解放確認（実測ベース）
        rss_before = _read_rss_kb(proc.pid)
        switch_ok = True

        if len(models) >= 2:
            a = models[0]
            b = models[1]
            for i in range(6):
                m = a if i % 2 == 0 else b
                s = InferenceSettings(model_name=m, pitch_shift=12, f0_method="rmvpe", backend="rvc-python")
                if not client.load_model(m, s, timeout=LOAD_TIMEOUT_SEC):
                    switch_ok = False
                    break
            switch_detail = f"switched between '{a}' and '{b}'"
        else:
            # 単一モデル環境では再ロード反復でリーク兆候を確認
            m = models[0]
            s = InferenceSettings(model_name=m, pitch_shift=12, f0_method="rmvpe", backend="rvc-python")
            for _ in range(6):
                if not client.load_model(m, s, timeout=LOAD_TIMEOUT_SEC):
                    switch_ok = False
                    break
            switch_detail = f"single model reload x6 ('{m}')"

        rss_after = _read_rss_kb(proc.pid)
        rss_delta_mb = (rss_after - rss_before) / 1024.0 if rss_before and rss_after else 0.0

        # 厳密なゼロ増加は求めず、異常な単調増加を検知する閾値を設定
        memory_ok = True if (rss_before == 0 or rss_after == 0) else (rss_delta_mb < MEMORY_DELTA_LIMIT_MB)
        _check("model switch/reload stability", switch_ok, switch_detail)
        _check("memory growth check", memory_ok, f"rss_delta={rss_delta_mb:.1f}MB")

        client.disconnect()
        _check("disconnect", not client.is_connected)

        print("\n=== Phase 6 validation passed ===")
    finally:
        try:
            client.disconnect()
        except Exception:
            pass
        _stop_server_process(proc)


if __name__ == "__main__":
    main()
