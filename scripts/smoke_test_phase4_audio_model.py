#!/usr/bin/env python3
"""Phase 4 smoke test: AudioModel.process_audio uses RPC inference path.

このテストは以下を確認する。
1) InferenceClient でサーバへ接続できる
2) モデルロード後、AudioModel に client を注入できる
3) process_audio(normal + rvc_enabled) が RPC 推論経由で出力を返す

注意:
- 推論ウィンドウ要件のため 48kHz / 1秒チャンクで検証する
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.client import InferenceClient
from src.models import AudioModel
from src.protocol import InferenceSettings


_HOST = "127.0.0.1"
_PORT = 18766
_URL = f"ws://{_HOST}:{_PORT}/ws"
_MODEL_NAME = "05 つくよみちゃん公式RVCモデル 弱"


def _start_server_process() -> subprocess.Popen:
    """WSL推論サーバを別プロセスで起動し、ヘルス待ちする。"""
    cmd = [
        sys.executable,
        "-m",
        "src.server.inference_server",
        "--host",
        _HOST,
        "--port",
        str(_PORT),
    ]
    proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent.parent))

    deadline = time.time() + 20.0
    health_url = f"http://{_HOST}:{_PORT}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early with code {proc.returncode}")
        try:
            resp = httpx.get(health_url, timeout=1.0)
            if resp.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise SystemExit("server health check timed out")


def _check(label: str, condition: bool, detail: str = "") -> None:
    status = "[ok]" if condition else "[FAIL]"
    print(f"{status} {label}{' - ' + detail if detail else ''}")
    if not condition:
        raise SystemExit(f"failed: {label}")


def main() -> None:
    print("=== Phase 4 smoke test: AudioModel RPC path ===\n")
    server_proc = _start_server_process()

    client = InferenceClient(_URL)
    try:
        _check("connect", client.connect(timeout=10.0))

        models = client.list_models()
        _check("list_models not empty", len(models) > 0, str(models))

        model_name = _MODEL_NAME if _MODEL_NAME in models else models[0]
        settings = InferenceSettings(
            model_name=model_name,
            pitch_shift=12,
            f0_method="rmvpe",
            backend="rvc-python",
        )
        _check("load_model", client.load_model(model_name, settings, timeout=120.0), model_name)

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

        model = AudioModel()
        model.samplerate = sr
        model.blocksize = frame_count
        model.rvc_processing_timeout = 30.0
        model.set_inference_client(client)
        model.set_rvc_model(model_name)
        model.set_rvc_pitch_shift(12)
        model.enable_rvc(True)

        indata = audio.reshape(-1, 1).astype(np.float32)
        outdata = np.zeros_like(indata)

        model.process_audio(
            indata=indata,
            outdata=outdata,
            frames=frame_count,
            time_info=None,
            status=None,
            mode="normal",
        )

        _check("output shape", outdata.shape == indata.shape, f"{outdata.shape}")
        _check("output not all zeros", float(np.max(np.abs(outdata))) > 0.0)

        client.disconnect()
        _check("disconnect", not client.is_connected)

        print("\n=== Phase 4 smoke test passed ===")
    finally:
        try:
            client.disconnect()
        except Exception:
            pass
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
