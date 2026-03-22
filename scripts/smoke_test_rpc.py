#!/usr/bin/env python3
"""Basic in-process smoke test for the WSL inference WebSocket server."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.server import create_app


def main() -> None:
    app = create_app()
    model_name = "05 つくよみちゃん公式RVCモデル 弱"
    input_path = Path("test/test.wav")

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    audio, sample_rate = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)

    # rvc-python backend needs a longer window than real-time chunks for this phase-2 smoke test.
    frame_count = int(sample_rate)
    audio = audio[:frame_count]
    if len(audio) < frame_count:
        audio = np.pad(audio, (0, frame_count - len(audio)))

    payload = np.asarray(audio, dtype="<f4").tobytes()

    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        websocket.send_text(json.dumps({"type": "hello", "client": "smoke-test", "protocol_version": 1}))
        print(websocket.receive_json())

        websocket.send_text(json.dumps({"type": "health"}))
        print(websocket.receive_json())

        websocket.send_text(json.dumps({"type": "list_models"}))
        print(websocket.receive_json())

        websocket.send_text(
            json.dumps(
                {
                    "type": "load_model",
                    "model_name": model_name,
                    "params": {
                        "model_name": model_name,
                        "pitch_shift": 12,
                        "f0_method": "rmvpe",
                        "index_rate": 0.75,
                        "protect": 0.33,
                        "filter_radius": 3,
                        "rms_mix_rate": 1.0,
                        "backend": "rvc-python",
                    },
                }
            )
        )
        print(websocket.receive_json())

        websocket.send_text(
            json.dumps(
                {
                    "type": "infer_chunk",
                    "sequence": 1,
                    "sample_rate": sample_rate,
                    "channels": 1,
                    "dtype": "float32",
                    "frame_count": frame_count,
                }
            )
        )
        websocket.send_bytes(payload)
        result_header = websocket.receive_json()
        result_audio = websocket.receive_bytes()
        print(result_header)
        print(f"audio_bytes={len(result_audio)}")


if __name__ == "__main__":
    main()