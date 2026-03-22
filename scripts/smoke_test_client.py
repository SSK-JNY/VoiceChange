#!/usr/bin/env python3
"""Phase 3 smoke test: InferenceClient against a live WSL inference server.

このテストはサーバをインプロセスで起動し、InferenceClient が
connect → health → list_models → load_model → disconnect の
一連の操作を正常に完了できることを確認する。

実行方法:
    ./venv310/bin/python scripts/smoke_test_client.py

前提条件:
    - src/models/rvc/ 以下に .pth ファイルが存在すること
    - test/test.wav が存在すること（infer_chunk 検証に使用）
"""

import asyncio
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn

from src.server import create_app
from src.client import InferenceClient
from src.protocol import InferenceSettings


_HOST = "127.0.0.1"
_PORT = 18765  # フェーズ3テスト用ポート（8765 との衝突を避ける）
_URL = f"ws://{_HOST}:{_PORT}/ws"
_MODEL_NAME = "05 つくよみちゃん公式RVCモデル 弱"


def _start_server_in_thread() -> threading.Thread:
    """uvicorn をバックグラウンドスレッドで起動する。"""
    app = create_app()
    config = uvicorn.Config(app, host=_HOST, port=_PORT, log_level="warning")
    server = uvicorn.Server(config)

    def _run():
        asyncio.run(server.serve())

    t = threading.Thread(target=_run, name="test-server", daemon=True)
    t.start()
    # サーバが立ち上がるまで少し待つ
    time.sleep(2.0)
    return t


def _check(label: str, condition: bool, detail: str = "") -> None:
    status = "[ok]" if condition else "[FAIL]"
    msg = f"{status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not condition:
        raise SystemExit(f"Smoke test failed at: {label}")


def main() -> None:
    print("=== Phase 3 smoke test: InferenceClient ===\n")

    # --- サーバ起動 ---
    print(f"Starting server on {_URL} ...")
    _start_server_in_thread()

    # --- InferenceClient 接続 ---
    client = InferenceClient(_URL)

    connected = client.connect(timeout=10.0)
    _check("connect()", connected)

    _check("is_connected", client.is_connected)

    # --- health ---
    health = client.health()
    _check("health()", health is not None and health.get("ok") is True, str(health))

    # --- list_models ---
    models = client.list_models()
    _check("list_models() returns list", isinstance(models, list))
    _check("list_models() not empty", len(models) > 0, f"models={models}")
    print(f"       モデル一覧: {models}")

    # --- load_model ---
    if _MODEL_NAME in models:
        settings = InferenceSettings(
            model_name=_MODEL_NAME,
            pitch_shift=12,
            f0_method="rmvpe",
            backend="rvc-python",
        )
        ok = client.load_model(_MODEL_NAME, settings, timeout=120.0)
        _check(f"load_model({_MODEL_NAME!r})", ok)
    else:
        print(f"[skip] load_model — '{_MODEL_NAME}' not found, using first model")
        first = models[0]
        ok = client.load_model(first, timeout=120.0)
        _check(f"load_model({first!r})", ok)

    # --- disconnect ---
    client.disconnect()
    _check("disconnect() → is_connected==False", not client.is_connected)

    print(f"\n=== Phase 3 smoke test passed ===")


if __name__ == "__main__":
    main()
