"""WSL 推論サーバへの WebSocket RPC クライアント。

asyncio イベントループをバックグラウンドのデーモンスレッドで実行し、
同期メソッドとして各 RPC 操作を公開する。
Tkinter などのメインスレッドからそのまま呼び出せる。

フェーズ 3 実装範囲: connect / disconnect / health / list_models / load_model / update_params
フェーズ 4 実装範囲: infer_chunk（ストリーミング推論）
"""

import asyncio
import json
import logging
import threading
from typing import List, Optional

import websockets

from src.protocol import (
    AudioChunkSpec,
    AudioDType,
    ErrorMessage,
    HealthMessage,
    HealthResultMessage,
    HelloAckMessage,
    HelloMessage,
    InferChunkMessage,
    InferChunkResultMessage,
    InferenceSettings,
    ListModelsMessage,
    ListModelsResultMessage,
    LoadModelMessage,
    LoadModelResultMessage,
    UpdateParamsMessage,
    UpdateParamsResultMessage,
    deserialize_message,
    serialize_message,
)

logger = logging.getLogger(__name__)

_DEFAULT_URL = "ws://127.0.0.1:8765/ws"


class InferenceClient:
    """WSL 推論サーバへの同期 API を持つ WebSocket クライアント。

    内部では asyncio ループをバックグラウンドスレッドで実行する。
    すべての公開メソッドは同期的で、タイムアウト付きで結果を待つ。

    使い方:
        client = InferenceClient()
        if client.connect():
            models = client.list_models()
            client.load_model("05 つくよみちゃん", InferenceSettings(...))
        client.disconnect()
    """

    def __init__(self, url: str = _DEFAULT_URL) -> None:
        self._url = url
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="inference-client-loop",
            daemon=True,
        )
        self._thread.start()

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._rpc_lock: Optional[asyncio.Lock] = None  # ループスレッド内で初期化
        self._connected = False

    # ------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """現在 WebSocket が接続中かどうか。"""
        return self._connected

    # ------------------------------------------------------------------
    # 公開同期 API
    # ------------------------------------------------------------------

    def connect(self, timeout: float = 10.0) -> bool:
        """サーバへ接続して hello handshake を行う。

        Returns:
            True: 接続成功、False: 失敗
        """
        fut = asyncio.run_coroutine_threadsafe(self._async_connect(), self._loop)
        try:
            result = fut.result(timeout=timeout)
            self._connected = result
            return result
        except Exception as exc:
            logger.warning("connect failed: %s", exc)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """接続を閉じる。"""
        fut = asyncio.run_coroutine_threadsafe(self._async_disconnect(), self._loop)
        try:
            fut.result(timeout=5.0)
        except Exception:
            pass
        self._connected = False

    def health(self, timeout: float = 5.0) -> Optional[dict]:
        """サーバのヘルスチェックを実行する。

        Returns:
            {"ok": bool, "device": str, "model_loaded": bool} または None（失敗）
        """
        fut = asyncio.run_coroutine_threadsafe(self._async_health(), self._loop)
        try:
            return fut.result(timeout=timeout)
        except Exception as exc:
            logger.warning("health failed: %s", exc)
            return None

    def list_models(self, timeout: float = 5.0) -> List[str]:
        """サーバからモデル名一覧を取得する。

        Returns:
            モデル名のリスト（失敗時は空リスト）
        """
        fut = asyncio.run_coroutine_threadsafe(self._async_list_models(), self._loop)
        try:
            return fut.result(timeout=timeout)
        except Exception as exc:
            logger.warning("list_models failed: %s", exc)
            return []

    def load_model(
        self,
        model_name: str,
        settings: Optional[InferenceSettings] = None,
        timeout: float = 120.0,
    ) -> bool:
        """指定したモデルをサーバにロードする。

        Args:
            model_name: ロードするモデル名。
            settings:   推論パラメータ（省略時はデフォルト設定）。
            timeout:    初回ロードは fairseq 初期化で 60 秒以上かかる場合がある。

        Returns:
            True: ロード成功、False: 失敗
        """
        if settings is None:
            settings = InferenceSettings(model_name=model_name)
        fut = asyncio.run_coroutine_threadsafe(
            self._async_load_model(model_name, settings), self._loop
        )
        try:
            return fut.result(timeout=timeout)
        except Exception as exc:
            logger.warning("load_model failed: %s", exc)
            return False

    def update_params(
        self,
        settings: InferenceSettings,
        timeout: float = 5.0,
    ) -> bool:
        """推論パラメータのみを更新する（モデル再ロードなし）。"""
        fut = asyncio.run_coroutine_threadsafe(
            self._async_update_params(settings), self._loop
        )
        try:
            return fut.result(timeout=timeout)
        except Exception as exc:
            logger.warning("update_params failed: %s", exc)
            return False

    def infer_chunk(
        self,
        payload: bytes,
        sample_rate: int = 48000,
        frame_count: int = 48000,
        sequence: int = 0,
        timeout: float = 30.0,
    ) -> Optional[bytes]:
        """PCM バイト列を推論してサーバから変換済み PCM バイト列を受け取る。

        payload は float32 little-endian モノラルを想定。
        フェーズ 3 では直接呼び出しはしないが、フェーズ 4 で使用する。
        """
        fut = asyncio.run_coroutine_threadsafe(
            self._async_infer_chunk(payload, sample_rate, frame_count, sequence),
            self._loop,
        )
        try:
            return fut.result(timeout=timeout)
        except Exception as exc:
            logger.warning("infer_chunk failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 内部非同期実装
    # ------------------------------------------------------------------

    def _ensure_rpc_lock(self) -> asyncio.Lock:
        """イベントループのスレッド内で asyncio.Lock を遅延初期化する。"""
        if self._rpc_lock is None:
            self._rpc_lock = asyncio.Lock()
        return self._rpc_lock

    async def _async_connect(self) -> bool:
        self._rpc_lock = asyncio.Lock()
        try:
            self._ws = await websockets.connect(self._url)
            req = serialize_message(HelloMessage(client="windows-gui", protocol_version=1))
            await self._ws.send(json.dumps(req))
            raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
            resp = deserialize_message(json.loads(raw))
            if isinstance(resp, HelloAckMessage):
                logger.info("Connected to %s (server=%s)", self._url, resp.server)
                return True
            logger.warning("Unexpected response to hello: %s", type(resp).__name__)
            return False
        except Exception as exc:
            logger.warning("_async_connect error: %s", exc)
            if self._ws is not None:
                await self._safe_close()
            return False

    async def _async_disconnect(self) -> None:
        await self._safe_close()

    async def _safe_close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _rpc(self, request_msg, expected_type, recv_timeout: float = 5.0):
        """JSON 単発 request → response RPC を実行する。"""
        if self._ws is None:
            raise RuntimeError("not connected")
        lock = self._ensure_rpc_lock()
        async with lock:
            await self._ws.send(json.dumps(serialize_message(request_msg)))
            raw = await asyncio.wait_for(self._ws.recv(), recv_timeout)
        resp = deserialize_message(json.loads(raw))
        if isinstance(resp, ErrorMessage):
            logger.warning("server error [%s]: %s", resp.code, resp.message)
            return None
        if not isinstance(resp, expected_type):
            logger.warning("unexpected response type: %s", type(resp).__name__)
            return None
        return resp

    async def _async_health(self) -> Optional[dict]:
        resp = await self._rpc(HealthMessage(), HealthResultMessage)
        if resp is None:
            return None
        return {
            "ok": resp.ok,
            "device": resp.device,
            "model_loaded": resp.model_loaded,
            "active_model": resp.active_model,
        }

    async def _async_list_models(self) -> List[str]:
        resp = await self._rpc(ListModelsMessage(), ListModelsResultMessage)
        if resp is None:
            return []
        return [m.name for m in resp.models]

    async def _async_load_model(
        self, model_name: str, settings: InferenceSettings
    ) -> bool:
        if self._ws is None:
            raise RuntimeError("not connected")
        lock = self._ensure_rpc_lock()
        async with lock:
            req = LoadModelMessage(model_name=model_name, params=settings)
            await self._ws.send(json.dumps(serialize_message(req)))
            # モデルロードは長時間かかるため recv_timeout を長めに設定
            raw = await asyncio.wait_for(self._ws.recv(), timeout=120.0)
        resp = deserialize_message(json.loads(raw))
        if isinstance(resp, ErrorMessage):
            logger.warning("load_model error [%s]: %s", resp.code, resp.message)
            return False
        if isinstance(resp, LoadModelResultMessage):
            logger.info("load_model result: ok=%s device=%s", resp.ok, resp.device)
            return resp.ok
        return False

    async def _async_update_params(self, settings: InferenceSettings) -> bool:
        resp = await self._rpc(UpdateParamsMessage(params=settings), UpdateParamsResultMessage)
        return resp is not None and resp.ok

    async def _async_infer_chunk(
        self,
        payload: bytes,
        sample_rate: int,
        frame_count: int,
        sequence: int,
    ) -> Optional[bytes]:
        if self._ws is None:
            raise RuntimeError("not connected")
        spec = AudioChunkSpec(
            sample_rate=sample_rate,
            channels=1,
            dtype=AudioDType.FLOAT32,
            frame_count=frame_count,
        )
        header_msg = InferChunkMessage(sequence=sequence, audio=spec)
        lock = self._ensure_rpc_lock()
        async with lock:
            await self._ws.send(json.dumps(serialize_message(header_msg)))
            await self._ws.send(payload)
            raw_header = await asyncio.wait_for(self._ws.recv(), timeout=30.0)
            resp = deserialize_message(json.loads(raw_header))
            if isinstance(resp, ErrorMessage):
                logger.warning("infer_chunk error [%s]: %s", resp.code, resp.message)
                return None
            if not isinstance(resp, InferChunkResultMessage):
                logger.warning("unexpected infer_chunk response: %s", type(resp).__name__)
                return None
            raw_audio = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
        return raw_audio
