"""FastAPI-based WebSocket inference server running inside WSL."""

from __future__ import annotations

import argparse
import json
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.protocol import (
    ErrorCode,
    ErrorMessage,
    HealthMessage,
    HelloAckMessage,
    HelloMessage,
    InferChunkMessage,
    ListModelsMessage,
    ListModelsResultMessage,
    LoadModelMessage,
    ProtocolMessage,
    SessionSettings,
    UpdateParamsMessage,
    deserialize_message,
    serialize_message,
)

from .inference_engine import InferenceEngine
from .session_manager import SessionManager


def create_app() -> FastAPI:
    app = FastAPI(title="VoiceChange WSL Inference Server")
    engine = InferenceEngine()
    sessions = SessionManager()

    async def send_protocol_message(websocket: WebSocket, message: ProtocolMessage) -> None:
        await websocket.send_text(json.dumps(serialize_message(message), ensure_ascii=False))

    @app.get("/health")
    async def http_health() -> dict:
        return serialize_message(engine.health())

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        session_id = str(uuid4())
        sessions.create(session_id)

        try:
            while True:
                message = await websocket.receive()
                message_type = message.get("type")
                if message_type == "websocket.disconnect":
                    break

                if message.get("text") is not None:
                    try:
                        payload = json.loads(message["text"])
                        protocol_message = deserialize_message(payload)
                        await _handle_text_message(
                            websocket,
                            session_id,
                            protocol_message,
                            engine,
                            sessions,
                            send_protocol_message,
                        )
                    except Exception as exc:
                        await send_protocol_message(
                            websocket,
                            ErrorMessage(
                                code=ErrorCode.BACKEND_ERROR,
                                message=str(exc),
                            ),
                        )
                elif message.get("bytes") is not None:
                    await _handle_binary_message(
                        websocket,
                        session_id,
                        message["bytes"],
                        engine,
                        sessions,
                        send_protocol_message,
                    )
        except WebSocketDisconnect:
            sessions.remove(session_id)
        finally:
            sessions.remove(session_id)

    return app


async def _handle_text_message(
    websocket: WebSocket,
    session_id: str,
    message: ProtocolMessage,
    engine: InferenceEngine,
    sessions: SessionManager,
    send_protocol_message,
) -> None:
    if isinstance(message, HelloMessage):
        sessions.get(session_id).settings = SessionSettings(
            protocol_version=message.protocol_version,
            client_name=message.client,
            server_name="wsl-inference",
        )
        await send_protocol_message(websocket, HelloAckMessage(protocol_version=message.protocol_version))
        return

    if isinstance(message, HealthMessage):
        await send_protocol_message(websocket, engine.health())
        return

    if isinstance(message, ListModelsMessage):
        await send_protocol_message(
            websocket,
            ListModelsResultMessage(models=engine.list_models()),
        )
        return

    if isinstance(message, LoadModelMessage):
        result = engine.load_model(message.model_name, message.params)
        await send_protocol_message(websocket, result)
        return

    if isinstance(message, UpdateParamsMessage):
        result = engine.update_params(message.params)
        await send_protocol_message(websocket, result)
        return

    if isinstance(message, InferChunkMessage):
        sessions.set_pending_infer(session_id, message)
        return

    await send_protocol_message(
        websocket,
        ErrorMessage(code=ErrorCode.PROTOCOL_MISMATCH, message=f"Unsupported message: {message.type.value}"),
    )


async def _handle_binary_message(
    websocket: WebSocket,
    session_id: str,
    payload: bytes,
    engine: InferenceEngine,
    sessions: SessionManager,
    send_protocol_message,
) -> None:
    pending = sessions.pop_pending_infer(session_id)
    if pending is None:
        await send_protocol_message(
            websocket,
            ErrorMessage(
                code=ErrorCode.PROTOCOL_MISMATCH,
                message="Received binary payload without pending infer_chunk header",
            ),
        )
        return

    try:
        result_message, output_bytes = engine.infer_chunk(pending.sequence, pending.audio, payload)
        await send_protocol_message(websocket, result_message)
        await websocket.send_bytes(output_bytes)
    except FileNotFoundError as exc:
        await send_protocol_message(
            websocket,
            ErrorMessage(code=ErrorCode.MODEL_NOT_FOUND, message=str(exc), sequence=pending.sequence),
        )
    except RuntimeError as exc:
        code = ErrorCode.MODEL_NOT_LOADED if "No active model" in str(exc) else ErrorCode.BACKEND_ERROR
        await send_protocol_message(
            websocket,
            ErrorMessage(code=code, message=str(exc), sequence=pending.sequence),
        )
    except TimeoutError as exc:
        await send_protocol_message(
            websocket,
            ErrorMessage(code=ErrorCode.INFERENCE_TIMEOUT, message=str(exc), sequence=pending.sequence),
        )
    except ValueError as exc:
        await send_protocol_message(
            websocket,
            ErrorMessage(code=ErrorCode.INVALID_PARAMS, message=str(exc), sequence=pending.sequence),
        )
    except Exception as exc:
        await send_protocol_message(
            websocket,
            ErrorMessage(code=ErrorCode.BACKEND_ERROR, message=str(exc), sequence=pending.sequence),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceChange WSL inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()