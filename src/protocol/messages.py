"""Shared RPC message types for Windows client and WSL inference server."""

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from .audio_schema import AudioChunkSpec
from .settings_schema import InferenceSettings


class MessageType(str, Enum):
    HELLO = "hello"
    HELLO_ACK = "hello_ack"
    LIST_MODELS = "list_models"
    LIST_MODELS_RESULT = "list_models_result"
    LOAD_MODEL = "load_model"
    LOAD_MODEL_RESULT = "load_model_result"
    UPDATE_PARAMS = "update_params"
    UPDATE_PARAMS_RESULT = "update_params_result"
    INFER_CHUNK = "infer_chunk"
    INFER_CHUNK_RESULT = "infer_chunk_result"
    HEALTH = "health"
    HEALTH_RESULT = "health_result"
    ERROR = "error"


class ErrorCode(str, Enum):
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INVALID_PARAMS = "INVALID_PARAMS"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    UNSUPPORTED_SAMPLE_RATE = "UNSUPPORTED_SAMPLE_RATE"
    BACKEND_ERROR = "BACKEND_ERROR"
    PROTOCOL_MISMATCH = "PROTOCOL_MISMATCH"


def _default_audio_chunk() -> AudioChunkSpec:
    return AudioChunkSpec.from_dict(
        {
            "sample_rate": 48000,
            "channels": 1,
            "dtype": "float32",
            "frame_count": 960,
        }
    )


@dataclass(frozen=True)
class ModelInfo:
    name: str
    has_index: bool = False
    version: str = "v2"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        return cls(
            name=str(data["name"]),
            has_index=bool(data.get("has_index", False)),
            version=str(data.get("version", "v2")),
        )


@dataclass(frozen=True)
class HelloMessage:
    type: MessageType = MessageType.HELLO
    client: str = "windows-gui"
    protocol_version: int = 1
    session_id: Optional[str] = None


@dataclass(frozen=True)
class HelloAckMessage:
    type: MessageType = MessageType.HELLO_ACK
    protocol_version: int = 1
    server: str = "wsl-inference"
    features: List[str] = field(default_factory=lambda: ["infer_chunk", "load_model", "list_models"])


@dataclass(frozen=True)
class ListModelsMessage:
    type: MessageType = MessageType.LIST_MODELS


@dataclass(frozen=True)
class ListModelsResultMessage:
    type: MessageType = MessageType.LIST_MODELS_RESULT
    models: List[ModelInfo] = field(default_factory=list)


@dataclass(frozen=True)
class LoadModelMessage:
    type: MessageType = MessageType.LOAD_MODEL
    model_name: str = ""
    params: InferenceSettings = field(default_factory=InferenceSettings)


@dataclass(frozen=True)
class LoadModelResultMessage:
    type: MessageType = MessageType.LOAD_MODEL_RESULT
    ok: bool = True
    active_model: str = ""
    device: str = "cpu:0"


@dataclass(frozen=True)
class UpdateParamsMessage:
    type: MessageType = MessageType.UPDATE_PARAMS
    params: InferenceSettings = field(default_factory=InferenceSettings)


@dataclass(frozen=True)
class UpdateParamsResultMessage:
    type: MessageType = MessageType.UPDATE_PARAMS_RESULT
    ok: bool = True


@dataclass(frozen=True)
class InferChunkMessage:
    type: MessageType = MessageType.INFER_CHUNK
    sequence: int = 0
    audio: AudioChunkSpec = field(default_factory=_default_audio_chunk)


@dataclass(frozen=True)
class InferChunkResultMessage:
    type: MessageType = MessageType.INFER_CHUNK_RESULT
    sequence: int = 0
    audio: AudioChunkSpec = field(default_factory=_default_audio_chunk)
    processing_ms: float = 0.0
    fallback: bool = False


@dataclass(frozen=True)
class HealthMessage:
    type: MessageType = MessageType.HEALTH


@dataclass(frozen=True)
class HealthResultMessage:
    type: MessageType = MessageType.HEALTH_RESULT
    ok: bool = True
    device: str = "cpu:0"
    model_loaded: bool = False
    active_model: str = ""


@dataclass(frozen=True)
class ErrorMessage:
    type: MessageType = MessageType.ERROR
    code: ErrorCode = ErrorCode.BACKEND_ERROR
    message: str = ""
    sequence: Optional[int] = None


ProtocolMessage = Union[
    HelloMessage,
    HelloAckMessage,
    ListModelsMessage,
    ListModelsResultMessage,
    LoadModelMessage,
    LoadModelResultMessage,
    UpdateParamsMessage,
    UpdateParamsResultMessage,
    InferChunkMessage,
    InferChunkResultMessage,
    HealthMessage,
    HealthResultMessage,
    ErrorMessage,
]


_MESSAGE_TYPES: Dict[MessageType, Type[ProtocolMessage]] = {
    MessageType.HELLO: HelloMessage,
    MessageType.HELLO_ACK: HelloAckMessage,
    MessageType.LIST_MODELS: ListModelsMessage,
    MessageType.LIST_MODELS_RESULT: ListModelsResultMessage,
    MessageType.LOAD_MODEL: LoadModelMessage,
    MessageType.LOAD_MODEL_RESULT: LoadModelResultMessage,
    MessageType.UPDATE_PARAMS: UpdateParamsMessage,
    MessageType.UPDATE_PARAMS_RESULT: UpdateParamsResultMessage,
    MessageType.INFER_CHUNK: InferChunkMessage,
    MessageType.INFER_CHUNK_RESULT: InferChunkResultMessage,
    MessageType.HEALTH: HealthMessage,
    MessageType.HEALTH_RESULT: HealthResultMessage,
    MessageType.ERROR: ErrorMessage,
}


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if is_dataclass(value):
        return {key: _serialize_value(item) for key, item in asdict(value).items()}
    return value


def serialize_message(message: ProtocolMessage) -> dict:
    """Convert a protocol message dataclass into a JSON-friendly dict."""

    if isinstance(message, (InferChunkMessage, InferChunkResultMessage)):
        data = {
            "type": message.type.value,
            "sequence": message.sequence,
        }
        data.update(message.audio.to_dict())
        if isinstance(message, InferChunkResultMessage):
            data["processing_ms"] = message.processing_ms
            data["fallback"] = message.fallback
        return data

    data = {}
    for field_name in message.__dataclass_fields__:
        data[field_name] = _serialize_value(getattr(message, field_name))
    return data


def deserialize_message(payload: dict) -> ProtocolMessage:
    """Create a protocol message dataclass from a decoded JSON payload."""

    message_type = MessageType(payload["type"])
    message_cls = _MESSAGE_TYPES[message_type]

    if message_cls is ListModelsMessage or message_cls is HealthMessage:
        return message_cls()
    if message_cls is HelloMessage:
        return HelloMessage(
            client=str(payload.get("client", "windows-gui")),
            protocol_version=int(payload.get("protocol_version", 1)),
            session_id=payload.get("session_id"),
        )
    if message_cls is HelloAckMessage:
        return HelloAckMessage(
            protocol_version=int(payload.get("protocol_version", 1)),
            server=str(payload.get("server", "wsl-inference")),
            features=list(payload.get("features", ["infer_chunk", "load_model", "list_models"])),
        )
    if message_cls is ListModelsResultMessage:
        return ListModelsResultMessage(
            models=[ModelInfo.from_dict(item) for item in payload.get("models", [])]
        )
    if message_cls is LoadModelMessage:
        params_payload = dict(payload.get("params", {}))
        params_payload.setdefault("model_name", payload.get("model_name", ""))
        return LoadModelMessage(
            model_name=str(payload.get("model_name", "")),
            params=InferenceSettings.from_dict(params_payload),
        )
    if message_cls is LoadModelResultMessage:
        return LoadModelResultMessage(
            ok=bool(payload.get("ok", True)),
            active_model=str(payload.get("active_model", "")),
            device=str(payload.get("device", "cpu:0")),
        )
    if message_cls is UpdateParamsMessage:
        return UpdateParamsMessage(
            params=InferenceSettings.from_dict(payload.get("params", {}))
        )
    if message_cls is UpdateParamsResultMessage:
        return UpdateParamsResultMessage(ok=bool(payload.get("ok", True)))
    if message_cls is InferChunkMessage:
        audio_payload = payload.get("audio", payload)
        return InferChunkMessage(
            sequence=int(payload["sequence"]),
            audio=AudioChunkSpec.from_dict(audio_payload),
        )
    if message_cls is InferChunkResultMessage:
        audio_payload = payload.get("audio", payload)
        return InferChunkResultMessage(
            sequence=int(payload["sequence"]),
            audio=AudioChunkSpec.from_dict(audio_payload),
            processing_ms=float(payload.get("processing_ms", 0.0)),
            fallback=bool(payload.get("fallback", False)),
        )
    if message_cls is HealthResultMessage:
        return HealthResultMessage(
            ok=bool(payload.get("ok", True)),
            device=str(payload.get("device", "cpu:0")),
            model_loaded=bool(payload.get("model_loaded", False)),
            active_model=str(payload.get("active_model", "")),
        )
    if message_cls is ErrorMessage:
        return ErrorMessage(
            code=ErrorCode(payload.get("code", ErrorCode.BACKEND_ERROR.value)),
            message=str(payload.get("message", "")),
            sequence=payload.get("sequence"),
        )

    raise ValueError(f"Unsupported message type: {message_type}")