"""Inference parameter schemas shared by the Windows client and WSL server."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InferenceSettings:
    """Per-model inference parameters sent to the WSL inference server."""

    model_name: str = ""
    pitch_shift: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.75
    protect: float = 0.33
    filter_radius: int = 3
    rms_mix_rate: float = 1.0
    backend: str = "rvc-python"

    def validate(self) -> None:
        if self.model_name and not self.model_name.strip():
            raise ValueError("model_name must not be blank")
        if not -24 <= self.pitch_shift <= 24:
            raise ValueError("pitch_shift must be between -24 and 24")
        if not 0.0 <= self.index_rate <= 1.0:
            raise ValueError("index_rate must be between 0.0 and 1.0")
        if not 0.0 <= self.protect <= 1.0:
            raise ValueError("protect must be between 0.0 and 1.0")
        if self.filter_radius < 0:
            raise ValueError("filter_radius must be >= 0")
        if self.rms_mix_rate < 0.0:
            raise ValueError("rms_mix_rate must be >= 0.0")
        if not self.f0_method:
            raise ValueError("f0_method is required")
        if not self.backend:
            raise ValueError("backend is required")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceSettings":
        settings = cls(
            model_name=str(data.get("model_name", "")),
            pitch_shift=int(data.get("pitch_shift", 0)),
            f0_method=str(data.get("f0_method", "rmvpe")),
            index_rate=float(data.get("index_rate", 0.75)),
            protect=float(data.get("protect", 0.33)),
            filter_radius=int(data.get("filter_radius", 3)),
            rms_mix_rate=float(data.get("rms_mix_rate", 1.0)),
            backend=str(data.get("backend", "rvc-python")),
        )
        settings.validate()
        return settings


@dataclass(frozen=True)
class SessionSettings:
    """Non-audio RPC session settings shared by client and server."""

    protocol_version: int = 1
    client_name: str = "windows-gui"
    server_name: str = "wsl-inference"

    def validate(self) -> None:
        if self.protocol_version <= 0:
            raise ValueError("protocol_version must be positive")
        if not self.client_name:
            raise ValueError("client_name is required")
        if not self.server_name:
            raise ValueError("server_name is required")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionSettings":
        settings = cls(
            protocol_version=int(data.get("protocol_version", 1)),
            client_name=str(data.get("client_name", "windows-gui")),
            server_name=str(data.get("server_name", "wsl-inference")),
        )
        settings.validate()
        return settings