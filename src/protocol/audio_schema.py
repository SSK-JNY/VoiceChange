"""Audio payload metadata shared between the Windows client and WSL server."""

from dataclasses import asdict, dataclass
from enum import Enum


class AudioDType(str, Enum):
    """Supported PCM payload element types."""

    FLOAT32 = "float32"
    INT16 = "int16"

    @property
    def bytes_per_sample(self) -> int:
        if self is AudioDType.FLOAT32:
            return 4
        return 2


@dataclass(frozen=True)
class AudioChunkSpec:
    """Metadata for one audio chunk transferred over RPC."""

    sample_rate: int
    channels: int
    dtype: AudioDType
    frame_count: int

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")

    @property
    def payload_nbytes(self) -> int:
        return self.frame_count * self.channels * self.dtype.bytes_per_sample

    def to_dict(self) -> dict:
        self.validate()
        data = asdict(self)
        data["dtype"] = self.dtype.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "AudioChunkSpec":
        spec = cls(
            sample_rate=int(data["sample_rate"]),
            channels=int(data["channels"]),
            dtype=AudioDType(data["dtype"]),
            frame_count=int(data["frame_count"]),
        )
        spec.validate()
        return spec