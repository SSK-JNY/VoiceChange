"""推論設定（WSL 側へ送る設定）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from src.protocol import InferenceSettings


@dataclass
class InferenceRuntimeSettings:
    """WSL 推論サーバへ送る設定。"""

    model_name: str = ""
    pitch_shift: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.75
    protect: float = 0.33
    filter_radius: int = 3
    rms_mix_rate: float = 1.0
    backend: str = "rvc-python"

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceRuntimeSettings":
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
        # 既存のバリデーションを再利用
        settings.to_protocol_settings()
        return settings

    def to_protocol_settings(self, model_name: Optional[str] = None) -> InferenceSettings:
        return InferenceSettings(
            model_name=(model_name if model_name is not None else self.model_name),
            pitch_shift=self.pitch_shift,
            f0_method=self.f0_method,
            index_rate=self.index_rate,
            protect=self.protect,
            filter_radius=self.filter_radius,
            rms_mix_rate=self.rms_mix_rate,
            backend=self.backend,
        )

    def to_dict(self) -> dict:
        return asdict(self)
