"""GUI ローカル設定（Windows 側のみで完結する設定）。"""

from dataclasses import asdict, dataclass


@dataclass
class GuiLocalSettings:
    """GUI・ローカル音声処理向け設定。"""

    window_width: int = 900
    window_height: int = 1200
    window_resizable: bool = True
    window_title: str = "リアルタイムボイスチェンジャー"

    # ローカル音声設定
    samplerate: int = 44100
    blocksize: int = 1024
    initial_pitch_shift: int = 3
    initial_formant_shift: int = 0
    initial_input_gain: float = 1.0
    initial_output_gain: float = 1.0
    initial_noise_gate_threshold: int = -40
    rvc_processing_timeout_sec: float = 0.18
    stream_input_buffer_seconds: float = 0.5
    stream_output_buffer_seconds: float = 0.5
    output_delay_ms: float = 0.0
    # 高速モードの挙動調整
    fast_mode_rpc_every_n_chunks: int = 3
    fast_mode_rpc_timeout_sec: float = 0.12
    fast_mode_rpc_bootstrap_timeout_sec: float = 0.35
    fast_mode_local_mix: float = 0.35

    # GUIローカル状態
    default_input_device: str = ""
    default_output_device: str = ""
    server_url: str = "ws://127.0.0.1:8765/ws"
    server_connect_retry_count: int = 3
    server_connect_retry_interval_sec: float = 1.5
    server_connect_show_error_dialog: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "GuiLocalSettings":
        return cls(
            window_width=int(data.get("window_width", 900)),
            window_height=int(data.get("window_height", 1200)),
            window_resizable=bool(data.get("window_resizable", True)),
            window_title=str(data.get("window_title", "リアルタイムボイスチェンジャー")),
            samplerate=int(data.get("samplerate", 44100)),
            blocksize=int(data.get("blocksize", 1024)),
            initial_pitch_shift=int(data.get("initial_pitch_shift", 3)),
            initial_formant_shift=int(data.get("initial_formant_shift", 0)),
            initial_input_gain=float(data.get("initial_input_gain", 1.0)),
            initial_output_gain=float(data.get("initial_output_gain", 1.0)),
            initial_noise_gate_threshold=int(data.get("initial_noise_gate_threshold", -40)),
            rvc_processing_timeout_sec=float(data.get("rvc_processing_timeout_sec", 0.18)),
            stream_input_buffer_seconds=float(data.get("stream_input_buffer_seconds", 0.5)),
            stream_output_buffer_seconds=float(data.get("stream_output_buffer_seconds", 0.5)),
            output_delay_ms=float(data.get("output_delay_ms", 0.0)),
            fast_mode_rpc_every_n_chunks=int(data.get("fast_mode_rpc_every_n_chunks", 3)),
            fast_mode_rpc_timeout_sec=float(data.get("fast_mode_rpc_timeout_sec", 0.12)),
            fast_mode_rpc_bootstrap_timeout_sec=float(data.get("fast_mode_rpc_bootstrap_timeout_sec", 0.35)),
            fast_mode_local_mix=float(data.get("fast_mode_local_mix", 0.35)),
            default_input_device=str(data.get("default_input_device", "")),
            default_output_device=str(data.get("default_output_device", "")),
            server_url=str(data.get("server_url", "ws://127.0.0.1:8765/ws")),
            server_connect_retry_count=int(data.get("server_connect_retry_count", 3)),
            server_connect_retry_interval_sec=float(data.get("server_connect_retry_interval_sec", 1.5)),
            server_connect_show_error_dialog=bool(data.get("server_connect_show_error_dialog", True)),
        )

    def to_dict(self) -> dict:
        return asdict(self)
