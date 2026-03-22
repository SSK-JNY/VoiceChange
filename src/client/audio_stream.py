"""sounddevice ストリームと RingBuffer を組み合わせた音声 I/O ラッパー。

マイク入力を input_buf へ書き込み、output_buf から再生する。
InferenceClient との接続はフェーズ 4 以降で行う。
"""

import logging
from typing import Optional

import numpy as np
import sounddevice as sd

from .ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


class AudioStream:
    """sounddevice.Stream ラッパー。

    - マイク入力は input_buf へノンブロッキングに書き込む。
    - 再生時は output_buf からノンブロッキングに読み取る。
      output_buf が空の場合はゼロ埋めで出力する（グリッチを防ぐ）。

    フェーズ 3 時点では InferenceClient との連携は行わず、
    バッファの充放電のみを担当する。
    """

    def __init__(
        self,
        input_device: int,
        output_device: int,
        samplerate: int = 48000,
        blocksize: int = 960,
        input_buffer_seconds: float = 0.5,
        output_buffer_seconds: float = 0.5,
    ) -> None:
        """
        Args:
            input_device:  sounddevice のデバイスインデックス（入力側）。
            output_device: sounddevice のデバイスインデックス（出力側）。
            samplerate:    サンプリングレート（Hz）。
            blocksize:     1 コールバックあたりのフレーム数。
            input_buffer_seconds:  入力リングバッファ容量（秒）。
            output_buffer_seconds: 出力リングバッファ容量（秒）。
        """
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._input_device = input_device
        self._output_device = output_device
        in_cap = max(blocksize, int(samplerate * max(0.1, input_buffer_seconds)))
        out_cap = max(blocksize, int(samplerate * max(0.1, output_buffer_seconds)))
        self.input_buf: RingBuffer = RingBuffer(in_cap)
        self.output_buf: RingBuffer = RingBuffer(out_cap)

        self._stream: Optional[sd.Stream] = None
        self._running = False

    # ------------------------------------------------------------------
    # 起動 / 停止
    # ------------------------------------------------------------------

    def start(self) -> None:
        """ストリームを開始する。すでに起動済みの場合は何もしない。"""
        if self._running:
            return
        self.input_buf.clear()
        self.output_buf.clear()
        self._stream = sd.Stream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=1,
            dtype="float32",
            device=(self._input_device, self._output_device),
            callback=self._callback,
        )
        self._stream.start()
        self._running = True
        logger.info(
            "AudioStream started: in=%s out=%s sr=%s bs=%s",
            self._input_device,
            self._output_device,
            self.samplerate,
            self.blocksize,
        )

    def stop(self) -> None:
        """ストリームを停止する。"""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        logger.info("AudioStream stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # sounddevice コールバック
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.debug("AudioStream callback status: %s", status)

        # マイク入力をリングバッファへ書き込む（モノラル前提）
        self.input_buf.put(indata[:, 0])

        # 再生バッファから読み取って出力
        chunk = self.output_buf.get(frames)
        if chunk is not None:
            outdata[:, 0] = chunk
        else:
            outdata[:] = 0.0
