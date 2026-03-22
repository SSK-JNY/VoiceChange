"""スレッドセーフなリングバッファ（Float32 モノラル PCM 用）。"""

import threading
from typing import Optional

import numpy as np


class RingBuffer:
    """固定容量のスレッドセーフ環状バッファ。

    マイク入力チャンクの蓄積と、再生チャンクの取り出しに使う。
    書き込みが容量を超えた場合は古いデータを上書きする（オーバーフロー）。
    """

    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity: バッファが保持できる最大サンプル数。
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._write = 0   # 次に書き込む位置
        self._count = 0   # 現在の有効サンプル数
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def available(self) -> int:
        """現在読み取り可能なサンプル数。"""
        with self._lock:
            return self._count

    # ------------------------------------------------------------------
    # 書き込み
    # ------------------------------------------------------------------

    def put(self, data: np.ndarray) -> None:
        """data を末尾に書き込む。

        容量を超える場合は最新 capacity サンプルのみ保持する。
        """
        data = np.asarray(data, dtype=np.float32).ravel()
        n = len(data)
        if n == 0:
            return

        with self._lock:
            if n >= self._capacity:
                # 容量以上のデータは末尾だけ保持
                self._buf[:] = data[-self._capacity:]
                self._write = 0
                self._count = self._capacity
                return

            space_to_end = self._capacity - self._write
            if space_to_end >= n:
                self._buf[self._write : self._write + n] = data
            else:
                self._buf[self._write :] = data[:space_to_end]
                self._buf[: n - space_to_end] = data[space_to_end:]
            self._write = (self._write + n) % self._capacity
            self._count = min(self._count + n, self._capacity)

    # ------------------------------------------------------------------
    # 読み取り
    # ------------------------------------------------------------------

    def get(self, n: int) -> Optional[np.ndarray]:
        """先頭から n サンプル読み取って返す。

        有効サンプル数が n 未満の場合は None を返す（ブロックしない）。
        """
        with self._lock:
            if self._count < n:
                return None
            read_pos = (self._write - self._count) % self._capacity
            if read_pos + n <= self._capacity:
                out = self._buf[read_pos : read_pos + n].copy()
            else:
                wrap = self._capacity - read_pos
                out = np.concatenate([self._buf[read_pos:], self._buf[: n - wrap]])
            self._count -= n
            return out

    def get_all(self) -> np.ndarray:
        """有効サンプルをすべて読み取って返す。空の場合は長さ 0 の配列を返す。"""
        with self._lock:
            n = self._count
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            read_pos = (self._write - n) % self._capacity
            if read_pos + n <= self._capacity:
                out = self._buf[read_pos : read_pos + n].copy()
            else:
                wrap = self._capacity - read_pos
                out = np.concatenate([self._buf[read_pos:], self._buf[: n - wrap]])
            self._count = 0
            return out

    def clear(self) -> None:
        """バッファをリセットする。"""
        with self._lock:
            self._count = 0
            self._write = 0
