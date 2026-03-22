#!/usr/bin/env python3
"""Model層: オーディオ処理ロジック。

フェーズ4ではローカルRVC推論（RVCModel）を使わず、
WSL 推論サーバ（InferenceClient）経由の RPC 推論に責務を移す。
"""

from __future__ import annotations

from pathlib import Path
import logging
import os
import sys
import threading
import time
from typing import Optional

import librosa
import numpy as np
from pedalboard import Pedalboard, PitchShift
from scipy import signal
import sounddevice as sd

# 親ディレクトリの app モジュールをインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import config
from src.app.gui_local_settings import GuiLocalSettings
from src.app.inference_runtime_settings import InferenceRuntimeSettings


class AudioModel:
    """オーディオモデル - デバイス管理とエフェクト処理"""

    def __init__(self, gui_settings: Optional[GuiLocalSettings] = None, inference_settings: Optional[InferenceRuntimeSettings] = None):
        self.gui_settings = gui_settings or GuiLocalSettings()
        self.inference_runtime_settings = inference_settings or InferenceRuntimeSettings()

        # ロガー設定
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # コンソールハンドラーがない場合は追加
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.samplerate = self.gui_settings.samplerate
        self.blocksize = self.gui_settings.blocksize

        # パラメータ
        self.pitch_shift = self.gui_settings.initial_pitch_shift
        self.formant_shift = self.gui_settings.initial_formant_shift
        self.input_gain = self.gui_settings.initial_input_gain
        self.output_gain = self.gui_settings.initial_output_gain
        self.noise_gate_threshold = self.gui_settings.initial_noise_gate_threshold
        self.output_delay_ms = max(0.0, float(getattr(self.gui_settings, "output_delay_ms", 0.0)))

        # RVC設定
        self.rvc_enabled = False
        self.rvc_model_name = self.inference_runtime_settings.model_name
        self.rvc_pitch_shift = self.inference_runtime_settings.pitch_shift
        self.rvc_fast_mode = False

        # RPC推論関連
        self.inference_client = None
        self._rvc_sequence = 0
        self.rvc_processing_timeout = max(0.05, float(self.gui_settings.rvc_processing_timeout_sec))
        self.last_rvc_processing_time = 0
        self._last_params_sync = 0.0
        self._fast_chunk_counter = 0
        self._fast_last_rpc_output: Optional[np.ndarray] = None
        self._fast_log_last_ts = 0.0
        self._last_stream_status_log_ts = 0.0
        
        # ボトルネック検出用: 処理時間計測
        self._bottleneck_stats = {
            "total_ms": [],
            "input_gain_ms": [],
            "noise_reduce_ms": [],
            "formant_ms": [],
            "rvc_ms": [],
            "pedalboard_ms": [],
            "output_gain_ms": [],
            "callback_status_count": 0,
            "last_report_ts": time.time(),
        }
        self._max_stats_samples = 100  # 直近100フレームを保持
        self._output_delay_buffer = np.zeros(0, dtype=np.float32)

        # デバイス情報
        self.devices = sd.query_devices()
        self.input_devices = []
        self.output_devices = []
        self._load_devices()

        # Pedalboard
        self.board = None
        self.lock = threading.Lock()
        self._update_board()
    
    def _load_devices(self):
        """利用可能なデバイスをロード"""
        self.input_devices = [
            (i, d['name']) 
            for i, d in enumerate(self.devices) 
            if d.get('max_input_channels', 0) > 0
        ]
        self.output_devices = [
            (i, d['name']) 
            for i, d in enumerate(self.devices) 
            if d.get('max_output_channels', 0) > 0
        ]
    
    def _update_board(self):
        """Pedalboardを更新"""
        with self.lock:
            self.board = Pedalboard([PitchShift(semitones=self.pitch_shift)])
    
    def set_pitch_shift(self, semitones):
        """ピッチシフトを設定"""
        self.pitch_shift = int(semitones)
        self._update_board()
    
    def set_formant_shift(self, shift):
        """フォルマントシフトを設定（-24～+12）"""
        self.formant_shift = int(shift)
    
    def set_input_gain(self, gain):
        """入力ゲインを設定"""
        self.input_gain = float(gain)
    
    def set_output_gain(self, gain):
        """出力ゲインを設定"""
        self.output_gain = float(gain)
    
    def set_noise_gate_threshold(self, threshold_db):
        """ノイズゲート閾値を設定（dB, -80～-20）"""
        self.noise_gate_threshold = float(threshold_db)

    def set_output_delay_ms(self, delay_ms: float):
        """出力遅延(ms)を設定する。変更時は内部バッファをリセットする。"""
        self.output_delay_ms = max(0.0, float(delay_ms))
        self._output_delay_buffer = np.zeros(0, dtype=np.float32)

    def enable_rvc(self, enabled):
        """RVCを有効/無効化"""
        self.rvc_enabled = bool(enabled)

    def set_inference_client(self, client):
        """推論サーバクライアントを設定する。None で切断扱い。"""
        self.inference_client = client

    def set_rvc_model(self, model_path):
        """RVCモデル名を設定する。

        引数はパス/モデル名どちらでも受け付け、内部では stem 名で保持する。
        """
        if not model_path:
            self.rvc_model_name = ""
            self.inference_runtime_settings.model_name = ""
            return
        self.rvc_model_name = Path(str(model_path)).stem
        self.inference_runtime_settings.model_name = self.rvc_model_name

    def set_rvc_fast_mode(self, enabled):
        """RVC高速モードを設定"""
        self.rvc_fast_mode = bool(enabled)
        self._fast_chunk_counter = 0
        self._fast_last_rpc_output = None

    def set_rvc_pitch_shift(self, pitch_shift):
        """RVCピッチシフトを設定"""
        self.rvc_pitch_shift = int(pitch_shift)
        self.inference_runtime_settings.pitch_shift = self.rvc_pitch_shift

    def get_current_inference_settings(self):
        """現在の推論設定を protocol 用 InferenceSettings として返す。"""
        return self.inference_runtime_settings.to_protocol_settings(
            model_name=self.rvc_model_name
        )

    def _apply_rvc_fast_mode(self, audio, sr, pitch_shift):
        """RVC高速モード - ピッチシフト + 簡易エフェクト"""
        try:
            # ピッチシフト適用
            if abs(pitch_shift) > 0.1:
                pitch_shifted = librosa.effects.pitch_shift(
                    audio, sr=sr, n_steps=pitch_shift, bins_per_octave=12
                )
            else:
                pitch_shifted = audio
            
            # 簡易的な声質変換（ディストーション + フィルタ）
            from scipy import signal
            
            # 軽いディストーション（声質変化をシミュレート）
            distorted = np.tanh(pitch_shifted * 2.0) * 0.8
            
            # 周波数特性変更
            if pitch_shift > 0:
                # 高音化時は高域を少しブースト
                b, a = signal.butter(1, 3000/(sr/2), btype='high')
                voice_modified = signal.filtfilt(b, a, distorted)
                voice_modified = voice_modified * 1.2
            elif pitch_shift < 0:
                # 低音化時は低域を少しブースト
                b, a = signal.butter(1, 500/(sr/2), btype='low')
                voice_modified = signal.filtfilt(b, a, distorted)
                voice_modified = voice_modified * 1.3
            else:
                voice_modified = distorted
            
            return voice_modified
            
        except Exception as e:
            self.logger.error(f"Fast RVC mode failed: {e}")
            return self._simple_convert(audio, sr, pitch_shift)
    
    def _simple_convert(self, audio, sr, pitch_shift):
        """簡易音声変換（ピッチシフトのみ）"""
        if abs(pitch_shift) > 0.1:
            try:
                converted_audio = librosa.effects.pitch_shift(
                    audio, sr=sr, n_steps=pitch_shift, bins_per_octave=12
                )
                return converted_audio
            except Exception as e:
                self.logger.error(f"Simple conversion failed: {e}")
                return audio
        return audio

    def get_available_rvc_models(self):
        """利用可能なRVCモデル一覧を返す。

        優先順位:
        1) サーバ接続中: list_models RPC
        2) 未接続時: ローカル src/models/rvc/*.pth のファイル名
        """
        try:
            if self.inference_client is not None and self.inference_client.is_connected:
                return self.inference_client.list_models()
        except Exception as exc:
            self.logger.warning("list_models via server failed: %s", exc)

        models_dir = Path(__file__).resolve().parent / "rvc"
        if not models_dir.exists():
            return []
        return sorted([p.stem for p in models_dir.glob("*.pth")])

    def download_rvc_pretrained_models(self):
        """フェーズ4ではローカルダウンロード機能を無効化。"""
        raise RuntimeError(
            "フェーズ4では RVC 事前学習モデルのローカルダウンロードは非対応です。"
            "WSL 側にモデルを配置してサーバ経由で利用してください。"
        )
    
    def process_audio(self, indata, outdata, frames, time_info, status, mode='normal'):
        """オーディオ処理コールバック
        
        Args:
            mode: 'normal' (エフェクト), 'passthrough' (エフェクト無し), 'test-tone' (テスト音)
        """
        frame_start = time.perf_counter()
        
        if status:
            now = time.time()
            if now - self._last_stream_status_log_ts > 1.0:
                self.logger.warning("Audio callback status: %s", status)
                self._last_stream_status_log_ts = now

            # コールバック過負荷時は重い処理を避け、グリッチ連鎖を抑える。
            self._bottleneck_stats["callback_status_count"] += 1
            input_overflow = bool(getattr(status, "input_overflow", False))
            output_underflow = bool(getattr(status, "output_underflow", False))
            if input_overflow or output_underflow:
                quick = indata * self.input_gain
                if mode == 'normal':
                    quick = self._apply_pedalboard_effects(quick)
                delayed = self._apply_output_delay(quick * self.output_gain)
                outdata[:] = delayed
                return
        
        with self.lock:
            if mode == 'test-tone':
                # テスト音生成: config.AUDIO_MODE_TEST_TONE_FREQ Hz正弦波
                frame_idx = getattr(self, '_test_frame_idx', 0)
                t = np.arange(frames) / self.samplerate + frame_idx / self.samplerate
                tone = config.AUDIO_MODE_TEST_TONE_GAIN * np.sin(2 * np.pi * config.AUDIO_MODE_TEST_TONE_FREQ * t)
                delayed = self._apply_output_delay((tone.reshape(-1, 1)) * self.output_gain)
                outdata[:] = delayed
                self._test_frame_idx = frame_idx + frames
            
            elif mode == 'passthrough':
                # パススルー: エフェクト無し
                delayed = self._apply_output_delay(indata * self.input_gain * self.output_gain)
                outdata[:] = delayed
            
            else:  # normal
                # 通常: Pedalboard エフェクト適用（入力ゲイン→ノイズ除去→フォルマント→Pedalboard→出力ゲイン）
                t0 = time.perf_counter()
                signal_in = indata * self.input_gain
                self._record_timing("input_gain_ms", t0)
                
                # ノイズ除去を適用
                t1 = time.perf_counter()
                noise_reduced = self._apply_noise_reduction(signal_in)
                self._record_timing("noise_reduce_ms", t1)
                
                # フォルマント処理を適用
                t2 = time.perf_counter()
                if abs(self.formant_shift) > 0.5:
                    formant_processed = self._apply_formant(noise_reduced)
                else:
                    formant_processed = noise_reduced
                self._record_timing("formant_ms", t2)
                
                # RVC/エフェクト適用
                t3 = time.perf_counter()
                if self.rvc_enabled and (self.rvc_model_name or self.rvc_fast_mode):
                    rvc_input = formant_processed if abs(self.formant_shift) > 0.5 else noise_reduced
                    try:
                        if self.rvc_fast_mode:
                            processed_signal = self._apply_rvc_hybrid_fast_mode(rvc_input)
                        else:
                            if not self.rvc_model_name:
                                raise RuntimeError("rvc model is not selected")
                            processed_signal = self._apply_rvc_rpc(rvc_input)
                    except Exception as e:
                        self.logger.warning("RVC RPC failed, fallback to local effect: %s", e)
                        processed_signal = self._apply_pedalboard_effects(formant_processed)
                else:
                    # 通常のエフェクト適用
                    processed_signal = self._apply_pedalboard_effects(formant_processed)
                self._record_timing("rvc_ms", t3)

                # Pedalboard（コールバック内で即座に実行されるエフェクト）
                t4 = time.perf_counter()
                # Pedalboard は RVC/フォルマント処理に含められているため、ここでは計測のみ
                self._record_timing("pedalboard_ms", t4)
                
                # 出力ゲイン適用
                t5 = time.perf_counter()
                delayed = self._apply_output_delay(processed_signal * self.output_gain)
                outdata[:] = delayed
                self._record_timing("output_gain_ms", t5)
        
        # 総処理時間を計測
        frame_end = time.perf_counter()
        total_ms = (frame_end - frame_start) * 1000.0
        expected_ms = (frames / self.samplerate) * 1000.0
        
        self._bottleneck_stats["total_ms"].append(total_ms)
        if len(self._bottleneck_stats["total_ms"]) > self._max_stats_samples:
            self._bottleneck_stats["total_ms"].pop(0)
        
        # 定期的にボトルネック情報をログ出力
        now = time.time()
        if now - self._bottleneck_stats["last_report_ts"] > 3.0:
            self._report_bottleneck_stats(expected_ms)
            self._bottleneck_stats["last_report_ts"] = now
    
    def _apply_pedalboard_effects(self, signal_in):
        """Pedalboardエフェクトを適用"""
        if self.board is not None:
            effected = self.board(signal_in.T, self.samplerate)
            return effected.T
        else:
            return signal_in

    def _apply_output_delay(self, signal_in: np.ndarray) -> np.ndarray:
        """設定された遅延ぶんだけ出力を遅らせる。"""
        delay_samples = int(self.samplerate * (self.output_delay_ms / 1000.0))
        mono = np.asarray(signal_in[:, 0], dtype=np.float32)

        if delay_samples <= 0:
            return signal_in

        merged = np.concatenate([self._output_delay_buffer, mono]).astype(np.float32, copy=False)
        available = max(0, len(merged) - delay_samples)
        take = min(len(mono), available)

        if take > 0:
            out_mono = merged[:take]
        else:
            out_mono = np.zeros(0, dtype=np.float32)

        if take < len(mono):
            out_mono = np.pad(out_mono, (0, len(mono) - take))

        self._output_delay_buffer = merged[take:]
        return out_mono.reshape(-1, 1)

    def _record_timing(self, key: str, start_time: float):
        """処理時間を記録"""
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if key not in self._bottleneck_stats:
            self._bottleneck_stats[key] = []
        self._bottleneck_stats[key].append(elapsed_ms)
        if len(self._bottleneck_stats[key]) > self._max_stats_samples:
            self._bottleneck_stats[key].pop(0)

    def _report_bottleneck_stats(self, expected_frame_ms: float):
        """ボトルネック統計情報をログ出力"""
        stats = self._bottleneck_stats
        
        def avg_ms(key):
            vals = stats.get(key, [])
            return np.mean(vals) if vals else 0.0
        
        def max_ms(key):
            vals = stats.get(key, [])
            return np.max(vals) if vals else 0.0
        
        total_avg = avg_ms("total_ms")
        total_max = max_ms("total_ms")
        status_count = stats.get("callback_status_count", 0)
        
        # ボトルネック判定フラグ
        is_bottlenecked = total_avg > expected_frame_ms * 0.8
        warning_level = "BOTTLENECK" if is_bottlenecked else "OK"
        
        msg = (
            f"[AudioStats/{warning_level}] "
            f"Total avg={total_avg:.1f}ms max={total_max:.1f}ms (expect={expected_frame_ms:.1f}ms) | "
            f"Gain={avg_ms('input_gain_ms'):.2f}ms "
            f"NR={avg_ms('noise_reduce_ms'):.2f}ms "
            f"Formant={avg_ms('formant_ms'):.2f}ms "
            f"RVC={avg_ms('rvc_ms'):.2f}ms "
            f"Pedalboard={avg_ms('pedalboard_ms'):.2f}ms | "
            f"StatusErrors={status_count}"
        )
        print(msg)
        self.logger.info(msg)
        
        # リセット
        stats["callback_status_count"] = 0
    
    def get_bottleneck_info(self) -> dict:
        """ボトルネック情報を辞書で返す（GUI表示用）"""
        stats = self._bottleneck_stats
        
        def avg_ms(key):
            vals = stats.get(key, [])
            return np.mean(vals) if vals else 0.0
        
        def max_ms(key):
            vals = stats.get(key, [])
            return np.max(vals) if vals else 0.0
        
        expected_ms = (self.blocksize / self.samplerate) * 1000.0
        total_avg = avg_ms("total_ms")
        
        return {
            "expected_ms": expected_ms,
            "total_avg_ms": total_avg,
            "total_max_ms": max_ms("total_ms"),
            "input_gain_ms": avg_ms("input_gain_ms"),
            "noise_reduce_ms": avg_ms("noise_reduce_ms"),
            "formant_ms": avg_ms("formant_ms"),
            "rvc_ms": avg_ms("rvc_ms"),
            "pedalboard_ms": avg_ms("pedalboard_ms"),
            "output_gain_ms": avg_ms("output_gain_ms"),
            "callback_status_count": stats.get("callback_status_count", 0),
            "is_bottlenecked": total_avg > expected_ms * 0.8,
        }

    def _apply_rvc_rpc(self, signal_in):
        """WSL推論サーバへRPC推論を依頼し、float32モノラルを返す。"""
        if self.inference_client is None or not self.inference_client.is_connected:
            raise RuntimeError("inference client is not connected")
        if not self.rvc_model_name:
            raise RuntimeError("rvc model is not selected")

        audio = np.asarray(signal_in[:, 0], dtype=np.float32)
        self._sync_params_if_needed()

        payload = np.asarray(audio, dtype="<f4").tobytes()
        self._rvc_sequence += 1
        result = self.inference_client.infer_chunk(
            payload,
            sample_rate=self.samplerate,
            frame_count=len(audio),
            sequence=self._rvc_sequence,
            timeout=self.rvc_processing_timeout,
        )
        if result is None:
            raise RuntimeError("infer_chunk returned no data")

        converted = np.frombuffer(result, dtype="<f4").astype(np.float32, copy=False)
        if len(converted) < len(audio):
            converted = np.pad(converted, (0, len(audio) - len(converted)))
        elif len(converted) > len(audio):
            converted = converted[: len(audio)]
        return converted.reshape(-1, 1)

    def _apply_rvc_hybrid_fast_mode(self, signal_in):
        """高速モード: ローカル高速変換 + 間引きRPCのハイブリッド。"""
        audio = np.asarray(signal_in[:, 0], dtype=np.float32)
        local_fast = self._apply_rvc_fast_mode(audio, self.samplerate, self.rvc_pitch_shift)

        interval = max(1, int(getattr(self.gui_settings, "fast_mode_rpc_every_n_chunks", 3)))
        mix = float(getattr(self.gui_settings, "fast_mode_local_mix", 0.35))
        mix = min(1.0, max(0.0, mix))
        rpc_timeout = max(0.03, float(getattr(self.gui_settings, "fast_mode_rpc_timeout_sec", 0.12)))
        bootstrap_timeout = max(
            rpc_timeout,
            float(getattr(self.gui_settings, "fast_mode_rpc_bootstrap_timeout_sec", 0.35)),
        )

        self._fast_chunk_counter += 1
        should_try_rpc = (
            self._fast_last_rpc_output is None
            or (self._fast_chunk_counter % interval == 0)
        )

        if should_try_rpc and self.inference_client is not None and self.inference_client.is_connected:
            prev_timeout = self.rvc_processing_timeout
            try:
                rpc_out = None

                # 1st try: strict timeout for low latency
                self.rvc_processing_timeout = rpc_timeout
                try:
                    rpc_out = self._apply_rvc_rpc(signal_in)
                except Exception:
                    rpc_out = None

                # 2nd try (bootstrap): when cache is empty, allow a slightly longer timeout
                if rpc_out is None and self._fast_last_rpc_output is None and self.rvc_model_name:
                    self.rvc_processing_timeout = bootstrap_timeout
                    rpc_out = self._apply_rvc_rpc(signal_in)

                if rpc_out is not None:
                    self._fast_last_rpc_output = np.asarray(rpc_out[:, 0], dtype=np.float32)
            except Exception as exc:
                # 高速モードでは間欠失敗を許容し、過度な warning スパムを抑える
                now = time.time()
                if now - self._fast_log_last_ts > 2.0:
                    self.logger.info("Fast mode RPC skipped/fallback: %s", exc)
                    self._fast_log_last_ts = now
            finally:
                self.rvc_processing_timeout = prev_timeout

        if self._fast_last_rpc_output is not None:
            cached = self._fit_audio_length(self._fast_last_rpc_output, len(local_fast))
            blended = (mix * local_fast) + ((1.0 - mix) * cached)
            return blended.reshape(-1, 1)

        return local_fast.reshape(-1, 1)

    def _fit_audio_length(self, audio: np.ndarray, target_len: int) -> np.ndarray:
        """長さ不一致時にゼロ詰め/切り詰めで target_len へ整形する。"""
        if len(audio) == target_len:
            return audio
        if len(audio) < target_len:
            return np.pad(audio, (0, target_len - len(audio)))
        return audio[:target_len]

    def _sync_params_if_needed(self):
        """推論パラメータをサーバへ反映する（間引きあり）。"""
        if self.inference_client is None or not self.inference_client.is_connected:
            return
        now = time.time()
        min_sync_interval = 0.5 if self.rvc_fast_mode else 0.25
        if now - self._last_params_sync < min_sync_interval:
            return

        settings = self.get_current_inference_settings()
        ok = self.inference_client.update_params(settings)
        if ok:
            self._last_params_sync = now
        else:
            self.logger.warning("update_params failed")
    
    def _apply_noise_reduction(self, indata):
        """スペクトル領域でのノイズ除去（Spectral Subtraction）"""
        try:
            from scipy.fft import fft, ifft
            
            # 単一チャンネル処理
            signal_1d = indata[:, 0]
            
            # FFT計算（周波数領域へ）
            X = fft(signal_1d)
            magnitude = np.abs(X)
            phase = np.angle(X)
            
            # パワースペクトラムに変換
            power = magnitude ** 2
            
            # 閾値を線形値に変換 (dB → 線形)
            threshold_linear = 10 ** (self.noise_gate_threshold / 10)
            
            # パワーが閾値以下の周波数成分を減衰
            # Spectral Subtraction: ノイズスペクトラムを推定して差し引く
            noise_power = np.where(power < threshold_linear, power * 0.1, 0)  # ノイズ部分を推定
            subtracted_power = np.maximum(power - noise_power, power * 0.001)  # 下限を設定
            
            # 新しいマグニチュードを計算
            magnitude_new = np.sqrt(subtracted_power)
            
            # 位相は保持して復元
            X_denoised = magnitude_new * np.exp(1j * phase)
            
            # 逆FFT（時間領域へ）
            result = np.real(ifft(X_denoised))
            
            return result.reshape(-1, 1)
        except Exception as e:
            # エラーが発生した場合は無処理で返す
            print(f"Noise reduction error: {e}")
            return indata
    
    def _apply_formant(self, indata):
        """フォルマント処理を適用（周波数領域での周波数特性調整）"""
        try:
            from scipy.fft import fft, ifft
            
            # FFT計算（周波数領域へ）
            X = fft(indata[:, 0])
            freqs = np.fft.fftfreq(len(indata), 1 / self.samplerate)
            freqs_abs = np.abs(freqs)
            
            # formant_shift に基づいてゲイン調整（-24～+12）
            shift_factor = self.formant_shift / 24.0  # -1.0 to +0.5
            
            # 低周波（Bass）と高周波（Treble）のゲイン
            bass_gain = np.power(10, -shift_factor * 2.0 / 20)  # -24 で +2dB, +12 で -1dB
            treble_gain = np.power(10, shift_factor * 3.0 / 20)  # -24 で -3dB, +12 で +1.5dB
            
            # 周波数帯別にゲインを適用（マルチバンドEQ）
            # 低域: 0-300Hz, 中低域: 300-1000Hz, 中高域: 1000-4000Hz, 高域: 4000Hz以上
            mask = np.ones_like(freqs_abs)
            
            # 低域強調（formant_shift < 0 の時）
            mask = np.where(freqs_abs < 300, bass_gain, mask)
            
            # 中域は段階的に遷移
            mask = np.where((freqs_abs >= 300) & (freqs_abs < 1000),
                          bass_gain + (1.0 - bass_gain) * (freqs_abs - 300) / 700, mask)
            
            # 高域強調（formant_shift > 0 の時）
            mask = np.where(freqs_abs >= 4000, treble_gain, mask)
            
            # マスクを適用して逆FFT
            X_modified = X * mask
            result = np.real(ifft(X_modified))
            
            return result.reshape(-1, 1)
        except Exception as e:
            # エラーが発生した場合は無処理で返す
            print(f"Formant processing error: {e}")
            return indata
    
    def validate_device_pair(self, input_idx, output_idx):
        """デバイスペアの検証（ホストAPI確認）"""
        if input_idx is None or output_idx is None:
            return True, "OK"
        
        try:
            in_dev = self.devices[input_idx]
            out_dev = self.devices[output_idx]
            
            in_api = in_dev.get('hostapi')
            out_api = out_dev.get('hostapi')
            
            if in_api != out_api:
                apis = sd.query_hostapis()
                in_api_name = apis[in_api]['name']
                out_api_name = apis[out_api]['name']
                return False, f"HostAPI不一致: 入力={in_api_name}, 出力={out_api_name}"
            
            return True, "OK"
        except Exception as e:
            return False, str(e)
