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
from pedalboard import Chorus, Distortion, Pedalboard, PitchShift
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
        self.robot_distortion_drive_db = float(getattr(self.gui_settings, "robot_distortion_drive_db", 0.0))
        self.robot_chorus_mix = float(getattr(self.gui_settings, "robot_chorus_mix", 0.0))

        # RVC設定
        self.rvc_enabled = False
        self.rvc_model_name = self.inference_runtime_settings.model_name
        self.rvc_pitch_shift = self.inference_runtime_settings.pitch_shift
        self.rvc_fast_mode = False
        self.strict_rvc_only = not bool(getattr(self.gui_settings, "allow_dry_fallback_on_rvc_fail", True))

        # RPC推論関連
        self.inference_client = None
        self._rvc_sequence = 0
        self.rvc_processing_timeout = max(0.05, float(self.gui_settings.rvc_processing_timeout_sec))
        self.last_rvc_processing_time = 0
        self._last_params_sync = 0.0
        self._fast_chunk_counter = 0
        self._fast_last_rpc_output: Optional[np.ndarray] = None
        self._last_rvc_success_output: Optional[np.ndarray] = None
        self._fast_log_last_ts = 0.0
        self._last_stream_status_log_ts = 0.0
        
        # ボトルネック検出用: 処理時間計測
        self._bottleneck_stats = {
            "total_ms": [],
            "input_gain_ms": [],
            "noise_reduce_ms": [],
            "formant_ms": [],
            "post_formant_suppress_ms": [],
            "rvc_ms": [],
            "pedalboard_ms": [],
            "output_gain_ms": [],
            "callback_status_count": 0,
            "last_report_ts": time.time(),
        }
        self._max_stats_samples = 100  # 直近100フレームを保持
        self._output_delay_buffer = np.zeros(0, dtype=np.float32)

        self._noise_gate_last_gain = 1.0
        self._noise_gate_hold_samples_remaining = 0  # ホールド残サンプル数

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
            self.board = Pedalboard([
                PitchShift(semitones=self.pitch_shift),
                Distortion(drive_db=self.robot_distortion_drive_db),
                Chorus(
                    rate_hz=1.6,
                    depth=0.95,
                    centre_delay_ms=14.0,
                    feedback=0.60,
                    mix=self.robot_chorus_mix,
                ),
            ])
    
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

    def set_robot_distortion_drive_db(self, drive_db: float):
        """ロボット風ディストーション量(dB)を設定する。"""
        self.robot_distortion_drive_db = float(min(60.0, max(0.0, drive_db)))
        self._update_board()

    def set_robot_chorus_mix(self, mix: float):
        """ロボット風コーラス混合比(0..1)を設定する。"""
        self.robot_chorus_mix = float(min(1.0, max(0.0, mix)))
        self._update_board()

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

    def set_strict_rvc_only(self, enabled: bool):
        """RVC失敗時に原音フォールバックを許可しないモードを設定する。"""
        self.strict_rvc_only = bool(enabled)

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
                held = self._get_held_rvc_output(len(indata)) if self.rvc_enabled else None
                if held is not None:
                    quick = held.reshape(-1, 1)
                elif self.strict_rvc_only and self.rvc_enabled:
                    quick = np.zeros_like(indata, dtype=np.float32)
                else:
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
                # 通常: 入力ゲイン→ノイズ除去→(RVC)→フォルマント→Pedalboard→出力ゲイン
                # RVC 有効時にフォルマント加工済み音声を先に食わせると、母音が誤認されやすい。
                # そのため RVC へはノイズ除去後の比較的素の音声を渡し、質感加工は後段で行う。
                t0 = time.perf_counter()
                signal_in = indata * self.input_gain
                self._record_timing("input_gain_ms", t0)
                
                # ノイズ除去を適用
                t1 = time.perf_counter()
                noise_reduced = self._apply_noise_reduction(signal_in)
                self._record_timing("noise_reduce_ms", t1)
                
                # RVC/エフェクト適用
                t3 = time.perf_counter()
                if self.rvc_enabled:
                    if self.rvc_model_name or self.rvc_fast_mode:
                        rvc_input = noise_reduced
                        try:
                            if self.rvc_fast_mode:
                                core_processed = self._apply_rvc_hybrid_fast_mode(rvc_input)
                            else:
                                if not self.rvc_model_name:
                                    raise RuntimeError("rvc model is not selected")
                                core_processed = self._apply_rvc_rpc(rvc_input)
                        except Exception as e:
                            self.logger.warning("RVC RPC failed: %s", e)
                            held = self._get_held_rvc_output(len(rvc_input))
                            if held is not None:
                                core_processed = held.reshape(-1, 1)
                            elif self.strict_rvc_only:
                                core_processed = np.zeros_like(noise_reduced, dtype=np.float32)
                            else:
                                core_processed = noise_reduced
                    else:
                        held = self._get_held_rvc_output(len(noise_reduced))
                        if held is not None:
                            core_processed = held.reshape(-1, 1)
                        elif self.strict_rvc_only:
                            core_processed = np.zeros_like(noise_reduced, dtype=np.float32)
                        else:
                            core_processed = noise_reduced
                else:
                    core_processed = noise_reduced
                self._record_timing("rvc_ms", t3)

                # フォルマント処理を適用
                t2 = time.perf_counter()
                if abs(self.formant_shift) > 0.5:
                    formant_processed = self._apply_formant(core_processed)
                else:
                    formant_processed = core_processed
                self._record_timing("formant_ms", t2)

                # フォルマント後のノイズサプレッション（軽い抑制）
                t2b = time.perf_counter()
                if abs(self.formant_shift) > 0.5:
                    post_formant_processed = self._apply_post_formant_noise_suppression(formant_processed)
                else:
                    post_formant_processed = formant_processed
                self._record_timing("post_formant_suppress_ms", t2b)

                # 通常のエフェクト適用
                processed_signal = self._apply_pedalboard_effects(post_formant_processed)

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
            f"SupprPost={avg_ms('post_formant_suppress_ms'):.2f}ms "
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
            "post_formant_suppress_ms": avg_ms("post_formant_suppress_ms"),
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
        self._last_rvc_success_output = np.asarray(converted, dtype=np.float32)
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

        held = self._get_held_rvc_output(len(local_fast))
        if held is not None:
            return held.reshape(-1, 1)

        if self.strict_rvc_only:
            return np.zeros((len(local_fast), 1), dtype=np.float32)

        return local_fast.reshape(-1, 1)

    def _get_held_rvc_output(self, target_len: int) -> Optional[np.ndarray]:
        """直近のRVC成功出力を target_len に整形して返す。無ければ None。"""
        if self._last_rvc_success_output is None:
            return None
        return self._fit_audio_length(self._last_rvc_success_output, target_len)

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
        """ノイズゲート相当の減衰を適用して静音帯のノイズを抑える。"""
        try:
            signal_1d = np.asarray(indata[:, 0], dtype=np.float32)

            # 閾値は dBFS の絶対レベルとして扱う（-80 ～ -20 dB）
            threshold = max(1e-6, 10 ** (self.noise_gate_threshold / 20.0))

            # 短時間エネルギー包絡を使ってゲート判定を安定化
            env_win = max(8, int(self.samplerate * 0.004))
            env_kernel = np.ones(env_win, dtype=np.float32) / float(env_win)
            rms_envelope = np.sqrt(np.convolve(signal_1d * signal_1d, env_kernel, mode="same") + 1e-12)

            # ピーク包絡（1ms窓最大値）: 破裂音・摩擦音の短い過渡成分を検出
            peak_win = max(4, int(self.samplerate * 0.001))
            abs_signal = np.abs(signal_1d)
            peak_kernel = np.ones(peak_win, dtype=np.float32) / float(peak_win)
            peak_envelope = np.convolve(abs_signal, peak_kernel, mode="same")

            # RMS と ピーク の大きい方でゲート判定（子音でもどちらかが閾値を超えればゲートを開く）
            envelope = np.maximum(rms_envelope, peak_envelope)

            # ヒステリシス付きソフトゲート（語尾保護寄り）
            # 閉じる閾値を下げ、完全に閉じ切らないことで語尾の母音が潰れにくくなる。
            floor_gain = 0.05
            close_th = threshold * 0.55
            open_th = threshold * 1.05
            target_gain = np.ones_like(signal_1d, dtype=np.float32)

            low_region = envelope <= close_th
            mid_region = (envelope > close_th) & (envelope < open_th)
            target_gain[low_region] = floor_gain
            if np.any(mid_region):
                t = (envelope[mid_region] - close_th) / (open_th - close_th + 1e-12)
                target_gain[mid_region] = floor_gain + (t * t) * (1.0 - floor_gain)

            # ごく弱い語尾も一気に落とさず、薄く残す。
            tail_region = (envelope > (close_th * 0.45)) & (envelope < close_th)
            if np.any(tail_region):
                t_tail = (envelope[tail_region] - (close_th * 0.45)) / (close_th * 0.55 + 1e-12)
                tail_floor = 0.12 + 0.18 * np.clip(t_tail, 0.0, 1.0)
                target_gain[tail_region] = np.maximum(target_gain[tail_region], tail_floor)

            # アタック/リリースで時間方向を平滑化（語尾の減衰を長めに保持）
            attack_samples = max(1.0, self.samplerate * 0.002)
            release_samples = max(1.0, self.samplerate * 0.090)
            # ホールド: ゲートが開いたら最低 180ms は閉じない（語尾と子音の過渡成分保護）
            hold_len = int(self.samplerate * 0.180)
            attack_alpha = float(np.exp(-1.0 / attack_samples))
            release_alpha = float(np.exp(-1.0 / release_samples))

            gain = np.empty_like(target_gain)
            g = float(self._noise_gate_last_gain)
            hold_remaining = int(self._noise_gate_hold_samples_remaining)
            for i, tg in enumerate(target_gain):
                # ゲートが開いたらホールドカウンタをリセット
                if tg >= 0.5:
                    hold_remaining = hold_len
                # ホールド中は target を十分高く保つ
                if hold_remaining > 0:
                    tg = max(float(tg), 0.85)
                    hold_remaining -= 1
                alpha = attack_alpha if tg > g else release_alpha
                g = alpha * g + (1.0 - alpha) * float(tg)
                gain[i] = g
            self._noise_gate_last_gain = float(g)
            self._noise_gate_hold_samples_remaining = hold_remaining

            gated = signal_1d * gain
            return gated.reshape(-1, 1)
        except Exception as e:
            # エラーが発生した場合は無処理で返す
            print(f"Noise reduction error: {e}")
            return indata
    
    def _apply_formant(self, indata):
        """フォルマント感を強めるための帯域シェルフ補正を適用する。"""
        try:
            from scipy.fft import fft, ifft

            signal_1d = np.asarray(indata[:, 0], dtype=np.float32)
            src_rms = float(np.sqrt(np.mean(signal_1d * signal_1d) + 1e-12))

            # FFT計算（周波数領域へ）
            X = fft(signal_1d)
            freqs = np.fft.fftfreq(len(signal_1d), 1 / self.samplerate)
            freqs_abs = np.abs(freqs)

            # -24..+12 を -1.0..+1.0 に正規化（+側の変化を強める）
            shift_norm = float(np.clip(self.formant_shift / 12.0, -1.0, 1.0))

            # 体感差を出すため、帯域ごとのゲイン幅を設定
            # 注意: presence band 開始を2kHzに上げて日本語母音の混濁を防ぐ
            # （えのF2=1700Hz, おのF2=800Hz が presence band に入らないようにする）
            low_db = -8.0 * shift_norm
            presence_db = 7.0 * shift_norm   # 10dB→7dBに抑えて母音誤認を軽減
            air_db = 6.0 * shift_norm

            low_gain = 10 ** (low_db / 20.0)
            presence_gain = 10 ** (presence_db / 20.0)
            air_gain = 10 ** (air_db / 20.0)

            mask = np.ones_like(freqs_abs)

            # Low shelf: < 350Hz full, 350-1200Hz transition
            # Low shelf: < 150Hz full, 150-650Hz transition
            # 遷移域を 650Hz で打ち切り、おのF2(800Hz)を減衰させない
            # （800Hz への影響が え(F2=1700Hz) との混濁の主因）
            mask = np.where(freqs_abs < 150.0, low_gain, mask)
            low_t = np.clip((freqs_abs - 150.0) / (650.0 - 150.0), 0.0, 1.0)
            low_interp = low_gain + (1.0 - low_gain) * low_t
            mask = np.where((freqs_abs >= 150.0) & (freqs_abs < 650.0), low_interp, mask)
            # 650Hz 以上は low shelf の影響なし（日本語母音F2をすべて保護）

            # Presence boost/cut: 2.0k-5.0kHz
            mask = np.where((freqs_abs >= 2000.0) & (freqs_abs <= 5000.0), mask * presence_gain, mask)

            # Air shelf: > 6kHz
            mask = np.where(freqs_abs >= 6000.0, mask * air_gain, mask)

            # マスクを適用して逆FFT
            X_modified = X * mask
            result = np.real(ifft(X_modified)).astype(np.float32, copy=False)

            # 出力レベルを揃えて比較しやすくする
            out_rms = float(np.sqrt(np.mean(result * result) + 1e-12))
            if out_rms > 1e-8:
                result = result * min(2.0, src_rms / out_rms)

            return result.reshape(-1, 1)
        except Exception as e:
            # エラーが発生した場合は無処理で返す
            print(f"Formant processing error: {e}")
            return indata
    
    def _apply_post_formant_noise_suppression(self, indata):
        """フォルマント後の軽いノイズサプレッション（スペクトラルサブトラクション）。
        
        フォルマント処理で周波数帯域を増幅する際に、背景ノイズも一緒に増幅されるため、
        その後に軽いノイズを抑制する。
        """
        try:
            from scipy.fft import fft, ifft

            signal_1d = np.asarray(indata[:, 0], dtype=np.float32)

            # FFT計算
            X = fft(signal_1d)
            freqs = np.fft.fftfreq(len(signal_1d), 1 / self.samplerate)
            freqs_abs = np.abs(freqs)

            # 周波数領域でノイズスペクトラムを推定
            # 低周波と高周波はノイズが多いと仮定し、軽く減衰させる
            X_mag = np.abs(X)
            X_phase = np.angle(X)

            # ノイズゲートしきい値（フォルマント処理前のノイズゲートから推定）
            noise_floor = max(1e-6, 10 ** ((self.noise_gate_threshold - 15) / 20.0))

            # 軽いスペクトラルサブトラクション: ノイズフロアの10%を差し引く
            suppression_factor = 0.1
            X_mag_suppressed = np.maximum(X_mag - (noise_floor * suppression_factor), X_mag * 0.95)

            # ノイズが多い帯域（0-300Hzと高周波）をさらに軽く抑制
            low_suppress = 0.98  # 低周波: 2%減衰
            high_suppress = 0.97  # 高周波: 3%減衰

            # 低周波（0-300Hz）
            low_mask = np.ones_like(freqs_abs)
            low_mask = np.where(freqs_abs <= 300.0, low_suppress, low_mask)

            # 高周波（9kHz以上）
            high_mask = np.ones_like(freqs_abs)
            high_mask = np.where(freqs_abs >= 9000.0, high_suppress, high_mask)

            # マスク適用
            X_mag_suppressed = X_mag_suppressed * low_mask * high_mask

            # 位相を保持して複素数に変換
            X_suppressed = X_mag_suppressed * np.exp(1j * X_phase)

            # 逆FFT
            result = np.real(ifft(X_suppressed)).astype(np.float32, copy=False)

            return result.reshape(-1, 1)
        except Exception as e:
            # エラーが発生した場合は無処理で返す
            print(f"Post-formant noise suppression error: {e}")
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
