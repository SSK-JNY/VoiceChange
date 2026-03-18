#!/usr/bin/env python3
"""
Model層: オーディオ処理ロジック
"""
import sounddevice as sd
import numpy as np
from pedalboard import Pedalboard, PitchShift
import threading
from scipy import signal
import sys
import os
import time
import logging
import librosa

# 親ディレクトリの app モジュールをインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
import config

# RVCモデルをインポート
try:
    from .rvc_model import RVCModel
    RVC_AVAILABLE = True
except ImportError as e:
    RVC_AVAILABLE = False
    print(f"Warning: RVC model not available. Using simplified implementation. Error: {e}")


class AudioModel:
    """オーディオモデル - デバイス管理とエフェクト処理"""

    def __init__(self):
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # コンソールハンドラーがない場合は追加
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.samplerate = config.SAMPLERATE
        self.blocksize = config.BLOCKSIZE

        # パラメータ
        self.pitch_shift = config.INITIAL_PITCH_SHIFT
        self.formant_shift = config.INITIAL_FORMANT_SHIFT
        self.input_gain = config.INITIAL_INPUT_GAIN
        self.output_gain = config.INITIAL_OUTPUT_GAIN
        self.noise_gate_threshold = config.INITIAL_NOISE_GATE_THRESHOLD

        # RVC設定
        self.rvc_enabled = False
        self.rvc_model_path = None
        self.rvc_pitch_shift = 0
        self.rvc_fast_mode = False  # デフォルトで高速モード無効

        # 処理最適化設定
        self.rvc_processing_timeout = 0.02  # 20ms以内に処理完了させる（1024サンプル対応）
        self.last_rvc_processing_time = 0

        # RVCモデル初期化
        if RVC_AVAILABLE:
            self.rvc_model = RVCModel()
        else:
            self.rvc_model = None

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

    def enable_rvc(self, enabled):
        """RVCを有効/無効化"""
        self.rvc_enabled = bool(enabled)

    def set_rvc_model(self, model_path):
        """RVCモデルを設定"""
        if self.rvc_model and model_path:
            self.rvc_model_path = model_path
            self.rvc_model.load_rvc_model(model_path)

    def set_rvc_fast_mode(self, enabled):
        """RVC高速モードを設定"""
        self.rvc_fast_mode = enabled

    def set_rvc_pitch_shift(self, pitch_shift):
        """RVCピッチシフトを設定"""
        self.rvc_pitch_shift = int(pitch_shift)

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
        """利用可能なRVCモデルの一覧を取得"""
        if self.rvc_model:
            return self.rvc_model.get_available_models()
        return []

    def download_rvc_pretrained_models(self):
        """RVC事前学習モデルをダウンロード"""
        if self.rvc_model:
            self.rvc_model.download_pretrained_models()
    
    def process_audio(self, indata, outdata, frames, time_info, status, mode='normal'):
        """オーディオ処理コールバック
        
        Args:
            mode: 'normal' (エフェクト), 'passthrough' (エフェクト無し), 'test-tone' (テスト音)
        """
        if status:
            print('Audio callback status:', status)
        
        with self.lock:
            if mode == 'test-tone':
                # テスト音生成: config.AUDIO_MODE_TEST_TONE_FREQ Hz正弦波
                frame_idx = getattr(self, '_test_frame_idx', 0)
                t = np.arange(frames) / self.samplerate + frame_idx / self.samplerate
                tone = config.AUDIO_MODE_TEST_TONE_GAIN * np.sin(2 * np.pi * config.AUDIO_MODE_TEST_TONE_FREQ * t)
                outdata[:, 0] = tone * self.output_gain
                self._test_frame_idx = frame_idx + frames
            
            elif mode == 'passthrough':
                # パススルー: エフェクト無し
                outdata[:] = indata * self.input_gain * self.output_gain
            
            else:  # normal
                # 通常: Pedalboard エフェクト適用（入力ゲイン→ノイズ除去→フォルマント→Pedalboard→出力ゲイン）
                signal_in = indata * self.input_gain
                
                # ノイズ除去を適用
                noise_reduced = self._apply_noise_reduction(signal_in)
                
                # フォルマント処理を適用
                if abs(self.formant_shift) > 0.5:
                    formant_processed = self._apply_formant(noise_reduced)
                else:
                    formant_processed = noise_reduced
                
                # エフェクト適用
                if self.rvc_enabled and self.rvc_model and self.rvc_model_path:
                    # RVC有効時はRVC変換を使用
                    try:
                        # 処理時間が長すぎる場合や高速モードの場合は簡易変換
                        if self.rvc_fast_mode or self.last_rvc_processing_time > self.rvc_processing_timeout:
                            # 高速モードまたはタイムアウト時は簡易RVC変換
                            rvc_input = formant_processed if abs(self.formant_shift) > 0.5 else noise_reduced
                            processed_signal = self._apply_rvc_fast_mode(
                                rvc_input[:, 0], self.samplerate, self.rvc_pitch_shift
                            ).reshape(-1, 1)
                        else:
                            # 通常のRVC変換
                            rvc_input = formant_processed if abs(self.formant_shift) > 0.5 else noise_reduced
                            start_time = time.time()
                            rvc_converted = self.rvc_model.convert_voice(
                                rvc_input[:, 0],
                                self.samplerate,
                                self.rvc_model_path,
                                self.rvc_pitch_shift
                            )
                            self.last_rvc_processing_time = time.time() - start_time
                            processed_signal = rvc_converted.reshape(-1, 1)
                    except Exception as e:
                        print(f"RVC conversion failed: {e}")
                        # RVC失敗時はPedalboardエフェクトを使用
                        processed_signal = self._apply_pedalboard_effects(formant_processed)
                else:
                    # 通常のエフェクト適用
                    processed_signal = self._apply_pedalboard_effects(formant_processed)

                outdata[:] = processed_signal * self.output_gain
    
    def _apply_pedalboard_effects(self, signal_in):
        """Pedalboardエフェクトを適用"""
        if self.board is not None:
            effected = self.board(signal_in.T, self.samplerate)
            return effected.T
        else:
            return signal_in
    
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
