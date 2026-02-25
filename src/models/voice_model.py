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

# 親ディレクトリの app モジュールをインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
import config


class AudioModel:
    """オーディオモデル - デバイス管理とエフェクト処理"""
    
    def __init__(self):
        self.samplerate = config.SAMPLERATE
        self.blocksize = config.BLOCKSIZE
        
        # パラメータ
        self.pitch_shift = config.INITIAL_PITCH_SHIFT
        self.formant_shift = config.INITIAL_FORMANT_SHIFT
        self.input_gain = config.INITIAL_INPUT_GAIN
        self.output_gain = config.INITIAL_OUTPUT_GAIN
        self.noise_gate_threshold = config.INITIAL_NOISE_GATE_THRESHOLD
        
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
                
                # Pedalboard エフェクト（PitchShift）を適用
                if self.board is not None:
                    effected = self.board(formant_processed.T, self.samplerate)
                    outdata[:] = effected.T * self.output_gain
                else:
                    outdata[:] = formant_processed * self.output_gain
    
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
