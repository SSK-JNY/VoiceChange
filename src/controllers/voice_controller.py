#!/usr/bin/env python3
"""MVC連携とイベント処理"""
import logging
import threading
import time
from typing import Optional

import sounddevice as sd
from tkinter import messagebox

logger = logging.getLogger(__name__)


class AudioController:
    """オーディオコントローラー - Model と View の連携"""
    
    def __init__(self, model, view):
        self.model = model
        self.view = view
        
        # ストリーム制御
        self.stream = None
        self.stream_thread = None
        self.is_running = False
        self.current_mode = 'normal'  # normal, passthrough, test-tone
        self._passthrough_mode_requested = False

        # WSL 推論サーバクライアント（接続後に設定される）
        self.inference_client = None

        # UI イベントをバインド
        self._bind_events()

        # RVCモデル一覧を初期化
        self._initialize_rvc_models()
    
    def _initialize_rvc_models(self):
        """RVCモデル一覧を初期化。"""
        try:
            models = self.model.get_available_rvc_models()
            self.view.update_rvc_models(models)
        except Exception as e:
            logger.warning("RVC model initialization failed: %s", e)
    
    def _bind_events(self):
        """UI イベントをバインド"""
        self.view.start_button.config(command=self.start_stream)
        self.view.stop_button.config(command=self.stop_stream)
        self.view.passthrough_button.config(command=self.passthrough_stream)
        self.view.pitch_var.trace('w', self._on_pitch_change)
        self.view.input_gain_var.trace('w', self._on_input_gain_change)
        self.view.output_gain_var.trace('w', self._on_output_gain_change)
        self.view.formant_var.trace('w', self._on_formant_change)
        self.view.noise_gate_var.trace('w', self._on_noise_gate_change)

        # RVC関連イベント
        self.view.rvc_enabled_var.trace('w', self._on_rvc_enabled_change)
        self.view.rvc_model_var.trace('w', self._on_rvc_model_change)
        self.view.rvc_pitch_var.trace('w', self._on_rvc_pitch_change)

        # 推論サーバ接続ボタン
        self.view.connect_button.config(command=self.connect_to_server)
    
    def _on_pitch_change(self, *args):
        """ピッチシフト変更時"""
        value = self.view.pitch_var.get()
        self.model.set_pitch_shift(value)
        self.view.update_pitch_label(value)
    
    def _on_input_gain_change(self, *args):
        """入力ゲイン変更時"""
        value = self.view.input_gain_var.get()
        self.model.set_input_gain(value)
        self.view.update_input_gain_label(value)
    
    def _on_output_gain_change(self, *args):
        """出力ゲイン変更時"""
        value = self.view.output_gain_var.get()
        self.model.set_output_gain(value)
        self.view.update_output_gain_label(value)
    
    def _on_formant_change(self, *args):
        """フォルマントシフト変更時"""
        value = self.view.formant_var.get()
        self.model.set_formant_shift(value)
        self.view.update_formant_label(value)
    
    def _on_noise_gate_change(self, *args):
        """ノイズゲート変更時"""
        value = self.view.noise_gate_var.get()
        self.model.set_noise_gate_threshold(value)
        self.view.update_noise_gate_label(value)

    def _on_rvc_enabled_change(self, *args):
        """RVC有効/無効変更時"""
        enabled = self.view.rvc_enabled_var.get()
        print(f"RVC有効化変更: {enabled}")
        self.model.enable_rvc(enabled)
        print(f"モデルRVC有効状態: {self.model.rvc_enabled}")

    def _on_rvc_model_change(self, *args):
        """RVCモデル変更時。サーバ接続中はサーバ側でロードを試みる。"""
        model_name = self.view.rvc_model_var.get()
        if not model_name:
            return

        # AudioModel 側には常に選択モデル名を保持しておく
        self.model.set_rvc_model(model_name)

        if self.inference_client is not None and self.inference_client.is_connected:
            # サーバ経由でロード（バックグラウンドスレッドで実行）
            self._load_model_via_server(model_name)

    def _on_rvc_pitch_change(self, *args):
        """RVCピッチシフト変更時"""
        value = self.view.rvc_pitch_var.get()
        self.model.set_rvc_pitch_shift(value)
        self.view.update_rvc_pitch_label(value)

    def _on_rvc_fast_mode_change(self, *args):
        """RVC高速モード変更時"""
        enabled = self.view.rvc_fast_mode_var.get()
        self.model.set_rvc_fast_mode(enabled)

    def on_rvc_fast_mode_change(self):
        """RVC高速モード変更時 (Checkbutton用)"""
        enabled = self.view.rvc_fast_mode_var.get()
        self.model.set_rvc_fast_mode(enabled)
    
    def start_stream(self):
        """ストリーム開始"""
        # passthroughモード以外は normalに統一
        if self.current_mode != 'passthrough':
            self.current_mode = 'normal'
        self._passthrough_mode_requested = False
        
        input_idx, output_idx = self.view.get_selected_devices()
        
        if input_idx is None or output_idx is None:
            messagebox.showerror("エラー", "入力・出力デバイスを選択してください")
            return
        
        # デバイスペア検証
        valid, msg = self.model.validate_device_pair(input_idx, output_idx)
        if not valid:
            messagebox.showerror("デバイスエラー", msg)
            return
        
        self.is_running = True
        self.view.disable_start_button()
        self.view.disable_passthrough_button()
        self.view.enable_stop_button()
        self.view.set_status("実行中", "green")
        
        self.stream_thread = threading.Thread(
            target=self._stream_worker,
            args=(input_idx, output_idx),
            daemon=True
        )
        self.stream_thread.start()
    
    def _stream_worker(self, input_idx, output_idx):
        """ストリーム処理スレッド"""
        try:
            device_pair = (input_idx, output_idx)
            
            # コールバック関数を定義（Model とのバインド）
            def audio_callback(indata, outdata, frames, time_info, status):
                self.model.process_audio(indata, outdata, frames, time_info, status, self.current_mode)
            
            with sd.Stream(
                samplerate=self.model.samplerate,
                blocksize=self.model.blocksize,
                channels=1,
                callback=audio_callback,
                device=device_pair
            ):
                while self.is_running:
                    time.sleep(0.1)
        
        except Exception as e:
            self.view.set_status(f"エラー: {str(e)}", "red")
        
        finally:
            self.is_running = False
            self.current_mode = 'normal'
            self.view.enable_start_button()
            self.view.enable_passthrough_button()
            self.view.disable_stop_button()
            self.view.set_status("停止中", "red")
    
    def stop_stream(self):
        """ストリーム停止"""
        self.is_running = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
    
    def passthrough_stream(self):
        """パススルーモードでストリーム開始"""
        self._passthrough_mode_requested = True
        self.current_mode = 'passthrough'
        self.start_stream()
    
    def set_mode(self, mode):
        """処理モード設定（normal, passthrough, test-tone）"""
        self.current_mode = mode
    
    def connect_to_server(self) -> None:
        """推論サーバへの接続を開始する（バックグラウンドスレッドで実行）。"""
        from src.client import InferenceClient

        url = self.view.server_url_var.get().strip()
        if not url:
            url = "ws://127.0.0.1:8765/ws"

        # 既存の接続を切断してから再接続
        if self.inference_client is not None and self.inference_client.is_connected:
            self.inference_client.disconnect()
            self.inference_client = None
            self.model.set_inference_client(None)
            self.view.update_server_status("未接続", "gray")
            self.view.update_server_detail("手動で切断しました", "gray")
            self.view.connect_button.config(text="接続")
            return

        retry_count = max(1, int(self.model.gui_settings.server_connect_retry_count))
        retry_interval = max(0.1, float(self.model.gui_settings.server_connect_retry_interval_sec))

        self.view.update_server_status("接続中...", "orange")
        self.view.update_server_detail(f"最大 {retry_count} 回リトライします", "orange")
        self.view.connect_button.config(state="disabled")

        def _work() -> None:
            last_error = "接続できませんでした"
            for attempt in range(1, retry_count + 1):
                client = InferenceClient(url)
                connected = client.connect(timeout=8.0)

                if connected:
                    models = client.list_models()
                    self.inference_client = client
                    self.model.set_inference_client(client)

                    def _update_success() -> None:
                        self.view.update_server_status("接続済み", "green")
                        self.view.update_server_detail(f"接続成功（試行 {attempt}/{retry_count}）", "green")
                        self.view.connect_button.config(state="normal", text="切断")
                        self.view.update_rvc_models(models)

                    self.view.root.after(0, _update_success)
                    return

                client.disconnect()
                last_error = f"試行 {attempt}/{retry_count} で接続失敗"
                if attempt < retry_count:
                    wait_left = retry_count - attempt

                    def _update_retry() -> None:
                        self.view.update_server_status("再接続中...", "orange")
                        self.view.update_server_detail(
                            f"{last_error}。{retry_interval:.1f}秒後に再試行（残り {wait_left} 回）",
                            "orange",
                        )

                    self.view.root.after(0, _update_retry)
                    time.sleep(retry_interval)

            self.inference_client = None
            self.model.set_inference_client(None)

            def _update_fail() -> None:
                self.view.update_server_status("接続失敗", "red")
                self.view.update_server_detail(
                    f"{last_error}。URLとWSLサーバ状態を確認して再試行してください。",
                    "red",
                )
                self.view.connect_button.config(state="normal", text="再試行")
                if self.model.gui_settings.server_connect_show_error_dialog:
                    messagebox.showwarning(
                        "推論サーバ接続失敗",
                        "推論サーバに接続できませんでした。\n"
                        "URL設定・WSLサーバ起動状態・ポート番号を確認してください。",
                    )

            self.view.root.after(0, _update_fail)

        threading.Thread(target=_work, daemon=True).start()

    def _load_model_via_server(self, model_name: str) -> None:
        """サーバ経由でモデルをロードする（バックグラウンドスレッドで実行）。"""
        self.view.set_status(f"モデルロード中: {model_name}", "orange")

        def _work() -> None:
            settings = self.model.get_current_inference_settings()
            ok = self.inference_client.load_model(model_name, settings)

            def _update() -> None:
                if ok:
                    self.model.set_rvc_model(model_name)
                    self.view.set_status(f"モデル準備完了: {model_name}", "green")
                else:
                    self.view.set_status("モデルロード失敗", "red")

            self.view.root.after(0, _update)

        threading.Thread(target=_work, daemon=True).start()

    def on_window_closing(self):
        """ウィンドウ閉じる時の処理"""
        self.stop_stream()
        if self.inference_client is not None:
            self.inference_client.disconnect()
