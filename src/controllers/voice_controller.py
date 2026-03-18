#!/usr/bin/env python3
"""MVC連携とイベント処理"""


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
        
        # UI イベントをバインド
        self._bind_events()

        # RVCモデル一覧を初期化
        self._initialize_rvc_models()
    
    def _initialize_rvc_models(self):
        """RVCモデル一覧を初期化"""
        try:
            models = self.model.get_available_rvc_models()
            self.view.update_rvc_models(models)
        except Exception as e:
            print(f"RVC model initialization failed: {e}")
    
    def _bind_events(self):
        """UI イベントをバインド"""
        self.view.start_button.config(command=self.start_stream)
        self.view.stop_button.config(command=self.stop_stream)
        self.view.pitch_var.trace('w', self._on_pitch_change)
        self.view.input_gain_var.trace('w', self._on_input_gain_change)
        self.view.output_gain_var.trace('w', self._on_output_gain_change)
        self.view.formant_var.trace('w', self._on_formant_change)
        self.view.noise_gate_var.trace('w', self._on_noise_gate_change)

        # RVC関連イベント
        self.view.rvc_enabled_var.trace('w', self._on_rvc_enabled_change)
        self.view.rvc_model_var.trace('w', self._on_rvc_model_change)
        self.view.rvc_pitch_var.trace('w', self._on_rvc_pitch_change)
        self.view.rvc_download_button.config(command=self._on_rvc_download)
    
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
        """RVCモデル変更時"""
        model_name = self.view.rvc_model_var.get()
        if model_name:
            # モデルパスを構築（models/rvc/ 配下）
            import os
            from pathlib import Path
            model_path = Path(__file__).parent.parent / 'models' / 'rvc' / f"{model_name}.pth"
            self.model.set_rvc_model(str(model_path))

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

    def _on_rvc_download(self):
        """RVCモデルダウンロード"""
        try:
            self.view.rvc_download_button.config(state="disabled", text="ダウンロード中...")
            self.model.download_rvc_pretrained_models()
            # モデル一覧を更新
            models = self.model.get_available_rvc_models()
            self.view.update_rvc_models(models)
            messagebox.showinfo("完了", "RVC事前学習モデルのダウンロードが完了しました")
        except Exception as e:
            messagebox.showerror("エラー", f"ダウンロードに失敗しました: {e}")
        finally:
            self.view.rvc_download_button.config(state="normal", text="モデルダウンロード")
    
    def start_stream(self):
        """ストリーム開始"""
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
            self.view.enable_start_button()
            self.view.disable_stop_button()
            self.view.set_status("停止中", "red")
    
    def stop_stream(self):
        """ストリーム停止"""
        self.is_running = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
    
    def set_mode(self, mode):
        """処理モード設定（normal, passthrough, test-tone）"""
        self.current_mode = mode
    
    def on_window_closing(self):
        """ウィンドウ閉じる時の処理"""
        self.stop_stream()
