#!/usr/bin/env python3
"""
View層: Tkinter UIビュー
"""
import tkinter as tk
from tkinter import ttk
import sys
import os

# 親ディレクトリの app モジュールをインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
import config


class AudioView:
    """オーディオGUI - ビュー層"""
    
    def __init__(self, root, input_devices, output_devices):
        self.root = root
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.resizable(config.WINDOW_RESIZABLE, config.WINDOW_RESIZABLE)
        
        # UI変数
        self.status_var = tk.StringVar(value="停止中")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.pitch_var = tk.IntVar(value=config.INITIAL_PITCH_SHIFT)
        self.input_gain_var = tk.DoubleVar(value=config.INITIAL_INPUT_GAIN)
        self.output_gain_var = tk.DoubleVar(value=config.INITIAL_OUTPUT_GAIN)
        self.formant_var = tk.IntVar(value=config.INITIAL_FORMANT_SHIFT)
        self.noise_gate_var = tk.IntVar(value=config.INITIAL_NOISE_GATE_THRESHOLD)
        
        # ラベル
        self.pitch_label = None
        self.input_gain_label = None
        self.output_gain_label = None
        self.formant_label = None
        self.noise_gate_label = None
        
        # ボタン
        self.start_button = None
        self.stop_button = None
        
        # デバイス情報
        self.input_devices = input_devices
        self.output_devices = output_devices
        
        self._build_ui()
    
    def _build_ui(self):
        """UI構築"""
        # タイトル
        title = ttk.Label(self.root, text=config.WINDOW_TITLE, 
                          font=config.FONT_TITLE)
        title.pack(pady=10)
        
        # ステータス表示
        status_label = ttk.Label(self.root, textvariable=self.status_var, 
                                 font=config.FONT_LABEL, foreground="red")
        status_label.pack(pady=5)
        
        # デバイス選択フレーム
        device_frame = ttk.LabelFrame(self.root, text="デバイス設定", padding=10)
        device_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(device_frame, text="入力デバイス:").grid(row=0, column=0, sticky="w")
        input_combo = ttk.Combobox(
            device_frame, 
            textvariable=self.input_var,
            values=[f"[{i}] {name[:45]}" for i, name in self.input_devices],
            state="readonly",
            width=50
        )
        if self.input_devices:
            input_combo.current(0)
        input_combo.grid(row=0, column=1, sticky="ew", padx=5)
        
        ttk.Label(device_frame, text="出力デバイス:").grid(row=1, column=0, sticky="w")
        output_combo = ttk.Combobox(
            device_frame,
            textvariable=self.output_var,
            values=[f"[{i}] {name[:45]}" for i, name in self.output_devices],
            state="readonly",
            width=50
        )
        if self.output_devices:
            output_combo.current(0)
        output_combo.grid(row=1, column=1, sticky="ew", padx=5)
        
        device_frame.columnconfigure(1, weight=1)
        
        # エフェクト設定フレーム
        effect_frame = ttk.LabelFrame(self.root, text="エフェクト設定", padding=10)
        effect_frame.pack(fill="x", padx=10, pady=10)
        
        # ピッチシフト
        ttk.Label(effect_frame, text="ピッチシフト (セミトーン):").grid(row=0, column=0, sticky="w")
        self.pitch_label = ttk.Label(effect_frame, text="3", font=config.FONT_VALUE)
        self.pitch_label.grid(row=0, column=2, sticky="e")
        
        pitch_slider = ttk.Scale(
            effect_frame,
            from_=config.PITCH_SHIFT_MIN,
            to=config.PITCH_SHIFT_MAX,
            orient="horizontal",
            variable=self.pitch_var
        )
        pitch_slider.grid(row=0, column=1, sticky="ew", padx=5)
        
        # 入力ゲイン
        ttk.Label(effect_frame, text="入力ゲイン (倍率):").grid(row=1, column=0, sticky="w")
        self.input_gain_label = ttk.Label(effect_frame, text="1.00", font=config.FONT_VALUE)
        self.input_gain_label.grid(row=1, column=2, sticky="e")
        
        input_gain_slider = ttk.Scale(
            effect_frame,
            from_=config.INPUT_GAIN_MIN,
            to=config.INPUT_GAIN_MAX,
            orient="horizontal",
            variable=self.input_gain_var
        )
        input_gain_slider.grid(row=1, column=1, sticky="ew", padx=5)
        
        # 出力ゲイン
        ttk.Label(effect_frame, text="出力ゲイン (倍率):").grid(row=2, column=0, sticky="w")
        self.output_gain_label = ttk.Label(effect_frame, text="1.00", font=config.FONT_VALUE)
        self.output_gain_label.grid(row=2, column=2, sticky="e")
        
        output_gain_slider = ttk.Scale(
            effect_frame,
            from_=config.OUTPUT_GAIN_MIN,
            to=config.OUTPUT_GAIN_MAX,
            orient="horizontal",
            variable=self.output_gain_var
        )
        output_gain_slider.grid(row=2, column=1, sticky="ew", padx=5)
        
        # フォルマントシフト
        ttk.Label(effect_frame, text="フォルマントシフト (セント相当):").grid(row=3, column=0, sticky="w")
        self.formant_label = ttk.Label(effect_frame, text="0", font=config.FONT_VALUE)
        self.formant_label.grid(row=3, column=2, sticky="e")
        
        formant_slider = ttk.Scale(
            effect_frame,
            from_=config.FORMANT_SHIFT_MIN,
            to=config.FORMANT_SHIFT_MAX,
            orient="horizontal",
            variable=self.formant_var
        )
        formant_slider.grid(row=3, column=1, sticky="ew", padx=5)
        
        # ノイズゲート
        ttk.Label(effect_frame, text="ノイズゲート (dB):").grid(row=4, column=0, sticky="w")
        self.noise_gate_label = ttk.Label(effect_frame, text="-40", font=config.FONT_VALUE)
        self.noise_gate_label.grid(row=4, column=2, sticky="e")
        
        noise_gate_slider = ttk.Scale(
            effect_frame,
            from_=config.NOISE_GATE_MIN,
            to=config.NOISE_GATE_MAX,
            orient="horizontal",
            variable=self.noise_gate_var
        )
        noise_gate_slider.grid(row=4, column=1, sticky="ew", padx=5)
        
        effect_frame.columnconfigure(1, weight=1)
        
        # コントロールボタンフレーム
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=15)
        
        self.start_button = ttk.Button(button_frame, text="開始")
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="停止", state="disabled")
        self.stop_button.pack(side="left", padx=5)
        
        # 情報フレーム
        info_frame = ttk.LabelFrame(self.root, text="情報", padding=10)
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        info_text = """
MVC アーキテクチャ:
- Model: オーディオ処理
- View: このUI
- Controller: イベント処理

設定値：
- ピッチシフト: -12～+12セミトーン
- 入力ゲイン: 0.1～20.0倍（マイク音量）
- 出力ゲイン: 0.1～20.0倍（スピーカー音量）
- フォルマントシフト: -24～+12（音声特性調整）
- ノイズゲート: -80～-20 dB（ノイズ除去閾値）

注意：
- デバイスは開始前に選択
- 入出力HostAPI一致が必須
        """
        info_display = tk.Text(info_frame, height=10, width=55, wrap="word")
        info_display.insert("1.0", info_text.strip())
        info_display.config(state="disabled")
        info_display.pack()
    
    def get_selected_devices(self):
        """選択デバイスのインデックスを取得"""
        input_str = self.input_var.get()
        output_str = self.output_var.get()
        
        try:
            input_idx = int(input_str.split(']')[0].strip('[')) if input_str else None
            output_idx = int(output_str.split(']')[0].strip('[')) if output_str else None
            return input_idx, output_idx
        except Exception:
            return None, None
    
    def update_pitch_label(self, value):
        """ピッチシフト表示を更新"""
        self.pitch_label.config(text=str(int(float(value))))
    
    def update_input_gain_label(self, value):
        """入力ゲイン表示を更新"""
        self.input_gain_label.config(text=f"{float(value):.2f}")
    
    def update_output_gain_label(self, value):
        """出力ゲイン表示を更新"""
        self.output_gain_label.config(text=f"{float(value):.2f}")
    
    def update_formant_label(self, value):
        """フォルマントシフト表示を更新"""
        self.formant_label.config(text=str(int(float(value))))
    
    def update_noise_gate_label(self, value):
        """ノイズゲート表示を更新"""
        self.noise_gate_label.config(text=str(int(float(value))))
    
    def set_status(self, status_text, color="red"):
        """ステータス表示を更新"""
        self.status_var.set(status_text)
        # 色は別途対応可能
    
    def enable_start_button(self):
        """開始ボタン有効化"""
        self.start_button.config(state="normal")
    
    def disable_start_button(self):
        """開始ボタン無効化"""
        self.start_button.config(state="disabled")
    
    def enable_stop_button(self):
        """停止ボタン有効化"""
        self.stop_button.config(state="normal")
    
    def disable_stop_button(self):
        """停止ボタン無効化"""
        self.stop_button.config(state="disabled")
    
    def enable_device_controls(self):
        """デバイス選択有効化"""
        # Comboboxの状態変更は別途対応（現在は read-only）
        pass
    
    def disable_device_controls(self):
        """デバイス選択無効化"""
        pass
