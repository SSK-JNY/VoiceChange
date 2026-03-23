#!/usr/bin/env python3
"""Tkinter UIビュー"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from typing import Optional
import config
from src.app.gui_local_settings import GuiLocalSettings


class AudioView:
    """オーディオGUI - ビュー層"""
    
    def __init__(self, root, input_devices, output_devices, gui_settings: Optional[GuiLocalSettings] = None):
        self.gui_settings = gui_settings or GuiLocalSettings()
        self.root = root
        self.root.title(self.gui_settings.window_title)

        self._setup_fonts()
        
        # Controller参照（後で設定される）
        self.controller = None
        
        # UI変数
        self.status_var = tk.StringVar(value="停止中")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.tuning_preset_var = tk.StringVar(value="低遅延")
        self.blocksize_var = tk.StringVar(value=str(self.gui_settings.blocksize))
        self.rvc_timeout_var = tk.StringVar(value=f"{self.gui_settings.rvc_processing_timeout_sec:.2f}")
        self.fast_rpc_every_var = tk.StringVar(value=str(self.gui_settings.fast_mode_rpc_every_n_chunks))
        self.fast_rpc_timeout_var = tk.StringVar(value=f"{self.gui_settings.fast_mode_rpc_timeout_sec:.2f}")
        self.fast_rpc_bootstrap_timeout_var = tk.StringVar(value=f"{self.gui_settings.fast_mode_rpc_bootstrap_timeout_sec:.2f}")
        self.fast_local_mix_var = tk.StringVar(value=f"{self.gui_settings.fast_mode_local_mix:.2f}")
        self.stream_in_buf_var = tk.StringVar(value=f"{self.gui_settings.stream_input_buffer_seconds:.2f}")
        self.stream_out_buf_var = tk.StringVar(value=f"{self.gui_settings.stream_output_buffer_seconds:.2f}")
        self.output_delay_ms_var = tk.StringVar(value=f"{self.gui_settings.output_delay_ms:.0f}")
        self.robot_distortion_drive_db_var = tk.StringVar(value=f"{float(getattr(self.gui_settings, 'robot_distortion_drive_db', 45.0)):.0f}")
        self.robot_chorus_mix_var = tk.StringVar(value=f"{float(getattr(self.gui_settings, 'robot_chorus_mix', 0.9)):.2f}")
        self.pitch_var = tk.IntVar(value=self.gui_settings.initial_pitch_shift)
        self.input_gain_var = tk.DoubleVar(value=self.gui_settings.initial_input_gain)
        self.output_gain_var = tk.DoubleVar(value=self.gui_settings.initial_output_gain)
        self.formant_var = tk.IntVar(value=self.gui_settings.initial_formant_shift)
        self.noise_gate_var = tk.IntVar(value=self.gui_settings.initial_noise_gate_threshold)

        # RVC関連変数
        self.rvc_enabled_var = tk.BooleanVar(value=False)
        self.rvc_fast_mode_var = tk.BooleanVar(value=False)  # デフォルトで高速モード無効
        self.allow_dry_fallback_var = tk.BooleanVar(value=bool(self.gui_settings.allow_dry_fallback_on_rvc_fail))
        self.rvc_model_var = tk.StringVar()
        self.rvc_pitch_var = tk.IntVar(value=0)

        # サーバ接続関連変数
        self.server_url_var = tk.StringVar(value=self.gui_settings.server_url)
        self.server_status_var = tk.StringVar(value="未接続")
        self.server_detail_var = tk.StringVar(value="")
        self.connect_button = None  # _build_ui 内で初期化
        
        # ボトルネック情報表示用変数
        self.bottleneck_info_var = tk.StringVar(value="")
        
        # ラベル
        self.pitch_label = None
        self.input_gain_label = None
        self.output_gain_label = None
        self.formant_label = None
        self.noise_gate_label = None
        
        # ボタン
        self.start_button = None
        self.stop_button = None
        self.passthrough_button = None
        
        # デバイス情報
        self.input_devices = input_devices
        self.output_devices = output_devices
        
        self._build_ui()

    def _setup_fonts(self):
        """環境に応じて日本語表示可能なフォントを選択"""
        available_families = set(tkfont.families(self.root))
        candidates = [
            "Noto Sans CJK JP",
            "Noto Sans JP",
            "IPAexGothic",
            "IPAGothic",
            "TakaoGothic",
            "VL Gothic",
            "Yu Gothic UI",
            "Meiryo",
            "MS Gothic",
            "DejaVu Sans",
            "Arial",
        ]

        selected = "TkDefaultFont"
        for family in candidates:
            if family in available_families:
                selected = family
                break

        self.font_title = (selected, 14, "bold")
        self.font_label = (selected, 11)
        self.font_value = (selected, 11)

        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family=selected, size=11)

        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(family=selected, size=11)

        style = ttk.Style(self.root)
        style.configure("TLabel", font=self.font_label)
        style.configure("TButton", font=self.font_label)
        style.configure("TCheckbutton", font=self.font_label)
        style.configure("TLabelframe.Label", font=self.font_label)
    
    def _build_ui(self):
        """UI構築"""
        # スクロール可能なキャンバスを作成（縦横両方向）
        # フレームコンテナ
        frame_container = ttk.Frame(self.root)
        frame_container.pack(fill="both", expand=True)
        
        # Canvas
        canvas = tk.Canvas(frame_container, highlightthickness=0, bg="white")
        
        # スクロールバー（縦）
        vscrollbar = ttk.Scrollbar(frame_container, orient="vertical", command=canvas.yview)
        
        # スクロールバー（横）
        hscrollbar = ttk.Scrollbar(frame_container, orient="horizontal", command=canvas.xview)
        
        # スクロール可能なフレーム
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def on_frame_configure(event=None):
            # Canvas のスクロール領域を更新
            canvas.configure(scrollregion=canvas.bbox("all"))
            # scrollable_frame をキャンバス幅に合わせる
            canvas_width = canvas.winfo_width()
            if canvas_width > 1:
                canvas.itemconfig(scrollable_frame_id, width=canvas_width)
        
        scrollable_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(scrollable_frame_id, width=e.width) or on_frame_configure())
        
        canvas.configure(yscrollcommand=vscrollbar.set, xscrollcommand=hscrollbar.set)
        
        # スクロール操作有効化（マウスホイール）
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # グリッドレイアウト設定
        frame_container.grid_rowconfigure(0, weight=1)
        frame_container.grid_columnconfigure(0, weight=1)
        
        # Canvas配置
        canvas.grid(row=0, column=0, sticky="nsew")
        vscrollbar.grid(row=0, column=1, sticky="ns")
        hscrollbar.grid(row=1, column=0, sticky="ew")
        
        # タイトル
        title = ttk.Label(scrollable_frame, text=self.gui_settings.window_title, 
                          font=self.font_title)
        title.pack(pady=10)
        
        # ステータス表示
        status_label = ttk.Label(scrollable_frame, textvariable=self.status_var, 
                                 font=self.font_label, foreground="red")
        status_label.pack(pady=5)
        
        # デバイス選択フレーム
        device_frame = ttk.LabelFrame(scrollable_frame, text="デバイス設定", padding=10)
        device_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(device_frame, text="入力デバイス:").grid(row=0, column=0, sticky="w")
        input_values = [f"[{i}] {name[:45]}" for i, name in self.input_devices]
        if not input_values:
            input_values = ["利用可能な入力デバイスが見つかりません"]

        self.input_combo = ttk.Combobox(
            device_frame, 
            textvariable=self.input_var,
            values=input_values,
            state="readonly",
            width=50
        )
        if self.input_devices:
            self.input_combo.current(0)
        else:
            self.input_var.set(input_values[0])
            self.input_combo.state(["disabled"])
        self.input_combo.grid(row=0, column=1, sticky="ew", padx=5)
        
        ttk.Label(device_frame, text="出力デバイス:").grid(row=1, column=0, sticky="w")
        output_values = [f"[{i}] {name[:45]}" for i, name in self.output_devices]
        if not output_values:
            output_values = ["利用可能な出力デバイスが見つかりません"]

        self.output_combo = ttk.Combobox(
            device_frame,
            textvariable=self.output_var,
            values=output_values,
            state="readonly",
            width=50
        )
        if self.output_devices:
            self.output_combo.current(0)
        else:
            self.output_var.set(output_values[0])
            self.output_combo.state(["disabled"])
        self.output_combo.grid(row=1, column=1, sticky="ew", padx=5)

        ttk.Label(device_frame, text="ブロックサイズ:").grid(row=2, column=0, sticky="w")
        self.blocksize_combo = ttk.Combobox(
            device_frame,
            textvariable=self.blocksize_var,
            values=["512", "1024", "2048", "4096", "8192", "12288", "16384", "24576", "32768", "44100"],
            state="readonly",
            width=20,
        )
        self.blocksize_combo.grid(row=2, column=1, sticky="w", padx=5)

        if not self.input_devices or not self.output_devices:
            warning_text = (
                "WSL から音声デバイスが検出できていません。"
                " Windows 側で実行するか、WSL の音声転送設定を確認してください。"
            )
            self.device_warning_label = ttk.Label(
                device_frame,
                text=warning_text,
                foreground="darkorange",
                wraplength=420,
                justify="left",
            )
            self.device_warning_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        
        device_frame.columnconfigure(1, weight=1)
        
        # エフェクト設定フレーム
        effect_frame = ttk.LabelFrame(scrollable_frame, text="エフェクト設定", padding=10)
        effect_frame.pack(fill="x", padx=10, pady=10)
        
        # ピッチシフト
        ttk.Label(effect_frame, text="ピッチシフト (セミトーン):").grid(row=0, column=0, sticky="w")
        self.pitch_label = ttk.Label(effect_frame, text="3", font=self.font_value)
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
        self.input_gain_label = ttk.Label(effect_frame, text="1.00", font=self.font_value)
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
        self.output_gain_label = ttk.Label(effect_frame, text="1.00", font=self.font_value)
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
        self.formant_label = ttk.Label(effect_frame, text="0", font=self.font_value)
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
        self.noise_gate_label = ttk.Label(effect_frame, text="-40", font=self.font_value)
        self.noise_gate_label.grid(row=4, column=2, sticky="e")
        
        noise_gate_slider = ttk.Scale(
            effect_frame,
            from_=config.NOISE_GATE_MIN,
            to=config.NOISE_GATE_MAX,
            orient="horizontal",
            variable=self.noise_gate_var
        )
        noise_gate_slider.grid(row=4, column=1, sticky="ew", padx=5)

        ttk.Label(effect_frame, text="ロボ歪み(dB):").grid(row=5, column=0, sticky="w")
        self.robot_distortion_combo = ttk.Combobox(
            effect_frame,
            textvariable=self.robot_distortion_drive_db_var,
            values=["0", "8", "16", "24", "32", "40", "45", "50", "60"],
            state="readonly",
            width=12,
        )
        self.robot_distortion_combo.grid(row=5, column=1, sticky="w", padx=5, pady=(4, 0))

        ttk.Label(effect_frame, text="ロボコーラスmix:").grid(row=5, column=2, sticky="w")
        self.robot_chorus_mix_combo = ttk.Combobox(
            effect_frame,
            textvariable=self.robot_chorus_mix_var,
            values=["0.00", "0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90", "1.00"],
            state="readonly",
            width=12,
        )
        self.robot_chorus_mix_combo.grid(row=5, column=3, sticky="w", padx=5, pady=(4, 0))
        
        effect_frame.columnconfigure(1, weight=1)

        # 推論サーバ接続フレーム
        server_frame = ttk.LabelFrame(scrollable_frame, text="推論サーバ (WSL)", padding=8)
        server_frame.pack(fill="x", padx=10, pady=(5, 0))

        ttk.Label(server_frame, text="URL:").grid(row=0, column=0, sticky="w")
        url_entry = ttk.Entry(server_frame, textvariable=self.server_url_var, width=36)
        url_entry.grid(row=0, column=1, sticky="ew", padx=5)
        self.connect_button = ttk.Button(server_frame, text="接続")
        self.connect_button.grid(row=0, column=2, padx=(0, 3))

        ttk.Label(server_frame, text="状態:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.server_status_label = ttk.Label(
            server_frame, textvariable=self.server_status_var, foreground="gray"
        )
        self.server_status_label.grid(row=1, column=1, columnspan=2, sticky="w", padx=5, pady=(4, 0))

        self.server_detail_label = ttk.Label(
            server_frame,
            textvariable=self.server_detail_var,
            foreground="gray",
            wraplength=330,
            justify="left",
        )
        self.server_detail_label.grid(row=2, column=1, columnspan=2, sticky="w", padx=5, pady=(2, 0))

        server_frame.columnconfigure(1, weight=1)

        # 速度チューニングフレーム
        tune_frame = ttk.LabelFrame(scrollable_frame, text="処理速度チューニング", padding=8)
        tune_frame.pack(fill="x", padx=10, pady=(5, 0))

        ttk.Label(tune_frame, text="チューニングプリセット:").grid(row=0, column=0, sticky="w")
        self.tuning_preset_combo = ttk.Combobox(
            tune_frame,
            textvariable=self.tuning_preset_var,
            values=["低遅延", "完全変換"],
            state="readonly",
            width=12,
        )
        self.tuning_preset_combo.grid(row=0, column=1, sticky="w", padx=5)
        self.apply_preset_button = ttk.Button(tune_frame, text="適用")
        self.apply_preset_button.grid(row=0, column=2, sticky="w", padx=(12, 0))

        ttk.Label(tune_frame, text="RVCタイムアウト(s):").grid(row=1, column=0, sticky="w")
        self.rvc_timeout_combo = ttk.Combobox(
            tune_frame,
            textvariable=self.rvc_timeout_var,
            values=["0.08", "0.12", "0.18", "0.25", "0.30", "0.40"],
            state="readonly",
            width=10,
        )
        self.rvc_timeout_combo.grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(tune_frame, text="高速RPC間隔(チャンク):").grid(row=1, column=2, sticky="w", padx=(12, 0))
        self.fast_rpc_every_combo = ttk.Combobox(
            tune_frame,
            textvariable=self.fast_rpc_every_var,
            values=["1", "2", "3", "4", "5", "6"],
            state="readonly",
            width=8,
        )
        self.fast_rpc_every_combo.grid(row=1, column=3, sticky="w", padx=5)

        ttk.Label(tune_frame, text="高速RPC timeout(s):").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.fast_rpc_timeout_combo = ttk.Combobox(
            tune_frame,
            textvariable=self.fast_rpc_timeout_var,
            values=["0.08", "0.10", "0.12", "0.15", "0.18", "0.22", "0.30"],
            state="readonly",
            width=10,
        )
        self.fast_rpc_timeout_combo.grid(row=2, column=1, sticky="w", padx=5, pady=(4, 0))

        ttk.Label(tune_frame, text="初回RPC timeout(s):").grid(row=2, column=2, sticky="w", padx=(12, 0), pady=(4, 0))
        self.fast_rpc_bootstrap_timeout_combo = ttk.Combobox(
            tune_frame,
            textvariable=self.fast_rpc_bootstrap_timeout_var,
            values=["0.20", "0.30", "0.35", "0.40", "0.45", "0.55", "0.70"],
            state="readonly",
            width=8,
        )
        self.fast_rpc_bootstrap_timeout_combo.grid(row=2, column=3, sticky="w", padx=5, pady=(4, 0))

        ttk.Label(tune_frame, text="ローカル混合比:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.fast_local_mix_combo = ttk.Combobox(
            tune_frame,
            textvariable=self.fast_local_mix_var,
            values=["0.00", "0.05", "0.10", "0.15", "0.20", "0.25", "0.35", "0.50", "0.60", "0.70", "0.80", "0.90", "1.00"],
            state="readonly",
            width=10,
        )
        self.fast_local_mix_combo.grid(row=3, column=1, sticky="w", padx=5, pady=(4, 0))

        ttk.Label(tune_frame, text="入力バッファ(s):").grid(row=3, column=2, sticky="w", padx=(12, 0), pady=(4, 0))
        self.stream_in_buf_combo = ttk.Combobox(
            tune_frame,
            textvariable=self.stream_in_buf_var,
            values=["0.30", "0.40", "0.50", "0.80", "1.00", "1.50"],
            state="readonly",
            width=8,
        )
        self.stream_in_buf_combo.grid(row=3, column=3, sticky="w", padx=5, pady=(4, 0))

        ttk.Label(tune_frame, text="出力バッファ(s):").grid(row=4, column=2, sticky="w", padx=(12, 0), pady=(4, 0))
        self.stream_out_buf_combo = ttk.Combobox(
            tune_frame,
            textvariable=self.stream_out_buf_var,
            values=["0.30", "0.40", "0.50", "0.80", "1.00", "1.50"],
            state="readonly",
            width=8,
        )
        self.stream_out_buf_combo.grid(row=4, column=3, sticky="w", padx=5, pady=(4, 0))

        ttk.Label(tune_frame, text="出力遅延(ms):").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self.output_delay_ms_combo = ttk.Combobox(
            tune_frame,
            textvariable=self.output_delay_ms_var,
            values=["0", "20", "40", "60", "80", "100", "150", "200", "300", "500"],
            state="readonly",
            width=10,
        )
        self.output_delay_ms_combo.grid(row=4, column=1, sticky="w", padx=5, pady=(4, 0))

        # RVC設定フレーム
        rvc_frame = ttk.LabelFrame(scrollable_frame, text="RVC設定 (AI音声変換)", padding=10)
        rvc_frame.pack(fill="x", padx=10, pady=10)

        # RVC有効/無効
        ttk.Label(rvc_frame, text="RVC有効:").grid(row=0, column=0, sticky="w")
        rvc_enabled_check = ttk.Checkbutton(rvc_frame, variable=self.rvc_enabled_var)
        rvc_enabled_check.grid(row=0, column=1, sticky="w", padx=5)

        # RVC高速モード
        ttk.Label(rvc_frame, text="高速モード:").grid(row=1, column=0, sticky="w")
        self.rvc_fast_mode_check = ttk.Checkbutton(rvc_frame, variable=self.rvc_fast_mode_var)
        self.rvc_fast_mode_check.grid(row=1, column=1, sticky="w", padx=5)

        # RVC失敗時の原音フォールバック可否
        ttk.Label(rvc_frame, text="RVC失敗時に原音を混ぜる:").grid(row=1, column=2, sticky="w", padx=(12, 0))
        self.allow_dry_fallback_check = ttk.Checkbutton(rvc_frame, variable=self.allow_dry_fallback_var)
        self.allow_dry_fallback_check.grid(row=1, column=3, sticky="w", padx=5)

        # RVCモデル選択
        ttk.Label(rvc_frame, text="モデル:").grid(row=2, column=0, sticky="w")
        self.rvc_model_combo = ttk.Combobox(
            rvc_frame,
            textvariable=self.rvc_model_var,
            values=[],  # 動的に設定
            state="readonly",
            width=40
        )
        self.rvc_model_combo.grid(row=2, column=1, sticky="ew", padx=5)

        # RVCピッチシフト
        ttk.Label(rvc_frame, text="ピッチシフト:").grid(row=3, column=0, sticky="w")
        self.rvc_pitch_label = ttk.Label(rvc_frame, text="0", font=self.font_value)
        self.rvc_pitch_label.grid(row=3, column=2, sticky="e")

        rvc_pitch_slider = ttk.Scale(
            rvc_frame,
            from_=-24,
            to=24,
            orient="horizontal",
            variable=self.rvc_pitch_var
        )
        rvc_pitch_slider.grid(row=3, column=1, sticky="ew", padx=5)

        rvc_frame.columnconfigure(1, weight=1)

        # コントロールボタンフレーム
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(pady=15)
        
        self.start_button = ttk.Button(button_frame, text="開始")
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="停止", state="disabled")
        self.stop_button.pack(side="left", padx=5)
        
        self.passthrough_button = ttk.Button(button_frame, text="パススルー", state="disabled")
        self.passthrough_button.pack(side="left", padx=5)
        
        # 情報フレーム
        info_frame = ttk.LabelFrame(scrollable_frame, text="情報", padding=10)
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

RVC (Retrieval-based Voice Conversion):
- AIによる音声変換
- 学習済みモデルを使用
- リアルタイム変換可能

注意：
- デバイスは開始前に選択
- 入出力HostAPI一致が必須
- RVC使用時はGPU推奨
        """
        info_display = tk.Text(info_frame, height=10, width=55, wrap="word", font=self.font_label)
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

    def update_rvc_pitch_label(self, value):
        """RVCピッチシフト表示を更新"""
        self.rvc_pitch_label.config(text=str(int(value)))

    def update_rvc_models(self, models):
        """RVCモデル一覧を更新"""
        self.rvc_model_combo['values'] = models
        if models:
            self.rvc_model_combo.current(0)
    
    def set_status(self, status_text, color="red"):
        """ステータス表示を更新"""
        self.status_var.set(status_text)
        # 色は別途対応可能
    
    def enable_start_button(self):
        """開始ボタン有効化"""
        if self.input_devices and self.output_devices:
            self.start_button.config(state="normal")
        else:
            self.start_button.config(state="disabled")
    
    def disable_start_button(self):
        """開始ボタン無効化"""
        self.start_button.config(state="disabled")
    
    def enable_stop_button(self):
        """停止ボタン有効化"""
        self.stop_button.config(state="normal")
    
    def disable_stop_button(self):
        """停止ボタン無効化"""
        self.stop_button.config(state="disabled")
    
    def enable_passthrough_button(self):
        """パススルーボタン有効化"""
        if self.input_devices and self.output_devices:
            self.passthrough_button.config(state="normal")
        else:
            self.passthrough_button.config(state="disabled")
    
    def disable_passthrough_button(self):
        """パススルーボタン無効化"""
        self.passthrough_button.config(state="disabled")
    
    def enable_device_controls(self):
        """デバイス選択有効化"""
        if self.input_devices:
            self.input_combo.state(["!disabled", "readonly"])
        if self.output_devices:
            self.output_combo.state(["!disabled", "readonly"])
    
    def set_controller(self, controller):
        """Controllerを設定（View初期化後に呼び出す）"""
        self.controller = controller
        
        # Controller設定後にイベントハンドラを設定
        if hasattr(self, 'rvc_fast_mode_check'):
            self.rvc_fast_mode_check.config(command=self.controller.on_rvc_fast_mode_change)

    def update_server_status(self, text: str, color: str = "gray") -> None:
        """サーバ接続状態ラベルを更新する。

        Args:
            text:  表示するテキスト（例: "接続済み", "未接続"）。
            color: tkinter 色文字列（例: "green", "red", "gray"）。
        """
        self.server_status_var.set(text)
        if hasattr(self, "server_status_label"):
            self.server_status_label.config(foreground=color)

    def update_server_detail(self, text: str, color: str = "gray") -> None:
        """サーバ接続詳細（試行回数、失敗理由など）を更新する。"""
        self.server_detail_var.set(text)
        if hasattr(self, "server_detail_label"):
            self.server_detail_label.config(foreground=color)
    
    def display_bottleneck_info(self, bottleneck_info: dict) -> str:
        """ボトルネック情報を文字列にフォーマットして返す（コンソール出力用）
        
        Args:
            bottleneck_info: get_bottleneck_info() の戻り値
        
        Returns:
            フォーマットされたボトルネック情報文字列
        """
        expected = bottleneck_info.get("expected_ms", 0)
        total_avg = bottleneck_info.get("total_avg_ms", 0)
        is_bottleneck = bottleneck_info.get("is_bottlenecked", False)
        
        status = "⚠️  BOTTLENECK" if is_bottleneck else "✓ OK"
        
        msg = f"""
{status} Audio Processing Performance Report:
  Expected frame time: {expected:.2f}ms
  
Processing breakdown (avg ms):
  Total:       {total_avg:.2f}ms (max: {bottleneck_info.get('total_max_ms', 0):.2f}ms)
  Input gain:  {bottleneck_info.get('input_gain_ms', 0):.2f}ms
  Noise reduce: {bottleneck_info.get('noise_reduce_ms', 0):.2f}ms
  Formant:     {bottleneck_info.get('formant_ms', 0):.2f}ms
  RVC:         {bottleneck_info.get('rvc_ms', 0):.2f}ms
  Pedalboard:  {bottleneck_info.get('pedalboard_ms', 0):.2f}ms
  Output gain: {bottleneck_info.get('output_gain_ms', 0):.2f}ms
  
Callback status errors: {bottleneck_info.get('callback_status_count', 0)}
"""
        return msg
