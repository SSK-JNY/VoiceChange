#!/usr/bin/env python3
"""Tkinter UIビュー"""
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
import config


class AudioView:
    """オーディオGUI - ビュー層"""
    
    def __init__(self, root, input_devices, output_devices):
        self.root = root
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.resizable(config.WINDOW_RESIZABLE, config.WINDOW_RESIZABLE)

        self._setup_fonts()
        
        # Controller参照（後で設定される）
        self.controller = None
        
        # UI変数
        self.status_var = tk.StringVar(value="停止中")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.pitch_var = tk.IntVar(value=config.INITIAL_PITCH_SHIFT)
        self.input_gain_var = tk.DoubleVar(value=config.INITIAL_INPUT_GAIN)
        self.output_gain_var = tk.DoubleVar(value=config.INITIAL_OUTPUT_GAIN)
        self.formant_var = tk.IntVar(value=config.INITIAL_FORMANT_SHIFT)
        self.noise_gate_var = tk.IntVar(value=config.INITIAL_NOISE_GATE_THRESHOLD)

        # RVC関連変数
        self.rvc_enabled_var = tk.BooleanVar(value=False)
        self.rvc_fast_mode_var = tk.BooleanVar(value=False)  # デフォルトで高速モード無効
        self.rvc_model_var = tk.StringVar()
        self.rvc_pitch_var = tk.IntVar(value=0)
        
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
        # タイトル
        title = ttk.Label(self.root, text=config.WINDOW_TITLE, 
                          font=self.font_title)
        title.pack(pady=10)
        
        # ステータス表示
        status_label = ttk.Label(self.root, textvariable=self.status_var, 
                                 font=self.font_label, foreground="red")
        status_label.pack(pady=5)
        
        # デバイス選択フレーム
        device_frame = ttk.LabelFrame(self.root, text="デバイス設定", padding=10)
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
            self.device_warning_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        
        device_frame.columnconfigure(1, weight=1)
        
        # エフェクト設定フレーム
        effect_frame = ttk.LabelFrame(self.root, text="エフェクト設定", padding=10)
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
        
        effect_frame.columnconfigure(1, weight=1)

        # RVC設定フレーム
        rvc_frame = ttk.LabelFrame(self.root, text="RVC設定 (AI音声変換)", padding=10)
        rvc_frame.pack(fill="x", padx=10, pady=10)

        # RVC有効/無効
        ttk.Label(rvc_frame, text="RVC有効:").grid(row=0, column=0, sticky="w")
        rvc_enabled_check = ttk.Checkbutton(rvc_frame, variable=self.rvc_enabled_var)
        rvc_enabled_check.grid(row=0, column=1, sticky="w", padx=5)

        # RVC高速モード
        ttk.Label(rvc_frame, text="高速モード:").grid(row=1, column=0, sticky="w")
        rvc_fast_check = ttk.Checkbutton(rvc_frame, variable=self.rvc_fast_mode_var)
        rvc_fast_check.grid(row=1, column=1, sticky="w", padx=5)

        # RVCモデル選択
        ttk.Label(rvc_frame, text="モデル:").grid(row=2, column=0, sticky="w")
        self.rvc_model_combo = ttk.Combobox(
            rvc_frame,
            textvariable=self.rvc_model_var,
            values=[],  # 動的に設定
            state="readonly",
            width=40
        )
        self.rvc_model_combo.grid(row=1, column=1, sticky="ew", padx=5)

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

        # RVCモデルダウンロードボタン
        self.rvc_download_button = ttk.Button(rvc_frame, text="モデルダウンロード")
        self.rvc_download_button.grid(row=4, column=0, columnspan=3, pady=5)

        # RVC高速モードチェックボックス
        self.rvc_fast_mode_var = tk.BooleanVar(value=False)
        self.rvc_fast_mode_check = ttk.Checkbutton(
            rvc_frame,
            text="高速モード",
            variable=self.rvc_fast_mode_var
        )
        self.rvc_fast_mode_check.grid(row=5, column=0, columnspan=3, pady=5)

        rvc_frame.columnconfigure(1, weight=1)

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
