#!/usr/bin/env python3
"""
設定ファイル: UI上で変更しないパラメータを管理
"""

# オーディオパラメータ
SAMPLERATE = 44100  # Hz
BLOCKSIZE = 1024    # サンプル数（途切れ対策でさらに小さく）

# 初期値
INITIAL_PITCH_SHIFT = 3       # セミトーン
INITIAL_FORMANT_SHIFT = 0     # セント相当
INITIAL_INPUT_GAIN = 1.0      # 倍率
INITIAL_OUTPUT_GAIN = 1.0     # 倍率
INITIAL_NOISE_GATE_THRESHOLD = -40  # dB

# パラメータ範囲
PITCH_SHIFT_MIN = -12
PITCH_SHIFT_MAX = 12

FORMANT_SHIFT_MIN = -24
FORMANT_SHIFT_MAX = 12

INPUT_GAIN_MIN = 0.1
INPUT_GAIN_MAX = 20.0

OUTPUT_GAIN_MIN = 0.1
OUTPUT_GAIN_MAX = 20.0

NOISE_GATE_MIN = -80
NOISE_GATE_MAX = -20

# GUI設定
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800
WINDOW_RESIZABLE = False

# ウィンドウタイトル
WINDOW_TITLE = "リアルタイムボイスチェンジャー"

# オーディオモード
AUDIO_MODE_NORMAL = 'normal'
AUDIO_MODE_PASSTHROUGH = 'passthrough'
AUDIO_MODE_TEST_TONE = 'test-tone'
AUDIO_MODE_TEST_TONE_FREQ = 440  # Hz
AUDIO_MODE_TEST_TONE_GAIN = 0.2  # 振幅

# フォントサイズ
FONT_TITLE = ("Arial", 14, "bold")
FONT_LABEL = ("Arial", 12)
FONT_VALUE = ("Arial", 12)
