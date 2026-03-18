#!/usr/bin/env python3
"""
RVC処理速度テストスクリプト
"""
import sys
import os
import time
import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(__file__))

from src.models.rvc_model import RVCModel

def test_rvc_speed():
    """RVC処理速度をテスト"""
    print("=== RVC Processing Speed Test ===")

    # RVCモデルインスタンス作成
    rvc = RVCModel()
    print(f"Device: {rvc.device}")

    # テスト音声生成（1024サンプル、約23ms）
    sr = 44100
    duration = 1024 / sr  # ブロックサイズ分の長さ
    t = np.linspace(0, duration, 1024, False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hzのサイン波

    print(f"Test audio shape: {audio.shape}, duration: {duration:.3f}s")

    # 利用可能なモデル確認
    models = rvc.get_available_models()
    print(f"Available models: {models}")

    if not models:
        print("No RVC models found, testing simple conversion only")
        # 簡易変換テスト
        times = []
        for i in range(10):
            start = time.time()
            result = rvc._simple_convert(audio, sr, pitch_shift=2.0)
            elapsed = time.time() - start
            times.append(elapsed)
            print(".4f")

        avg_time = np.mean(times)
        print(".4f")
        print(".1f")
        return

    # モデル読み込みテスト
    model_name = models[0]
    model_path = rvc.models_dir / f"{model_name}.pth"

    print(f"\nLoading model: {model_name}")
    start = time.time()
    success = rvc.load_rvc_model(str(model_path))
    load_time = time.time() - start

    if not success:
        print("Model loading failed")
        return

    print(".4f")

    # RVC変換速度テスト
    print("\nTesting RVC conversion speed...")
    times = []
    for i in range(10):
        start = time.time()
        result = rvc.convert_voice(audio, sr, str(model_path), pitch_shift=2.0)
        elapsed = time.time() - start
        times.append(elapsed)
        print(".4f")

    avg_time = np.mean(times)
    max_time = np.max(times)
    print(".4f")
    print(".4f")
    print(".1f")

    # リアルタイム要件チェック
    block_duration = 1024 / 44100  # 約23ms
    if avg_time < block_duration:
        print(".1f")
    else:
        print(".1f")

if __name__ == "__main__":
    test_rvc_speed()