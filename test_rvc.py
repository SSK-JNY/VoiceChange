#!/usr/bin/env python3
"""
RVCモデルテストスクリプト
"""
import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# fairseqの警告を抑制
warnings.filterwarnings("ignore", message=".*fairseq.*")

try:
    from src.models.rvc_model import RVCModel
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

import numpy as np

def test_rvc_model():
    """RVCモデルのテスト"""
    print("=== RVC Model Test ===")

    # RVCモデル初期化
    rvc = RVCModel()

    print(f"Model directory: {rvc.models_dir}")
    print(f"Device: {rvc.device}")

    # テスト音声生成
    sr = 16000
    duration = 1.0  # 1秒
    t = np.linspace(0, duration, int(sr * duration), False)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hzサイン波

    print(f"Test audio shape: {test_audio.shape}, sr: {sr}")

    # HuBERT特徴抽出テスト
    print("\n--- Testing Feature Extraction ---")
    try:
        feats = rvc.extract_features(test_audio, sr)
        if feats is not None:
            print(f"✅ Features extracted successfully: shape {feats.shape}")
        else:
            print("❌ Feature extraction failed")
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")

    # F0抽出テスト
    print("\n--- Testing F0 Extraction ---")
    try:
        f0 = rvc.extract_f0(test_audio, sr)
        if f0 is not None:
            print(f"✅ F0 extracted successfully: shape {f0.shape}, mean: {np.mean(f0):.2f}")
        else:
            print("❌ F0 extraction failed")
    except Exception as e:
        print(f"❌ F0 extraction error: {e}")

    # 簡易変換テスト
    print("\n--- Testing Simple Conversion ---")
    try:
        converted = rvc._simple_convert(test_audio, sr, pitch_shift=2.0)
        print(f"✅ Simple conversion successful: shape {converted.shape}")
    except Exception as e:
        print(f"❌ Simple conversion error: {e}")

    # 完全変換テスト（モデルなし）
    print("\n--- Testing Full Voice Conversion (without model) ---")
    try:
        converted = rvc.convert_voice(test_audio, sr, pitch_shift=2.0)
        print(f"✅ Full voice conversion successful: shape {converted.shape}")
    except Exception as e:
        print(f"❌ Full voice conversion error: {e}")

    # 利用可能なモデル確認
    print("\n--- Available Models ---")
    models = rvc.get_available_models()
    print(f"Available models: {models}")

    if models:
        # 最初のモデルをテスト
        model_name = models[0]
        model_path = rvc.models_dir / f"{model_name}.pth"
        config_path = rvc.models_dir / f"{model_name}.json"

        print(f"\n--- Testing RVC Model Loading: {model_name} ---")
        print(f"Model path: {model_path}")
        print(f"Config path: {config_path}")
        print(f"Model exists: {model_path.exists()}")
        print(f"Config exists: {config_path.exists()}")

        success = rvc.load_rvc_model(str(model_path))

        if success:
            print("✅ RVC model loading successful")

            # モデルを使った変換テスト
            try:
                converted = rvc.convert_voice(test_audio, sr, str(model_path), pitch_shift=2.0)
                print(f"✅ RVC voice conversion successful: shape {converted.shape}")
            except Exception as e:
                print(f"❌ RVC voice conversion error: {e}")
        else:
            print("❌ RVC model loading failed")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_rvc_model()