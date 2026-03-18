#!/usr/bin/env python3
"""RVC音声変換スクリプト: 設定ファイルに基づいて音声を変換"""
import json
import sys
import os
import argparse
import numpy as np
import soundfile as sf
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.models.voice_model import AudioModel

try:
    from rvc_python.infer import RVCInference
    RVC_PYTHON_AVAILABLE = True
except Exception:
    RVCInference = None
    RVC_PYTHON_AVAILABLE = False


def load_config(config_path):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config_path, config):
    """設定ファイルを保存する"""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def convert_with_rvc_python(input_path, output_path, model_path, pitch_shift, config):
    """rvc-python を使った実モデル推論経路"""
    if not RVC_PYTHON_AVAILABLE:
        raise RuntimeError("rvc-python is not available in current environment")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    index_path = model_path.replace('.pth', '.index')

    # PyTorch 2.6+ の weights_only 既定値変更に対応
    old_env = os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

    try:
        infer = RVCInference(device=device)
        infer.load_model(
            model_path,
            version=config.get('version', 'v2'),
            index_path=index_path if os.path.exists(index_path) else ''
        )
        infer.set_params(
            f0up_key=int(pitch_shift),
            f0method=config.get('f0_method', 'rmvpe'),
            index_rate=float(config.get('index_rate', 0.75)),
            filter_radius=int(config.get('filter_radius', 3)),
            rms_mix_rate=float(config.get('rms_mix_rate', 1.0)),
            protect=float(config.get('protect', 0.33)),
        )
        infer.infer_file(input_path, output_path)
    finally:
        if old_env is None:
            os.environ.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', None)
        else:
            os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = old_env


def main():
    parser = argparse.ArgumentParser(description='RVC音声変換スクリプト')
    parser.add_argument('--config', '-c', default='rvc_config.json',
                       help='設定ファイルのパス (デフォルト: rvc_config.json)')
    parser.add_argument('--input', '-i', help='入力ファイル (設定ファイルより優先)')
    parser.add_argument('--output', '-o', help='出力ファイル (設定ファイルより優先)')
    parser.add_argument('--model', '-m', help='RVCモデル名 (設定ファイルより優先)')
    parser.add_argument('--pitch', '-p', type=int, help='ピッチシフト (設定ファイルより優先)')
    parser.add_argument('--fast', action='store_true', help='高速モードを使用')
    parser.add_argument('--backend', choices=['auto', 'legacy', 'rvc-python'], default='auto',
                       help='変換バックエンド (auto/legacy/rvc-python)')

    args = parser.parse_args()

    # 設定ファイルを読み込む
    try:
        config = load_config(args.config)
        print(f"設定ファイルを読み込みました: {args.config}")
    except FileNotFoundError:
        print(f"設定ファイルが見つかりません: {args.config}")
        return 1
    except json.JSONDecodeError as e:
        print(f"設定ファイルの解析エラー: {e}")
        return 1

    # 設定ファイルのディレクトリを基準とした相対パスを解決
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if config_dir and not os.path.isabs(config['input_file']):
        config['input_file'] = os.path.join(config_dir, config['input_file'])
    if config_dir and not os.path.isabs(config['output_file']):
        config['output_file'] = os.path.join(config_dir, config['output_file'])

    # コマンドライン引数で設定を上書き
    if args.input:
        config['input_file'] = args.input
    if args.output:
        config['output_file'] = args.output
    if args.model:
        config['rvc_model'] = args.model
    if args.pitch is not None:
        config['pitch_shift'] = args.pitch
    if args.fast:
        config['fast_mode'] = True

    # 設定を表示
    print("=== RVC変換設定 ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print()

    selected_backend = args.backend
    if selected_backend == 'auto':
        selected_backend = 'rvc-python' if RVC_PYTHON_AVAILABLE else 'legacy'
    print(f"backend: {selected_backend}")

    # 入力ファイルの存在確認
    if not os.path.exists(config['input_file']):
        print(f"入力ファイルが見つかりません: {config['input_file']}")
        return 1

    # AudioModelを初期化
    print("AudioModelを初期化中...")
    model = AudioModel()

    # RVC設定
    print("RVC設定を適用中...")
    model.enable_rvc(True)
    model.set_rvc_fast_mode(config.get('fast_mode', False))

    # 利用可能なモデルを確認
    available_models = model.get_available_rvc_models()
    if config['rvc_model'] not in available_models:
        print(f"指定されたモデルが見つかりません: {config['rvc_model']}")
        print(f"利用可能なモデル: {available_models}")
        return 1

    # RVCモデルを設定
    model_path = f"src/models/rvc/{config['rvc_model']}.pth"
    if not os.path.exists(model_path):
        print(f"RVCモデルファイルが見つかりません: {model_path}")
        return 1

    model.set_rvc_model(model_path)
    model.set_rvc_pitch_shift(config.get('pitch_shift', 0))

    print(f"RVCモデルを設定しました: {config['rvc_model']}")

    if selected_backend == 'rvc-python':
        print("実モデル推論 (rvc-python) を実行中...")
        try:
            # 出力ディレクトリが存在しない場合は作成
            output_dir = os.path.dirname(config['output_file'])
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            convert_with_rvc_python(
                config['input_file'],
                config['output_file'],
                model_path,
                config.get('pitch_shift', 0),
                config,
            )
            print(f"変換完了！出力ファイル: {config['output_file']}")

            config_output_path = os.path.splitext(config['output_file'])[0] + '.json'
            config['backend'] = selected_backend
            save_config(config_output_path, config)
            print(f"設定ファイルを保存しました: {config_output_path}")
            return 0
        except Exception as e:
            print(f"rvc-python変換エラー: {e}")
            return 1

    # legacy 経路: 既存の簡易RVC変換
    print(f"音声ファイルを読み込み中: {config['input_file']}")
    try:
        audio, sr = sf.read(config['input_file'])
        print(f"音声データ: {len(audio)}サンプル, {sr}Hz, {audio.shape}")

        # ステレオの場合はモノラルに変換
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            print("ステレオをモノラルに変換しました")

        # 正規化（オプション）
        if config.get('normalize_input', True):
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
                print("入力を正規化しました")

    except Exception as e:
        print(f"音声ファイルの読み込みエラー: {e}")
        return 1

    # RVC変換を実行
    print("RVC変換を実行中...")
    try:

        if config.get('fast_mode', False):
            # 高速モード
            converted = model._apply_rvc_fast_mode(audio, sr, config.get('pitch_shift', 0))
            print("高速モードで変換しました")
        else:
            # 通常モード
            converted = model.rvc_model.convert_voice(
                audio, sr, model.rvc_model_path, config.get('pitch_shift', 0)
            )
            print("通常モードで変換しました")

        print(f"変換結果: {len(converted)}サンプル")

        # フォルマント/EQ調整（formant_shiftパラメータがあれば適用）
        formant_shift = config.get('formant_shift', 0)
        if abs(formant_shift) > 0.5:
            temp_model = AudioModel()
            temp_model.samplerate = sr
            temp_model.formant_shift = formant_shift
            converted = temp_model._apply_formant(converted.reshape(-1, 1)).flatten()
            print(f"フォルマント/EQ調整を適用しました (formant_shift={formant_shift})")

        # 正規化（オプション）
        if config.get('normalize_output', True):
            max_val = np.max(np.abs(converted))
            if max_val > 0:
                converted = converted / max_val
                print("出力を正規化しました")

    except Exception as e:
        print(f"RVC変換エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 出力ファイルを保存
    print(f"出力ファイルを保存中: {config['output_file']}")
    try:
        # 出力ディレクトリが存在しない場合は作成
        output_dir = os.path.dirname(config['output_file'])
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sf.write(config['output_file'], converted, sr)
        print(f"変換完了！出力ファイル: {config['output_file']}")

        # 変換結果と同名の設定ファイルを保存
        config_output_path = os.path.splitext(config['output_file'])[0] + '.json'
        save_config(config_output_path, config)
        print(f"設定ファイルを保存しました: {config_output_path}")

    except Exception as e:
        print(f"出力ファイル保存エラー: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())