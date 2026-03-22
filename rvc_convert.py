#!/usr/bin/env python3
"""RVC音声変換スクリプト: rvc-python による実モデル推論を実行する。"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

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


def list_available_models() -> list[str]:
    """src/models/rvc 配下の .pth を列挙する。"""
    models_dir = Path(__file__).resolve().parent / 'src' / 'models' / 'rvc'
    if not models_dir.exists():
        return []
    return sorted(path.stem for path in models_dir.glob('*.pth'))


def resolve_model_path(model_name: str) -> Path:
    """モデル名から .pth パスを解決する。"""
    return Path(__file__).resolve().parent / 'src' / 'models' / 'rvc' / f'{model_name}.pth'


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
    parser.add_argument('--backend', choices=['rvc-python'], default='rvc-python',
                       help='変換バックエンド (rvc-python のみ対応)')

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
    config.pop('fast_mode', None)

    # 設定を表示
    print("=== RVC変換設定 ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print()

    selected_backend = 'rvc-python'
    print(f"backend: {selected_backend}")

    if not RVC_PYTHON_AVAILABLE:
        print('rvc-python が現在の環境にインストールされていません')
        return 1

    # 入力ファイルの存在確認
    if not os.path.exists(config['input_file']):
        print(f"入力ファイルが見つかりません: {config['input_file']}")
        return 1

    # 利用可能なモデルを確認
    available_models = list_available_models()
    if config['rvc_model'] not in available_models:
        print(f"指定されたモデルが見つかりません: {config['rvc_model']}")
        print(f"利用可能なモデル: {available_models}")
        return 1

    # RVCモデルを設定
    model_path = resolve_model_path(config['rvc_model'])
    if not model_path.exists():
        print(f"RVCモデルファイルが見つかりません: {model_path}")
        return 1

    print(f"RVCモデルを設定しました: {config['rvc_model']}")
    try:
        # 出力ディレクトリが存在しない場合は作成
        output_dir = os.path.dirname(config['output_file'])
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("実モデル推論 (rvc-python) を実行中...")
        convert_with_rvc_python(
            config['input_file'],
            config['output_file'],
            str(model_path),
            config.get('pitch_shift', 0),
            config,
        )
        print(f"変換完了！出力ファイル: {config['output_file']}")

        # 変換結果と同名の設定ファイルを保存
        config_output_path = os.path.splitext(config['output_file'])[0] + '.json'
        config['backend'] = selected_backend
        save_config(config_output_path, config)
        print(f"設定ファイルを保存しました: {config_output_path}")

    except Exception as e:
        print(f"rvc-python変換エラー: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())