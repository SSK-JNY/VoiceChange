# RVC音声変換ガイド

`rvc_convert.py` を使って、WAVファイルをRVCモデルで変換するための手順です。

## 現在のバックエンド

- `rvc-python`: 実モデル推論経路（推奨）
- `legacy`: 既存の互換経路
- `auto`: `rvc-python` が使える場合は自動選択

## 事前準備

1. 仮想環境（推奨: Python 3.10）

```bash
python3.10 -m venv venv310
source venv310/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install rvc-python==0.1.5
```

2. モデル配置

```text
src/models/rvc/
  05 つくよみちゃん公式RVCモデル 弱.pth
  (任意) 05 つくよみちゃん公式RVCモデル 弱.index
```

3. 入力音声

- WAV形式を推奨
- 例: `test/test.wav`

## 設定ファイル

`rvc_config.json` の主な項目:

```json
{
  "input_file": "test/test.wav",
  "output_file": "test/test_female.wav",
  "rvc_model": "05 つくよみちゃん公式RVCモデル 弱",
  "pitch_shift": 12,
  "fast_mode": false,
  "sample_rate": 44100,
  "normalize_input": true,
  "normalize_output": true,
  "f0_method": "rmvpe",
  "index_rate": 0.75,
  "protect": 0.33
}
```

## 実行方法

### 実モデル推論（推奨）

```bash
./venv310/bin/python rvc_convert.py \
  -i test/test.wav \
  -o test/test_female_real.wav \
  -m "05 つくよみちゃん公式RVCモデル 弱" \
  -p 12 \
  --backend rvc-python
```

### 互換経路で実行

```bash
./venv310/bin/python rvc_convert.py --backend legacy
```

### 自動選択

```bash
./venv310/bin/python rvc_convert.py --backend auto
```

## CLIオプション

- `-c, --config`: 設定ファイル（デフォルト: `rvc_config.json`）
- `-i, --input`: 入力WAVパス
- `-o, --output`: 出力WAVパス
- `-m, --model`: モデル名（拡張子なし）
- `-p, --pitch`: ピッチシフト（半音）
- `--fast`: 高速モード（legacy経路用）
- `--backend`: `auto | legacy | rvc-python`

## 生成ファイル

- 変換後のWAV
- 同名の設定JSON（実行時パラメータを保存）

例:

```text
test/
  test_female_real.wav
  test_female_real.json
```

## トラブルシューティング

### `weights_only` 関連エラー

- PyTorch 2.6+ で発生する既知問題
- `rvc_convert.py --backend rvc-python` 経路では互換設定を内部で適用済み

### モデルはあるのにJSONがない

- 本リポジトリでは `.pth` 内蔵設定の読み取りに対応
- ただし、実モデル推論は `rvc-python` 経路の利用を推奨

### WSLで音声デバイスが見えない

- リアルタイムI/OはWindowsネイティブ実行が安定
- WSLはモデル推論とバッチ変換に使用
