# RVC音声変換スクリプト

このスクリプトを使用すると、Retrieval-based Voice Conversion (RVC) を用いて音声を変換できます。

## 準備

1. RVCモデルをダウンロードして `src/models/rvc/` ディレクトリに配置
2. 変換したい音声ファイルを準備

## 設定ファイル

`rvc_config.json` で変換パラメータを設定できます。または、**WAVファイルと同じディレクトリに同名のJSONファイルを配置**することで、個別の設定ファイルを使用できます。

"output_file": "test_final.wav", // 設定ファイルからの相対パス

```json
{
  "input_file": "test/test.wav", // 入力音声ファイル
  "output_file": "test/output.wav", // 出力音声ファイル
  "rvc_model": "05 つくよみちゃん公式RVCモデル 弱", // 使用するRVCモデル名
  "pitch_shift": 0, // ピッチシフト（半音単位、-24～+24）
  "fast_mode": false, // 高速モードを使用するか
  "sample_rate": 44100, // サンプルレート
  "normalize_input": true, // 入力音声を正規化するか
  "normalize_output": true // 出力音声を正規化するか
}
```

### 個別設定ファイル

WAVファイルと同じディレクトリに同名のJSONファイルを配置することで、個別の設定を使用できます：

```
test/
├── test.wav      // 入力ファイル
├── test.json     // 個別設定ファイル
├── test_final.wav   // 出力ファイル
└── test_final.json  // 自動保存される変換設定
```

個別設定ファイルの内容例：

```json
{
  "input_file": "test.wav", // 設定ファイルからの相対パス
  "output_file": "test_output.wav", // 設定ファイルからの相対パス
  "rvc_model": "05 つくよみちゃん公式RVCモデル 弱",
  "pitch_shift": 12, // 女性声変換用
  "fast_mode": false
}
```

## 使用方法

### 基本的な使用

```bash
python rvc_convert.py
```

### コマンドラインオプション

```bash
python rvc_convert.py [オプション]
```

#### オプション

- `-c, --config CONFIG`: 設定ファイルのパス（デフォルト: rvc_config.json）
- `-i, --input INPUT`: 入力ファイル（設定ファイルより優先）
- `-o, --output OUTPUT`: 出力ファイル（設定ファイルより優先）
- `-m, --model MODEL`: RVCモデル名（設定ファイルより優先）
- `-p, --pitch PITCH`: ピッチシフト（設定ファイルより優先）
- `--fast`: 高速モードを使用

### 使用例

```bash
# 基本的な変換
python rvc_convert.py

# ピッチを+5半音シフト
python rvc_convert.py --pitch 5 --output output_pitch5.wav

# 高速モードでピッチを-3半音シフト
python rvc_convert.py --fast --pitch -3 --output output_fast.wav

# 別のモデルを使用
python rvc_convert.py --model "別のモデル名" --input input.wav --output output.wav

# 女性声変換（男性声から女性声へ）
python rvc_convert.py --pitch 12 --output female_voice.wav

# さまざまな女性声バリエーション
python rvc_convert.py --pitch 6 --output female_mild.wav    # 穏やかな女性声
python rvc_convert.py --pitch 8 --output female_soft.wav    # 柔らかい女性声
python rvc_convert.py --pitch 12 --output female_high.wav   # 高めの女性声

# 個別設定ファイルを使用
python rvc_convert.py --config test/test.json
```

## モードの違い

### 通常モード（fast_mode: false）

- 完全なRVCモデルを使用
- 高品質な声質変換が可能
- 処理時間が長い

### 高速モード（fast_mode: true）

- 簡易的なピッチシフト + エフェクトを使用
- 処理速度が速い
- 品質は通常モードより劣る

## 注意事項

- 入力音声はWAV形式を推奨
- RVCモデルファイル（.pth）は `src/models/rvc/` に配置してください
- 出力ディレクトリは自動的に作成されます

## 女性声変換のヒント

現在の環境では男性声モデルしか利用できませんが、ピッチシフトを使って女性声に近づけることができます：

- **+6～+8半音**: 穏やかで自然な女性声
- **+10～+12半音**: しっかりした女性声
- **+12半音以上**: 高めの女性声（声質が変わりすぎる可能性あり）

最適なピッチシフト値は、元の声の性別や個性によって異なります。複数の値を試して最適なものを選んでください。
