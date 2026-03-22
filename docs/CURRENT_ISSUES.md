# 現状の問題点と解決方針

最終更新: 2026-03-22

このドキュメントは、README とは別に、現時点で確認できている問題点、暫定対応、今後の解決方針を整理するための作業メモです。

## 現在の到達点

- Python 3.10 仮想環境 `venv310` で `fairseq` の import は成功
- `rvc-python` バックエンドを追加し、実モデル推論によるオフライン変換は成功
- フェーズ1として `src/protocol/` に共有メッセージ型、音声スキーマ、推論設定スキーマを追加済み
- フェーズ2として `src/server/` に WSL 推論サーバを追加済み
- WebSocket 経由で `health`, `list_models`, `load_model`, `infer_chunk` の単発疎通を確認済み
- フェーズ3として `src/client/` に Windows クライアント層を追加済み
- `InferenceClient` からサーバへの connect / health / list_models / load_model / disconnect を smoke test で確認済み
- GUI（`AudioView`）に「推論サーバ (WSL)」接続パネルを追加済み
- `test/test.wav` から以下の出力を生成済み
  - `test/test_female.wav`: 既存経路の変換結果
  - `test/test_female_real.wav`: 実モデル推論の変換結果
  - `test/test_female_real2.wav`: `rvc_convert.py --backend rvc-python` 経路の変換結果
- GUI の日本語フォント選択を改善し、WSL 側に日本語フォントを導入済み

## 課題一覧

### 1. WSL では音声デバイスを取得できない

状況:

- `sounddevice` / PortAudio から入力・出力デバイスが 0 件として見えている
- `/dev/snd` が存在しない
- WSL 上の GUI は表示できても、リアルタイム音声 I/O は安定して使えない

影響:

- WSL 上で GUI を起動しても、入力デバイスと出力デバイスを選択できない
- リアルタイムボイスチェンジャーとしては使用不可

暫定対応:

- UI 側で「デバイス未検出」メッセージを表示するように修正済み
- デバイスが 0 件のときは開始ボタンを無効化

本命の解決案:

- GUI と音声 I/O は Windows ネイティブで実行する
- WSL は fairseq / RVC 推論 / バッチ変換専用にする
- Windows 側クライアントと WSL 側推論サーバを localhost 通信で接続する

優先度:

- 高

### 2. GUI の RVC 経路は RPC 化済みだが、リアルタイム最適化は継続課題

状況:

- `rvc_convert.py` は `rvc-python` による実モデル推論へ一本化済み
- GUI のリアルタイム RVC は WSL 推論サーバへの RPC 経路へ移行済み
- ただし短チャンク連続推論の遅延・バッファ設計は継続的な調整対象

影響:

- GUI 側で RVC を有効にしても、常に本物のモデル推論が走る保証はない
- 状況によって簡易フォールバックに落ちる

暫定対応:

- オフライン変換用途は `rvc-python` バックエンドを使う

本命の解決案:

- GUI から直接 `RVCModel` を呼ばない構成へ移行する
- WSL 側に推論サーバを用意し、GUI 側は RPC 経由で推論を要求する
- `rvc_convert.py` も将来的には同じ推論エンジンを使うように統一する

優先度:

- 高

### 3. 旧ローカル RVC 実装は撤去済み

状況:

- 旧 `src/models/rvc_model.py` は削除済み
- 旧 legacy テストと補助ドキュメントも整理済み
- 推論確認は `rvc-python` と WSL 推論サーバのスモークテストへ集約した

影響:

- ローカル独自実装との二重保守が解消された
- 推論経路は `rvc-python` ベースの実装に統一された

暫定対応:

- なし（撤去完了）

本命の解決案:

- 現行の `rvc-python` / RPC 構成を維持しつつ、運用チューニングを続ける

優先度:

- 高

### 4. PyTorch 2.6+ で `weights_only` 互換問題がある

状況:

- HuBERT / RMVPE / RVC checkpoint 読み込みで `weights_only=True` 既定値変更の影響を受ける
- `fairseq` 系の checkpoint 読み込みで失敗する場合がある

影響:

- そのままでは HuBERT / モデルロードが失敗する

暫定対応:

- 必要な箇所で `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` を使う互換処理を追加済み

本命の解決案:

- 読み込みを一元化し、checkpoint ロード周りを共通ユーティリティへ切り出す
- 使用する推論バックエンドを一本化して互換処理の分散を減らす

優先度:

- 中

### 5. `rvc-python` 導入で依存関係が固定される

状況:

- `rvc-python==0.1.5` の依存で `numpy` が `1.23.5` に固定される
- 既存の他ライブラリと将来的に競合する可能性がある

影響:

- 同一仮想環境で他の用途まで抱えると依存衝突が起きやすい

暫定対応:

- 現在は `venv310` を RVC 向け仮想環境として利用

本命の解決案:

- `venv310` を「RVC/推論専用環境」として固定する
- GUI 用 Windows 環境と、WSL 推論用環境を分離する

優先度:

- 中

### 6. フォント問題は解消済みだが、環境依存は残る

状況:

- 日本語フォント未導入環境では文字化けする可能性がある
- 既存コードは利用可能フォントを自動選択するように修正済み

影響:

- 新しい環境では再度フォント不足が起こる可能性がある

暫定対応:

- WSL 側に `fonts-noto-cjk` と `fonts-ipafont` を導入済み

本命の解決案:

- Windows 実行を GUI の標準運用にする
- フォントに依存しすぎない UI を維持する

優先度:

- 低

## 推奨アーキテクチャ

今後の構成は以下を推奨する。

### Windows 側

- GUI
- 音声デバイス列挙
- マイク入力
- スピーカー出力
- リングバッファ
- 推論クライアント

### WSL 側

- fairseq
- HuBERT
- RMVPE
- RVC モデルロード
- 推論サーバ
- オフライン変換

### 共通層

- プロトコル定義
- 設定スキーマ
- モデル一覧 API
- 推論パラメータ更新 API

## 具体的なディレクトリ構成案

単一リポジトリのまま、Windows 側クライアント、WSL 側推論サーバ、共通仕様を分離する。

```text
VoiceChange/
├── docs/
│   ├── CURRENT_ISSUES.md
│   ├── ARCHITECTURE.md
│   └── RPC_PROTOCOL.md
├── gui.py
├── main.py
├── rvc_convert.py
├── requirements.txt
├── requirements-windows.txt
├── requirements-wsl.txt
├── src/
│   ├── app/
│   │   ├── config.py
│   │   ├── gui.py
│   │   └── main.py
│   ├── client/
│   │   ├── __init__.py
│   │   ├── inference_client.py
│   │   ├── audio_stream.py
│   │   └── ring_buffer.py
│   ├── controllers/
│   │   ├── __init__.py
│   │   └── voice_controller.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── voice_model.py
│   │   └── rvc/
│   ├── protocol/
│   │   ├── __init__.py
│   │   ├── messages.py
│   │   ├── audio_schema.py
│   │   └── settings_schema.py
│   ├── server/
│   │   ├── __init__.py
│   │   ├── inference_server.py
│   │   ├── session_manager.py
│   │   ├── inference_engine.py
│   │   └── model_registry.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── list_devices.py
│   └── views/
│       ├── __init__.py
│       └── voice_view.py
├── scripts/
│   ├── start_windows_gui.ps1
│   ├── start_wsl_server.sh
│   ├── check_models.py
│   └── smoke_test_rpc.py
└── test/
```

### Windows 側に置くもの

- `src/views/`
- `src/controllers/`
- `src/client/`
- `src/app/gui.py`
- `src/models/voice_model.py` のうち、音声デバイス I/O とローカルエフェクト処理

### WSL 側に置くもの

- `src/server/`
- `rvc_convert.py`
- fairseq / HuBERT / RMVPE / RVC checkpoint 読み込み

### 共通化するもの

- `src/protocol/`
- `src/app/config.py` のうち共有可能な設定
- 推論パラメータ、音声チャンク形式、エラーコード

### 既存コードからの対応関係

- [src/app/gui.py](src/app/gui.py#L8): Windows 側 GUI 起動の入口として維持
- [src/models/voice_model.py](src/models/voice_model.py#L23): Windows 側の音声処理モデルに整理
- [rvc_convert.py](rvc_convert.py#L33): WSL 側バッチ変換・推論確認用 CLI として維持

## RPC メッセージ仕様案

初期実装は WebSocket を想定する。

### 接続方針

- Windows クライアント: `ws://127.0.0.1:8765`
- WSL サーバ: localhost で待受
- 音声本体はバイナリ、制御は JSON

### セッション開始

クライアント送信:

```json
{
  "type": "hello",
  "client": "windows-gui",
  "protocol_version": 1,
  "session_id": "optional-uuid"
}
```

サーバ応答:

```json
{
  "type": "hello_ack",
  "protocol_version": 1,
  "server": "wsl-inference",
  "features": ["infer_chunk", "load_model", "list_models"]
}
```

### モデル一覧取得

クライアント送信:

```json
{
  "type": "list_models"
}
```

サーバ応答:

```json
{
  "type": "list_models_result",
  "models": [
    {
      "name": "05 つくよみちゃん公式RVCモデル 弱",
      "has_index": false,
      "version": "v2"
    }
  ]
}
```

### モデル読み込み

クライアント送信:

```json
{
  "type": "load_model",
  "model_name": "05 つくよみちゃん公式RVCモデル 弱",
  "params": {
    "f0_method": "rmvpe",
    "index_rate": 0.75,
    "protect": 0.33,
    "pitch_shift": 12
  }
}
```

サーバ応答:

```json
{
  "type": "load_model_result",
  "ok": true,
  "active_model": "05 つくよみちゃん公式RVCモデル 弱",
  "device": "cuda:0"
}
```

### パラメータ更新

クライアント送信:

```json
{
  "type": "update_params",
  "params": {
    "pitch_shift": 10,
    "index_rate": 0.6,
    "protect": 0.4,
    "rms_mix_rate": 1.0
  }
}
```

サーバ応答:

```json
{
  "type": "update_params_result",
  "ok": true
}
```

### 音声チャンク推論

制御ヘッダ:

```json
{
  "type": "infer_chunk",
  "sequence": 42,
  "sample_rate": 48000,
  "channels": 1,
  "dtype": "float32",
  "frame_count": 960
}
```

直後に送るバイナリ:

- PCM float32 little-endian
- `frame_count * channels` 分の連続データ

サーバ応答ヘッダ:

```json
{
  "type": "infer_chunk_result",
  "sequence": 42,
  "sample_rate": 48000,
  "channels": 1,
  "dtype": "float32",
  "frame_count": 960,
  "processing_ms": 14.7,
  "fallback": false
}
```

直後に返すバイナリ:

- 変換済み PCM float32 little-endian

### ヘルスチェック

クライアント送信:

```json
{
  "type": "health"
}
```

サーバ応答:

```json
{
  "type": "health_result",
  "ok": true,
  "device": "cuda:0",
  "model_loaded": true,
  "active_model": "05 つくよみちゃん公式RVCモデル 弱"
}
```

### エラー応答

```json
{
  "type": "error",
  "code": "MODEL_NOT_LOADED",
  "message": "No active model",
  "sequence": 42
}
```

### エラーコード案

- `MODEL_NOT_FOUND`
- `MODEL_NOT_LOADED`
- `INVALID_PARAMS`
- `INFERENCE_TIMEOUT`
- `UNSUPPORTED_SAMPLE_RATE`
- `BACKEND_ERROR`
- `PROTOCOL_MISMATCH`

### 運用上のルール

- `sequence` は必須
- 音声チャンクは順不同再送をしない
- タイムアウト時は Windows 側で原音または前フレームを再利用
- サーバ側は 1 セッション 1 アクティブモデルから開始する

## 移行チェックリスト

次回以降の作業は、以下の順に進める。

### フェーズ 1: 共有プロトコルの切り出し

- [x] `src/protocol/messages.py` を追加する
- [x] モデル一覧、モデル読み込み、パラメータ更新、推論、ヘルスチェックのメッセージ型を定義する
- [x] `sample_rate`, `channels`, `dtype`, `frame_count` の必須項目を固定する
- [x] `sequence` の扱いを全メッセージで統一する

完了条件:

- [x] クライアント・サーバで同じメッセージ定義を import できる

### フェーズ 2: WSL 推論サーバの追加

- [x] `src/server/inference_server.py` を追加する
- [x] `src/server/inference_engine.py` に `rvc-python` 呼び出しを集約する
- [x] `health`, `list_models`, `load_model`, `update_params`, `infer_chunk` を実装する
- [x] `rvc_convert.py` と同等の backend 設定をサーバ側でも再利用できるようにする
- [x] `scripts/start_wsl_server.sh` と `scripts/smoke_test_rpc.py` を追加する

完了条件:

- [x] CLI なしで、サーバ API 経由でモデル読み込みと単発推論が成功する

補足:

- 現在の `infer_chunk` 実装は temp WAV を介した単発確認用であり、リアルタイム最適化は未着手
- 48kHz / 1秒窓の単発推論で疎通確認済み
- 次フェーズでは Windows クライアントから短い音声チャンクを扱う前提で、バッファリング戦略と継続推論設計が必要

### フェーズ 3: Windows クライアント層の追加

- [x] `src/client/inference_client.py` を追加する
- [x] `src/client/audio_stream.py` にマイク入力とスピーカー出力を寄せる
- [x] `src/client/ring_buffer.py` を追加する
- [x] GUI に「推論サーバ (WSL)」接続パネルを追加する
- [x] `AudioController` に `connect_to_server()` / `_load_model_via_server()` を追加する
- [x] `scripts/smoke_test_client.py` を追加する

完了条件:

- [x] GUI からサーバへ接続できる
- [x] モデル一覧取得とモデル切替が GUI から行える

補足:

- `InferenceClient` は asyncio ループをバックグラウンドスレッドで実行し、Tkinter メインスレッドからの同期呼び出しをサポート
- `RingBuffer` はフェーズ 4 のストリーミング推論で使用するテープラインバッファ
- `AudioStream` はフェーズ 4 で sounddevice と InferenceClient を接続する層
- smoke test 結果: connect / health / list_models / load_model が全てパス（デバイス cuda:0、モデルロード ok=True）

### フェーズ 4: 既存 `AudioModel` の責務整理

- [x] [src/models/voice_model.py](src/models/voice_model.py#L23) から、推論本体の責務を外す
- [x] `process_audio()` では RPC クライアント呼び出しを行う
- [x] ローカルに残す処理は以下に限定する
  - 入出力ゲイン
  - ノイズ除去
  - フォルマント処理
  - バイパス
  - タイムアウト時フォールバック

完了条件:

- [x] GUI 側に fairseq / HuBERT / RVC checkpoint 依存が残らない

補足:

- `AudioModel` は `RVCModel` の import / load / convert を行わず、`InferenceClient` 注入方式に変更済み
- `AudioController.connect_to_server()` でクライアントを `AudioModel` へ注入し、切断時は `None` に戻す
- `scripts/smoke_test_phase4_audio_model.py` で `process_audio(normal)` の RPC 経路を確認済み
- 短チャンクのリアルタイム最適化はフェーズ6検証で継続（現状はタイムアウト時にローカルエフェクトへフォールバック）

### フェーズ 5: 設定分離

- [x] GUI ローカル設定と推論設定を別ファイルに分離する
- [x] GUI 側設定の例
  - 使用デバイス
  - ウィンドウ状態
  - ローカルエフェクトの初期値
- [x] 推論設定の例
  - モデル名
  - pitch_shift
  - f0_method
  - index_rate
  - protect

完了条件:

- [x] Windows 側だけで完結する設定と、WSL 側に送る設定が混在しない

補足:

- 追加ファイル:
  - `gui_local_settings.json`（GUIローカル設定）
  - `inference_settings.json`（推論設定）
  - `src/app/gui_local_settings.py`
  - `src/app/inference_runtime_settings.py`
  - `src/app/settings_loader.py`
- `src/app/gui.py` で設定をロードし、`AudioModel` と `AudioView` に注入する構成へ変更
- `AudioModel` は `get_current_inference_settings()` で推論設定のみを組み立て、GUIローカル設定とは分離
- `scripts/smoke_test_phase5_settings.py` で設定分離を確認済み

### フェーズ 6: 検証

- [x] サーバ単体のヘルスチェック
- [x] 単発ファイル推論
- [x] 音声チャンク 100 回連続推論
- [x] タイムアウト時のバイパス確認
- [x] モデル切替時のメモリ解放確認

完了条件:

- [x] GUI からのリアルタイム変換がタイムアウト時も破綻しない

補足:

- `scripts/smoke_test_phase6_validation.py` を追加し、フェーズ6の5項目を統合検証
- 実行結果:
  - `health` OK
  - 単発推論 OK
  - 100連続推論 OK（100/100）
  - timeout bypass OK（`AudioModel` がフォールバック出力を返す）
  - モデル再ロード安定性 OK
- メモリ確認は `/proc/<pid>/status` の `VmRSS` 差分で実測（今回 `rss_delta=323.6MB`）
- モデルが1つのみの環境では切替検証の代替として同一モデル再ロード反復で確認
- 実運用向けの初期チューニングを適用済み
  - `AudioModel.rvc_processing_timeout`: 0.30s → 0.18s（`gui_local_settings.json` で調整可能）
  - `AudioStream` リングバッファ: 入力/出力とも 2.0s → 0.5s（コンストラクタ引数で調整可能）
  - フェーズ6メモリ許容値: 1024MB → 640MB（`PHASE6_MEMORY_DELTA_LIMIT_MB` で上書き可能）

## 次回の優先作業

1. [x] フェーズ6結果を基に、実運用向けの閾値（timeout・バッファサイズ）を調整する
2. 必要ならモデル切替時のGPUメモリ計測を `nvidia-smi` ベースで追加する
3. [x] GUIの運用導線（接続失敗時リトライ/通知）を強化する

## 当面の運用ルール

1. オフライン変換は `./venv310/bin/python rvc_convert.py` を使う
2. WSL 上の GUI をリアルタイム音声用途では使わない
3. 実モデル推論の確認は `test/test_female_real*.wav` のように別名出力で残す
4. モデル本体・生成音声・テスト JSON は Git 管理対象にしない