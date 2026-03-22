# リアルタイムボイスチェンジャー

リアルタイムで声を変換・加工する Python アプリケーションです。

現在の推奨構成は以下です。

- Windows 側: GUI と音声入出力
- WSL 側: 推論サーバ (RVC / fairseq / HuBERT)

## 現在の実装概要

### リアルタイム GUI

- MVC 構成
- エフェクト処理
   - ピッチシフト
   - フォルマント
   - ノイズ除去
   - 入出力ゲイン
- RVC 変換
   - GUI から WSL 推論サーバへ WebSocket RPC で推論要求
   - タイムアウトや推論失敗時はローカル処理へフォールバック
   - 高速モード時は「間引き RPC + ローカル高速変換」のハイブリッド処理
- 推論サーバ接続パネル
   - 接続状態表示
   - 接続失敗時の自動リトライ
   - 失敗時の再試行導線と通知

### オフライン変換

- rvc_convert.py で rvc-python による実モデル推論に対応

## 推奨環境

- Python 3.10 (venv310)
- CUDA 対応 GPU (WSL 推論側)

## セットアップ

依存は以下の 3 ファイルに分割されています。

- requirements.txt
   - 共通依存 (Windows / WSL 共通)
- requirements-windows.txt
   - Windows GUI / リアルタイム音声 I/O 向け依存
- requirements-wsl.txt
   - WSL 推論サーバ / RVC 推論スタック向け依存

### Windows PowerShell 版 (GUI 実行環境)

`Activate.ps1` が実行ポリシーでブロックされる場合は、同じ PowerShell セッションで以下を先に実行してください。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```powershell
py -3.10 -m venv venv310
.\venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-windows.txt
```

### WSL bash 版 (推論サーバ実行環境)

```bash
python3.10 -m venv venv310
source venv310/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-wsl.txt
```

注意:

- rvc-python の依存により numpy は 1.23.5 系に固定されます
- 依存衝突を避けるため、推論専用の仮想環境運用を推奨します

## クイックスタート

### 1. WSL 推論サーバを起動

WSL bash 版:

```bash
bash scripts/start_wsl_server.sh
```

または:

```bash
./venv310/bin/python -m src.server.inference_server --host 127.0.0.1 --port 8765
```

### 2. GUI を起動

Windows PowerShell 版:

```powershell
.\venv310\Scripts\python.exe gui.py
```

WSL bash 版:

```bash
./venv310/bin/python gui.py
```

### 3. GUI で接続

1. 推論サーバ URL を確認 (既定: ws://127.0.0.1:8765/ws)
2. 接続ボタンを押す
3. 状態が接続済みになることを確認
4. RVC 有効化 + モデル選択
5. 開始ボタンでストリーム開始

## 設定ファイル

設定は以下の 2 ファイルに分離されています。

- gui_local_settings.json
   - GUI ローカル設定
   - 音声デバイス寄り設定
   - サーバ接続設定
- inference_settings.json
   - 推論サーバへ送る設定
   - model_name, pitch_shift, f0_method, index_rate, protect など

## 実運用向け閾値

### gui_local_settings.json

- rvc_processing_timeout_sec
   - RVC RPC 1 チャンクあたりのタイムアウト秒
   - 既定値: 0.18
- stream_input_buffer_seconds
   - 入力リングバッファ秒数
   - 既定値: 0.5
- stream_output_buffer_seconds
   - 出力リングバッファ秒数
   - 既定値: 0.5
- server_connect_retry_count
   - サーバ接続時の最大リトライ回数
   - 既定値: 3
- server_connect_retry_interval_sec
   - サーバ接続リトライ間隔秒
   - 既定値: 1.5
- server_connect_show_error_dialog
   - 接続失敗時ダイアログ表示
   - 既定値: true
- fast_mode_rpc_every_n_chunks
   - 高速モード時に RPC を実行する間隔（Nチャンクごと）
   - 既定値: 3
- fast_mode_rpc_timeout_sec
   - 高速モード時の RPC タイムアウト秒
   - 既定値: 0.12
- fast_mode_local_mix
   - 高速モード時のローカル高速変換の混合比 (0.0〜1.0)
   - 既定値: 0.35

### フェーズ6検証スクリプト

scripts/smoke_test_phase6_validation.py は以下の環境変数で閾値上書き可能です。

- PHASE6_ITERATIONS
- PHASE6_CONNECT_TIMEOUT_SEC
- PHASE6_LOAD_TIMEOUT_SEC
- PHASE6_INFER_TIMEOUT_SEC
- PHASE6_FORCED_TIMEOUT_SEC
- PHASE6_MEMORY_DELTA_LIMIT_MB

既定のメモリ許容値は 640 MB です。

## RVC モデル準備

- 学習済み .pth を src/models/rvc/ に配置
- 必要であれば対応する .index も同階層に配置

補足:

- GUI の「モデルダウンロード」ボタンによる事前学習モデル取得は現行構成では非対応です
- モデルは WSL 側に配置し、サーバ経由で利用してください

## 使用例

### オフライン変換

Windows PowerShell 版:

```powershell
.\venv310\Scripts\python.exe rvc_convert.py `
   -i test/test.wav `
   -o test/test_female_real.wav `
   -m "05 つくよみちゃん公式RVCモデル 弱" `
   -p 12
```

WSL bash 版:

```bash
./venv310/bin/python rvc_convert.py \
   -i test/test.wav \
   -o test/test_female_real.wav \
   -m "05 つくよみちゃん公式RVCモデル 弱" \
   -p 12
```

### デバイス確認

Windows PowerShell 版:

```powershell
.\venv310\Scripts\python.exe list_devices.py
```

WSL bash 版:

```bash
./venv310/bin/python list_devices.py
```

## 動作確認スクリプト

主要スモークテスト:

- scripts/smoke_test_protocol.py
- scripts/smoke_test_rpc.py
- scripts/smoke_test_client.py
- scripts/smoke_test_phase4_audio_model.py
- scripts/smoke_test_phase5_settings.py
- scripts/smoke_test_phase6_validation.py

詳細は scripts/README.md を参照してください。

## トラブルシューティング

### 接続失敗する

1. WSL 推論サーバが起動しているか確認
2. URL とポートが一致しているか確認
3. GUI の状態表示で再試行回数と失敗理由を確認

### WSL でデバイスが見えない

- WSL 環境では音声デバイス列挙ができない場合があります
- 音声 I/O が必要なリアルタイム運用は Windows ネイティブ実行を推奨します

## ライセンス

MIT

## 最終更新

2026-03-22
