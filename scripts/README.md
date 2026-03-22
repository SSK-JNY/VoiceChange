# scripts/

各フェーズの動作確認に使うスクリプト群です。  
すべてプロジェクトルート（`VoiceChange/`）から実行してください。

## 依存関係の分割方針

このリポジトリでは依存を以下の 3 ファイルに分割しています。

- `requirements.txt`: 共通依存（Windows / WSL 共通）
- `requirements-windows.txt`: Windows GUI / 音声I/O 用依存
- `requirements-wsl.txt`: WSL 推論サーバ / RVC 推論スタック用依存

### インストール例

Windows PowerShell 版:

`Activate.ps1` が実行ポリシーでブロックされる場合は、同じ PowerShell セッションで以下を先に実行してください。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```powershell
python -m pip install -r requirements-windows.txt
```

WSL bash 版:

```bash
python -m pip install -r requirements-wsl.txt
```

---

## スクリプト一覧

| ファイル | 対象フェーズ | 必要なもの | 実行時間 |
|---|---|---|---|
| `smoke_test_protocol.py` | Phase 1（プロトコル層） | 共通依存（`requirements.txt`） | 数秒 |
| `smoke_test_rpc.py` | Phase 2（WSL 推論サーバ） | WSL依存（`requirements-wsl.txt`）+ GPU + RVC モデル | 数十秒〜数分 |
| `smoke_test_client.py` | Phase 3（Windows クライアント層） | WSL依存（`requirements-wsl.txt`）+ GPU + RVC モデル | 数十秒〜数分 |
| `smoke_test_phase4_audio_model.py` | Phase 4（AudioModel責務整理） | WSL依存（`requirements-wsl.txt`）+ GPU + RVC モデル | 数十秒〜数分 |
| `smoke_test_phase5_settings.py` | Phase 5（設定分離） | 共通依存（`requirements.txt`） | 数秒 |
| `smoke_test_phase6_validation.py` | Phase 6（検証） | WSL依存（`requirements-wsl.txt`）+ GPU + RVC モデル | 数十秒〜数分 |
| `start_wsl_server.sh` | Phase 2 以降（サーバ起動） | WSL依存（`requirements-wsl.txt`）+ GPU + RVC モデル | — |

---

## smoke_test_protocol.py — Phase 1

`src/protocol/` の全 13 メッセージ型について、**サーバや GPU 不要**でシリアライズ／デシリアライズのラウンドトリップを検証します。

### 実行方法

Windows PowerShell 版:

```powershell
.\venv310\Scripts\python.exe scripts/smoke_test_protocol.py
```

WSL bash 版:

```bash
./venv310/bin/python scripts/smoke_test_protocol.py
```

### 期待される出力

```
=== Phase 1 smoke test: protocol layer ===

[ok] hello
[ok] hello_ack
[ok] health
[ok] health_result
[ok] list_models
[ok] list_models_result
[ok] load_model
[ok] load_model_result
[ok] update_params
[ok] update_params_result
[ok] infer_chunk
[ok] infer_chunk_result
[ok] error

=== All 13 message types passed roundtrip check ===
```

### 確認内容

- 全メッセージ型が `src.protocol` からインポートできる
- `serialize_message()` で dict に変換できる
- JSON 往復後に `deserialize_message()` で元の型に戻せる

---

## smoke_test_rpc.py — Phase 2

FastAPI の **インプロセス WebSocket テストクライアント**を使い、サーバを別プロセスで起動せずに以下の RPC シーケンスを一括検証します。

```
hello → health → list_models → load_model → infer_chunk（PCM バイナリ送受信）
```

### 前提条件

| 要件 | 詳細 |
|---|---|
| Python 環境 | `venv310`（rvc-python, fastapi, uvicorn, soundfile, numpy 等がインストール済み） |
| GPU | CUDA 対応 GPU（RTX 4060 Ti 等）が利用可能であること |
| テスト音声 | `test/test.wav`（モノラルまたはステレオ、サンプルレート任意） |
| RVC モデル | `src/models/rvc/` 以下に `.pth` ファイルが配置済みであること |

### 実行方法

Windows PowerShell 版:

```powershell
.\venv310\Scripts\python.exe scripts/smoke_test_rpc.py
```

WSL bash 版:

```bash
./venv310/bin/python scripts/smoke_test_rpc.py
```

スクリプト内の `model_name` 変数を使用するモデル名に合わせて変更してください（デフォルト: `05 つくよみちゃん公式RVCモデル 弱`）。

```python
# scripts/smoke_test_rpc.py 内
model_name = "05 つくよみちゃん公式RVCモデル 弱"  # ← ここを変更
```

### 期待される出力

```
{'type': 'hello_ack', 'protocol_version': 1, 'server': 'wsl-inference', ...}
{'type': 'health_result', 'ok': True, 'device': 'cuda:0', 'model_loaded': False, 'active_model': ''}
{'type': 'list_models_result', 'models': [{'name': '05 つくよみちゃん公式RVCモデル 弱', ...}]}
{'type': 'load_model_result', 'ok': True, 'active_model': '05 つくよみちゃん公式RVCモデル 弱', 'device': 'cuda:0'}
{'type': 'infer_chunk_result', 'sequence': 1, 'sample_rate': 48000, 'channels': 1, 'dtype': 'float32', 'frame_count': 48000, 'processing_ms': ..., 'fallback': False}
audio_bytes=192000
```

### 注意点

- `infer_chunk` は 1 秒（48000 サンプル）単位で送信しています。rvc-python の推論ウィンドウ要件のため、60 ms 程度の短いチャンクは失敗します
- 初回モデルロードは fairseq / HuBERT / RMVPE の初期化で数十秒かかります
- PyTorch 2.6+ 向けに `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` を自動設定しています

---

## smoke_test_client.py — Phase 3

`src/client/inference_client.py` の同期 API を使って、以下の RPC シーケンスを検証します。

```
connect → health → list_models → load_model → disconnect
```

このスクリプトは内部でテスト用サーバ（`uvicorn`）を起動し、クライアント接続を確認します。

### 実行方法

Windows PowerShell 版:

```powershell
.\venv310\Scripts\python.exe scripts/smoke_test_client.py
```

WSL bash 版:

```bash
./venv310/bin/python scripts/smoke_test_client.py
```

### 期待される出力

```
=== Phase 3 smoke test: InferenceClient ===
[ok] connect()
[ok] is_connected
[ok] health() ...
[ok] list_models() returns list
[ok] list_models() not empty
[ok] load_model('...')
[ok] disconnect() → is_connected==False
=== Phase 3 smoke test passed ===
```

### 確認内容

- InferenceClient で WSL 推論サーバへ接続できる
- `health` / `list_models` / `load_model` の基本 RPC が正常に返る
- 切断処理（`disconnect`）後に接続状態が False になる

---

## smoke_test_phase4_audio_model.py — Phase 4

`src/models/voice_model.py` の `process_audio()` が、
RVC有効時にローカル `RVCModel` ではなく `InferenceClient.infer_chunk()` を使う経路へ
切り替わっていることを検証します。

### 実行方法

Windows PowerShell 版:

```powershell
.\venv310\Scripts\python.exe scripts/smoke_test_phase4_audio_model.py
```

WSL bash 版:

```bash
./venv310/bin/python scripts/smoke_test_phase4_audio_model.py
```

### 期待される出力

```
=== Phase 4 smoke test: AudioModel RPC path ===
[ok] connect
[ok] list_models not empty
[ok] load_model - ...
[ok] output shape - (48000, 1)
[ok] output not all zeros
[ok] disconnect
=== Phase 4 smoke test passed ===
```

### 確認内容

- `AudioModel` に `InferenceClient` を注入して推論を実行できる
- `process_audio(normal)` で RPC 推論結果を出力に反映できる
- 失敗時フォールバック経路を残したまま、RPC経路が有効である

---

## smoke_test_phase5_settings.py — Phase 5

設定分離の実装（`gui_local_settings.json` と `inference_settings.json`）が、
`AudioModel` 上で混在せずに反映されることを検証します。

### 実行方法

Windows PowerShell 版:

```powershell
.\venv310\Scripts\python.exe scripts/smoke_test_phase5_settings.py
```

WSL bash 版:

```bash
./venv310/bin/python scripts/smoke_test_phase5_settings.py
```

### 期待される出力

```
=== Phase 5 smoke test: settings separation ===
[ok] gui settings loaded
[ok] inference settings loaded
[ok] samplerate from GUI settings
[ok] blocksize from GUI settings
[ok] local pitch from GUI settings
[ok] inference model name separated
[ok] inference pitch separated
[ok] inference f0_method kept
[ok] inference backend kept
=== Phase 5 smoke test passed ===
```

### 確認内容

- GUI ローカル設定（サンプルレート、ブロックサイズ、UI初期値）が `AudioModel` に反映される
- 推論設定（model/pitch/f0/backend）が別系統として保持される
- 推論設定を更新しても GUI ローカル設定とは独立して扱われる

---

## smoke_test_phase6_validation.py — Phase 6

フェーズ6の完了条件に対応する統合検証スクリプトです。

### 実行方法

Windows PowerShell 版:

```powershell
.\venv310\Scripts\python.exe scripts/smoke_test_phase6_validation.py
```

WSL bash 版:

```bash
./venv310/bin/python scripts/smoke_test_phase6_validation.py
```

### 検証内容

- サーバ単体のヘルスチェック（HTTP `/health`）
- 単発ファイル推論（`infer_chunk`）
- 音声チャンク 100 回連続推論
- タイムアウト時のバイパス確認（`AudioModel.process_audio()` のフォールバック）
- モデル切替/再ロード時の安定性とメモリ増加量の確認

### 備考

- モデルが1つしかない環境では、モデル切替の代わりに同一モデル再ロードを複数回実行して確認します
- メモリ確認は `/proc/<pid>/status` の `VmRSS` 差分による実測です
- デフォルト閾値（実運用向け）:
	- 推論タイムアウト: `12.0` 秒
	- タイムアウトフォールバック誘発: `0.02` 秒
	- メモリ増分許容: `640` MB
- 環境変数で上書き可能:
	- `PHASE6_ITERATIONS`
	- `PHASE6_CONNECT_TIMEOUT_SEC`
	- `PHASE6_LOAD_TIMEOUT_SEC`
	- `PHASE6_INFER_TIMEOUT_SEC`
	- `PHASE6_FORCED_TIMEOUT_SEC`
	- `PHASE6_MEMORY_DELTA_LIMIT_MB`

---

## start_wsl_server.sh — WSL 推論サーバ起動

フェーズ 2 以降で Windows 側クライアントから接続するためのサーバを起動します。  
WebSocket エンドポイント: `ws://127.0.0.1:8765/ws`

### 実行方法

Windows PowerShell 版:

```powershell
wsl bash scripts/start_wsl_server.sh
```

WSL bash 版:

```bash
bash scripts/start_wsl_server.sh
```

または直接 uvicorn を起動する場合:

Windows PowerShell 版:

```powershell
wsl ./venv310/bin/python -m src.server.inference_server --host 127.0.0.1 --port 8765
```

WSL bash 版:

```bash
./venv310/bin/python -m src.server.inference_server --host 127.0.0.1 --port 8765
```

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--host` | `127.0.0.1` | バインドする IP アドレス |
| `--port` | `8765` | ポート番号 |

---

## フェーズ対応表

```
Phase 1  プロトコル層          → smoke_test_protocol.py  ✅ 完了
Phase 2  WSL 推論サーバ        → smoke_test_rpc.py       ✅ 完了
Phase 3  Windows クライアント層 → smoke_test_client.py    ✅ 完了
Phase 4  AudioModel 責務整理   → smoke_test_phase4_audio_model.py ✅ 完了
Phase 5  設定分離              → smoke_test_phase5_settings.py ✅ 完了
Phase 6  検証                  → smoke_test_phase6_validation.py ✅ 完了
```

詳細は [docs/CURRENT_ISSUES.md](../docs/CURRENT_ISSUES.md) を参照してください。
