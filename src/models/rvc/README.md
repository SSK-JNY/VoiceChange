# RVCモデル配置ディレクトリ

#

# このディレクトリに以下のファイルを配置してください：

#

# 1. 学習済みRVCモデルファイル (.pth)

# - 例: my_voice_model.pth

# - Hugging FaceやRVCコミュニティから入手

#

# 2. 対応する設定ファイル (.json)

# - 例: my_voice_model.json

# - モデルと同じ名前で配置

#

# 3. 事前学習モデル (初回ダウンロード時に自動取得)

# - hubert_base.pt (HuBERT特徴抽出)

# - rmvpe.pt (RMVPEピッチ抽出)

#

# 配置例:

# ├── my_voice_model.pth

# ├── my_voice_model.json

# ├── hubert_base.pt

# └── rmvpe.pt

#

# テスト実行: python test_rvc.py
