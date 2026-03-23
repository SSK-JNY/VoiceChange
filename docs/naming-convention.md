# Naming Convention

## 目的

ドキュメント名の表記ゆれを防ぎ、検索性と運用性を高める。

## 適用対象

- docs 配下の .md ファイル

## 基本ルール

1. 文字種は小文字英数字とハイフンのみを使う
2. 区切りはハイフンを使う
3. 単語は kebab-case で連結する
4. 拡張子は .md 固定
5. 日本語、空白、アンダースコア、大文字は使わない

## パターン

1. 恒久ドキュメント

- 形式: topic-name.md
- 例: current-issues.md, rpc-protocol.md

2. 日付付き作業ログ・レポート

- 形式: yyyy-mm-dd-topic-name.md
- 例: 2026-03-23-worklog-and-improvement-proposals.md

3. 単発調査レポート

- 形式: yyyy-mm-dd-topic-name.md
- 例: 2026-03-22-blocksize-stability-report.md

## 禁止例

- CURRENT_ISSUES.md
- 2026-03-23\_作業ログと改善提案.md
- BlockSizeReport.md

## 命名の粒度

1. 先頭はファイルの分類を示す語を置く
2. topic-name は 3-8 語程度に収める
3. 内容変更で主題が変わったらファイル名も更新する

## リネーム時の運用

1. 参照リンクを同時更新する
2. 同日に複数版を作る場合は末尾に -v2, -v3 を付ける
3. 既存ルールに合わない新規ファイルは作成時にこの規約へ合わせる
