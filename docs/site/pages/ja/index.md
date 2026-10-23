# 埋め込み表現を、原型と混合比から理解する

Arc_nlpは、埋め込みを原型分析（Archetypal Analysis; AA）で分解し、再構成・係数の使われ方・ラベルとの関係を調べる研究プロジェクトです。このサイトは配布用パッケージ `neureps_arc_nlp` の実装と、より広い研究の方向性を紹介します。

<div class="equation">X ≈ AZ, &nbsp; Z = BX</div>

Xは元の埋め込み、Zは原型、Aは各観測の混合比です。Bは、どの訓練観測を混ぜて原型を作ったかを表します。

## はじめる

| 目的 | ページ |
|---|---|
| 小さな合成データで動かす | [導入](../getting-started.html)・[最初のAA](../tutorial.html) |
| 数学を理解する | [日本語の数理解説](mathematics.html) |
| v2・v3・v4を選ぶ | [ソルバと診断](../solvers.html) |
| 保存結果から係数を推定する | [SavedFit](../api/artifacts.html#SavedFit) |
| ソースを読む | [構成](../architecture.html)・[APIとソース](../api/index.html) |
| 発表のQRから来た | [発表用入口](../conference/index.html) |

## 配布される機能

Standard AAのv2・v3・v4、配列入力からのfit、保存状態を使った係数推定、再構成・係数使用・ラベル指標を含みます。既存結果の再現性のため既定値はv2を維持し、`solver=standard_v3` と `solver=standard_v4` を明示して追加の実装を選びます。

v4はfloat64を要し、停止条件の近似診断や追加探索を提供します。診断やrefinementで計算量が増える場合があり、一般的な高速化・優位性を保証するものではありません。

## 研究と実装範囲

Robust／Scalable AA、Mahalanobis AA、near-convex拡張、球面辞書、モデル間対応などは、より広い研究プログラムの方向性です。この配布用パッケージに、それらの実行APIが含まれるという意味ではありません。[研究方向](../research.html)で区別しています。

## 結果を読むとき

実行完了、停留条件の通過、大域最適性は別です。前処理と原型はTrainで学習し、Validationで設定を選び、Testの閲覧履歴を保持します。分類性能の有意差が出ないことも同等性の証明ではありません。
