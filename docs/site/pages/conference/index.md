# 原型と混合比から、埋め込み表現を理解する

Arc_nlpの研究課題、動かせる実行例、その実装を紹介します。

<div class="equation" role="math" aria-label="観測は原型の混合として表され、原型も訓練観測の混合から作られます">観測 ≈ 原型の混合<br>X ≈ AZ &nbsp; · &nbsp; Z = BX</div>

**研究課題：** 埋め込み空間を少数の原型で表すと、どのような構造や情報が保たれるか。

<div class="hero-actions"><a class="button-primary" href="../index.html">プロジェクトを見る →</a><a class="button-secondary" href="../tutorial.html">実行例を試す</a><a class="button-secondary" href="https://github.com/ryuuua/Arc_nlp_public">ソースを読む ↗</a><a class="button-secondary" href="../ja/index.html">日本語の概要</a></div>

## 原型分析の考え方

各観測を、非負で和が1になる重みを使って原型の混合として表します。原型も訓練観測の凸結合なので、両方の係数に幾何学的な意味があります。学習後は原型を固定し、新しい観測の混合比を推定します。

[数理モデルの解説](../mathematics.html)から、[APIと主要ソース](../api/index.html)へ進めます。

## 試せる機能

配布パッケージには、Standard AA v2・v3・v4、入力配列からの再現可能なモデル学習、保存状態を使った推論、各種評価指標が含まれます。小規模な実行例は、データセットのダウンロードやGPUなしで実行できます。互換性のため既定のソルバはv2を維持し、v3とv4は明示的に選択します。

より広い[研究の方向性](../research.html)と、このパッケージで利用できる機能は分けて紹介しています。
