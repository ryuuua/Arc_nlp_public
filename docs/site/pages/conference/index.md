<div class="poster-hero">
<p class="result-kicker">ポスター発表 · 補足資料</p>
<h1>アーキタイプ分析による文埋め込みの幾何学的構造の検討</h1>
<p class="author-line"><a href="profile.html">阿部龍之介</a><span>久保孝富，池田和司</span></p>
<p class="author-affiliation">奈良先端科学技術大学院大学</p>
<p class="hero-deck">文埋め込みを代表点の凸結合で近似し、混合係数とラベルの整合性、およびモデル間の凸混合関係を分析した。</p>
<p>混合係数とラベルの整合性は、線形分類性能と強い正の関連を示した。また、Qwen3とNemotronの比較では、混合係数から得た置換対応に沿って、代表点の方向も対応することが示唆された。このページでは、ポスターに掲載した結果の概要と、検証文書を用いた追加の可視化を紹介する。</p>
</div>
<nav class="companion-links" aria-label="補足資料と関連ページ">
<a class="companion-link research-link" href="#research-overview"><span class="link-category">01 · 研究内容</span><strong>発表内容・補足結果 <span aria-hidden="true">↓</span></strong><span>研究の背景、実験条件、結果と解釈</span></a>
<a class="companion-link" href="profile.html"><span class="link-category">02 · 発表者</span><strong>阿部龍之介 <span aria-hidden="true">→</span></strong><span>所属、研究関心、連絡先</span></a>
<a class="companion-link tool-link" href="https://fermatter.github.io/" target="_blank" rel="noopener noreferrer"><span class="link-category">03 · ソフトウェア ↗</span><strong><img class="fermatter-mark" src="../assets/conference/fermat-light.svg" alt="" width="30" height="30">FerMatter</strong><span>数式・図の編集と発表資料への出力</span></a>
</nav>
<nav class="section-jumps" aria-label="補足資料内の移動"><a href="#research-overview">背景・方法</a><a href="#label-results">ラベルとの関連</a><a href="#poster-reading">モデル間比較</a><a href="#cross-model-validation">検証文書の補足結果</a><a href="#poster-faq">結果の解釈</a></nav>

<section class="poster-context" aria-labelledby="research-overview">
<p class="result-kicker">01 · ポスター掲載内容</p>
<h2 id="research-overview">文埋め込みの幾何構造と凸混合関係</h2>
<p>線形同定可能性の理論を背景に、代表点の凸結合で表される点は、可逆線形変換後も同じ混合係数で記述できるという性質に着目した。アーキタイプ分析（Archetypal Analysis; AA）を用いて文埋め込みを代表点の凸結合で近似し、その混合係数を分析した。</p>
<p class="figure-note">線形同定可能性の理論は、分布の一致・同一次元・多様性などの条件を前提とする。本実験はモデル間の同定可能性そのものを証明するものではない。</p>
<div class="equation" role="math" aria-label="文埋め込みXを混合係数Aと代表点Zの積で近似する">X ≈ AZ &nbsp; · &nbsp; Z = BX<sup>(train)</sup></div>
<dl class="notation-guide"><div><dt>X · 文埋め込み</dt><dd>各文を表すベクトル</dd></div><div><dt>A · 混合係数</dt><dd>各文における代表点の混合割合</dd></div><div><dt>Z · 代表点</dt><dd>訓練データの凸包内に推定するK個の点</dd></div></dl>
<p>AとBの各行の要素を非負かつ総和1に制約し、再構成誤差が小さくなるように代表点と混合係数を求める。AAの学習にはクラスラベルを用いず、学習後に係数とラベルの関係を評価した。</p>
<div class="equation result-equation" role="math" aria-label="可逆線形変換の下で凸混合係数を保持する">X′ = XT ≈ (AZ)T = A(ZT) = AZ′</div>
<p class="figure-note">この式は、与えられた代表点と係数による凸混合関係の保存を表す。異なるモデルで独立に最適化したAAが、同じ代表点・混合係数を得ることを保証するものではない。</p>
<h3>実験条件</h3>
<div class="result-table-wrap"><table class="result-table"><caption>ポスター全体の実験設定</caption><tbody><tr><th scope="row">データセット</th><td>20 Newsgroups、AG News、Go Emotions、dair-ai/Emotion</td></tr><tr><th scope="row">モデル</th><td>埋め込みモデル4種・その他モデル4種の計8モデル</td></tr><tr><th scope="row">条件数</th><td>4データセット × 8モデル = 32条件</td></tr><tr><th scope="row">アーキタイプ数・初期化</th><td>K = 90、初期化10回</td></tr></tbody></table></div>
<details class="result-details"><summary>対象モデルと表現の抽出</summary><p>埋め込みモデルにはQwen3-Embedding-8B、llama-embed-nemotron-8b、EmbeddingGemma、Sentence-BERTを用いた。その他のモデルにはRoBERTa-base、Gemma-2-27B、Gemma-3-27B、DeepSeek-R1-Distill-Qwen-14Bを用いた。埋め込みモデルではエンコード出力を、その他のモデルでは最終層のmean poolingによる出力を表現として取得した。</p></details>
<p class="figure-note">以下に詳述するモデル間比較は、20 NewsgroupsのQwen3・Nemotron、K = 90、seed 0の一例である。初期化10回の平均値ではない。</p>
</section>

<section class="poster-context" aria-labelledby="label-results">
<h2 id="label-results">ラベルと代表点の対応および線形分類性能</h2>
<p>ラベル別の平均係数と正規化相互情報を用いて、混合係数とクラスラベルの整合性を評価した。また、訓練文書の係数とラベルを使って線形分類器を学習し、検証文書の係数からの予測をmacro-F1で評価した。</p>
<p><strong>代表点群がラベルに選択的に対応する条件ほど、混合係数による線形分類性能も高い傾向がみられた。</strong>埋め込みモデルでは、ラベルごとに異なる代表点群が選択的に利用されることが観察された。</p>
<p>この結果は、混合係数による単体上の表現に、文埋め込みのラベル情報が線形に読み出せる形で含まれていることを示唆する。アーキタイプ分析は、そのような情報がどのように組織化されているかを記述する補助的な分析となる。</p>
<p class="figure-note">本節はポスター掲載結果の要約である。以下のモデル間cosine類似度は、ラベル整合性や線形分類性能とは異なる評価である。</p>
</section>

<section class="poster-context" aria-labelledby="poster-reading">
<h2 id="poster-reading">探索的分析：モデル間の凸混合関係</h2>
<p>Qwen3とNemotronによる文埋め込みにそれぞれAAを適用した。推定した混合係数から代表点インデックスの対応Pを、対応する文埋め込みから線形写像Tを推定した。その後、混合係数Aと、Tを適用した後の対応する代表点Zを、それぞれcosine類似度で比較した。</p>
<ol class="method-steps"><li><strong>代表点の対応Pの推定。</strong>訓練文書（Train）の混合係数の列をcosine類似度で比較し、総和が最大となる一対一対応をHungarian法で求めた。</li><li><strong>文埋め込み間の線形写像Tの推定。</strong>対応する訓練文書の埋め込みから、直交プロクラステス変換に一様なスケールを含めたscaled orthogonal ProcrustesによりTを求めた。</li></ol>
<div class="equation result-equation" role="math" aria-label="混合係数はA QとA N P、代表点はZ Q TとP転置Z Nを比較する">A<sub>Q</sub> と A<sub>N</sub>P<br>Z<sub>Q</sub>T と P<sup>⊤</sup>Z<sub>N</sub></div>
<p class="figure-note">Q、NはそれぞれQwen3、Nemotronを表す。PはNemotronの代表点番号をQwen3に対応づける置換行列であり、係数の列と代表点の行を同じ対応で並べ替える。</p>
<p class="metric-caption">20 Newsgroups · K = 90 · 訓練文書から得た代表点Zの平均cosine類似度</p>
<div class="result-metrics"><div><span>① 元の番号同士</span><strong>−0.005</strong><small>Z<sub>Q</sub> と Z<sub>N</sub></small></div><div><span>② 代表点の対応Pのみ</span><strong>−0.003</strong><small>Z<sub>Q</sub> と P<sup>⊤</sup>Z<sub>N</sub></small></div><div class="metric-emphasis"><span>③ 線形写像T＋対応P</span><strong>0.854</strong><small>Z<sub>Q</sub>T と P<sup>⊤</sup>Z<sub>N</sub></small></div></div>
<p>代表点の対応を揃えた混合係数の平均cosine類似度は0.687であった。代表点の平均cosine類似度は、元の番号同士では−0.005、置換Pのみでは−0.003であり、同じ対応を保って線形写像Tを適用すると0.854となった。<strong>この結果は、係数から得た対応に沿って、変換後の代表点の方向も類似することを示唆する。</strong></p>
<details class="result-details" open><summary>代表点のcosine類似度：元の番号／対応Pのみ／線形写像T＋対応P</summary><a href="../assets/conference/20newsgroups_k90_raw_P_TP_cork.svg" target="_blank" rel="noopener"><img class="full-heatmap" src="../assets/conference/20newsgroups_k90_raw_P_TP_cork.svg" alt="K=90の代表点Zのcosine行列。元の番号、Pのみ、TとP適用後。対角平均は−0.005、−0.003、0.854。" loading="lazy"></a><p>縦軸はQwen3、横軸はNemotronの代表点を表す。②と③では、対角が訓練係数から求めた対応に一致する。代表点Zは訓練文書から推定した辞書であり、本図は検証文書そのものの評価ではない。</p><p><a href="../assets/conference/20newsgroups_k90_raw_P_TP_cork.svg" download>代表点Zの3連SVGを保存</a></p></details>
<p class="figure-note">係数Aの0.687は検証文書（Validation）での値である。一方、代表点Zの0.854は訓練文書から得た90組の代表点の比較であり、評価対象が異なる。</p>
<div class="reading-bridge"><strong>追加分析：検証文書における対応</strong><p>訓練時に推定したPとTを固定し、検証文書の混合係数Aと文埋め込みXを比較した。以下に、保存済み結果から作成した補足図を示す。</p></div>
</section>

<section class="research-result" aria-labelledby="cross-model-validation">
<p class="result-kicker">02 · 補足結果 · 20 Newsgroups · K = 90 · seed 0</p>
<h2 id="cross-model-validation">検証文書における混合係数と文埋め込みの対応</h2>
<p class="result-lead">訓練文書から求めたPとTを固定した条件で、検証文書においても<strong>混合係数と文埋め込みの方向の対応</strong>が観察された。</p>
<p>ポスターに掲載した代表点の比較を補足するため、検証文書の係数と埋め込みを比較した。訓練文書から推定した代表点Zを固定し、検証文書の混合係数Aを用いた。以下はK = 90の実験で保存された結果である。</p>
<div class="result-scope"><span>Qwen3 → Llama Embed Nemotron 8B</span><span>訓練文書：16,019</span><span>検証文書：2,827</span></div>
<p class="metric-caption">検証文書2,827件 · 同一文書の平均cosine類似度</p>
<div class="result-metrics"><div><span>① 代表点番号を揃えた混合係数A</span><strong>0.687</strong><small>訓練文書から求めたPを適用</small></div><div><span>② 変換前の文埋め込みX</span><strong>−0.002</strong><small>元の座標における参考値</small></div><div class="metric-emphasis"><span>③ 変換後の文埋め込みXT</span><strong>0.790</strong><small>訓練文書から求めたTを適用</small></div></div>
<h3 id="validation-heatmaps">検証文書間のcosine類似度行列</h3>
<p>縦軸はQwen3、横軸はNemotronの検証文書を表す。3つの図では共通の文書順を用いており、<strong>対角は同一文書、非対角は異なる行の比較</strong>に対応する。①の混合係数と③の変換後の埋め込みでは、対角に高いcosine類似度が観察された。</p>
<p class="figure-note">可視化には共通の200文書を等間隔に抽出した。類似度に基づく選別・並べ替えは行っていない。配色はcork、全図共通の線形スケール（−1〜1）を用いた。</p>
<div class="heatmap-grid"><figure class="heatmap-panel"><h4>① 混合係数の対応（P適用後）</h4><a href="../assets/conference/validation_a_aligned_cork_transparent.svg" target="_blank" rel="noopener" aria-label="① 混合係数の対応（P適用後）の図を拡大"><img src="../assets/conference/validation_a_aligned_cork_transparent.svg" alt="① 混合係数の対応（P適用後）。200文書のcosine行列。対角平均は0.674。" width="696" height="576"></a><div class="panel-formula">cos(a<sub>Q,i</sub>, a<sup>π</sup><sub>N,j</sub>)</div><figcaption>代表点の番号を揃えた混合係数の比較。<br><span>表示200文書の対角平均：<strong>0.674</strong></span></figcaption></figure><figure class="heatmap-panel"><h4>② 文埋め込み・変換前</h4><a href="../assets/conference/validation_x_raw_cork_transparent.svg" target="_blank" rel="noopener" aria-label="② 文埋め込み・変換前の図を拡大"><img src="../assets/conference/validation_x_raw_cork_transparent.svg" alt="② 文埋め込み・変換前。200文書のcosine行列。対角平均は−0.003。" width="696" height="576"></a><div class="panel-formula">cos(x<sub>Q,i</sub>, x<sub>N,j</sub>)</div><figcaption>文埋め込みの元の座標における比較。<br><span>表示200文書の対角平均：<strong>−0.003</strong></span></figcaption></figure><figure class="heatmap-panel"><h4>③ 文埋め込み・変換後</h4><a href="../assets/conference/validation_x_transformed_cork_transparent.svg" target="_blank" rel="noopener" aria-label="③ 文埋め込み・変換後の図を拡大"><img src="../assets/conference/validation_x_transformed_cork_transparent.svg" alt="③ 文埋め込み・変換後。200文書のcosine行列。対角平均は0.789。" width="696" height="576"></a><div class="panel-formula">cos(x<sub>Q,i</sub>T, x<sub>N,j</sub>)</div><figcaption>訓練文書から求めた線形写像Tを適用。<br><span>表示200文書の対角平均：<strong>0.789</strong></span></figcaption></figure></div>
<p class="figure-note">各図をクリックすると拡大表示できる。目盛りの1〜200は表示順を表す。Pは混合係数Aの代表点列に適用する置換であり、文埋め込みXには適用しない。</p>
<details class="result-details"><summary>全2,827文書のcosine類似度行列</summary><p>全行・全列を用いた2,827 × 2,827の行列を示す。縮小表示では対角線が見えにくいため、下記の数値を併記した。</p><a href="../assets/conference/validation_full_heatmaps_cork.svg" target="_blank" rel="noopener"><img class="full-heatmap" loading="lazy" src="../assets/conference/validation_full_heatmaps_cork.svg" alt="全検証2,827文書の混合係数対応、変換前X、変換後Xの3連ヒートマップ。対角平均は0.687、−0.002、0.790。"></a><p><a href="../assets/conference/validation_full_heatmaps_cork.svg" download>全体版SVGを保存</a> · <a href="../assets/conference/validation_three_heatmaps_cork.svg" download>200文書の3連SVGを保存</a></p></details>
<h3 id="validation-numbers">同一文書と異なる行の比較</h3>
<div class="result-table-wrap" tabindex="0" role="region" aria-label="全検証文書のcosine類似度"><table class="result-table"><caption>全2,827文書の平均cosine類似度</caption><thead><tr><th scope="col">比較対象</th><th scope="col">同一文書<br>対角平均</th><th scope="col">異なる行<br>非対角平均</th></tr></thead><tbody><tr><th scope="row">① 混合係数の対応（P適用後）</th><td>0.687</td><td>0.041</td></tr><tr><th scope="row">② 文埋め込み・変換前</th><td>−0.002</td><td>−0.002</td></tr><tr><th scope="row">③ 文埋め込み・変換後</th><td>0.790</td><td>0.225</td></tr></tbody></table></div>
<p>変換後の文埋め込みでは、同一文書の平均cosine類似度は<strong>0.790</strong>、非対角の平均は<strong>0.225</strong>であった。共通方向への偏りに加え、同一文書に沿った方向の対応が示唆された。非対角は異なる行の比較であり、同一本文を除外した値ではない。この対角・非対角平均の差に対する有意差検定は行っていない。</p>
<h3 id="validation-method">推定手順と検証時の固定条件</h3>
<ol class="method-steps"><li><strong>代表点と混合係数の推定。</strong>各モデルの訓練文書にAAを適用し、90個の代表点を求めた。検証文書では代表点Zを固定し、混合係数Aを推定した。</li><li><strong>代表点の対応Pの推定。</strong>訓練文書から決定的に抽出した2,048件の混合係数を用い、列間cosine類似度の総和を最大化する一対一対応をHungarian法で求めた。同じPを検証係数の列に適用した。</li><li><strong>線形写像Tの推定。</strong>対応する訓練文書の埋め込みから、scaled orthogonal Procrustesにより回転・反転と一様なスケールを求めた。中心化および切片は用いていない。</li><li><strong>検証文書における評価。</strong>PとTを固定し、混合係数および変換前後の文埋め込みのcosine類似度を算出した。検証文書に合わせたP・Tの再推定は行っていない。</li></ol>
<div class="equation result-equation" role="math" aria-label="訓練文書から推定した線形写像Tを検証文書に適用する">訓練：X<sub>Q</sub><sup>(train)</sup>T ≈ X<sub>N</sub><sup>(train)</sup><br>検証：cos(x<sub>Q,i</sub><sup>(valid)</sup>T, x<sub>N,i</sub><sup>(valid)</sup>)</div>
<p>cosine類似度はベクトルの<strong>方向</strong>を評価する。変換後のXの相対誤差は<strong>0.614</strong>であり、座標が完全に一致していることを示すものではない。変換前のcosine類似度は、モデルごとに異なる座標軸をそのまま比較した参考値である。</p>
<aside class="scope-note" aria-labelledby="interpretation-scope"><h3 id="interpretation-scope">評価の範囲と制約</h3><p>本補足は<strong>K = 90・seed 0の保存済み結果を用いた、既閲覧の検証文書に対する探索的評価</strong>である。モデル対とデータセットは対応を観測した後に選択した。独立した最終Testによる確認や、別モデル・別seedへの一般化を示すものではない。訓練・検証間の本文重複は今回再検査していない。</p><p>検証文書のA・X、訓練文書から推定した代表点Z、修正版Nemotronキャッシュを用いたK = 128実験は、それぞれ評価対象・条件が異なる。</p></aside>
<details class="result-details"><summary>集計値と可視化の来歴</summary><p>保存済みのA・X・P・Tからcosine類似度を算出した。入力ハッシュ、全対角と固定セルの検算、過去の平均値の再現を確認済みである。本可視化に伴うAAの再学習、係数の再推論、P・Tの再推定、Test評価は行っていない。</p><p><a href="../assets/conference/summary.json">集計値（JSON）</a> · <a href="../assets/conference/cmcrameri_LICENSE.txt">配色corkのライセンス</a></p></details>
</section>

<section class="poster-faq" aria-labelledby="poster-faq">
<p class="result-kicker">補足 · 結果の解釈</p>
<h2 id="poster-faq">対応の評価対象と解釈の範囲</h2>
<details class="result-details"><summary>代表点と検証文書の評価対象</summary><p>平均cosine類似度<strong>0.854は、訓練文書（Train）から推定した90個の代表点Z</strong>の比較である。一方、<strong>0.790は、検証文書（Validation）2,827件の文埋め込みX</strong>の比較である。混合係数Aの0.687も検証文書に対する評価値であり、代表点の評価値とは対象が異なる。</p></details>
<details class="result-details"><summary>混合係数の対応と代表点の対応</summary><p>混合係数Aの類似から、代表点Zの一致が自動的に導かれるわけではない。同じ混合係数で記述できても、代表点の位置は異なり得る。本分析では、混合係数の対応、線形写像適用後の代表点の方向、検証文書の文埋め込みの対応をそれぞれ評価した。</p></details>
<details class="result-details"><summary>方向の類似と座標の一致</summary><p>cosine類似度はベクトルの方向を測る指標であり、長さが異なる場合にも1となり得る。検証文書の文埋め込みXでは、線形写像適用後の相対誤差は0.614であった。したがって、本結果は方向の対応を示すものであり、座標の完全な一致を示すものではない。</p></details>
<details class="result-details"><summary>訓練文書による推定と検証文書による評価</summary><p>対応Pと線形写像Tは訓練文書で推定し、検証文書では固定した。ただし、検証結果は既に閲覧済みであり、モデル対とデータセットは対応を観測した後に選択した。本分析は探索的評価であり、独立した最終Testによる確認ではない。</p></details>
<details class="result-details"><summary>共通方向への偏りと対応点の類似</summary><p>検証文書の文埋め込みXでは、同一文書の平均cosine類似度は0.790、異なる行の平均は0.225であった。代表点Zでは、線形写像適用後の対応点の平均は0.854、ランダムな代表点対応の期待値は0.380、その差は0.473であった。共通方向への偏りに加え、対応点に沿った類似が観察された。ただし、これらの平均差を統計的有意差の検定結果として扱うものではない。</p></details>
</section>

<section class="further-reading" aria-labelledby="poster-future">
<h2 id="poster-future">課題と今後の発展</h2>
<ul>
<li><strong>推定の安定性：</strong>代表点数・初期化の調整、および球面正規化を考慮した誤差の設定を検討する。</li>
<li><strong>モデル間対応：</strong>理論的な保証とともに、対応の再現性と一般化を検証する。</li>
<li><strong>代表点と混合係数の解釈：</strong>データ点を代表点に用いるアーキタイポイド分析を検討する。</li>
</ul>
</section>

<section class="further-reading" aria-labelledby="further-reading">
<h2 id="further-reading">実装例・関連情報</h2>
<p>プロジェクトの概要、数理モデルの解説、実装例やAPIを紹介しています。</p>
<div class="hero-actions"><a class="button-secondary" href="../ja/index.html">プロジェクト概要</a><a class="button-primary" href="../ja/mathematics.html">数理モデル</a><a class="button-secondary" href="../tutorial.html">実行例</a><a class="button-secondary" href="../api/index.html">API・主要ソース</a></div>
</section>

<section class="related-work" aria-labelledby="related-work">
<h2 id="related-work">発表者・関連ソフトウェア</h2>
<div class="related-columns"><div><h3>阿部龍之介 <small>Ryunosuke Abe</small></h3><p>自然言語処理・神経科学・表現学習に関心があります。学部では文系分野を学び、修士課程から情報系に進みました！</p><a href="profile.html">プロフィールを見る →</a></div><div><h3><img class="fermatter-mark" src="../assets/conference/fermat-light.svg" alt="" width="32" height="32">FerMatter</h3><p>LaTeX / Typst、TikZ / CeTZを扱い、SVG・PDF・画像を出力できるツールです。こちらもぜひご覧ください！</p><a href="https://fermatter.github.io/" target="_blank" rel="noopener noreferrer">FerMatter公式サイト ↗</a></div></div>
</section>
