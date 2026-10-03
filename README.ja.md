# 育児川柳 Transformer 学習ラボ（WASM）

ブラウザ内で実際に学習する、小さなデコーダー専用 Transformer の可視化教材です。
学習データは画面上部の「データ」で、育児川柳（5,000件）と SF 起承転結掌編（5,000話）から選べます。
外部通信・外部ライブラリは使用しません。

## そのまま使う

`dist/index.html` を現行の Chrome / Edge / Firefox / Safari で開き、
「学習開始」を押してください。WebAssembly SIMD、Web Worker、
DecompressionStream に対応するブラウザが必要です。
HTML はデータ・WASM・コードを埋め込んだ単一ファイルです。
タブを閉じると学習状態は失われます。

## 再ビルド

Python 3.9 以降を使用します。pip / npm のインストール、ネットワーク接続、
C/C++ コンパイラーは不要です。リポジトリのルートで実行してください。

```sh
python3 build-wasm.py
python3 package.py
```

生成物は `dist/index.html` と `dist/transformer-lab.zip` です。
`combined*.js` はビルド時に生成する、テスト用の結合コードです。

## 検証

Node.js 22 以降を使用します。先に再ビルドしてください。

```sh
node test-engine.cjs
node test-page.cjs
node test-completion.cjs
```

- エンジン：独立した JavaScript 推論との一致、数値微分による勾配確認、実学習、全15構成。
- ページ：DOM を模擬し、実際の Worker と WASM で操作・生成・停止・復元を検証。
- 完走：4,286件を4周学習し、追加学習と停止を検証。

ページの検証は実ブラウザでの外観確認とは異なります。
`all-themes-verification.json` はデータ分割変更時の検証結果です。

## ファイル構成

| ファイル | 内容 |
| --- | --- |
| `build-wasm.py` | 13個の数値演算カーネルを直接 WASM バイナリとして生成 |
| `engine.js` | Transformer、逆伝播、Adam の実行・メモリー管理 |
| `train-worker.js` | ミニバッチ学習、評価、チェックポイント復元 |
| `decoder.js` / `beam.js` / `infer-worker.js` | 可視化用の推論と生成候補5件の探索 |
| `live-loader.js` / `live-app.js` / `details.js` | 起動、画面操作、ヒートマップ、セル詳細 |
| `live-page.html` / `style.css` | HTML テンプレートと画面のスタイル |
| `corpus-common.json` | 本文・読み・テーマ・文字語彙・トークン・分割情報を含む5,000件 |
| `split-all-themes.py` | 全14テーマを含む学習／評価分割を再現 |
| `generate-sf-story.py` / `corpus-sf.json` | SF起承転結掌編5,000話の生成器と生成結果 |
| `datasets.json` | 画面で選べるデータセットの一覧（表示名・説明・CSV名） |
| `package.py` | 単一HTMLとそのZIPを作成 |
| `test-*.cjs` | 検証コード |

## データと評価

データは教材用の生成作品5,000件、公開作品0件です。
漢字かな交じりの本文、読み、5・7・5の音数情報を収録しています。
14場面・350種類の候補句を組み合わせたもので、不自然な表現も含みます。

各テーマ51件、計714件を評価用に残し、4,286件を学習します。
同じテーマで「上五＋中七」が一致する作品群を同じ側にまとめます。
同一作品の重複と評価専用の文字はありません。
候補句そのものは両側に含まれるため、既知テーマ内の新しい組み合わせを評価します。

分割を再生成する場合は次を実行し、続いて再ビルドしてください。

```sh
python3 split-all-themes.py
python3 package.py
```

緑は固定した訓練128件、橙は固定した評価128件の平均次文字損失です。
川柳全体の品質や5・7・5の成功率を直接表す指標ではありません。
評価値で構成を選ぶ場合は、別の最終評価データが必要です。
以前の「未学習テーマ」分割とは条件が異なり、損失は直接比較できません。

## 設定

ブロック数：2 / 4 / 6。次元数：8 / 16 / 32 / 48 / 64。
バッチサイズ：8 / 16 / 32 / 64。構成変更では学習を初期化します。
データセットを切り替えると、語彙と文脈長（川柳21・SF掌編111位置）が変わるため学習を初期化します。
SF掌編を再生成する場合は `python3 generate-sf-story.py` のあと `python3 package.py` を実行してください。
学習率は `train-worker.js` と `live-loader.js` の既定値 0.003 です。

このZIPは現在のソース一式です。過去のGit履歴は含みません。

## ライセンス

[MIT](LICENSE)
