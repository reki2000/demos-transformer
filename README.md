# Ikuji Senryu Transformer Lab (WASM)

[日本語版 / Japanese](README.ja.md)

**Live demo: https://reki2000.github.io/demos-transformer/**

An interactive visualization of a tiny decoder-only Transformer that is actually
trained inside your browser. The corpus is a set of senryu (5-7-5 Japanese
poems) about child-rearing, covering 14 themes. There are no external requests
and no external libraries.

## Usage

Open the live demo (or `dist/index.html` after building) in a current Chrome,
Edge, Firefox or Safari and press the start-training button. The browser must
support WebAssembly SIMD, Web Workers and `DecompressionStream`.
The page is a single HTML file with the data, WASM binary and code embedded.
Training state is lost when the tab is closed.

## Build

Requires Python 3.9+. No pip/npm packages, network access or C/C++ compiler
are needed. Run from the repository root:

```sh
python3 build-wasm.py
python3 package.py
```

This produces `dist/index.html` and `dist/transformer-lab.zip`.
`combined*.js` are intermediate files for tests.

Pushes to `main` are built, tested and deployed to GitHub Pages by
`.github/workflows/pages.yml`.

## Tests

Requires Node.js 22+. Build first.

```sh
node test-engine.cjs
node test-page.cjs
node test-completion.cjs
```

- Engine: agreement with an independent JavaScript implementation, numerical
  gradient checks, real training, all 15 configurations.
- Page: simulated DOM with real Workers and WASM; covers interaction,
  generation, stop and restore.
- Completion: four epochs over 4,286 samples, plus continued training and stop.

The page tests are not a substitute for checking the appearance in real browsers.

## Files

| File | Description |
| --- | --- |
| `build-wasm.py` | Emits the 13 numeric kernels directly as `engine.wasm` (build output, git-ignored) |
| `engine.js` | Transformer forward/backward, Adam, memory management |
| `train-worker.js` | Mini-batch training, evaluation, checkpoint restore |
| `decoder.js` / `beam.js` / `infer-worker.js` | Inference for visualization and search for 5 candidate generations |
| `live-loader.js` / `live-app.js` / `details.js` | Startup, UI, heatmaps, cell details |
| `live-page.html` / `style.css` | HTML template and styles |
| `corpus-common.json` | 5,000 samples with text, readings, themes, character vocabulary, tokens and split info |
| `split-all-themes.py` | Reproduces the train/eval split covering all 14 themes |
| `package.py` | Builds the single HTML file and its ZIP |
| `test-*.cjs` | Tests |

## Data and evaluation

The data is 5,000 synthetic samples made for teaching, and no published
works. It contains kanji-kana text, readings and 5-7-5 mora information,
assembled from 350 candidate phrases across 14 scenes, so some phrases are
unnatural.

51 samples per theme (714 in total) are held out for evaluation and 4,286 are
used for training. Samples sharing the same first two lines are kept on the
same side. There are no duplicated samples and no evaluation-only characters.
The candidate phrases themselves appear on both sides, so evaluation measures
new combinations within known themes.

To regenerate the split, run the following and then rebuild:

```sh
python3 split-all-themes.py
python3 package.py
```

The green curve is the mean next-character loss on a fixed set of 128 training
samples, and the orange curve is the same for a fixed set of 128 evaluation
samples. These are not direct measures of senryu quality or 5-7-5 success rate.
If you choose a configuration by the evaluation loss, a separate final test set
is needed. The loss is not comparable with the earlier "unseen theme" split.

## Configuration

Blocks: 2 / 4 / 6. Dimensions: 8 / 16 / 32 / 48 / 64.
Batch size: 8 / 16 / 32 / 64. Changing the configuration resets training.
The default learning rate is 0.003 (in `train-worker.js` and `live-loader.js`).

## License

[MIT](LICENSE)
