# Ikuji Senryu Transformer Lab (WASM)

[日本語版 / Japanese](README.ja.md)

**Live demo: https://reki2000.github.io/demos-transformer/**

An interactive visualization of a tiny decoder-only Transformer that is actually
trained inside your browser. The corpus is a set of senryu (5-7-5 Japanese
poems) about child-rearing, covering 14 themes; a "Data" selector switches to
5,000 generated four-act SF short stories. A "Model" selector switches between
five model families (below), and a structure diagram shows the selected one.
There are no external requests and no external libraries.

## Usage

Open the live demo (or `dist/index.html` after building) in a current Chrome,
Edge, Firefox or Safari and press the start-training button. The browser must
support WebAssembly SIMD, Web Workers and `DecompressionStream`.
The page is a single HTML file with the data, WASM binary and code embedded.
Training state is lost when the tab is closed.

## Models

| Model | What it is |
| --- | --- |
| Standard Transformer | The original layout: learned position table, LayerNorm, tanh MLP, causal attention |
| Modern Transformer | RoPE (rotary positions), RMSNorm and SwiGLU, with about the same parameter count |
| Diffusion LM | Modern backbone with bidirectional attention, trained to recover randomly masked characters (masked diffusion). Generation starts from all masks and reveals the most confident positions step by step |
| Looped Transformer | One modern block applied L times with shared weights; the number of loops can also be changed at generation time |
| Mamba-3 | Modern backbone with the attention mixer replaced by a Mamba-3 state-space mixer (exponential-trapezoidal discretization, rotation for the complex state, output gate) |

The last three change exactly one thing from the modern Transformer; the
diagram marks the changed parts in orange. Mamba-3 is trained in its quadratic
form, which has the same shape as attention, so the page shows its mixing
matrix M where the others show attention. Generation uses the recurrent form
with a fixed-size state. The diffusion loss averages over hidden characters
only, so it is not directly comparable with the next-character loss of the
other models.

Generation starts from the text typed in the generation box; when the box is
empty it starts from the example up to the selected position.

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
node test-kvcache.cjs
```

- Engine (also checks 2 and 4 heads, and every model family): agreement with an independent JavaScript
  implementation, numerical gradient checks of every parameter tensor, real training, all 15 configurations,
  shared weights of the looped model, and diffusion masking.
- Page: simulated DOM with real Workers and WASM; covers interaction,
  generation, stop and restore.
- Completion: four epochs over 4,286 samples, plus continued training and stop.
- KV cache: incremental decoding matches full recomputation (for Mamba-3, the
  recurrent state matches the quadratic form), and beam search returns the same
  candidates with and without the cache.

The page tests are not a substitute for checking the appearance in real browsers.

## Files

| File | Description |
| --- | --- |
| `build-wasm.py` | Emits the 22 numeric kernels directly as `engine.wasm` (build output, git-ignored) |
| `engine.js` | Forward/backward of all five model families, Adam, memory management |
| `train-worker.js` | Mini-batch training, evaluation, checkpoint restore |
| `decoder.js` / `beam.js` / `diffuse.js` / `infer-worker.js` | Inference for visualization, search for 5 candidate generations, and diffusion sampling |
| `live-loader.js` / `live-app.js` / `architecture.js` / `details.js` | Startup, UI, heatmaps, model descriptions and structure diagram, cell details |
| `live-page.html` / `style.css` | HTML template and styles |
| `corpus-common.json` | 5,000 samples with text, readings, themes, character vocabulary, tokens and split info |
| `split-all-themes.py` | Reproduces the train/eval split covering all 14 themes |
| `generate-sf-story.py` / `corpus-sf.json` | Generator and output for 5,000 four-act SF short stories |
| `datasets.json` | Datasets offered in the page (label, description, CSV name) |
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

Blocks: 2 / 4 / 6. Dimensions: 8 / 16 / 32 / 48 / 64. Heads: 1 / 2 / 4 (the
dimension is split evenly; each head needs at least 4 columns). With several
heads each block shows one attention map per head. The "mean distance" in each
block header is how many characters back attention reaches on the shown example.
Batch size: 8 / 16 / 32 / 64. Changing the configuration resets training.
Switching the dataset also resets training, since the vocabulary and context
length change (21 positions for senryu, 128 for SF stories). To regenerate the
SF stories, run `python3 generate-sf-story.py` and then `python3 package.py`.
The default learning rate is 0.003 (in `train-worker.js` and `live-loader.js`).

## Experiment: blocks and heads

`experiments/depth-heads.md` compares 2, 4 and 6 blocks with 1 and 4 heads
(64 dimensions) on the SF stories. To reproduce (10–20 minutes per configuration):

```sh
node experiments/depth-heads.cjs 4 64 4   # blocks dimensions heads
python3 experiments/evaluate.py
```

## License

[MIT](LICENSE)
