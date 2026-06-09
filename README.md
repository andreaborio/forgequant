# forgequant

**Config-driven asymmetric quantization for DeepSeek-V4-Flash.**

`forgequant` turns a small JSON **recipe** — per-tensor-family quant types, an
importance matrix (imatrix), and optionally a per-layer expert **boost** — into a
quantized GGUF, reproducibly, with a manifest. It wraps [ds4](#built-on-ds4)'s
`deepseek4-quantize` (plus `ds4 --imatrix-dataset`, `ds4-server --imatrix-out` and
the GGUF splicer), so you stop hand-assembling long quantizer command lines.

The point: a model's 2-bit budget can be spent **asymmetrically**, at three depths —

1. **family** — keep attention/shared/output near-lossless (Q8), push the routed
   experts to 2 bits;
2. **expert** — an imatrix re-allocates those 2 bits *inside every tensor* toward the
   experts your workload actually activates (ds4 records per-(layer, expert)
   activation statistics);
3. **layer** — `boost` upcasts the routed experts of the layers your workload lives
   in (e.g. Q4_K on the 6 hottest), via `--tensor-type` overrides — or `splice`
   copies them from a donor GGUF in minutes, without requantizing.

forgequant makes that recipe a file you can version, diff, and re-run — and gives
you the tools to *see* the activation paths before you spend the bits.

> Specific to **DeepSeek-V4-Flash** and ds4's quantizer — not a general GGUF tool.

## Install

Python 3.8+ (standard library only; numpy used opportunistically if present). You
need a built ds4 checkout (provides `ds4`, `ds4-server` and
`gguf-tools/deepseek4-quantize`), the FP model source, and a template GGUF.

```sh
git clone --recursive https://github.com/andreaborio/forgequant.git   # --recursive pulls benchy
# already cloned? grab the benchy submodule:
git submodule update --init

export DS4_DIR=~/BEEP/ds4               # your ds4 checkout (default ~/ds4)
export MODELS_DIR=~/BEEP/ds4-models     # models/imatrices ({models} in recipes)
```

`benchy` (the benchmark source) is vendored as a git submodule, so a recursive
clone is self-contained. `git submodule update --remote benchy` bumps it to
benchy's latest registry.

## Use

```sh
python3 forgequant.py list                  # available recipes
python3 forgequant.py show coder-q4boost    # resolved recipe + EXACT commands (nothing runs)
python3 forgequant.py verify coder-q4boost  # preflight: paths, imatrix, disk space
python3 forgequant.py build coder-q4boost   # full pipeline: imatrix (if missing) -> quantize
```

`<recipe>` is a preset name (`recipes/<name>.json`) or a path to your own `.json`.

## Getting an imatrix (four ways)

The imatrix is the activation-path record that steers the bits. Pick your source:

```sh
# 1. from REAL benchmarks (the questions a domain expert faces) — fetched from benchy
python3 forgequant.py build coder-q4boost     # the `bench` block builds the corpus first

# 2. from a corpus you already have (rendered prompt dataset)
python3 forgequant.py imatrix medical-iq2

# 3. from ANY raw prompt list — render it first, no ds4 python tooling needed
python3 forgequant.py render my_prompts.txt -o coder_corpus.txt   # .txt or .jsonl
python3 forgequant.py imatrix coder-q4boost

# 4. from LIVE inference — serve the model, use it for real, Ctrl-C when done
python3 forgequant.py capture coder-q4boost --port 8000
```

`capture` wraps `ds4-server --imatrix-out`: it records *only* aggregate per-expert
activation statistics from your real traffic — no prompt text is ever stored
(see ds4's `ONEDGE_IMATRIX.md`), and snapshots are written periodically. Ctrl-C is a
graceful stop: forgequant waits for ds4-server to flush its final snapshot.

## Calibrating on real benchmarks (via benchy)

Calibrate a domain imatrix on the questions a domain *expert* actually faces.
Benchmarks come from **benchy** (github.com/andreaborio/benchy), vendored as a git
submodule so everyone who clones forgequant gets the same source — a registry of
real, **non-saturated** evals (MMLU-Pro, SuperGPQA, HumanEval, MBPP, MedXpertQA,
MedQA, …) fetched live from the HuggingFace datasets-server and normalized.

```sh
git submodule update --init                  # first time: pull benchy in
python3 forgequant.py bench list             # the registry (current vs saturated)
python3 forgequant.py bench bundles          # domain bundles: code / medical / reasoning / …
python3 forgequant.py bench corpus code -o bench/corpora/code.txt --answers --mix reasoning
```

Or declare it in a recipe and let `build` do everything:

```json
"bench": {"keys": ["humaneval","mbpp","mmlu_cs"], "answers": true, "mix": "reasoning", "cap": 400}
```

`--answers` adds the gold answer as an assistant turn (so the imatrix sees the
activation paths of *answering*, not just reading); `--mix DOMAIN` interleaves a
general set so a domain imatrix doesn't over-specialize. Every corpus build records
its provenance (benchmark keys, row SHAs, options) under `bench/runs/` — tracked in
the repo, so a calibration is always traceable to the exact benchmark snapshot.
forgequant never redistributes benchmark data: rows are fetched from HF on demand.

## Seeing the brain paths

```sh
python3 forgequant.py paths coder-q4boost          # per-layer/per-expert heatmap
python3 forgequant.py paths a.dat --diff b.dat     # what does CODE light up that MEDICAL doesn't?
python3 forgequant.py suggest coder-q4boost --top 6 --type q4_k   # boost proposal + size cost
```

`paths` parses the `.dat` directly (the format packs one importance vector per
expert per routed tensor) and shows where the workload concentrates. `suggest`
turns that into a ready-to-paste `boost` block with an estimated size delta.
Values are count-normalized activation energy: how hard an expert works when
routed; never-routed experts show as cold (zero).

## Dashboard (UI)

A single-file web dashboard drives forgequant from the browser — template gallery,
recipe builder (families + boost + imatrix), build/quantize/imatrix/capture/splice
actions, live progress (per-tensor, ETA), an interactive **brain map** of any
imatrix (43×256 heatmap, hot layers, diff between two imatrices, one-click boost
suggestion), past-runs browser, and the table of forged models.

```sh
python3 forge_ui.py            # -> http://localhost:8060
```

Stdlib only; same `DS4_DIR` / `MODELS_DIR` config.

## Recipe format

```json
{
  "name": "coder-q4boost",
  "description": "...",
  "hf": "{models}/DeepSeek-V4-Flash-FP",      // FP safetensors source
  "template": "{models}/<base>.gguf",          // metadata/order/shapes; non-listed families copied from here
  "imatrix": "{models}/coder.dat",             // legacy .dat imatrix (applied per expert)
  "corpus": "{models}/coder_corpus.txt",       // optional: build the imatrix from this if missing
  "imatrix_max_tokens": 120000,
  "quant": {                                    // family -> quant type (only what you change)
    "routed_w1": "iq2_xxs",   // gate experts
    "routed_w3": "iq2_xxs",   // up   experts
    "routed_w2": "q2_k"       // down experts
  },
  "boost": {                                    // per-layer expert upcast (optional)
    "layers": "auto:6",       // N hottest layers from the imatrix — or "37-42", or [37,40]
    "type": "q4_k",
    "families": ["w1","w2","w3"]                // optional subset
  },
  "tensor_types": {"blk.0.": "q8_0"},          // raw --tensor-type prefix overrides (optional)
  "splice": {                                   // fast layer boost without requantizing (optional)
    "donor": "{models}/<q4-variant>.gguf",
    "layers": "auto:6"
  },
  "threads": 16
}
```

Families: `routed_w1/w2/w3` (gate/down/up experts), `experts` (all three),
`attention`, `attn_proj`, `shared`, `embedding`, `output`, `dense`. **Anything you
omit is copied verbatim from `template`.**

**Producible quant types:** deepseek4-quantize can only *generate* `iq2_xxs`,
`q2_k`, `q4_k`, `q8_0` (plus `f16`/`bf16`/`f32` passthrough) — these are the
`ds4q_can_quantize()` types in ds4's `quants.c`. Other names (`q3_k`, `iq3_xxs`,
`iq2_s`, `q5_k`, `q6_k`, …) parse but the quantizer rejects them with "unsupported
quant target type", so forgequant validates recipes up front.

`{models}` expands to `$MODELS_DIR`, `{name}` to the recipe name, `~` to your home.

### Where each knob acts

| Granularity | Mechanism | Cost to test |
|---|---|---|
| family (all experts) | `quant` → `--routed-w1/w2/w3` | full requantize |
| expert (within a tensor) | `imatrix` — per-expert bit steering, automatic | imatrix run |
| layer (chosen experts ×3 tensors) | `boost` → `--tensor-type` overrides | full requantize |
| layer, instantly | `splice` — copy from donor GGUF | **minutes** |

Per-expert *types* inside one fused tensor aren't possible (GGUF stores one type
per tensor; verified in `deepseek4-quantize.c`) — the imatrix's per-expert bit
steering plus layer boost is the practical equivalent.

## Presets

| Recipe | Idea | For |
|---|---|---|
| `medical-iq2` | IQ2_XXS · Q2_K + medical imatrix | the proven BeepMed recipe |
| `coder-iq2` | same budget, code-calibrated imatrix | coding workloads |
| `coder-q4boost` | coder-iq2 + Q4_K on the 6 code-hottest layers | **"keep my coding expert sharp"** |
| `medical-q4boost` | medical-iq2 + Q4_K on the 6 med-hottest layers | BeepMed, higher fidelity |
| `last6-q4boost` | static Q4_K on layers 37-42 | ds4's proven mixed experiment |
| `splice-fast` | copy hot layers from a Q4 donor | fastest A/B loop |
| `balanced` | Q4_K gate/up · Q2_K down | bigger, higher fidelity |
| `aggressive` | IQ2_XXS everywhere | smallest, most lossy |

## A/B with benchy

Each forgequant output is a drop-in model. Serve it and measure with
[benchy](https://github.com/andreaborio/benchy):

```sh
ds4-server -m ~/BEEP/ds4-models/DeepSeek-V4-Flash-coder-q4boost.gguf --ssd-streaming --port 8000 &
python3 ../benchy/eval_mcq.py data/humaneval.jsonl 60 think coder-q4boost
```

For quick quality signals without a benchmark run, ds4's
`gguf-tools/quality-testing/` scores GGUF variants by NLL against official
DeepSeek continuations.

## Reproducibility

Quantization is deterministic: the same recipe + the same imatrix produce the same
GGUF. Every `quantize`/`splice` writes `<out>.manifest.json` — the resolved recipe,
the exact command, the ds4 git revision, duration, and SHA-256 + size of both the
imatrix and the output — so a result is always traceable to its inputs.

## Tests

```sh
python3 test_forge.py    # stdlib unittest; covers the .dat parser, renderer, recipes, UI guards
```

## Built on ds4

`forgequant` is a thin orchestrator over **ds4 / DwarfStar** by Salvatore Sanfilippo
([antirez](https://github.com/antirez)) — specifically `gguf-tools/deepseek4-quantize`,
`ds4 --imatrix-dataset`, `ds4-server --imatrix-out` and
`gguf-tools/mixed/splice_mixed_expert_layers_gguf.py`. All the real quantization
work is theirs; forgequant only turns a recipe into the right invocation and
records what it did. ds4 is a separate project under its own license.

## License

MIT — see [LICENSE](LICENSE). Does not cover ds4 or any model weights.
