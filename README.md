# forgequant

**Config-driven asymmetric quantization for DeepSeek-V4-Flash.**

`forgequant` turns a small JSON **recipe** — per-tensor-family quant types plus an
importance matrix (imatrix) — into a quantized GGUF, reproducibly, with a manifest. It
wraps [ds4](#built-on-ds4)'s `deepseek4-quantize` (and `ds4 --imatrix-dataset` for the
optional imatrix step), so you stop hand-assembling long quantizer command lines and
re-prompting an assistant for the flags.

The point: a model's 2-bit budget can be spent **asymmetrically** — keep attention and
shared/output weights near-lossless (Q8), push the routed experts to 2 bits — and an
imatrix re-allocates those 2 bits toward the weights your workload actually activates.
forgequant makes that recipe a file you can version, diff, and re-run.

> Specific to **DeepSeek-V4-Flash** and ds4's quantizer — not a general GGUF tool.

## Install

Python 3.8+ (standard library only). You need a built ds4 checkout (provides `ds4` and
`gguf-tools/deepseek4-quantize`), the FP model source, and a template GGUF.

```sh
export DS4_DIR=~/ds4               # your ds4 checkout (default ~/ds4)
export MODELS_DIR=~/ds4-models     # where models/imatrices live (usable as {models} in recipes)
```

## Use

```sh
python3 forgequant.py list                 # available recipes
python3 forgequant.py show medical-iq2     # resolved recipe + the EXACT commands (nothing runs)
python3 forgequant.py imatrix medical-iq2  # build the imatrix from the recipe's corpus
python3 forgequant.py quantize medical-iq2 # recipe -> GGUF + manifest
python3 forgequant.py build medical-iq2    # full pipeline: imatrix (if missing) then quantize
```

`<recipe>` is a preset name (`recipes/<name>.json`) or a path to your own `.json`.

## Dashboard (UI)

A single-file web dashboard drives forgequant from the browser — pick or **build a recipe**
(per-family quant types + an imatrix to load or extract from a corpus), launch
build/quantize/imatrix, and watch the quantization **live** (per-tensor progress, ETA, log)
plus a table of the models you've forged.

```sh
python3 forge_ui.py            # -> http://localhost:8060
```

Stdlib only; same `DS4_DIR` / `MODELS_DIR` config.

## Recipe format

```json
{
  "name": "medical-iq2",
  "description": "...",
  "hf": "{models}/DeepSeek-V4-Flash-FP",            // FP safetensors source (re-quantized families read from here)
  "template": "{models}/<base>.gguf",                // metadata/order/shapes; non-listed families copied from here
  "imatrix": "{models}/medical.dat",                 // legacy .dat imatrix (applied to routed experts)
  "corpus": "~/.../rendered_corpus.txt",             // optional: build the imatrix from this if missing
  "imatrix_max_tokens": 120000,
  "quant": {                                          // family -> quant type (only what you want to change)
    "routed_w1": "iq2_xxs",   // gate expert
    "routed_w3": "iq2_xxs",   // up   expert
    "routed_w2": "q2_k"       // down expert
  }
}
```

Families: `routed_w1/w2/w3` (gate/down/up experts), `attention`, `attn_proj`, `shared`,
`embedding`, `output`. **Anything you omit is copied verbatim from `template`.** Quant
type names are ds4's: `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `q2_k`, `q3_k`,
`q4_k`, `q5_k`, `q6_k`, `q8_0`, `bf16`, … (see `deepseek4-quantize --help`).

`{models}` expands to `$MODELS_DIR`, `{name}` to the recipe name, `~` to your home.

## Presets

| Recipe        | Experts (gate/up · down) | For |
|---------------|--------------------------|-----|
| `medical-iq2` | IQ2_XXS · Q2_K + medical imatrix | the proven BeepMed recipe |
| `coder-iq2`   | IQ2_XXS · Q2_K + code imatrix    | same budget, code-calibrated |
| `balanced`    | IQ3_XXS · Q3_K                   | bigger, higher fidelity |
| `aggressive`  | IQ2_XXS · IQ2_XXS                | smallest, most lossy |

## The imatrix step

`forgequant imatrix <recipe>` runs `ds4 -m <template> --imatrix-dataset <corpus>
--imatrix-out <imatrix>`. The **corpus must be a rendered prompt dataset** (ds4's
format). Two easy ways to get one:
- offline: render your prompts (e.g. domain Q&A, code) into that format;
- on-edge: collect it from live traffic with `ds4-server --imatrix-out` (privacy-safe,
  no prompts stored) and point the recipe's `imatrix` straight at the result — no corpus
  needed.

## A/B with benchy

Each forgequant output is a drop-in model. Serve it and measure with
[benchy](https://github.com/andreaborio/benchy):

```sh
ds4-server -m ~/ds4-models/DeepSeek-V4-Flash-medical-iq2.gguf --ssd-streaming --port 8000 &
python3 ../benchy/eval_mcq.py data/medqa_test.jsonl 60 think medical-iq2
# compare the `medical-iq2` tag against your baseline in the benchy dashboard
```

## Reproducibility

Quantization is deterministic: the same recipe + the same imatrix produce the same GGUF.
Every `quantize` writes `<out>.manifest.json` — the resolved recipe, the exact command,
and SHA-256 + size of both the imatrix and the output — so a result is always traceable
to its inputs.

## Built on ds4

`forgequant` is a thin orchestrator over **ds4 / DwarfStar** by Salvatore Sanfilippo
([antirez](https://github.com/antirez)) — specifically `gguf-tools/deepseek4-quantize`
and `ds4 --imatrix-dataset`. All the real quantization work is theirs; forgequant only
turns a recipe into the right invocation and records what it did. ds4 is a separate
project under its own license.

## License

MIT — see [LICENSE](LICENSE). Does not cover ds4 or any model weights.
