# bench/ — benchmark-driven calibration

forgequant calibrates a domain imatrix on the questions a domain *expert* actually
faces. The benchmarks come from **benchy** (vendored as the `../benchy` submodule),
which is the single source of truth: a registry of real, non-saturated evals fetched
live from the HuggingFace datasets-server. forgequant never re-implements or
redistributes a benchmark — it fetches from benchy and renders.

```sh
git submodule update --init                       # pull benchy in (first time)
python3 forge_bench.py list                        # the registry (current vs saturated)
python3 forge_bench.py bundles                      # domain bundles
python3 forge_bench.py fetch code                    # cache HumanEval+MBPP+MMLU-CS rows
python3 forge_bench.py corpus code -o bench/corpora/code.txt --answers --mix reasoning
```

Or declare it in a recipe and let `forgequant build` do it:

```json
"bench": {"keys": ["humaneval","mbpp","mmlu_cs"], "answers": true, "mix": "reasoning", "cap": 400}
```

## What's tracked here (and what isn't)

| Path | Tracked? | Why |
|---|---|---|
| `runs/*.json` | **yes** | provenance of every calibration: which benchmarks, row SHAs, options, resulting corpus SHA — so anyone using the repo can see/reproduce exactly what calibrated an imatrix |
| `packs/*.jsonl` | **yes** | small local prompt packs for domains benchy has no online benchmark for (e.g. `agentic` tool-use). These are hand-written supplements, not redistributed evals |
| `corpora/*.txt` | no (gitignored) | rendered corpora are derived from licensed benchmark data — rebuild them from benchy on demand |
| `../benchy/data/*.jsonl` | no (benchy gitignores) | the raw benchmark rows are fetched from HF, never committed |

## Domain bundles

| Bundle | Benchmarks (non-saturated) |
|---|---|
| `code` | HumanEval, MBPP, MMLU-CS |
| `medical` | MedXpertQA, MedMCQA, MedQA |
| `reasoning` | MMLU-Pro, SuperGPQA, MMLU formal-logic |
| `knowledge` | MMLU-Pro, OpenBookQA |
| `broad` | a cross-domain mix |

`--answers` adds the gold answer as an assistant turn, so the imatrix also sees the
activation paths of *answering*, not just reading. `--mix DOMAIN` interleaves a
general set so a domain imatrix doesn't over-specialize (benchy's own medical corpus
uses ~40% domain by characters).
