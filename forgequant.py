#!/usr/bin/env python3
"""forgequant — config-driven asymmetric quantization for DeepSeek-V4-Flash (ds4).

A *recipe* (recipes/<name>.json) declares the per-tensor-family quant types, an
imatrix (or a corpus to build one), and optionally a per-layer expert *boost*.
forgequant turns a recipe into a quantized GGUF, reproducibly, writing a manifest
next to the output. It wraps ds4's `deepseek4-quantize` (and `ds4
--imatrix-dataset` / `ds4-server --imatrix-out` for the imatrix step).

Commands:
  forgequant.py list                 list available recipes
  forgequant.py show <recipe>        resolved recipe + the exact commands (no run)
  forgequant.py verify <recipe>      preflight: paths, imatrix, disk space, plan
  forgequant.py bench [list|bundles] benchmark registry from benchy (the eval source)
  forgequant.py imatrix <recipe>     build the imatrix from the recipe's corpus
                                     (auto-builds the corpus from a `bench` block)
  forgequant.py capture <recipe>     serve the model live and collect the imatrix
                                     from real inference traffic (the activation
                                     paths of YOUR workload; Ctrl-C to stop)
  forgequant.py render <in> -o <out> render raw prompts (.txt/.jsonl) into a corpus
  forgequant.py paths <recipe|.dat>  show the activation paths in an imatrix:
                                     per-layer/per-expert heatmap, hot layers
  forgequant.py suggest <recipe>     suggest a per-layer boost from the imatrix
  forgequant.py quantize <recipe>    run the quantization (recipe -> GGUF + manifest)
  forgequant.py splice <recipe>      fast boost WITHOUT requantizing: copy selected
                                     expert layers from a higher-precision donor GGUF
  forgequant.py build <recipe>       full pipeline: imatrix (if needed) then quantize

<recipe> is a preset name (recipes/<name>.json) or a path to a .json file.

Recipe quant families: routed_w1/w2/w3 (gate/down/up experts), experts (all three),
attention, attn_proj, shared, embedding, output, dense. Omitted families are copied
verbatim from `template`. Extra keys:
  "bench":        {"keys": ["humaneval","mbpp","mmlu_cs"] | "domain": "code",
                   "answers": true, "mix": "reasoning", "cap": 400}
                  calibrate the imatrix on real, non-saturated benchmarks fetched
                  from benchy — the corpus is built automatically before `imatrix`
  "boost":        {"layers": "37-42" | [37,42] | "auto:6", "type": "q4_k",
                   "families": ["w1","w2","w3"]}   per-layer expert upcast, expands
                  to --tensor-type overrides; "auto:N" picks the N hottest layers
                  from the recipe's imatrix (run `imatrix`/`capture` first)
  "reuse":        "{models}/DeepSeek-V4-Flash-coder-iq2.gguf"   copy byte-identical
                  tensors from a prior build (same hf+imatrix) instead of regenerating —
                  only changed (e.g. boosted) tensors are quantized. Missing prior or a
                  key mismatch safely falls back to a full quantize.
  "tensor_types": {"blk.0.": "q8_0", ...}          raw --tensor-type prefix overrides
  "splice":       {"donor": "...gguf", "layers": "37-42" | "auto:6", "out": "..."}
  "threads": N, "imatrix_ctx": N, "imatrix_cache": "40GB", "imatrix_max_tokens": N,
  "imatrix_strict": true, "overwrite": true

Config via env:
  DS4_DIR     ds4 checkout (default ~/ds4) — provides ds4 + gguf-tools
  MODELS_DIR  where models/imatrices live (default ~/ds4-models); {models} in paths

Stdlib only. Quantization is deterministic: same recipe + same imatrix => same GGUF.
"""
import json, os, signal, sys, subprocess, hashlib, datetime, time

import forge_imatrix
import forge_corpus
import forge_bench

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPES = os.path.join(HERE, "recipes")
DS4_DIR = os.path.expanduser(os.environ.get("DS4_DIR", "~/ds4"))
MODELS_DIR = os.path.expanduser(os.environ.get("MODELS_DIR", "~/ds4-models"))
QUANTIZE = os.path.join(DS4_DIR, "gguf-tools", "deepseek4-quantize")
SPLICER = os.path.join(DS4_DIR, "gguf-tools", "mixed", "splice_mixed_expert_layers_gguf.py")
DS4 = os.path.join(DS4_DIR, "ds4")
DS4_SERVER = os.path.join(DS4_DIR, "ds4-server")
SCHEMA = 2

# recipe family key -> deepseek4-quantize flag
FLAG = {"routed_w1": "--routed-w1", "routed_w2": "--routed-w2", "routed_w3": "--routed-w3",
        "experts": "--experts", "attention": "--attention", "attn_proj": "--attn-proj",
        "shared": "--shared", "embedding": "--embedding", "output": "--output",
        "dense": "--dense"}
# boost family -> tensor name fragment
BOOST_TENSOR = {"w1": "ffn_gate_exps", "w2": "ffn_down_exps", "w3": "ffn_up_exps"}
N_LAYERS = 43          # DeepSeek-V4-Flash routed layers
# quant TARGET types deepseek4-quantize can actually produce (ds4q_can_quantize() in
# quants.c); plus f16/f32/bf16 passthrough. Anything else dies "unsupported quant target".
PRODUCIBLE = {"iq2_xxs", "q2_k", "q4_k", "q8_0", "f16", "f32", "bf16"}

def die(msg): sys.exit("forgequant: " + msg)

def check_type(t, where):
    if str(t).lower() not in PRODUCIBLE:
        die(f"{where}: deepseek4-quantize cannot produce '{t}' "
            f"(producible: {', '.join(sorted(PRODUCIBLE))})")

def resolve(p, name):
    if not p: return p
    return os.path.expanduser(p.replace("{name}", name).replace("{models}", MODELS_DIR))

def load_recipe(arg):
    path = arg if os.path.exists(arg) else os.path.join(RECIPES, arg + ".json")
    if not os.path.exists(path):
        die(f"recipe not found: {arg}  (presets live in {RECIPES})")
    try:
        r = json.load(open(path))
    except Exception as e:
        die(f"invalid recipe JSON {path}: {e}")
    r["name"] = r.get("name") or os.path.basename(path)[:-5]
    for k in ("hf", "template", "imatrix", "corpus", "out", "reuse"):
        if r.get(k): r[k] = resolve(r[k], r["name"])
    if not r.get("out"):
        r["out"] = os.path.join(MODELS_DIR, f"DeepSeek-V4-Flash-{r['name']}.gguf")
    for fam, t in (r.get("quant") or {}).items():
        if fam not in FLAG:
            die(f"unknown family '{fam}' in recipe (allowed: {', '.join(FLAG)})")
        check_type(t, f"quant.{fam}")
    for pfx, t in (r.get("tensor_types") or {}).items():
        check_type(t, f"tensor_types[{pfx}]")
    b = r.get("boost")
    if b:
        for fam in b.get("families", []):
            if fam not in BOOST_TENSOR:
                die(f"unknown boost family '{fam}' (allowed: {', '.join(BOOST_TENSOR)})")
        if not b.get("type"): die("boost needs a `type` (e.g. q4_k)")
        check_type(b["type"], "boost.type")
        if "layers" not in b: die("boost needs `layers` (e.g. \"37-42\" or \"auto:6\")")
        if b.get("mode") not in (None, "energy", "contrast"):
            die("boost.mode must be 'energy' (default) or 'contrast'")
        if b.get("mode") == "contrast":
            b["baseline"] = resolve(b.get("baseline"), r["name"])
            if not b["baseline"]:
                die("contrast boost needs `boost.baseline` (a general/other-domain imatrix .dat "
                    "to contrast against — boosts the layers THIS domain uses differently)")
    return r

def _check_range(layers):
    bad = [l for l in layers if not 0 <= l < N_LAYERS]
    if bad: die(f"layer ids out of range 0..{N_LAYERS - 1}: {bad}")
    return layers

def parse_layers(spec, recipe=None):
    """'37-42' | '0-2,40-42' | [37,42] | 'auto:N' -> sorted unique layer ids."""
    if isinstance(spec, list):
        return _check_range(sorted(set(int(x) for x in spec)))
    spec = str(spec).strip()
    if spec.startswith("auto:"):
        n = int(spec.split(":", 1)[1])
        if not recipe or not recipe.get("imatrix"):
            die("boost layers 'auto:N' needs the recipe to have an `imatrix`")
        if not os.path.exists(recipe["imatrix"]):
            die(f"imatrix not found: {recipe['imatrix']} — run `imatrix` or `capture` first, "
                "or use a static layer list (e.g. \"37-42\")")
        stats = forge_imatrix.cached_stats(recipe["imatrix"])
        return _check_range(sorted(stats["hot_layers"][:n]))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    if not out: die(f"empty boost layer spec: {spec!r}")
    return _check_range(sorted(out))

def resolve_boost_layers(r, b):
    """Layers to upcast. mode 'energy' (default): the hottest layers of the recipe's imatrix.
    mode 'contrast': the layers this domain's imatrix uses most DIFFERENTLY from a baseline —
    i.e. the domain-distinctive layers, not the generally-hot ones."""
    spec = b["layers"]
    if b.get("mode") == "contrast" and isinstance(spec, str) and spec.startswith("auto:"):
        n = int(spec.split(":", 1)[1])
        if not r.get("imatrix") or not os.path.exists(r.get("imatrix") or ""):
            die("contrast boost 'auto:N' needs the recipe's domain imatrix — run `imatrix`/`capture` first")
        if not os.path.exists(b["baseline"]):
            die(f"contrast baseline not found: {b['baseline']} — build a general imatrix "
                "(e.g. `bench corpus broad` -> imatrix) to contrast against")
        dom = forge_imatrix.analyze(forge_imatrix.load_dat(r["imatrix"]))
        base = forge_imatrix.analyze(forge_imatrix.load_dat(b["baseline"]))
        return _check_range(forge_imatrix.suggest_contrast(dom, base, n))
    return parse_layers(spec, r)

def boost_overrides(r):
    """Expand the recipe's `boost` block into ordered (tensor-prefix, type) pairs."""
    b = r.get("boost")
    if not b: return []
    layers = resolve_boost_layers(r, b)
    fams = b.get("families") or ["w1", "w2", "w3"]
    return [(f"blk.{l}.{BOOST_TENSOR[f]}.weight", b["type"]) for l in layers for f in fams]

def quant_cmd(r, dry=False):
    for k in ("hf", "template"):
        if not r.get(k): die(f"recipe missing required field: {k}")
    cmd = [QUANTIZE, "--hf", r["hf"], "--template", r["template"], "--out", r["out"]]
    if r.get("imatrix"): cmd += ["--imatrix", r["imatrix"]]
    if r.get("imatrix_strict"): cmd += ["--imatrix-strict"]
    for fam, t in (r.get("quant") or {}).items():
        cmd += [FLAG[fam], str(t)]
    # deepseek4-quantize applies the FIRST matching prefix, so order most-specific first.
    # Tag source priority (user tensor_types=0, boost=1) so that for an EQUAL-length prefix
    # (the same exact tensor) the user's explicit override wins over the boost; longer
    # prefixes still sort ahead of shorter ones overall.
    tagged = [(p, t, 0) for p, t in (r.get("tensor_types") or {}).items()] + \
             [(p, t, 1) for p, t in boost_overrides(r)]
    for pfx, t, _ in sorted(tagged, key=lambda x: (-len(x[0]), x[2])):
        cmd += ["--tensor-type", f"{pfx}={t}"]
    # reuse: copy byte-identical tensors from a prior build (same hf+imatrix) instead of
    # regenerating — only the changed (e.g. boosted) tensors are quantized. The quantizer
    # verifies a matching reuse key, so a stale/mismatched prior safely falls back to a full
    # quantize. Skip silently if the prior isn't there yet.
    reuse = r.get("reuse")
    if reuse and not dry:
        if os.path.exists(reuse):
            cmd += ["--reuse", reuse]
        else:
            print(f"forgequant: reuse prior not found ({reuse}) — doing a full quantize", file=sys.stderr)
    if r.get("threads"): cmd += ["--threads", str(r["threads"])]
    if r.get("overwrite"): cmd += ["--overwrite"]
    if dry: cmd += ["--dry-run"]
    return cmd

def bench_corpus_path(r):
    """Where a `bench` recipe's corpus lives (without building it). None if no bench block."""
    if not r.get("bench"):
        return None
    return r.get("corpus") or os.path.join(MODELS_DIR, f"{r['name']}_corpus.txt")

def ensure_corpus(r):
    """If the recipe declares a `bench` block and its corpus is missing, build the corpus
    from benchy benchmarks (the activation paths of a domain expert). Side-effecting:
    fetches + renders. Call only when actually about to collect, never from `show`."""
    b = r.get("bench")
    if not b:
        return
    if not r.get("corpus"):
        r["corpus"] = bench_corpus_path(r)
    if os.path.exists(r["corpus"]):
        return
    sel = b.get("keys") or b.get("domain") or b.get("bundle")
    if isinstance(sel, list):
        sel = ",".join(sel)
    if not sel:
        die("recipe `bench` block needs `keys`, `domain` or `bundle`")
    print(f"forgequant: building calibration corpus from benchy [{sel}] "
          f"(answers={bool(b.get('answers'))}, mix={b.get('mix')}) ...")
    rec = forge_bench.build_corpus(sel, r["corpus"], answers=bool(b.get("answers")),
                                   mix_domain=b.get("mix"), cap=b.get("cap"),
                                   mode=b.get("mode", "both"))
    print(f"forgequant: corpus -> {r['corpus']} ({rec['prompts']} prompts from "
          f"{', '.join(rec['keys'])}); run recorded -> {rec.get('run_record')}")


def imatrix_cmd(r):
    if not r.get("corpus"):
        r["corpus"] = bench_corpus_path(r)   # display/use the path; building happens in cmd_imatrix
    if not r.get("corpus"): die("recipe has no `corpus` to build an imatrix from "
                                "(use a `bench` block, `render` to make one, or `capture` for live traffic)")
    if not r.get("imatrix"): die("recipe has no `imatrix` output path")
    model = r.get("imatrix_model") or r.get("template")
    if not model: die("need `imatrix_model` or `template` to run the model for imatrix collection")
    cmd = [DS4, "-m", resolve(model, r["name"]),
           "--imatrix-dataset", r["corpus"], "--imatrix-out", r["imatrix"],
           "--ssd-streaming", "--ssd-streaming-cache-experts", str(r.get("imatrix_cache", "40GB")),
           "--ctx", str(r.get("imatrix_ctx", 8192))]
    if r.get("imatrix_max_tokens"): cmd += ["--imatrix-max-tokens", str(r["imatrix_max_tokens"])]
    if r.get("imatrix_max_prompts"): cmd += ["--imatrix-max-prompts", str(r["imatrix_max_prompts"])]
    return cmd

def capture_cmd(r, port=8000):
    """ds4-server serving the forged (or template) model, collecting the imatrix live."""
    model = r["out"] if os.path.exists(r["out"]) else r.get("template")
    if not model: die("capture needs the recipe's `out` gguf or `template` to serve")
    out = r.get("imatrix") or os.path.join(MODELS_DIR, f"{r['name']}-live.dat")
    cmd = [DS4_SERVER, "-m", model, "--port", str(port),
           "--imatrix-out", out,
           "--imatrix-every", str(r.get("imatrix_every", 64)),
           "--imatrix-min-requests", str(r.get("imatrix_min_requests", 8)),
           "--ssd-streaming", "--ssd-streaming-cache-experts", str(r.get("imatrix_cache", "40GB"))]
    return cmd, out, model

def splice_cmd(r):
    s = r.get("splice")
    if not s: die("recipe has no `splice` block ({donor, layers, [base], [out]})")
    if not s.get("donor"): die("splice needs `donor` (a higher-precision GGUF)")
    base = resolve(s.get("base"), r["name"]) or (r["out"] if os.path.exists(r["out"])
                                                 else r.get("template"))
    if not base: die("splice needs a `base` gguf (or the recipe's out/template)")
    donor = resolve(s["donor"], r["name"])
    layers = parse_layers(s.get("layers", "auto:6"), r)
    out = resolve(s.get("out"), r["name"]) or os.path.join(
        MODELS_DIR, f"DeepSeek-V4-Flash-{r['name']}-splice.gguf")
    cmd = [sys.executable, SPLICER, "--base", base, "--donor", donor,
           "--q4-layers", ",".join(str(l) for l in layers), "--out", out]
    if s.get("force") or r.get("overwrite"): cmd += ["--force"]
    return cmd, out, layers

def sha256(path):
    if not os.path.exists(path): return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def ds4_git_rev():
    try:
        return subprocess.run(["git", "-C", DS4_DIR, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None

def run(cmd, cwd=DS4_DIR):
    print("forgequant$ " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=cwd).returncode
    if rc != 0:
        sig = f" (signal {signal.Signals(-rc).name})" if rc < 0 else ""
        print(f"forgequant: command exited with code {rc}{sig}", file=sys.stderr)
    return rc

def run_until_interrupt(cmd, cwd=DS4_DIR, grace=20):
    """Run a long-lived child (ds4-server capture) and treat Ctrl-C as a graceful stop:
    forward the SIGINT, then wait up to `grace` seconds for the child to flush its final
    imatrix snapshot before escalating. subprocess.run would SIGKILL the child ~0.25s
    after KeyboardInterrupt, truncating that snapshot — so we own the Popen here."""
    print("forgequant$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=cwd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\nforgequant: stopping capture — letting ds4-server flush its final snapshot…")
        try:
            proc.send_signal(signal.SIGINT)   # the child already got the terminal SIGINT; be explicit
            try:
                return proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try: return proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill(); return proc.wait()
        except Exception:
            proc.kill(); return -signal.SIGINT

def write_manifest(r, cmd, out, action, started):
    man = out + ".manifest.json"
    rec = {"schema": SCHEMA, "name": r["name"], "action": action,
           "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "duration_s": round(time.time() - started),
           "ds4_git": ds4_git_rev(),
           "recipe": r, "command": " ".join(cmd),
           "output": {"path": out, "bytes": os.path.getsize(out) if os.path.exists(out) else None,
                      "sha256": sha256(out)},
           "imatrix": {"path": r.get("imatrix"),
                       "sha256": sha256(r["imatrix"]) if r.get("imatrix") else None}}
    json.dump(rec, open(man, "w"), indent=2)
    print(f"forgequant: manifest -> {man}")

# ---- argv helpers for the free-form subcommands ----
def opt(extra, flag, default=None):
    if flag not in extra: return default
    i = extra.index(flag)
    if i + 1 >= len(extra): die(f"{flag} needs a value")
    return extra[i + 1]

def opt_int(extra, flag, default=None):
    v = opt(extra, flag, None)
    if v is None: return default
    try: return int(v)
    except ValueError: die(f"{flag} expects an integer, got {v!r}")

# ---- commands ----
def cmd_list(_a, _x):
    if not os.path.isdir(RECIPES): die(f"no recipes dir at {RECIPES}")
    for f in sorted(os.listdir(RECIPES)):
        if f.endswith(".json"):
            try:
                r = json.load(open(os.path.join(RECIPES, f)))
            except Exception:
                continue
            tags = "".join(t for t, k in ((" [boost]", "boost"), (" [splice]", "splice"))
                           if r.get(k))
            print(f"  {f[:-5]:18} {r.get('description', '')}{tags}")

def cmd_show(a, _x):
    r = load_recipe(a)
    print(json.dumps(r, indent=2))
    # 'auto:N' boost/splice can't resolve until the imatrix exists; show a placeholder
    # rather than dying, so `show` works before `imatrix`/`capture` has been run.
    try:
        print("\n# quantize command:\n" + " ".join(quant_cmd(r)))
        if r.get("boost"):
            ov = boost_overrides(r)
            layers = sorted(set(int(p.split(".")[1]) for p, _ in ov))
            print(f"\n# boost: layers {layers} -> {r['boost']['type']} "
                  f"({len(ov)} tensor overrides)")
    except SystemExit as e:
        print(f"\n# quantize command: (unresolved — {e})")
        print(f"# boost {r['boost']['layers']} -> {r['boost']['type']} "
              f"(layers resolved at forge time, once the imatrix exists)")
    if r.get("corpus") or r.get("bench"):
        try:
            print("\n# imatrix command:\n" + " ".join(imatrix_cmd(r)))
        except SystemExit as e:
            print(f"\n# imatrix command: (unresolved — {e})")
    if r.get("splice"):
        try:
            cmd, out, layers = splice_cmd(r)
            print(f"\n# splice command (layers {layers}):\n" + " ".join(cmd))
        except SystemExit as e:
            print(f"\n# splice command: (unresolved — {e})")

def cmd_verify(a, _x):
    r = load_recipe(a)
    ok = True
    def chk(cond, msg, warn=False):
        nonlocal ok
        mark = "ok " if cond else ("WARN" if warn else "FAIL")
        if not cond and not warn: ok = False
        print(f"  [{mark}] {msg}")
    chk(os.path.exists(QUANTIZE), f"quantizer: {QUANTIZE}")
    chk(bool(r.get("hf")) and os.path.exists(os.path.join(r.get("hf") or "", "model.safetensors.index.json")),
        f"hf source: {r.get('hf')}")
    chk(bool(r.get("template")) and os.path.exists(r.get("template") or ""), f"template: {r.get('template')}")
    if r.get("imatrix"):
        if os.path.exists(r["imatrix"]):
            try:
                stats = forge_imatrix.cached_stats(r["imatrix"])
                chk(True, f"imatrix: {r['imatrix']} ({stats['n_layers']} layers x "
                          f"{stats['n_experts']} experts, dataset={stats.get('dataset')})")
            except Exception as e:
                chk(False, f"imatrix unreadable: {e}")
        else:
            chk(bool(r.get("corpus")), f"imatrix missing: {r['imatrix']} "
                f"({'will be built from corpus' if r.get('corpus') else 'and no corpus set'})",
                warn=bool(r.get("corpus")))
    if r.get("corpus"):
        chk(os.path.exists(r["corpus"]), f"corpus: {r['corpus']}")
    boost_ok = True
    if r.get("boost"):
        try:
            ov = boost_overrides(r)
            chk(True, f"boost: {len(ov)} overrides -> {r['boost']['type']}")
        except SystemExit as e:   # 'auto:N' before the imatrix exists is a warn, not a hard fail
            boost_ok = False
            chk(bool(r.get("corpus") or r.get("bench")), f"boost: {e}",
                warn=bool(r.get("corpus") or r.get("bench")))
    if boost_ok and os.path.exists(QUANTIZE) and r.get("hf") and r.get("template") \
            and os.path.exists(r.get("template") or ""):
        try:
            probe = quant_cmd(r, dry=True)
            if "--overwrite" not in probe: probe += ["--overwrite"]  # --dry-run never writes
            o = subprocess.run(probe, cwd=DS4_DIR, capture_output=True,
                               text=True, timeout=120).stdout
            approx = next((int(l.split()[-1]) for l in o.splitlines()
                           if l.startswith("approx_file_bytes:")), None)
            if approx:
                st = os.statvfs(os.path.dirname(r["out"]) or ".")
                free = st.f_bavail * st.f_frsize
                chk(free > approx, f"disk: need ~{approx / 1e9:.0f} GB, "
                                   f"free {free / 1e9:.0f} GB at {os.path.dirname(r['out'])}")
        except (Exception, SystemExit) as e:
            chk(True, f"dry-run skipped ({e})", warn=True)
    print("forgequant: verify " + ("OK" if ok else "FAILED"))
    if not ok: sys.exit(1)

def cmd_imatrix(a, _x):
    r = load_recipe(a)
    ensure_corpus(r)
    if run(imatrix_cmd(r)) != 0: die("imatrix collection failed")
    print(f"forgequant: imatrix -> {r['imatrix']}  "
          f"({(os.path.getsize(r['imatrix']) >> 20) if os.path.exists(r['imatrix']) else '?'} MB)")

def cmd_capture(a, extra):
    r = load_recipe(a)
    port = opt_int(extra, "--port", 8000)
    cmd, out, model = capture_cmd(r, port)
    print(f"forgequant: serving {os.path.basename(model)} on :{port}, collecting the "
          f"activation paths of live traffic into {out}")
    print("forgequant: send your real workload (chat, evals, agents) — Ctrl-C to stop; "
          "snapshots are written periodically")
    rc = run_until_interrupt(cmd)
    if os.path.exists(out):
        print(f"forgequant: captured imatrix -> {out} ({os.path.getsize(out) >> 20} MB)")
        print(f"forgequant: inspect it with `forgequant.py paths {out}`")
    elif rc != 0:
        die("capture failed before writing an imatrix (needs >= imatrix-min-requests requests)")

def cmd_render(a, extra):
    out = opt(extra, "-o") or opt(extra, "--out")
    mode = opt(extra, "--mode", "both")
    if not out: die("usage: forgequant.py render <prompts.txt|.jsonl> -o <corpus.txt> [--mode nothink|think|both]")
    if mode not in ("nothink", "think", "both"):
        die(f"bad --mode {mode!r} (expected nothink|think|both)")
    modes = ("nothink", "think") if mode == "both" else (mode,)
    n, size = forge_corpus.build_corpus(os.path.expanduser(a), os.path.expanduser(out), modes)
    print(f"forgequant: {n} rendered prompts -> {out} ({size >> 10} KB, ~{size // 4} tokens est.)")
    print(f"forgequant: point a recipe's `corpus` at it and run `imatrix <recipe>`")

def _dat_of(a):
    if a.endswith(".dat"):
        p = os.path.expanduser(a)
        return p if os.path.exists(p) else die(f"imatrix not found: {p}")
    r = load_recipe(a)
    if not r.get("imatrix"): die(f"recipe {r['name']} has no `imatrix`")
    if not os.path.exists(r["imatrix"]): die(f"imatrix not found: {r['imatrix']}")
    return r["imatrix"]

def cmd_paths(a, extra):
    dat = _dat_of(a)
    stats = forge_imatrix.analyze(forge_imatrix.load_dat(dat))
    if "--json" in extra:
        print(json.dumps(forge_imatrix.to_json(stats), indent=2)); return
    if "--contrast" in extra:
        # the legible compare: one diverging bar per layer, not two 43x256 grids
        base_dat = _dat_of(opt(extra, "--contrast"))
        base = forge_imatrix.analyze(forge_imatrix.load_dat(base_dat))
        rows = forge_imatrix.contrast(stats, base)
        dn = os.path.basename(a) if a.endswith(".dat") else a
        bn = os.path.basename(opt(extra, "--contrast"))
        print(forge_imatrix.contrast_bars(rows, domain=dn, base=bn))
        print(f"\ncontrast boost candidates (most domain-distinctive): "
              f"{forge_imatrix.suggest_contrast(stats, base, 6)}")
        return
    if "--diff" in extra:
        other = forge_imatrix.analyze(forge_imatrix.load_dat(_dat_of(opt(extra, "--diff"))))
        print("layer  cosine  Δshare   experts only in domain")
        for row in sorted(forge_imatrix.diff(stats, other), key=lambda r: r["cosine"]):
            print(f"blk.{row['layer']:>2} {row['cosine']:7.4f} {row['share_delta'] * 100:+6.2f}%  "
                  f"{','.join(str(e) for e in row['experts_only_b']) or '—'}")
        return
    print(f"imatrix: {dat}  (dataset: {stats.get('dataset') or 'live capture'})\n")
    print(forge_imatrix.heatmap(stats))
    hot = forge_imatrix.hot_layers(stats, 6)
    print(f"\nhot layers: {sorted(r['layer'] for r in hot)}")

def cmd_suggest(a, extra):
    r = load_recipe(a)
    if not r.get("imatrix") or not os.path.exists(r["imatrix"]):
        die("suggest needs the recipe's imatrix — run `imatrix` or `capture` first")
    n = opt_int(extra, "--top", 6)
    to_type = opt(extra, "--type", "q4_k")
    check_type(to_type, "suggest --type")
    stats = forge_imatrix.analyze(forge_imatrix.load_dat(r["imatrix"]))
    layers = forge_imatrix.suggest_boost(stats, n)
    # size delta per family from its CURRENT type (w2 is often q2_k, not iq2_xxs)
    q = r.get("quant") or {}
    delta = sum(forge_imatrix.boost_size_delta(stats, layers, q.get(f"routed_{fam}", "iq2_xxs"),
                                               to_type, families=(fam,)) for fam in ("w1", "w2", "w3"))
    print(f"# {n} hottest layers in {os.path.basename(r['imatrix'])}; "
          f"estimated size delta -> {to_type}: +{delta / 1e9:.1f} GB")
    print(json.dumps({"boost": {"layers": ",".join(str(l) for l in layers), "type": to_type}},
                     indent=2))
    print(f"# or keep it adaptive:  \"boost\": {{\"layers\": \"auto:{n}\", \"type\": \"{to_type}\"}}")

def cmd_quantize(a, _x):
    r = load_recipe(a)
    started = time.time()
    if not os.path.exists(QUANTIZE): die(f"deepseek4-quantize not found at {QUANTIZE} (set DS4_DIR or build gguf-tools)")
    if r.get("imatrix") and not os.path.exists(r["imatrix"]):
        die(f"imatrix not found: {r['imatrix']} — run `forgequant.py imatrix {a}` first, "
            f"capture one live (`capture {a}`), or set a `corpus`")
    cmd = quant_cmd(r)
    if run(cmd) != 0: die("quantization failed")
    write_manifest(r, cmd, r["out"], "quantize", started)
    print(f"forgequant: done -> {r['out']}")

def cmd_splice(a, _x):
    r = load_recipe(a)
    started = time.time()
    if not os.path.exists(SPLICER): die(f"splicer not found at {SPLICER} (update ds4)")
    cmd, out, layers = splice_cmd(r)
    if run(cmd) != 0: die("splice failed")
    write_manifest(r, cmd, out, "splice", started)
    print(f"forgequant: spliced layers {layers} -> {out}")

def cmd_build(a, _x):
    r = load_recipe(a)
    if (r.get("corpus") or r.get("bench")) and r.get("imatrix") and not os.path.exists(r["imatrix"]):
        print("forgequant: imatrix missing, building it from corpus first...")
        cmd_imatrix(a, None)
    cmd_quantize(a, None)

def cmd_bench(a, extra):
    """Proxy to the benchy bridge: `forgequant.py bench [list|bundles|fetch|corpus] ...`."""
    forge_bench.main(([a] if a else ["list"]) + extra)

CMDS = {"list": cmd_list, "show": cmd_show, "verify": cmd_verify, "bench": cmd_bench,
        "imatrix": cmd_imatrix, "capture": cmd_capture, "render": cmd_render,
        "paths": cmd_paths, "suggest": cmd_suggest, "quantize": cmd_quantize,
        "splice": cmd_splice, "build": cmd_build}

# commands that take a free-form selection / no recipe instead of a <recipe> arg
NO_RECIPE = {"bench"}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__); sys.exit(0 if len(sys.argv) < 2 else 2)
    cmd = sys.argv[1]
    if cmd != "list" and cmd not in NO_RECIPE and len(sys.argv) < 3:
        die(f"usage: forgequant.py {cmd} <recipe>")
    CMDS[cmd](sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[3:])

if __name__ == "__main__":
    main()
