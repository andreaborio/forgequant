#!/usr/bin/env python3
"""forgequant — config-driven asymmetric quantization for DeepSeek-V4-Flash (ds4).

A *recipe* (recipes/<name>.json) declares the per-tensor-family quant types and an
imatrix (or a corpus to build one). forgequant turns a recipe into a quantized GGUF,
reproducibly, writing a manifest next to the output. It wraps ds4's
`deepseek4-quantize` (and `ds4 --imatrix-dataset` for the optional imatrix step) — so
you stop hand-assembling long quantizer command lines.

Families NOT listed in a recipe's `quant` block are copied verbatim from `template`.

Commands:
  forgequant.py list                 list available recipes
  forgequant.py show <recipe>        resolved recipe + the exact quantize command (no run)
  forgequant.py imatrix <recipe>     build the imatrix from the recipe's corpus
  forgequant.py quantize <recipe>    run the quantization (recipe -> GGUF + manifest)
  forgequant.py build <recipe>       full pipeline: imatrix (if needed) then quantize

<recipe> is a preset name (recipes/<name>.json) or a path to a .json file.

Config via env:
  DS4_DIR     ds4 checkout (default ~/ds4) — provides ds4 + gguf-tools/deepseek4-quantize
  MODELS_DIR  where models/imatrices live (default ~/ds4-models); usable as {models} in paths

Stdlib only. Quantization is deterministic: same recipe + same imatrix => same GGUF.
"""
import json, os, sys, subprocess, hashlib, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPES = os.path.join(HERE, "recipes")
DS4_DIR = os.path.expanduser(os.environ.get("DS4_DIR", "~/ds4"))
MODELS_DIR = os.path.expanduser(os.environ.get("MODELS_DIR", "~/ds4-models"))
QUANTIZE = os.path.join(DS4_DIR, "gguf-tools", "deepseek4-quantize")
DS4 = os.path.join(DS4_DIR, "ds4")

# recipe family key -> deepseek4-quantize flag
FLAG = {"routed_w1": "--routed-w1", "routed_w2": "--routed-w2", "routed_w3": "--routed-w3",
        "attention": "--attention", "attn_proj": "--attn-proj", "shared": "--shared",
        "embedding": "--embedding", "output": "--output"}

def die(msg): sys.exit("forgequant: " + msg)

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
    for k in ("hf", "template", "imatrix", "corpus", "out"):
        if r.get(k): r[k] = resolve(r[k], r["name"])
    if not r.get("out"):
        r["out"] = os.path.join(MODELS_DIR, f"DeepSeek-V4-Flash-{r['name']}.gguf")
    for fam in (r.get("quant") or {}):
        if fam not in FLAG:
            die(f"unknown family '{fam}' in recipe (allowed: {', '.join(FLAG)})")
    return r

def quant_cmd(r, dry=False):
    for k in ("hf", "template"):
        if not r.get(k): die(f"recipe missing required field: {k}")
    cmd = [QUANTIZE, "--hf", r["hf"], "--template", r["template"], "--out", r["out"]]
    if r.get("imatrix"): cmd += ["--imatrix", r["imatrix"]]
    if r.get("imatrix_strict"): cmd += ["--imatrix-strict"]
    for fam, t in (r.get("quant") or {}).items():
        cmd += [FLAG[fam], str(t)]
    if r.get("overwrite"): cmd += ["--overwrite"]
    if dry: cmd += ["--dry-run"]
    return cmd

def imatrix_cmd(r):
    if not r.get("corpus"): die("recipe has no `corpus` to build an imatrix from")
    if not r.get("imatrix"): die("recipe has no `imatrix` output path")
    model = r.get("imatrix_model") or r.get("template")
    if not model: die("need `imatrix_model` or `template` to run the model for imatrix collection")
    cmd = [DS4, "-m", resolve(model, r["name"]),
           "--imatrix-dataset", r["corpus"], "--imatrix-out", r["imatrix"],
           "--ssd-streaming", "--ssd-streaming-cache-experts", str(r.get("imatrix_cache", "40GB")),
           "--ctx", str(r.get("imatrix_ctx", 8192))]
    if r.get("imatrix_max_tokens"): cmd += ["--imatrix-max-tokens", str(r["imatrix_max_tokens"])]
    return cmd

def sha256(path):
    if not os.path.exists(path): return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def run(cmd, cwd=DS4_DIR):
    print("forgequant$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd).returncode

def write_manifest(r, cmd):
    out = r["out"]
    man = out + ".manifest.json"
    rec = {"name": r["name"], "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "recipe": r, "command": " ".join(cmd),
           "output": {"path": out, "bytes": os.path.getsize(out) if os.path.exists(out) else None,
                      "sha256": sha256(out)},
           "imatrix": {"path": r.get("imatrix"), "sha256": sha256(r["imatrix"]) if r.get("imatrix") else None}}
    json.dump(rec, open(man, "w"), indent=2)
    print(f"forgequant: manifest -> {man}")

# ---- commands ----
def cmd_list(_a):
    if not os.path.isdir(RECIPES): die(f"no recipes dir at {RECIPES}")
    for f in sorted(os.listdir(RECIPES)):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(RECIPES, f)))
            print(f"  {f[:-5]:18} {r.get('description', '')}")

def cmd_show(a):
    r = load_recipe(a)
    print(json.dumps(r, indent=2))
    print("\n# quantize command:\n" + " ".join(quant_cmd(r, dry=True)))
    if r.get("corpus"):
        print("\n# imatrix command:\n" + " ".join(imatrix_cmd(r)))

def cmd_imatrix(a):
    r = load_recipe(a)
    if run(imatrix_cmd(r)) != 0: die("imatrix collection failed")
    print(f"forgequant: imatrix -> {r['imatrix']}  ({(os.path.getsize(r['imatrix'])>>20) if os.path.exists(r['imatrix']) else '?'} MB)")

def cmd_quantize(a):
    r = load_recipe(a)
    if not os.path.exists(QUANTIZE): die(f"deepseek4-quantize not found at {QUANTIZE} (set DS4_DIR or build gguf-tools)")
    if r.get("imatrix") and not os.path.exists(r["imatrix"]):
        die(f"imatrix not found: {r['imatrix']} — run `forgequant.py imatrix {a}` first, or set a `corpus`")
    cmd = quant_cmd(r)
    if run(cmd) != 0: die("quantization failed")
    write_manifest(r, cmd)
    print(f"forgequant: done -> {r['out']}")

def cmd_build(a):
    r = load_recipe(a)
    if r.get("corpus") and r.get("imatrix") and not os.path.exists(r["imatrix"]):
        print("forgequant: imatrix missing, building it from corpus first...")
        cmd_imatrix(a)
    cmd_quantize(a)

CMDS = {"list": cmd_list, "show": cmd_show, "imatrix": cmd_imatrix,
        "quantize": cmd_quantize, "build": cmd_build}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__); sys.exit(0 if len(sys.argv) < 2 else 2)
    cmd = sys.argv[1]
    if cmd != "list" and len(sys.argv) < 3:
        die(f"usage: forgequant.py {cmd} <recipe>")
    CMDS[cmd](sys.argv[2] if len(sys.argv) > 2 else None)

if __name__ == "__main__":
    main()
