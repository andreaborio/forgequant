#!/usr/bin/env python3
"""forge_bench — the single bridge to benchy's benchmark registry.

forgequant does not ship or re-implement benchmarks. It fetches them from **benchy**
(github.com/andreaborio/benchy), vendored here as a **git submodule** so anyone who
clones forgequant gets the exact same benchmark source:

    git clone --recursive https://github.com/andreaborio/forgequant.git
    # or, in an existing clone:
    git submodule update --init                # pull benchy in
    git submodule update --remote benchy       # bump to benchy's latest

benchy's `fetch_benchmarks.py` is the one source of truth: a registry of real,
non-saturated evals (MMLU-Pro, SuperGPQA, HumanEval, MBPP, MedXpertQA, MedQA, …)
pulled live from the HuggingFace datasets-server and normalized. We import that
module from the submodule, fetch rows on demand (cached in benchy/data, which
benchy gitignores — licensed data is never redistributed), and render them into a
DeepSeek-V4 calibration corpus. Because benchy is the source, the set updates as
benchy updates.

Config:
  BENCHY_DIR   override the benchy checkout (default: the ./benchy submodule,
               then ~/BEEP/benchy, then ~/benchy)

Commands:
  forge_bench.py list                       the benchy registry (tier · domain · present)
  forge_bench.py bundles                     domain bundles (code/medical/reasoning/…)
  forge_bench.py fetch <sel>                 download rows into benchy/data (key|domain|bundle|current|all)
  forge_bench.py corpus <sel> -o OUT [opts]  render a calibration corpus from a selection

corpus options:
  --answers            include gold answers as assistant turns (calibrate the
                       ANSWERING paths too, not just reading)
  --mix DOMAIN         interleave a general corpus from another domain to avoid
                       over-specializing (e.g. code calibration mixed with reasoning)
  --cap N              rows per benchmark (default: benchy's per-set cap)
  --mode M             nothink | think | both (default both)

<sel> is a benchmark key (humaneval), a bundle/domain name (code), 'current'
(all non-saturated), or a comma list (humaneval,mbpp,mmlu_cs).

Every corpus build writes a tracked run record to bench/runs/<ts>__<sel>.json
(benchmark keys, per-file row counts + sha, options, resulting corpus sha) — so a
calibration is always traceable to the exact benchmark snapshot it came from.
"""
import datetime, hashlib, importlib.util, json, os, sys

import forge_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "bench", "runs")
PACKS = os.path.join(HERE, "bench", "packs")

# Domain bundles: an "expert" -> the non-saturated benchmarks that probe it. Keys must
# exist in benchy's REGISTRY; unknown keys are dropped with a warning at resolve time.
BUNDLES = {
    "code":      ["humaneval", "mbpp", "mmlu_cs"],
    "medical":   ["medxpertqa", "medmcqa", "medqa_test"],
    "reasoning": ["mmlu_pro", "supergpqa", "logic"],
    "knowledge": ["mmlu_pro", "openbookqa"],
    "truthful":  ["truthfulqa"],
    "broad":     ["mmlu_pro", "supergpqa", "humaneval", "medxpertqa", "truthfulqa"],
}


def die(msg):
    sys.exit("forge_bench: " + msg)


def benchy_dir():
    # the vendored submodule first, so a fresh clone Just Works; then overrides/fallbacks
    for cand in (os.environ.get("BENCHY_DIR"), os.path.join(HERE, "benchy"),
                 "~/BEEP/benchy", "~/benchy"):
        if cand:
            p = os.path.expanduser(cand)
            if os.path.exists(os.path.join(p, "fetch_benchmarks.py")):
                return p
    die("benchy not found — run `git submodule update --init`, or set BENCHY_DIR "
        "to your benchy checkout (github.com/andreaborio/benchy)")


def load_benchy():
    """Import benchy's fetch_benchmarks as a module (it is stdlib, importable)."""
    d = benchy_dir()
    spec = importlib.util.spec_from_file_location("benchy_fetch",
                                                  os.path.join(d, "fetch_benchmarks.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, d


def resolve_keys(fb, sel):
    """A selection string -> ordered unique benchmark keys known to benchy."""
    reg = fb.REGISTRY
    keys = []
    for tok in str(sel).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in reg:
            keys.append(tok)
        elif tok in BUNDLES:
            keys += BUNDLES[tok]
        elif tok == "current":
            keys += fb.current_keys()
        elif tok == "all":
            keys += list(reg)
        elif any(s["domain"] == tok for s in reg.values()):   # a raw domain name
            keys += [k for k, s in reg.items() if s["domain"] == tok]
        else:
            die(f"unknown selection '{tok}' (try: a key, a bundle {list(BUNDLES)}, "
                f"a domain, 'current', or 'all')")
    seen, out = set(), []
    for k in keys:
        if k in reg and k not in seen:
            seen.add(k); out.append(k)
    if not out:
        die(f"selection '{sel}' resolved to no benchmarks")
    return out


def data_path(fb, bdir, key):
    return os.path.join(getattr(fb, "DATA", os.path.join(bdir, "data")), key + ".jsonl")


def ensure_fetched(fb, bdir, keys):
    """Fetch any selected benchmark not already cached in benchy/data. Returns present keys."""
    present = []
    for k in keys:
        p = data_path(fb, bdir, k)
        if not os.path.exists(p):
            print(f"forge_bench: fetching {k} from benchy …")
            try:
                fb.fetch(k)
            except Exception as e:
                print(f"forge_bench: ! {k} fetch failed ({e}) — skipping")
        if os.path.exists(p):
            present.append(k)
    if not present:
        die("nothing fetched (network error, or all selected sets are gated/manual)")
    return present


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()[:12]


def build_corpus(sel, out_path, answers=False, mix_domain=None, cap=None,
                 mode="both", record_run=True):
    """Fetch the selection from benchy and render it into a calibration corpus.

    Returns the run record dict (also written under bench/runs/ when record_run).
    """
    fb, bdir = load_benchy()
    keys = resolve_keys(fb, sel)
    present = ensure_fetched(fb, bdir, keys)
    inputs = [data_path(fb, bdir, k) for k in present]
    modes = ("nothink", "think") if mode == "both" else (mode,)

    mix_path = None
    if mix_domain:
        mkeys = ensure_fetched(fb, bdir, resolve_keys(fb, mix_domain))
        mix_path = os.path.join(os.path.dirname(out_path) or ".",
                                f".forgemix_{mix_domain}.txt")
        forge_corpus.build_corpus_multi([data_path(fb, bdir, k) for k in mkeys],
                                        mix_path, modes=modes, cap=cap)

    n, size, per_file = forge_corpus.build_corpus_multi(
        inputs, os.path.expanduser(out_path), modes=modes, answers=answers,
        cap=cap, mix=mix_path)
    if mix_path and os.path.exists(mix_path):
        os.remove(mix_path)

    rec = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
           "selection": sel, "keys": present, "benchy_dir": bdir,
           "answers": answers, "mix": mix_domain, "cap": cap, "mode": mode,
           "sources": {k: {"rows_file": data_path(fb, bdir, k),
                           "sha": file_sha(data_path(fb, bdir, k))} for k in present},
           "prompts": n, "corpus": os.path.expanduser(out_path),
           "corpus_bytes": size, "corpus_sha": file_sha(os.path.expanduser(out_path)),
           "per_file": per_file}
    if record_run:
        os.makedirs(RUNS, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        safe = "".join(c for c in str(sel) if c.isalnum() or c in "-_,")[:40]
        path = os.path.join(RUNS, f"{ts}__{safe}.json")
        json.dump(rec, open(path, "w"), indent=2)
        rec["run_record"] = path
    return rec


# ---- commands ----
def cmd_list(_args):
    fb, _ = load_benchy()
    meta = fb.registry_meta()
    print("benchy registry — '✓' cached in benchy/data. fit [mcq]=letter [code]=executed\n")
    for tier, head in (("current", "CURRENT · still discriminating (use these for expert calibration)"),
                       ("legacy", "LEGACY · saturated (regression/sanity only)")):
        print(f"  == {head} ==")
        for b in meta["available"]:
            if b["tier"] != tier:
                continue
            print(f"   {'✓' if b['present'] else ' '} {b['key']:<14} [{b['fit']}] "
                  f"{b['domain']:<12} {b['name']}")
        print()
    if meta.get("manual"):
        print("  == gated / manual (need a HF token; see benchy DATA.md) ==")
        for m in meta["manual"]:
            print(f"      {m['key']:<14} {m['note']}")


def cmd_bundles(_args):
    fb, _ = load_benchy()
    reg = fb.REGISTRY
    print("domain bundles — calibrate an expert on its non-saturated benchmarks:\n")
    for name, keys in BUNDLES.items():
        known = [k for k in keys if k in reg]
        print(f"  {name:<11} {', '.join(known)}")
    if os.path.isdir(PACKS):
        packs = [f[:-6] for f in sorted(os.listdir(PACKS)) if f.endswith(".jsonl")]
        if packs:
            print(f"\n  local packs (no online equivalent): {', '.join(packs)}")


def cmd_fetch(args):
    if not args:
        die("usage: forge_bench.py fetch <key|bundle|domain|current|all>")
    fb, bdir = load_benchy()
    keys = resolve_keys(fb, args[0])
    present = ensure_fetched(fb, bdir, keys)
    print(f"forge_bench: cached {len(present)} set(s) in {getattr(fb, 'DATA', bdir+'/data')}")


def cmd_corpus(args):
    if not args:
        die("usage: forge_bench.py corpus <sel> -o OUT [--answers] [--mix DOMAIN] "
            "[--cap N] [--mode nothink|think|both]")
    sel = args[0]
    out = answers = mix = cap = None
    mode = "both"
    i = 1
    while i < len(args):
        a = args[i]
        if a in ("-o", "--out"): out = args[i + 1]; i += 2
        elif a == "--answers": answers = True; i += 1
        elif a == "--mix": mix = args[i + 1]; i += 2
        elif a == "--cap": cap = int(args[i + 1]); i += 2
        elif a == "--mode": mode = args[i + 1]; i += 2
        else: die(f"unknown option {a}")
    if not out:
        die("corpus needs -o OUT")
    rec = build_corpus(sel, out, answers=bool(answers), mix_domain=mix, cap=cap, mode=mode)
    print(f"forge_bench: {rec['prompts']} prompts from {', '.join(rec['keys'])} "
          f"-> {out} ({rec['corpus_bytes'] >> 10} KB, ~{rec['corpus_bytes'] // 4} tokens est.)")
    print(f"forge_bench: per-set {rec['per_file']}")
    print(f"forge_bench: run recorded -> {rec.get('run_record')}")
    print(f"forge_bench: point a recipe's `corpus` here, or use a `bench` block, then "
          f"`forgequant.py imatrix <recipe>`")


CMDS = {"list": cmd_list, "bundles": cmd_bundles, "fetch": cmd_fetch, "corpus": cmd_corpus}


def main(argv):
    if not argv or argv[0] not in CMDS:
        print(__doc__); sys.exit(0 if not argv else 2)
    CMDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
