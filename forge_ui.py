#!/usr/bin/env python3
"""forge-ui — a small web dashboard to drive + monitor forgequant.

Pick a recipe (or build one from a template), launch build/quantize/imatrix/
capture/splice from the browser, watch the long quantization live (per-tensor
progress, ETA, throughput) plus a tail of the log, browse past runs, and open the
*brain map* of any imatrix — the per-layer/per-expert activation heatmap that shows
which paths of the network a workload lights up, with one-click boost suggestions.

Run:  python3 forge_ui.py [port]      # default 8060  ->  http://localhost:8060
Env:  DS4_DIR, MODELS_DIR (same as forgequant).
"""
import json, os, sys, re, subprocess, signal, threading, time, datetime, glob, html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import forge_imatrix

HERE = os.path.dirname(os.path.abspath(__file__))
FORGEQUANT = os.path.join(HERE, "forgequant.py")
RECIPES = os.path.join(HERE, "recipes")
DS4_DIR = os.path.expanduser(os.environ.get("DS4_DIR", "~/ds4"))
MODELS_DIR = os.path.expanduser(os.environ.get("MODELS_DIR", "~/ds4-models"))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)

JOB = {}   # {"recipe","action","log","started","proc","samples":[(t,n)]}
JOB_LOCK = threading.Lock()

NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]")
ACTIONS = ("build", "quantize", "imatrix", "capture", "splice")

# ---------------- model / recipe data ----------------
def safe_name(name):
    """Collapse any user-supplied name to a single safe path component."""
    return NAME_RE.sub("", os.path.basename(str(name or "")))[:80]

def list_recipes():
    out = []
    for f in sorted(glob.glob(os.path.join(RECIPES, "*.json"))):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        out.append({"name": r.get("name", os.path.basename(f)[:-5]),
                    "description": r.get("description", ""),
                    "quant": r.get("quant", {}), "imatrix": r.get("imatrix"),
                    "corpus": r.get("corpus"), "boost": r.get("boost"),
                    "splice": r.get("splice"), "note": r.get("_note", "")})
    return out

def list_models():
    out = []
    for g in sorted(glob.glob(os.path.join(MODELS_DIR, "*.gguf")), key=os.path.getmtime, reverse=True):
        man = g + ".manifest.json"
        rec = {"path": os.path.basename(g),
               "gb": round(os.path.getsize(g) / 1e9, 1),
               "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(g)).isoformat(timespec="minutes")}
        if os.path.exists(man):
            try:
                m = json.load(open(man))
                rec["recipe"] = m.get("name")
                rec["created"] = m.get("created")
                rec["quant"] = (m.get("recipe", {}) or {}).get("quant", {})
                rec["boost"] = (m.get("recipe", {}) or {}).get("boost")
                rec["action"] = m.get("action", "quantize")
                rec["imatrix_sha"] = ((m.get("imatrix", {}) or {}).get("sha256") or "")[:12]
            except Exception:
                pass
        out.append(rec)
    return out[:30]

# quant TARGET types deepseek4-quantize can actually produce ("" = copy from template).
# Only these have ds4q_can_quantize()==true in ds4 quants.c, plus f16/bf16 passthrough;
# offering q3_k/iq3_xxs/etc would build recipes the quantizer rejects.
QUANT_TYPES = ["", "iq2_xxs", "q2_k", "q4_k", "q8_0", "bf16", "f16"]
BOOST_TYPES = ["q4_k", "q8_0", "bf16"]
FAMILIES = ["routed_w1", "routed_w2", "routed_w3", "attention", "attn_proj", "shared", "embedding", "output"]

# Evidence-based boost guidance (from a verified deep-research pass; see commit notes). Drives
# the dashboard's boost presets + notes so the control is self-explanatory and the defaults sound.
BOOST_GUIDE = {
    "tooltip": "Upcast the routed experts of the N most active layers to Q4_K. down-proj is the most "
               "quant-sensitive tensor, so boost it first; quality gains flatten above ~4-bit.",
    "default_n": 6,
    "presets": [
        {"key": "small", "n": 6, "type": "q4_k", "families": ["w2"],
         "label": "down-proj only · 6L",
         "when": "Tightest budget: down-proj (w2) is the most quant-sensitive expert tensor, so boosting "
                 "just it on the 6 hottest layers buys most of the gain for ~1/4 the size."},
        {"key": "medium", "n": 6, "type": "q4_k", "families": ["w1", "w2", "w3"],
         "label": "all experts · 6L · recommended",
         "when": "Recommended. The 6 hottest layers (~14% of 43), all three expert families to Q4_K — the "
                 "steep part of the 2→4-bit curve. Mirrors ds4's proven last-6 mixed experiment."},
        {"key": "aggressive", "n": 8, "type": "q4_k", "families": ["w1", "w2", "w3"],
         "label": "all experts · 8L",
         "when": "More headroom: 8 of 43 layers. Beyond ~10 layers / ~4-5 bpw the curve flattens "
                 "(4→5-bit adds little), so going higher mostly costs GB."},
    ],
    "selection": "Energy (default) ranks layers by activation intensity — the best single-imatrix proxy "
                 "for quant sensitivity. Contrast needs a domain + a baseline imatrix and instead finds "
                 "the layers a domain uses differently (for specializing a build).",
    "notes": [
        "down-proj (w2) is the most quant-sensitive expert tensor — if budget is tight, boost it first.",
        "2→4-bit is a big jump (~+3.9 pts Aider Polyglot); 4→5-bit adds ~+1.0 — so Q4_K is the target.",
        "Concentrate bits on a few layers, don't spread thin — layer-wise beats expert-wise.",
        "Prefer the first MoE blocks, then add the last 1-2 (model-specific, moderate evidence).",
        "N=6 is a sound heuristic, not a measured optimum for V4-Flash — A/B with benchy to find the knee.",
    ],
}

def list_imatrices():
    out = []
    for d in sorted(glob.glob(os.path.join(MODELS_DIR, "*.dat")), key=os.path.getmtime, reverse=True):
        out.append({"name": os.path.basename(d),
                    "mb": round(os.path.getsize(d) / 1e6),
                    "analyzed": os.path.exists(d + ".fqstats.json"),
                    "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(d)).isoformat(timespec="minutes")})
    return out

def imatrix_path(name):
    """Validated path of a .dat inside MODELS_DIR (None if fishy)."""
    name = safe_name(name)
    if not name.endswith(".dat"):
        return None
    p = os.path.join(MODELS_DIR, name)
    return p if os.path.exists(p) else None

def imatrix_stats(name):
    p = imatrix_path(name)
    if not p:
        return {"error": "unknown imatrix"}
    try:
        return forge_imatrix.cached_stats(p)
    except Exception as e:
        return {"error": str(e)}

def imatrix_diff(a, b):
    sa, sb = imatrix_stats(a), imatrix_stats(b)
    if sa.get("error") or sb.get("error"):
        return {"error": sa.get("error") or sb.get("error")}
    rows = forge_imatrix.diff_cached(sa, sb)
    boost = sorted(r["layer"] for r in sorted(rows, key=lambda r: r["cosine"])[:6])
    return {"rows": rows, "boost": boost}

# base type each expert family sits at before a boost (so the size delta is accurate)
BOOST_BASE = {"w1": "iq2_xxs", "w3": "iq2_xxs", "w2": "q2_k"}

def boost_gb(s, layers, to_type, families):
    total = sum(forge_imatrix.boost_delta_cached(s, layers, BOOST_BASE[f], to_type, families=(f,))
                for f in families if f in BOOST_BASE)
    return round(total / 1e9, 1)

def suggest(name, top, to_type, families=("w1", "w2", "w3")):
    s = imatrix_stats(name)
    if s.get("error"):
        return s
    layers = sorted(s["hot_layers"][:top])
    return {"layers": layers, "type": to_type, "families": list(families),
            "delta_gb": boost_gb(s, layers, to_type, families)}

def defaults():
    """hf/template defaults, lifted from an existing recipe so the builder is prefilled."""
    for r in list_recipes():
        try:
            raw = json.load(open(os.path.join(RECIPES, r["name"] + ".json")))
            if raw.get("hf") and raw.get("template"):
                return {"hf": raw["hf"], "template": raw["template"]}
        except Exception:
            pass
    return {"hf": "{models}/DeepSeek-V4-Flash-FP", "template": "{models}/BASE.gguf"}

def recipe_raw(name):
    name = safe_name(name)
    p = os.path.join(RECIPES, name + ".json")
    return json.load(open(p)) if name and os.path.exists(p) else {}

def save_recipe(d):
    name = safe_name(d.get("name"))[:40]
    if not name:
        return {"ok": False, "error": "name required"}
    rec = recipe_raw(name)  # merge: keep fields the builder doesn't render
    # quant: the builder only renders FAMILIES, so let it set/clear those, but keep
    # families it never shows (e.g. `experts`, `dense`) intact from the existing recipe.
    sent = {k: v for k, v in (d.get("quant") or {}).items() if k in FAMILIES and v in QUANT_TYPES and v}
    kept = {k: v for k, v in (rec.get("quant") or {}).items() if k not in FAMILIES}
    rec.update({"name": name, "description": (d.get("description") or rec.get("description") or "custom recipe")[:200],
                "hf": d.get("hf") or rec.get("hf") or defaults()["hf"],
                "template": d.get("template") or rec.get("template") or defaults()["template"],
                "quant": {**kept, **sent}})
    if d.get("imatrix"):
        rec["imatrix"] = d["imatrix"]
        rec.pop("corpus", None)
    elif d.get("corpus"):
        rec["corpus"] = d["corpus"]
        rec.setdefault("imatrix", "{models}/" + name + ".dat")
    b = d.get("boost") or {}
    if b.get("layers") and b.get("type"):
        if b["type"] not in BOOST_TYPES:
            return {"ok": False, "error": "boost type must be one of " + ", ".join(BOOST_TYPES)}
        if not re.fullmatch(r"(auto:\d+|\d+(-\d+)?(,\d+(-\d+)?)*)", str(b["layers"]).strip()):
            return {"ok": False, "error": "boost layers: use auto:N, 37-42, or a comma list"}
        nb = dict(rec.get("boost") or {})        # preserve fields the payload omits
        nb.update({"layers": b["layers"].strip(), "type": b["type"]})
        if "families" in b:                       # key present -> set/clear; absent -> keep existing
            fams = [f for f in (b.get("families") or []) if f in ("w1", "w2", "w3")]
            if fams and set(fams) != {"w1", "w2", "w3"}:
                nb["families"] = fams             # a subset (e.g. down-proj only)
            else:
                nb.pop("families", None)          # all three -> default, drop the key
        rec["boost"] = nb
    elif "boost" in d:  # builder sent an explicitly cleared boost
        rec.pop("boost", None)
    for k in ("imatrix_max_tokens", "threads"):
        if d.get(k):
            try:
                rec[k] = int(d[k])
            except (TypeError, ValueError):
                return {"ok": False, "error": f"{k} must be a number"}
        elif k in d:
            rec.pop(k, None)
    with open(os.path.join(RECIPES, name + ".json"), "w") as f:
        json.dump(rec, f, indent=2)
    return {"ok": True, "saved": name}

# ---------------- job control ----------------
# A launched job is also persisted to runs/active.json so the dashboard can RE-ATTACH to it
# after a restart (the job runs in its own session and outlives the UI). Without this, a UI
# restart loses the in-memory handle and a still-running quantize shows as "idle".
ACTIVE = os.path.join(RUNS, "active.json")

def _pid_alive(pid):
    if not pid: return False
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False

def _save_active(j):
    try:
        with open(ACTIVE, "w") as f:
            json.dump({"recipe": j["recipe"], "action": j["action"], "log": j["log"],
                       "pid": j["proc"].pid, "started": j["started"]}, f)
    except Exception:
        pass

def _clear_active():
    try: os.remove(ACTIVE)
    except OSError: pass

def _load_active():
    try: return json.load(open(ACTIVE))
    except Exception: return None

def _reattach_locked():
    """If a previously-launched job is still alive (per active.json), rebuild JOB for it
    with proc=None (we only have the pid across a restart). Caller holds JOB_LOCK."""
    try:
        a = json.load(open(ACTIVE))
    except Exception:
        return False
    if _pid_alive(a.get("pid")):
        JOB.clear()
        JOB.update({"recipe": a.get("recipe"), "action": a.get("action"), "log": a.get("log"),
                    "started": a.get("started", time.time()), "proc": None, "pid": a.get("pid"),
                    "samples": [], "reattached": True})
        return True
    _clear_active()
    return False

def _job_alive_locked():
    p = JOB.get("proc")
    return (p.poll() is None) if p is not None else _pid_alive(JOB.get("pid"))

def start_job(recipe, action):
    with JOB_LOCK:
        if not JOB: _reattach_locked()
        if JOB and _job_alive_locked():
            return {"ok": False, "error": "a job is already running"}
        if action not in ACTIONS:
            return {"ok": False, "error": "bad action"}
        if not any(r["name"] == recipe for r in list_recipes()):
            return {"ok": False, "error": "unknown recipe"}
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        log = os.path.join(RUNS, f"{recipe}__{action}__{ts}.log")
        lf = open(log, "w")
        proc = subprocess.Popen([sys.executable, "-u", FORGEQUANT, action, recipe],
                                cwd=HERE, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True, env=os.environ)
        JOB.clear()
        JOB.update({"recipe": recipe, "action": action, "log": log,
                    "started": time.time(), "proc": proc, "samples": []})
        _save_active(JOB)
    return {"ok": True, "started": {"recipe": recipe, "action": action}}

def stop_job():
    with JOB_LOCK:
        if not JOB: _reattach_locked()
        p = JOB.get("proc")
        pid = p.pid if p is not None else JOB.get("pid")
        action = JOB.get("action")
        alive = JOB and _job_alive_locked()
    if not (alive and pid):
        return {"ok": True, "stopped": False}
    # capture flushes a ~440 MB imatrix on SIGINT/SIGTERM — give it room before SIGKILL.
    grace = 25.0 if action == "capture" else 6.0
    def still():
        return (p.poll() is None) if p is not None else _pid_alive(pid)
    try:
        sig = signal.SIGINT if action == "capture" else signal.SIGTERM
        os.killpg(os.getpgid(pid), sig)   # own session -> the ds4 children get it too
        deadline = time.time() + grace
        while time.time() < deadline:
            if not still():
                _clear_active(); return {"ok": True, "stopped": True}
            time.sleep(0.2)
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        try: os.kill(pid, signal.SIGKILL)
        except Exception: pass
    _clear_active()
    return {"ok": True, "stopped": True}

TENSOR_RE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")
TOK_RE = re.compile(r"tokens[=: ]+(\d+)")
LAYER_RE = re.compile(r"generate_expert_tensor: layer (\d+)")   # the reliable progress during
N_LAYERS = 43                                                   # the long expert-quant phase (Flash)

def job_status():
    with JOB_LOCK:                       # consistent snapshot vs start_job's clear()+update()
        if not JOB:
            if not _reattach_locked():
                return {"running": False, "idle": True}
        elif not _job_alive_locked():
            # the tracked job ended — if a different live job is now active on disk, switch to it
            a = _load_active()
            if a and a.get("pid") != JOB.get("pid") and _pid_alive(a.get("pid")):
                JOB.clear(); _reattach_locked()
        p = JOB.get("proc"); pid = JOB.get("pid")
        action, recipe, log = JOB.get("action"), JOB.get("recipe"), JOB.get("log")
        started, last_phase = JOB.get("started", time.time()), JOB.get("phase")
        reattached = JOB.get("reattached", False)
    is_child = p is not None
    running = (p.poll() is None) if is_child else _pid_alive(pid)
    rc = (None if running else p.returncode) if is_child else None
    tail, n, m, toks, layer = "", None, None, None, None
    phase = last_phase or {"capture": "capture", "splice": "splice"}.get(action, "starting")
    try:
        with open(log, "rb") as f:
            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 16000)); data = f.read().decode("utf-8", "ignore")
        tail = data
        for ln in data.splitlines():
            if "deepseek4-quantize" in ln and "forgequant$" in ln: phase = "quantize"
            elif "--imatrix-dataset" in ln: phase = "imatrix"
            elif "splice_mixed" in ln: phase = "splice"
            elif "ds4-server" in ln and "forgequant$" in ln: phase = "capture"
            mt = TENSOR_RE.search(ln)
            if mt:
                a, b = int(mt.group(1)), int(mt.group(2))
                if 0 < b and a <= b: n, m = a, b
            lm = LAYER_RE.search(ln)
            if lm: layer = int(lm.group(1))           # expert-quant layer (0..42) — the long phase
            tk = TOK_RE.search(ln)
            if tk: toks = int(tk.group(1))
    except Exception:
        pass
    # the launch line may have scrolled out of the tail (esp. on re-attach) — infer from progress
    if phase == "starting":
        if layer is not None or m: phase = "quantize"
        elif toks: phase = "imatrix"
    # pick the progress signal the bar + ETA both use: during the long expert-quant phase the
    # layer counter is reliable (the tensor index stalls there); otherwise the tensor index.
    if layer is not None and phase == "quantize":
        prog, total, metric = layer + 1, N_LAYERS, "layer"
    else:
        prog, total, metric = n, m, "tensor"
    pct = (100.0 * prog / total) if (prog is not None and total) else None
    eta = rate = None
    with JOB_LOCK:
        if JOB.get("log") == log:                    # still the same job (child or re-attached)
            if phase != "starting":
                JOB["phase"] = phase                 # latch: survives the launch line scrolling away
            if not running:
                JOB.setdefault("ended", time.time())  # freeze elapsed at finish
                _clear_active()                       # job no longer alive -> drop persisted state
            ended = JOB.get("ended")
            if running and prog is not None and total:
                if metric == "layer":
                    # layers tick every ~minutes, so a short sample window never sees a change;
                    # use the stable overall average since the job started -> realistic, jitter-free
                    el = (ended or time.time()) - started
                    if prog > 0 and el > 2:
                        eta = round((total - prog) * (el / prog))
                else:
                    s = JOB.setdefault("samples", [])  # tensor index moves fast -> windowed rate
                    if not s or s[-1][1] != prog: s.append((time.time(), prog))
                    s[:] = s[-12:]
                    if len(s) >= 2 and s[-1][1] > s[0][1]:
                        rate = (s[-1][1] - s[0][1]) / max(1e-6, s[-1][0] - s[0][0])
                        if rate > 0: eta = round((total - prog) / rate)
        else:
            ended = None
    # terminal state: rc when we own the child; else infer from the log's end markers
    if is_child:
        done = (not running) and rc == 0
        failed = (not running) and rc not in (0, None) and rc > 0
        stopped = (not running) and rc is not None and rc < 0
    else:
        low = tail.lower()
        done = (not running) and ("forgequant: done ->" in tail or "forgequant: spliced" in tail
                                  or "forgequant: manifest ->" in tail)
        failed = (not running) and (not done) and ("fail" in low or "exited with code" in tail
                                                    or "traceback" in low)
        stopped = (not running) and (not done) and (not failed)
    sig = signal.Signals(-rc).name if (is_child and rc is not None and rc < 0) else None
    return {"running": running, "idle": False, "recipe": recipe, "reattached": reattached,
            "action": action, "rc": rc, "signal": sig,
            "phase": phase if running else ("done" if done else "stopped" if stopped else "failed"),
            "n": n, "m": m, "layer": layer, "n_layers": N_LAYERS,
            "pct": round(pct, 1) if pct is not None else None,
            "tokens": toks, "rate": round(rate, 2) if (rate and metric == "tensor") else None,
            "eta_s": eta, "elapsed_s": round((ended or time.time()) - started), "tail": tail[-3500:],
            "done": done, "failed": failed}

def list_runs():
    out = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*.log")), key=os.path.getmtime, reverse=True)[:50]:
        base = os.path.basename(f)
        parts = base[:-4].split("__")
        out.append({"file": base,
                    "recipe": parts[0] if parts else "?",
                    "action": parts[1] if len(parts) > 1 else "?",
                    "when": datetime.datetime.fromtimestamp(os.path.getmtime(f)).isoformat(timespec="minutes"),
                    "kb": os.path.getsize(f) >> 10})
    return out

def run_log(name):
    name = safe_name(name)
    p = os.path.join(RUNS, name)
    if not (name.endswith(".log") and os.path.exists(p)):
        return {"error": "unknown log"}
    with open(p, "rb") as f:
        f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 65536))
        return {"file": name, "text": f.read().decode("utf-8", "ignore")}

# ---------------- HTTP ----------------
PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>forge-ui</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#000;--card:#0a0a0a;--el:#121212;--bd:#232323;--bd2:#343434;--fg:#ededed;--mut:#9b9b9b;--mut2:#6a6a6a;--pri:#ededed;--prifg:#0a0a0a;--ok:#3fcf8e;--no:#f5544a;--info:#4aa3ff;--pur:#b07cff;--hot:#ffb454}
*{box-sizing:border-box}
html{-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;letter-spacing:-.006em}
.wrap{max-width:980px;margin:0 auto;padding:44px 24px 90px}
.mut{color:var(--mut)}.mono{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:.92em}
header{margin:0 0 30px}
header h1{font-size:20px;font-weight:600;margin:0;letter-spacing:-.02em;display:flex;align-items:center}
header .sub{color:var(--mut);font-size:14px;margin-top:7px;letter-spacing:-.01em}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:22px 24px;margin:16px 0}
.ch{display:flex;align-items:center;gap:10px;margin:0 0 20px}
.ch h2{font-size:15px;font-weight:600;margin:0;letter-spacing:-.015em}
.ch .sub{color:var(--mut2);font-size:13px;font-weight:400;letter-spacing:-.01em}
.step{width:21px;height:21px;border-radius:50%;background:var(--el);border:1px solid var(--bd2);color:var(--mut);font-size:11px;font-weight:600;display:inline-flex;align-items:center;justify-content:center;flex:none}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.grp{margin:20px 0}
.grp>.lbl{font-size:13px;font-weight:500;color:var(--fg);margin-bottom:5px;letter-spacing:-.01em}
.grp .cap{font-size:13px;color:var(--mut);margin:0 0 13px;max-width:72ch;line-height:1.5}
select,button,input{font:inherit;letter-spacing:inherit;height:36px;padding:0 12px;border-radius:8px;border:1px solid var(--bd2);background:var(--el);color:var(--fg);outline:none;transition:.12s ease}
select{cursor:pointer;-webkit-appearance:none;appearance:none;padding-right:30px;background-image:url("data:image/svg+xml;utf8,<svg width='10' height='6' xmlns='http://www.w3.org/2000/svg'><path d='M1 1l4 4 4-4' stroke='%238a8a8a' stroke-width='1.4' fill='none' stroke-linecap='round'/></svg>");background-repeat:no-repeat;background-position:right 11px center}
input::placeholder{color:var(--mut2)}
select:hover,input:hover{border-color:#454545}
input:focus,select:focus{border-color:#8a8a8a}
button{cursor:pointer;font-weight:500;display:inline-flex;align-items:center;justify-content:center;white-space:nowrap}
button:hover{background:#191919;border-color:#454545}
button.primary{background:var(--pri);color:var(--prifg);border-color:var(--pri)}
button.primary:hover{background:#fff;border-color:#fff}
button.ghost{background:transparent;border-color:var(--bd2);color:var(--fg)}
button.ghost:hover{background:#191919;border-color:#454545}
button:disabled{opacity:.45;cursor:not-allowed;background:var(--el);border-color:var(--bd);color:var(--mut2)}
.ic{width:15px;height:15px;stroke-width:1.75;fill:none;stroke:currentColor;stroke-linecap:round;stroke-linejoin:round;flex:none}
button .ic{margin-right:7px;margin-left:-2px}
.fld{display:flex;flex-direction:column;gap:6px;font-size:12px;color:var(--mut)}.fld select{height:34px;width:100%}
.famgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(116px,1fr));gap:10px}
.bar{height:6px;background:#1c1c1c;border-radius:999px;overflow:hidden}
.fill{height:100%;background:var(--fg);width:0;transition:width .5s ease;border-radius:999px}
.plab{font-size:13px;color:var(--mut);margin-top:10px;letter-spacing:-.01em}
.badge{display:inline-flex;align-items:center;gap:6px;padding:3px 10px 3px 9px;border-radius:999px;font-size:12px;font-weight:500;border:1px solid var(--bd2);color:var(--mut);background:var(--el);letter-spacing:-.01em}
.badge::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor;flex:none}
.stat{display:flex;gap:24px;flex-wrap:wrap;font-size:13px;color:var(--mut)}.stat b{color:var(--fg);font-weight:500}
pre.log{background:#000;border:1px solid var(--bd);border-radius:8px;padding:12px 14px;max-height:250px;overflow:auto;font-size:12px;line-height:1.6;white-space:pre-wrap;color:#7d7d7d;margin:0;font-family:ui-monospace,Menlo,monospace}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:11px 8px;border-bottom:1px solid var(--bd);text-align:left}
th{color:var(--mut2);font-weight:500;font-size:12px;letter-spacing:-.005em}
tr:last-child td{border-bottom:none}
.pill{display:inline-block;font-size:12px;font-family:ui-monospace,Menlo,monospace;background:var(--el);border:1px solid var(--bd);border-radius:6px;padding:1px 7px;margin:2px 5px 2px 0;color:var(--mut)}
.pill.hot{color:var(--hot);border-color:#4a3a1a}
.cmd{font-size:12px;color:var(--mut2);font-family:ui-monospace,Menlo,monospace;word-break:break-all;margin-top:12px;line-height:1.65}
.hint{font-size:13px;color:var(--mut);letter-spacing:-.01em}
.divider{height:1px;background:var(--bd);margin:22px -24px}
.tpl{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.tplc{background:var(--el);border:1px solid var(--bd2);border-radius:10px;padding:13px 14px;cursor:pointer;transition:.12s ease}
.tplc:hover{border-color:#5a5a5a;background:#161616}
.tplc.sel{border-color:var(--fg)}
.tplc b{font-size:13px;display:block;margin-bottom:4px}
.tplc .d{font-size:12px;color:var(--mut);line-height:1.45;max-height:54px;overflow:hidden}
.brain{position:relative;border:1px solid var(--bd);border-radius:8px;background:#050505;padding:10px}
canvas{display:block;width:100%;image-rendering:pixelated}
.legend{display:flex;gap:14px;font-size:12px;color:var(--mut2);margin-top:8px;align-items:center}
.lay{font-size:12px;color:var(--mut);font-family:ui-monospace,Menlo,monospace;margin-top:6px;min-height:17px}
.bb{font-size:12px;padding:4px 11px;border-radius:999px;margin-bottom:10px;border:1px solid var(--bd2);display:inline-block}
.bb.ok{color:var(--ok);border-color:#1e4d39;background:#0c1a14}
.bb.hot{color:var(--hot);border-color:#4a3a1a;background:#1a1407}
a.lnk{color:var(--info);text-decoration:none;cursor:pointer}
.preset{display:flex;flex-direction:column;gap:3px;align-items:flex-start;text-align:left;padding:9px 12px;height:auto;border-radius:9px;background:var(--el);border:1px solid var(--bd2);min-width:150px}
.preset:hover{border-color:#5a5a5a;background:#161616}
.preset.sel{border-color:var(--hot);background:#1a1407}
.preset b{font-size:12px;font-weight:600;color:var(--fg)}
.preset .pg{font-size:11px;color:var(--hot);font-family:ui-monospace,Menlo,monospace}
.preset.rec b::after{content:" ★";color:var(--hot)}
.gtk{font-size:12px;color:var(--mut);line-height:1.65;margin-top:10px;max-width:78ch}
.gtk b{color:var(--fg);font-weight:500}
.gtk li{margin:2px 0}
details.gtk>summary{cursor:pointer;color:var(--info);list-style:none}
details.gtk>summary::-webkit-details-marker{display:none}
</style></head><body><div class="wrap">
<header>
  <h1><svg class="ic" style="width:18px;height:18px;color:var(--fg);margin-right:10px" viewBox="0 0 24 24"><path d="m15 12-8.5 8.5a2.12 2.12 0 1 1-3-3L12 9"/><path d="M17.64 15 22 10.64"/><path d="m20.91 11.7-1.25-1.25c-.6-.6-.93-1.4-.93-2.25v-.86L16.01 4.6a5.56 5.56 0 0 0-3.94-1.64H9l.92.82A6.18 6.18 0 0 1 12 8.4v1.56l2 2h2.47l2.26 1.91"/></svg><b>forge</b>quant</h1>
  <div class="sub">Asymmetric quantization for DeepSeek-V4-Flash. <b>Forge</b> a model from a recipe and watch it. The cards below are independent tools — not steps to do in order: <b>build a recipe</b> or <b>explore an imatrix</b> only if you want to go beyond the presets; your outputs land at the bottom.</div>
</header>

<div class="card">
  <div class="ch"><h2>Forge a model</h2><span class="sub">pick a recipe + action and run it — presets work out of the box; live progress below</span></div>
  <div class="row">
    <select id="recipe" style="min-width:170px"></select>
    <select id="action">
      <option value="build">build · imatrix + quantize</option>
      <option value="quantize">quantize only</option>
      <option value="imatrix">imatrix · from corpus</option>
      <option value="capture">capture · imatrix from live inference</option>
      <option value="splice">splice · fast layer boost, no requant</option>
    </select>
    <button class="primary" id="go" onclick="forge()"><svg class="ic" viewBox="0 0 24 24"><path d="m15 12-8.5 8.5a2.12 2.12 0 1 1-3-3L12 9"/><path d="M17.64 15 22 10.64"/><path d="m20.91 11.7-1.25-1.25c-.6-.6-.93-1.4-.93-2.25v-.86L16.01 4.6a5.56 5.56 0 0 0-3.94-1.64H9l.92.82A6.18 6.18 0 0 1 12 8.4v1.56l2 2h2.47l2.26 1.91"/></svg>Forge</button>
    <button class="ghost" id="stop" onclick="stop()">Stop</button>
    <span class="hint" id="rdesc" style="margin:0 0 0 4px"></span>
  </div>
  <div class="cmd mono" id="cmd"></div>
  <div class="hint" id="acthint" style="margin-top:8px"></div>
  <div class="divider"></div>
  <div class="row" style="justify-content:space-between;margin-bottom:10px">
    <div class="row" style="gap:10px"><b style="font-size:13px">Progress</b><span id="phase" class="badge"></span></div>
    <div class="stat" style="margin:0"><span>recipe <b id="srecipe">—</b></span><span>elapsed <b id="selapsed">—</b></span><span>rate <b id="srate">—</b></span><span>ETA <b id="seta">—</b></span></div>
  </div>
  <div class="bar"><div class="fill" id="fill"></div></div>
  <div class="plab" id="plab">idle</div>
  <pre class="log" id="log" style="margin-top:14px">waiting…</pre>
</div>

<div class="card">
  <div class="ch"><h2>Brain map</h2><span class="sub">optional · explore where a workload activates, and send a boost down to the recipe builder</span></div>
  <div class="row" style="margin-bottom:12px">
    <select id="imsel" style="min-width:240px"></select>
    <span class="hint">compare with</span>
    <select id="imdiff" style="min-width:200px"><option value="">— nothing —</option></select>
    <button onclick="loadBrain()" id="brainbtn">Analyze</button>
    <span class="hint" id="brainmsg"></span>
  </div>
  <div class="brain" id="brainbox" style="display:none">
    <div id="bbanner" class="bb" style="display:none"></div>
    <canvas id="bmap" width="1024" height="172"></canvas>
    <div class="lay" id="bhover"></div>
    <div class="legend"><span id="blegend">rows = 43 layers (top→bottom) · columns = 256 experts · dim → bright = activation energy</span><span id="bds"></span></div>
  </div>
  <div class="grp" id="hotbox" style="display:none">
    <div class="lbl">Hot layers <span class="mut" style="font-weight:400">· where this workload concentrates</span></div>
    <div class="row" style="margin-bottom:14px"><span id="hotpills"></span></div>

    <div class="lbl">Boost</div>
    <div class="cap" id="boosttip"></div>
    <div class="row" id="presets" style="margin-bottom:12px"></div>
    <div class="row">
      <span class="hint">or tune:</span>
      <span class="hint">top</span><input id="sugn" value="6" style="width:50px">
      <span class="hint">layers ·</span>
      <select id="sugfam" title="which expert tensors to upcast">
        <option value="w1,w2,w3">all experts</option>
        <option value="w2">down-proj only (most sensitive)</option>
        <option value="w1,w3">gate + up only</option>
      </select>
      <span class="hint">to</span><select id="sugt"></select>
      <button onclick="applySuggest()">→ apply to recipe builder</button>
      <span class="hint hot" id="sugmsg"></span>
    </div>
    <div class="cap" id="boostsel" style="margin-top:10px"></div>
    <details class="gtk"><summary>good to know — how to choose</summary>
      <ul id="boostnotes" style="margin:8px 0 0;padding-left:18px"></ul></details>
  </div>
</div>

<div class="card">
  <div class="ch"><h2>Recipe builder</h2><span class="sub">optional · build or tune a recipe — it then appears in the Forge picker at the top</span></div>
  <div class="grp" style="margin-top:0">
    <div class="lbl">Templates</div>
    <div class="tpl" id="tpls"></div>
  </div>
  <div class="row" style="margin-bottom:6px">
    <input id="bname" placeholder="recipe name" style="width:170px">
    <input id="bdesc" placeholder="description" style="flex:1;min-width:200px">
  </div>

  <div class="grp">
    <div class="lbl">Quant per tensor family</div>
    <div class="cap"><b>(copy)</b> keeps the template's type. The routed experts are the 2-bit budget the imatrix re-allocates; attention/shared/output stay near-lossless (q8_0).</div>
    <div id="bfam"></div>
  </div>

  <div class="grp">
    <div class="lbl">Boost <span class="mut" style="font-weight:400">· per-layer expert upcast (the "keep my expert sharp" knob)</span></div>
    <div class="cap">Upcasts the routed experts of chosen layers via <span class="mono">--tensor-type</span> overrides. <b>auto:6</b> = the 6 hottest layers of this recipe's imatrix, resolved at forge time. Within every other tensor the imatrix still steers bits expert-by-expert.</div>
    <div class="row">
      <input id="blayers" placeholder='layers · e.g. auto:6 or 37-42' style="width:220px">
      <select id="btype"></select>
      <span class="hint" id="bdelta"></span>
    </div>
  </div>

  <div class="grp">
    <div class="lbl">Imatrix</div>
    <div class="cap">The importance matrix steers where bits go. <b>Load</b> an existing <span class="mono">.dat</span>, or <b>build from corpus</b> — and remember you can <b>capture</b> one from live inference (action above) or render a corpus from your own prompts: <span class="mono">forgequant.py render prompts.txt -o corpus.txt</span>.</div>
    <div class="row">
      <select id="bimat" style="min-width:240px"></select>
      <input id="bcorpus" placeholder="rendered corpus path" style="flex:1;min-width:220px;display:none">
      <input id="bmaxtok" placeholder="max tokens · e.g. 120000" style="width:200px;display:none">
    </div>
  </div>

  <div class="grp">
    <div class="lbl">Sources <span class="mut" style="text-transform:none;letter-spacing:0;font-weight:400">· FP weights + a template GGUF for shapes/metadata</span></div>
    <div class="row">
      <input id="bhf" placeholder="hf FP source dir" style="flex:1;min-width:200px">
      <input id="btmpl" placeholder="template gguf" style="flex:1;min-width:200px">
    </div>
  </div>

  <div class="row" style="margin-top:4px"><button class="primary" onclick="saveRecipe()"><svg class="ic" viewBox="0 0 24 24"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>Save recipe</button><span class="hint" id="bmsg"></span></div>
</div>

<div class="card">
  <div class="ch"><h2>Forged models</h2><span class="sub">outputs in MODELS_DIR with their manifests</span></div>
  <table><thead><tr><th>model</th><th>size</th><th>recipe</th><th>quant</th><th>created</th></tr></thead><tbody id="models"></tbody></table>
</div>

<div class="card">
  <div class="ch"><h2>Runs</h2><span class="sub">past launches and their logs</span></div>
  <table><thead><tr><th>when</th><th>recipe</th><th>action</th><th>log</th></tr></thead><tbody id="runs"></tbody></table>
  <pre class="log" id="runlog" style="margin-top:12px;display:none"></pre>
</div>

<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function fmt(s){if(s==null)return '—';s=+s;if(s<60)return s+'s';const m=Math.floor(s/60);return m+'m'+String(s%60).padStart(2,'0')+'s';}
const HINTS={build:'runs the imatrix step if missing, then quantizes',
  quantize:'recipe → GGUF (needs the imatrix to exist if the recipe names one)',
  imatrix:'runs the model over the recipe corpus and extracts the activation statistics',
  capture:'serves the forged model with ds4-server and collects the imatrix from the prompts you actually send — your real brain paths; Stop when you have enough traffic',
  splice:'copies the boost layers from a higher-precision donor GGUF — minutes instead of hours, great for A/B before a full requantize'};
async function loadRecipes(){const r=await fetch('/api/recipes').then(r=>r.json());const sel=$('recipe');const cur=sel.value;
  sel.innerHTML=r.map(x=>`<option value="${esc(x.name)}">${esc(x.name)}</option>`).join('');if(cur)sel.value=cur;
  window.RECIPES=r;updRecipe();drawTpls();}
function updRecipe(){const r=(window.RECIPES||[]).find(x=>x.name===$('recipe').value);if(!r)return;
  $('rdesc').textContent=r.description||'';
  const q=Object.entries(r.quant||{}).map(([k,v])=>`<span class="pill">${esc(k.replace('routed_',''))}=${esc(v)}</span>`).join('');
  const b=r.boost?` <span class="pill hot">boost ${esc(r.boost.layers)}→${esc(r.boost.type)}</span>`:'';
  const s=r.splice?` <span class="pill hot">splice ${esc(r.splice.layers||'')}</span>`:'';
  $('cmd').innerHTML=q+b+s+(r.imatrix?` <span class="pill">imatrix</span>`:'')+(r.corpus||r.bench?` <span class="pill">corpus</span>`:'');
  $('acthint').textContent=HINTS[$('action').value]||'';
  // guard incompatible actions so a click can't launch a guaranteed-failed job
  const A=$('action'),opt=v=>A.querySelector(`[value=${v}]`);
  if(opt('splice'))opt('splice').disabled=!r.splice;
  if(opt('imatrix'))opt('imatrix').disabled=!(r.corpus||r.bench);
  if(opt('capture'))opt('capture').disabled=false;
  if(A.selectedOptions[0]&&A.selectedOptions[0].disabled)A.value='build';}
async function forge(){const b=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({recipe:$('recipe').value,action:$('action').value})}).then(r=>r.json());
  if(!b.ok)alert(b.error);poll();}
async function stop(){await fetch('/api/stop',{method:'POST'});poll();}
const BC={quantize:'var(--info)',imatrix:'var(--pur)',capture:'var(--hot)',splice:'var(--hot)',done:'var(--ok)',failed:'var(--no)',stopped:'var(--mut)',starting:'var(--mut)'};
async function poll(){let s;try{s=await fetch('/api/status').then(r=>r.json());}catch(e){return;}
  const ph=s.phase||'idle';const pe=$('phase');pe.style.display=s.idle?'none':'inline-flex';pe.textContent=ph+(s.signal?' · '+s.signal:s.rc>0?' · rc '+s.rc:'');pe.style.color=BC[ph]||'var(--mut)';
  $('go').disabled=!!s.running;$('stop').disabled=!s.running;
  let pct=s.pct!=null?s.pct:0,lab='idle';
  if(s.layer!=null&&s.phase==='quantize'){lab=`layer ${s.layer+1} / ${s.n_layers||43} · quantizing experts · ${Math.round(pct)}%`;}
  else if(s.m){lab=`${s.n} / ${s.m} tensors · ${Math.round(pct)}%`;}
  else if(s.phase==='imatrix'){lab=s.tokens?`${s.tokens.toLocaleString()} tokens collected`:'collecting imatrix…';pct=s.tokens?Math.min(99,s.tokens/1200):5;}
  else if(s.phase==='capture'){lab=s.tokens?`${s.tokens.toLocaleString()} tokens observed — send traffic, Stop when done`:'serving · waiting for traffic…';pct=4;}
  else if(s.phase==='done')lab='done',pct=100;else if(s.phase==='failed')lab='failed'+(s.rc?` (rc ${s.rc})`:'');else if(s.phase==='stopped')lab='stopped';else if(s.running)lab='starting…',pct=2;
  $('fill').style.width=pct+'%';$('plab').textContent=lab;
  $('srecipe').textContent=s.recipe||'—';$('selapsed').textContent=fmt(s.elapsed_s);
  $('srate').textContent=s.rate?(s.rate.toFixed(1)+(s.phase==='quantize'?' tns/s':'')):'—';$('seta').textContent=fmt(s.eta_s);
  if(s.tail)$('log').textContent=s.tail;}
async function loadModels(){const m=await fetch('/api/models').then(r=>r.json());
  $('models').innerHTML=m.map(x=>{const q=Object.entries(x.quant||{}).map(([k,v])=>`<span class="pill">${esc(k.replace('routed_',''))}=${esc(v)}</span>`).join('')
   +(x.boost?`<span class="pill hot">boost ${esc(x.boost.layers)}→${esc(x.boost.type)}</span>`:'')+(x.action==='splice'?'<span class="pill hot">splice</span>':'');
   return `<tr><td class="mono">${esc(x.path)}</td><td>${esc(x.gb)} GB</td><td>${esc(x.recipe||'—')}</td><td>${q||'—'}</td><td class="mut">${esc(x.created||x.mtime)}</td></tr>`;}).join('')||'<tr><td colspan=5 class="mut">no forged models yet</td></tr>';}
let FAMS=[];
async function loadTypes(){const t=await fetch('/api/types').then(r=>r.json());FAMS=t.families;
  const nm={routed_w1:'gate (w1)',routed_w2:'down (w2)',routed_w3:'up (w3)',attn_proj:'attn-proj'};
  const fld=f=>`<div class="fld">${nm[f]||f}<select id="f_${f}">${t.types.map(x=>`<option value="${x}">${x||'(copy)'}</option>`).join('')}</select></div>`;
  const exp=['routed_w1','routed_w3','routed_w2'].filter(f=>FAMS.includes(f)),oth=FAMS.filter(f=>!exp.includes(f));
  $('bfam').innerHTML=`<div style="font-size:11px;color:var(--mut2);margin:0 0 6px">routed experts · the 2-bit budget</div><div class="famgrid">${exp.map(fld).join('')}</div>`+
    `<div style="font-size:11px;color:var(--mut2);margin:13px 0 6px">other tensors · keep near-lossless</div><div class="famgrid">${oth.map(fld).join('')}</div>`;
  $('btype').innerHTML=t.boost_types.map(x=>`<option>${x}</option>`).join('');$('btype').value='q4_k';
  $('sugt').innerHTML=t.boost_types.map(x=>`<option>${x}</option>`).join('');$('sugt').value='q4_k';
  const g=t.boost_guide;if(g){window.GUIDE=g;$('boosttip').textContent=g.tooltip;
    $('boostsel').innerHTML='<b class="mut">selection:</b> '+esc(g.selection);
    $('boostnotes').innerHTML=g.notes.map(x=>`<li>${esc(x)}</li>`).join('');
    $('presets').innerHTML=g.presets.map((p,i)=>`<button class="preset${p.key==='medium'?' rec':''}" data-i="${i}" title="${esc(p.when)}" onclick="applyPreset(${i})"><b>${esc(p.key)}</b><span>${esc(p.label)}</span><span class="pg" id="pg${i}">…</span></button>`).join('');
    $('sugn').value=g.default_n;}}
function applyPreset(i){const p=window.GUIDE.presets[i];
  document.querySelectorAll('.preset').forEach(e=>e.classList.toggle('sel',+e.dataset.i===i));
  $('sugn').value=p.n;$('sugt').value=p.type;
  $('sugfam').value=(p.families.length===3?'w1,w2,w3':p.families.join(','));
  applySuggest();}
async function loadImats(){const im=await fetch('/api/imatrices').then(r=>r.json());
  const opt=im.map(x=>`<option value="${esc(x.name)}">${esc(x.name)} · ${x.mb}MB${x.analyzed?' ·✓':''}</option>`).join('');
  const s=$('bimat'),cur=s.value;
  s.innerHTML='<option value="">— build from corpus —</option>'+im.map(x=>`<option value="{models}/${esc(x.name)}">${esc(x.name)} · ${x.mb}MB</option>`).join('');if(cur)s.value=cur;
  s.onchange=()=>{const build=!s.value;$('bcorpus').style.display=build?'':'none';$('bmaxtok').style.display=build?'':'none';};s.onchange();
  const i1=$('imsel'),c1=i1.value;i1.innerHTML=opt;if(c1)i1.value=c1;
  const i2=$('imdiff'),c2=i2.value;i2.innerHTML='<option value="">— nothing —</option>'+opt;if(c2)i2.value=c2;}
function drawTpls(){const r=window.RECIPES||[];
  $('tpls').innerHTML=r.map(x=>`<div class="tplc" data-n="${esc(x.name)}" onclick="useTpl(this.dataset.n)"><b>${esc(x.name)}</b><div class="d">${esc(x.description||'')}</div></div>`).join('');}
async function useTpl(n){document.querySelectorAll('.tplc').forEach(e=>e.classList.toggle('sel',e.dataset.n===n));
  const r=await fetch('/api/recipe?name='+encodeURIComponent(n)).then(r=>r.json());
  $('bname').value=r.name||n;$('bdesc').value=r.description||'';$('bhf').value=r.hf||'';$('btmpl').value=r.template||'';
  FAMS.forEach(f=>{const s=$('f_'+f);if(s)s.value=(r.quant||{})[f]||'';});
  $('blayers').value=(r.boost||{}).layers||'';$('btype').value=(r.boost||{}).type||'q4_k';
  window.BOOST_FAM=(r.boost||{}).families||[];
  $('bimat').value=r.imatrix||'';$('bcorpus').value=r.corpus||'';$('bmaxtok').value=r.imatrix_max_tokens||'';$('bimat').onchange();}
async function saveRecipe(){const quant={};FAMS.forEach(f=>{const v=$('f_'+f).value;if(v)quant[f]=v;});
  const body={name:$('bname').value,description:$('bdesc').value,hf:$('bhf').value,template:$('btmpl').value,quant,
    boost:{layers:$('blayers').value.trim(),type:$('btype').value,families:window.BOOST_FAM||[]}};
  // always send max-tokens key in corpus mode so clearing the field reaches the server's remove branch
  if($('bimat').value)body.imatrix=$('bimat').value;else if($('bcorpus').value){body.corpus=$('bcorpus').value;body.imatrix_max_tokens=$('bmaxtok').value;}
  const res=await fetch('/api/save_recipe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());
  $('bmsg').textContent=res.ok?('saved '+res.saved):('error — '+res.error);$('bmsg').style.color=res.ok?'var(--ok)':'var(--no)';if(res.ok){await loadRecipes();$('recipe').value=res.saved;updRecipe();}}
let BRAIN=null;
async function loadBrain(){const n=$('imsel').value;if(!n)return;
  $('brainmsg').textContent='analyzing… (first run on a big .dat takes a while, then cached)';$('brainbtn').disabled=true;
  try{
    const s=await fetch('/api/imatrix_stats?file='+encodeURIComponent(n)).then(r=>r.json());
    if(s.error){$('brainmsg').textContent='error — '+esc(s.error);BRAIN=null;$('brainbox').style.display='none';$('hotbox').style.display='none';return;}
    BRAIN=s;const baseName=$('imdiff').value;let base=null;
    if(baseName&&baseName!==n){
      base=await fetch('/api/imatrix_stats?file='+encodeURIComponent(baseName)).then(r=>r.json());
      if(base.error){$('brainmsg').textContent='error — '+esc(base.error);return;}
    }
    const info=drawBrain(s,base);
    const bn=$('bbanner');
    if(base){bn.style.display='';
      if(info.identical){bn.className='bb ok';bn.textContent='✓ exact match — these two imatrices light up identically';
        $('blegend').textContent=`comparing ${esc(n.replace('.dat',''))} vs ${esc(baseName.replace('.dat',''))} — no differences`;}
      else{bn.className='bb hot';bn.textContent=`differences highlighted · ${info.changed} of ${s.layers.length*s.n_experts} expert-cells changed`;
        $('blegend').textContent=`warm = ${esc(n.replace('.dat',''))}-leaning · cool = ${esc(baseName.replace('.dat',''))}-leaning · dark = same`;}
    }else{bn.style.display='none';
      $('blegend').textContent='rows = 43 layers (top→bottom) · columns = 256 experts · dim → bright = activation energy';}
    $('brainmsg').textContent='';$('bds').textContent=s.dataset?('corpus: '+(''+s.dataset).split('/').pop()):'live capture';
    $('hotpills').innerHTML=(s.hot_layers||[]).slice(0,8).map(l=>`<span class="pill hot">blk.${l}</span>`).join('');
    $('hotbox').style.display='';$('brainbox').style.display='';refreshPresetGB();
  }finally{$('brainbtn').disabled=false;}}
function drawBrain(s,base){const cv=$('bmap'),ctx=cv.getContext('2d');
  const L=s.layers.length,E=s.n_experts,cw=4,chh=4;cv.width=E*cw;cv.height=L*chh;
  ctx.fillStyle='#050505';ctx.fillRect(0,0,cv.width,cv.height);
  if(base){
    // compare: colour each cell by the SIGNED difference of global activation share, so only
    // the areas that change light up (warm = this-leaning, cool = baseline-leaning, dark = same)
    const bl={};base.layers.forEach(r=>bl[r.layer]=r);
    let maxd=0;const D=[];
    s.layers.forEach(row=>{const br=bl[row.layer];
      const ss=row.heat.reduce((a,b)=>a+b,0)||1,bs=br?(br.heat.reduce((a,b)=>a+b,0)||1):1;
      const dr=row.heat.map((h,e)=>(h/ss*row.share)-(br?(br.heat[e]/bs*br.share):0));
      D.push(dr);dr.forEach(d=>{if(Math.abs(d)>maxd)maxd=Math.abs(d);});});
    const identical=maxd<1e-5;let changed=0;
    // faint base grid: keep every untouched cell visible as a dark-grey square instead of losing
    // it in the black background, so the map still reads as a full grid with only the changes lit
    ctx.fillStyle='#242428';
    s.layers.forEach((row,i)=>{for(let e=0;e<E;e++)ctx.fillRect(e*cw,i*chh,cw-1,chh-1);});
    if(!identical)s.layers.forEach((row,i)=>{D[i].forEach((d,e)=>{const m=Math.abs(d)/maxd;if(m<0.06)return;changed++;
      const c=Math.round(45+m*210);
      ctx.fillStyle=d>0?`rgb(${c},${Math.round(c*0.5)},${Math.round(c*0.3)})`:`rgb(${Math.round(c*0.35)},${Math.round(c*0.65)},${c})`;
      ctx.fillRect(e*cw,i*chh,cw-1,chh-1);});});
    cv.onmousemove=ev=>{const r=cv.getBoundingClientRect();
      const e=Math.floor((ev.clientX-r.left)/r.width*E),l=Math.floor((ev.clientY-r.top)/r.height*L);
      if(!s.layers[l]||!D[l])return;const d=D[l][e]||0;
      $('bhover').textContent=`blk.${s.layers[l].layer} · expert ${e} · Δshare ${(d*100>=0?'+':'')}${(d*100).toFixed(3)}% · ${d>0?'this':'baseline'}-leaning`;};
    return {identical,changed};
  }
  s.layers.forEach((row,i)=>{row.heat.forEach((v,e)=>{if(v<=0)return;
    const c=Math.round(30+v*215);ctx.fillStyle=`rgb(${Math.round(c*0.55)},${Math.round(c*0.8)},${c})`;
    ctx.fillRect(e*cw,i*chh,cw-1,chh-1);});});
  cv.onmousemove=ev=>{const r=cv.getBoundingClientRect();
    const e=Math.floor((ev.clientX-r.left)/r.width*E),l=Math.floor((ev.clientY-r.top)/r.height*L);
    const row=s.layers[l];if(!row)return;
    $('bhover').textContent=`blk.${row.layer} · expert ${e} · heat ${(row.heat[e]||0).toFixed(2)} · layer share ${(row.share*100).toFixed(1)}% · ${row.active}/${E} experts active`;};
  return {identical:false,changed:0};}
async function applySuggest(){if(!BRAIN)return;const n=parseInt($('sugn').value)||6,t=$('sugt').value,fam=$('sugfam').value;
  const s=await fetch(`/api/suggest?imatrix=${encodeURIComponent($('imsel').value)}&top=${n}&type=${t}&families=${encodeURIComponent(fam)}`).then(r=>r.json());
  if(s.error){$('sugmsg').textContent=esc(s.error);return;}
  $('blayers').value='auto:'+n;$('btype').value=t;window.BOOST_FAM=s.families;
  $('bimat').value='{models}/'+$('imsel').value;$('bimat').onchange();
  const fl=s.families.length===3?'all experts':(fam==='w2'?'down-proj only':'gate+up');
  $('sugmsg').textContent=`auto:${n} · ${fl} · ≈ +${s.delta_gb} GB → applied to the builder`;
  $('bdelta').textContent=`≈ +${s.delta_gb} GB`;}
async function refreshPresetGB(){if(!window.GUIDE||!$('imsel').value)return;
  for(let i=0;i<window.GUIDE.presets.length;i++){const p=window.GUIDE.presets[i];
    const fam=p.families.length===3?'w1,w2,w3':p.families.join(',');
    try{const s=await fetch(`/api/suggest?imatrix=${encodeURIComponent($('imsel').value)}&top=${p.n}&type=${p.type}&families=${fam}`).then(r=>r.json());
      if(!s.error&&$('pg'+i))$('pg'+i).textContent='≈ +'+s.delta_gb+' GB';}catch(e){}}}
async function loadRuns(){const rs=await fetch('/api/runs').then(r=>r.json());
  $('runs').innerHTML=rs.map(x=>`<tr><td class="mut">${esc(x.when)}</td><td>${esc(x.recipe)}</td><td>${esc(x.action)}</td><td><a class="lnk" data-f="${esc(x.file)}" onclick="showLog(this.dataset.f)">${esc(x.file)}</a> <span class="mut">· ${x.kb} KB</span></td></tr>`).join('')||'<tr><td colspan=4 class="mut">no runs yet</td></tr>';}
async function showLog(f){const l=await fetch('/api/runlog?f='+encodeURIComponent(f)).then(r=>r.json());
  const p=$('runlog');p.style.display='';p.textContent=l.text||l.error;p.scrollTop=p.scrollHeight;}
$('recipe').onchange=updRecipe;$('action').onchange=updRecipe;
loadRecipes();loadModels();poll();loadTypes();loadImats();loadRuns();
setInterval(poll,1500);setInterval(loadModels,6000);setInterval(loadRecipes,15000);setInterval(loadImats,12000);setInterval(loadRuns,8000);
</script></div></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        if isinstance(body, str): body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def _json(self, o): self._send(200, json.dumps(o), "application/json")
    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try: return json.loads(self.rfile.read(n) or b"{}")
        except Exception: return {}
    def do_GET(self):
        u = urlparse(self.path)
        p, q = u.path, parse_qs(u.query)
        def arg(k): return q.get(k, [""])[0]
        def intarg(k, d):
            try: return int(arg(k))
            except ValueError: return d
        if p == "/": self._send(200, PAGE)
        elif p == "/api/recipes": self._json(list_recipes())
        elif p == "/api/models": self._json(list_models())
        elif p == "/api/status": self._json(job_status())
        elif p == "/api/types": self._json({"types": QUANT_TYPES, "families": FAMILIES,
                                            "boost_types": BOOST_TYPES, "boost_guide": BOOST_GUIDE})
        elif p == "/api/imatrices": self._json(list_imatrices())
        elif p == "/api/imatrix_stats": self._json(imatrix_stats(arg("file")))
        elif p == "/api/imatrix_diff": self._json(imatrix_diff(arg("a"), arg("b")))
        elif p == "/api/suggest":
            fams = tuple(f for f in arg("families").split(",") if f in ("w1", "w2", "w3")) or ("w1", "w2", "w3")
            self._json(suggest(arg("imatrix"), max(1, min(43, intarg("top", 6))),
                               arg("type") if arg("type") in BOOST_TYPES else "q4_k", fams))
        elif p == "/api/runs": self._json(list_runs())
        elif p == "/api/runlog": self._json(run_log(arg("f")))
        elif p == "/api/defaults": self._json(defaults())
        elif p == "/api/recipe": self._json(recipe_raw(arg("name")))
        else: self._send(404, "not found")
    def do_POST(self):
        p = self.path.split("?")[0]
        if p == "/api/run":
            b = self._body(); self._json(start_job(b.get("recipe", ""), b.get("action", "build")))
        elif p == "/api/stop": self._json(stop_job())
        elif p == "/api/save_recipe": self._json(save_recipe(self._body()))
        else: self._send(404, "not found")

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8060
    print(f"forge-ui -> http://127.0.0.1:{port}  (MODELS_DIR={MODELS_DIR})")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

if __name__ == "__main__":
    main()
