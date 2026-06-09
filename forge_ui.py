#!/usr/bin/env python3
"""forge-ui — a small web dashboard to drive + monitor forgequant.

Pick a recipe, launch build/quantize/imatrix from the browser, and watch the long
quantization live (per-tensor progress, ETA, throughput) plus a tail of the log. Lists
the models you've forged with their manifests. Stdlib http.server only; no deps.

Run:  python3 forge_ui.py [port]      # default 8060  ->  http://localhost:8060
Env:  DS4_DIR, MODELS_DIR (same as forgequant).
"""
import json, os, sys, re, subprocess, signal, time, datetime, glob, html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
FORGEQUANT = os.path.join(HERE, "forgequant.py")
RECIPES = os.path.join(HERE, "recipes")
MODELS_DIR = os.path.expanduser(os.environ.get("MODELS_DIR", "~/ds4-models"))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)

JOB = {}   # {"recipe","action","log","started","proc","samples":[(t,n)]}

# ---------------- model / recipe data ----------------
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
                    "corpus": r.get("corpus")})
    return out

def show_cmd(recipe):
    try:
        o = subprocess.run([sys.executable, FORGEQUANT, "show", recipe],
                           cwd=HERE, capture_output=True, text=True, timeout=20, env=os.environ)
        return o.stdout
    except Exception as e:
        return f"(show failed: {e})"

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
                rec["imatrix_sha"] = (m.get("imatrix", {}) or {}).get("sha256", "")[:12]
            except Exception:
                pass
        out.append(rec)
    return out[:30]

# quant types offered in the recipe builder ("" = copy from template). Full set: deepseek4-quantize --help
QUANT_TYPES = ["", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_xs",
               "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0", "bf16"]
FAMILIES = ["routed_w1", "routed_w2", "routed_w3", "attention", "attn_proj", "shared", "embedding", "output"]

def list_imatrices():
    out = []
    for d in sorted(glob.glob(os.path.join(MODELS_DIR, "*.dat")), key=os.path.getmtime, reverse=True):
        out.append({"name": os.path.basename(d),
                    "mb": round(os.path.getsize(d) / 1e6),
                    "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(d)).isoformat(timespec="minutes")})
    return out

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
    p = os.path.join(RECIPES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else {}

def save_recipe(d):
    name = re.sub(r"[^a-zA-Z0-9_.-]", "", str(d.get("name", "")))[:40]
    if not name:
        return {"ok": False, "error": "name required"}
    quant = {k: v for k, v in (d.get("quant") or {}).items() if v in QUANT_TYPES and v}
    rec = {"name": name, "description": (d.get("description") or "custom recipe")[:200],
           "hf": d.get("hf") or defaults()["hf"], "template": d.get("template") or defaults()["template"],
           "quant": quant}
    if d.get("imatrix"): rec["imatrix"] = d["imatrix"]
    if d.get("corpus"):
        rec["corpus"] = d["corpus"]
        if not rec.get("imatrix"):
            rec["imatrix"] = "{models}/" + name + ".dat"   # imatrix will be built here from the corpus
    try:
        if d.get("imatrix_max_tokens"): rec["imatrix_max_tokens"] = int(d["imatrix_max_tokens"])
    except Exception: pass
    json.dump(rec, open(os.path.join(RECIPES, name + ".json"), "w"), indent=2)
    return {"ok": True, "saved": name}

# ---------------- job control ----------------
def start_job(recipe, action):
    if JOB.get("proc") and JOB["proc"].poll() is None:
        return {"ok": False, "error": "a job is already running"}
    if action not in ("build", "quantize", "imatrix"):
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
    return {"ok": True, "started": {"recipe": recipe, "action": action}}

def stop_job():
    p = JOB.get("proc")
    if p and p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            try: p.kill()
            except Exception: pass
        return {"ok": True, "stopped": True}
    # also pkill the heavy children in case they detached
    for pat in ("deepseek4-quantize", "imatrix-dataset"):
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    return {"ok": True, "stopped": False}

PHASE_RE = re.compile(r"forgequant\$ .*?/(deepseek4-quantize|ds4)\b.*?(--imatrix-dataset)?")
TENSOR_RE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")
TOK_RE = re.compile(r"tokens[=: ]+(\d+)")

def job_status():
    if not JOB:
        return {"running": False, "idle": True}
    p = JOB.get("proc")
    running = p is not None and p.poll() is None
    log = JOB.get("log")
    tail, n, m, phase, toks = "", None, None, "starting", None
    try:
        with open(log, "rb") as f:
            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 6000)); data = f.read().decode("utf-8", "ignore")
        tail = data
        for ln in data.splitlines():
            if "deepseek4-quantize" in ln and "forgequant$" in ln: phase = "quantize"
            elif "--imatrix-dataset" in ln: phase = "imatrix"
            mt = TENSOR_RE.search(ln)
            if mt: n, m = int(mt.group(1)), int(mt.group(2))
            tk = TOK_RE.search(ln)
            if tk: toks = int(tk.group(1))
    except Exception:
        pass
    # ETA from recent (time, n) samples
    eta = rate = None
    if running and n is not None and m:
        s = JOB.setdefault("samples", [])
        if not s or s[-1][1] != n: s.append((time.time(), n))
        s[:] = s[-12:]
        if len(s) >= 2 and s[-1][1] > s[0][1]:
            rate = (s[-1][1] - s[0][1]) / max(1e-6, s[-1][0] - s[0][0])
            if rate > 0: eta = round((m - n) / rate)
    failed = (not running) and p is not None and p.returncode not in (0, None)
    done = (not running) and p is not None and p.returncode == 0
    return {"running": running, "idle": False, "recipe": JOB.get("recipe"),
            "action": JOB.get("action"), "phase": phase if running else ("done" if done else "failed" if failed else "stopped"),
            "n": n, "m": m, "tokens": toks, "rate": round(rate, 2) if rate else None, "eta_s": eta,
            "elapsed_s": round(time.time() - JOB["started"]), "tail": tail[-3500:], "done": done, "failed": failed}

# ---------------- HTTP ----------------
PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>forge-ui</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#000;--card:#0a0a0a;--el:#121212;--bd:#232323;--bd2:#343434;--fg:#ededed;--mut:#9b9b9b;--mut2:#6a6a6a;--pri:#ededed;--prifg:#0a0a0a;--ok:#3fcf8e;--no:#f5544a;--info:#4aa3ff;--pur:#b07cff}
*{box-sizing:border-box}
html{-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;letter-spacing:-.006em}
.wrap{max-width:860px;margin:0 auto;padding:44px 24px 90px}
.mut{color:var(--mut)}.mono{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:.92em}
header{margin:0 0 30px}
header h1{font-size:20px;font-weight:600;margin:0;letter-spacing:-.02em;display:flex;align-items:center}
header h1 b{font-weight:600}
header .sub{color:var(--mut);font-size:14px;margin-top:7px;letter-spacing:-.01em}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:22px 24px;margin:16px 0}
.ch{display:flex;align-items:center;gap:10px;margin:0 0 20px}
.ch h2{font-size:15px;font-weight:600;margin:0;letter-spacing:-.015em}
.ch .sub{color:var(--mut2);font-size:13px;font-weight:400;letter-spacing:-.01em}
.step{width:21px;height:21px;border-radius:50%;background:var(--el);border:1px solid var(--bd2);color:var(--mut);font-size:11px;font-weight:600;display:inline-flex;align-items:center;justify-content:center;flex:none}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.grp{margin:20px 0}
.grp>.lbl{font-size:13px;font-weight:500;color:var(--fg);margin-bottom:5px;letter-spacing:-.01em}
.grp .cap{font-size:13px;color:var(--mut);margin:0 0 13px;max-width:64ch;line-height:1.5}
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
.cmd{font-size:12px;color:var(--mut2);font-family:ui-monospace,Menlo,monospace;word-break:break-all;margin-top:12px;line-height:1.65}
.hint{font-size:13px;color:var(--mut);letter-spacing:-.01em}
.divider{height:1px;background:var(--bd);margin:22px -24px}
</style></head><body><div class="wrap">
<header>
  <h1><svg class="ic" style="width:18px;height:18px;color:var(--fg);margin-right:10px" viewBox="0 0 24 24"><path d="m15 12-8.5 8.5a2.12 2.12 0 1 1-3-3L12 9"/><path d="M17.64 15 22 10.64"/><path d="m20.91 11.7-1.25-1.25c-.6-.6-.93-1.4-.93-2.25v-.86L16.01 4.6a5.56 5.56 0 0 0-3.94-1.64H9l.92.82A6.18 6.18 0 0 1 12 8.4v1.56l2 2h2.47l2.26 1.91"/></svg><b>forge</b>quant</h1>
  <div class="sub">Asymmetric quantization for DeepSeek-V4-Flash — pick a recipe, run it, watch it.</div>
</header>

<div class="card">
  <div class="ch"><span class="step">1</span><h2>Forge</h2><span class="sub">choose a recipe and an action, then run</span></div>
  <div class="row">
    <select id="recipe" style="min-width:170px"></select>
    <select id="action"><option value="build">build · imatrix + quantize</option><option value="quantize">quantize only</option><option value="imatrix">imatrix only</option></select>
    <button class="primary" id="go" onclick="forge()"><svg class="ic" viewBox="0 0 24 24"><path d="m15 12-8.5 8.5a2.12 2.12 0 1 1-3-3L12 9"/><path d="M17.64 15 22 10.64"/><path d="m20.91 11.7-1.25-1.25c-.6-.6-.93-1.4-.93-2.25v-.86L16.01 4.6a5.56 5.56 0 0 0-3.94-1.64H9l.92.82A6.18 6.18 0 0 1 12 8.4v1.56l2 2h2.47l2.26 1.91"/></svg>Forge</button>
    <button class="ghost" id="stop" onclick="stop()">Stop</button>
    <span class="hint" id="rdesc" style="margin:0 0 0 4px"></span>
  </div>
  <div class="cmd mono" id="cmd"></div>
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
  <div class="ch"><span class="step">2</span><h2>Recipe</h2><span class="sub">create or edit how the model is quantized</span></div>
  <div class="row" style="margin-bottom:6px">
    <input id="bname" placeholder="recipe name" style="width:170px">
    <input id="bdesc" placeholder="description" style="flex:1;min-width:200px">
    <span class="hint">start from</span><select id="bload"></select><button onclick="loadIntoEditor()"><svg class="ic" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>load</button>
  </div>

  <div class="grp">
    <div class="lbl">Quant per tensor family</div>
    <div class="cap"><b>(copy)</b> keeps the template's type. The routed experts are the 2-bit budget the imatrix re-allocates; attention/shared/output stay near-lossless (q8_0).</div>
    <div id="bfam"></div>
  </div>

  <div class="grp">
    <div class="lbl">Imatrix</div>
    <div class="cap">The importance matrix steers where bits go. <b>Load</b> an existing <span class="mono">.dat</span>, or <b>build from corpus</b> — forge runs the model over the corpus to extract one.</div>
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

<script>
const $=id=>document.getElementById(id);
function fmt(s){if(s==null)return '—';s=+s;if(s<60)return s+'s';const m=Math.floor(s/60);return m+'m'+String(s%60).padStart(2,'0')+'s';}
async function loadRecipes(){const r=await fetch('/api/recipes').then(r=>r.json());const sel=$('recipe');const cur=sel.value;
  sel.innerHTML=r.map(x=>`<option value="${x.name}">${x.name}</option>`).join('');if(cur)sel.value=cur;
  window.RECIPES=r;updRecipe();}
function updRecipe(){const r=(window.RECIPES||[]).find(x=>x.name===$('recipe').value);if(!r)return;
  $('rdesc').textContent=r.description||'';
  const q=Object.entries(r.quant||{}).map(([k,v])=>`<span class="pill">${k.replace('routed_','')}=${v}</span>`).join('');
  $('cmd').innerHTML=q+(r.imatrix?` <span class="pill">imatrix</span>`:'')+(r.corpus?` <span class="pill">corpus</span>`:'');}
async function forge(){const b=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({recipe:$('recipe').value,action:$('action').value})}).then(r=>r.json());
  if(!b.ok)alert(b.error);poll();}
async function stop(){await fetch('/api/stop',{method:'POST'});poll();}
const BC={quantize:'var(--info)',imatrix:'var(--pur)',done:'var(--ok)',failed:'var(--no)',stopped:'var(--mut)',starting:'var(--mut)'};
async function poll(){let s;try{s=await fetch('/api/status').then(r=>r.json());}catch(e){return;}
  const ph=s.phase||'idle';const pe=$('phase');pe.style.display=s.idle?'none':'inline-flex';pe.textContent=ph;pe.style.color=BC[ph]||'var(--mut)';
  $('go').disabled=!!s.running;$('stop').disabled=!s.running;
  let pct=0,lab='idle';
  if(s.m){pct=Math.round(100*s.n/s.m);lab=`${s.n} / ${s.m} tensors · ${pct}%`;}
  else if(s.phase==='imatrix'){lab=s.tokens?`${s.tokens.toLocaleString()} tokens collected`:'collecting imatrix…';pct=s.tokens?Math.min(99,s.tokens/1200):5;}
  else if(s.phase==='done')lab='done',pct=100;else if(s.phase==='failed')lab='failed';else if(s.running)lab='starting…',pct=2;
  $('fill').style.width=pct+'%';$('plab').textContent=lab;
  $('srecipe').textContent=s.recipe||'—';$('selapsed').textContent=fmt(s.elapsed_s);
  $('srate').textContent=s.rate?(s.rate.toFixed(1)+(s.phase==='quantize'?' tns/s':'')):'—';$('seta').textContent=fmt(s.eta_s);
  if(s.tail)$('log').textContent=s.tail;}
async function loadModels(){const m=await fetch('/api/models').then(r=>r.json());
  $('models').innerHTML=m.map(x=>{const q=Object.entries(x.quant||{}).map(([k,v])=>`<span class="pill">${k.replace('routed_','')}=${v}</span>`).join('');
   return `<tr><td class="mono">${x.path}</td><td>${x.gb} GB</td><td>${x.recipe||'—'}</td><td>${q||'—'}</td><td class="mut">${x.created||x.mtime}</td></tr>`;}).join('')||'<tr><td colspan=5 class="mut">no forged models yet</td></tr>';}
let FAMS=[];
async function loadTypes(){const t=await fetch('/api/types').then(r=>r.json());FAMS=t.families;
  const nm={routed_w1:'gate (w1)',routed_w2:'down (w2)',routed_w3:'up (w3)',attn_proj:'attn-proj'};
  const fld=f=>`<div class="fld">${nm[f]||f}<select id="f_${f}">${t.types.map(x=>`<option value="${x}">${x||'(copy)'}</option>`).join('')}</select></div>`;
  const exp=['routed_w1','routed_w3','routed_w2'].filter(f=>FAMS.includes(f)),oth=FAMS.filter(f=>!exp.includes(f));
  $('bfam').innerHTML=`<div style="font-size:11px;color:var(--mut2);margin:0 0 6px">routed experts · the 2-bit budget</div><div class="famgrid">${exp.map(fld).join('')}</div>`+
    `<div style="font-size:11px;color:var(--mut2);margin:13px 0 6px">other tensors · keep near-lossless</div><div class="famgrid">${oth.map(fld).join('')}</div>`;}
async function loadImats(){const im=await fetch('/api/imatrices').then(r=>r.json());const s=$('bimat'),cur=s.value;
  s.innerHTML='<option value="">— build from corpus —</option>'+im.map(x=>`<option value="{models}/${x.name}">${x.name} · ${x.mb}MB</option>`).join('');if(cur)s.value=cur;
  s.onchange=()=>{const build=!s.value;$('bcorpus').style.display=build?'':'none';$('bmaxtok').style.display=build?'':'none';};s.onchange();}
async function loadEd(){const r=await fetch('/api/recipes').then(r=>r.json());$('bload').innerHTML='<option value="">— load preset —</option>'+r.map(x=>`<option>${x.name}</option>`).join('');
  const d=await fetch('/api/defaults').then(r=>r.json());if(!$('bhf').value)$('bhf').value=d.hf;if(!$('btmpl').value)$('btmpl').value=d.template;}
async function loadIntoEditor(){const n=$('bload').value;if(!n)return;const r=await fetch('/api/recipe?name='+encodeURIComponent(n)).then(r=>r.json());
  $('bname').value=r.name||n;$('bdesc').value=r.description||'';$('bhf').value=r.hf||'';$('btmpl').value=r.template||'';
  FAMS.forEach(f=>{const s=$('f_'+f);if(s)s.value=(r.quant||{})[f]||'';});
  $('bimat').value=r.imatrix||'';$('bcorpus').value=r.corpus||'';$('bmaxtok').value=r.imatrix_max_tokens||'';$('bimat').onchange();}
async function saveRecipe(){const quant={};FAMS.forEach(f=>{const v=$('f_'+f).value;if(v)quant[f]=v;});
  const body={name:$('bname').value,description:$('bdesc').value,hf:$('bhf').value,template:$('btmpl').value,quant};
  if($('bimat').value)body.imatrix=$('bimat').value;else if($('bcorpus').value){body.corpus=$('bcorpus').value;if($('bmaxtok').value)body.imatrix_max_tokens=$('bmaxtok').value;}
  const res=await fetch('/api/save_recipe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());
  $('bmsg').textContent=res.ok?('saved '+res.saved):('error — '+res.error);$('bmsg').style.color=res.ok?'var(--ok)':'var(--no)';if(res.ok){await loadRecipes();await loadEd();$('recipe').value=res.saved;updRecipe();}}
$('recipe').onchange=updRecipe;
loadRecipes();loadModels();poll();loadTypes();loadImats();loadEd();
setInterval(poll,1500);setInterval(loadModels,6000);setInterval(loadRecipes,15000);setInterval(loadImats,12000);
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
        p = self.path.split("?")[0]
        if p == "/": self._send(200, PAGE)
        elif p == "/api/recipes": self._json(list_recipes())
        elif p == "/api/models": self._json(list_models())
        elif p == "/api/status": self._json(job_status())
        elif p == "/api/types": self._json({"types": QUANT_TYPES, "families": FAMILIES})
        elif p == "/api/imatrices": self._json(list_imatrices())
        elif p == "/api/defaults": self._json(defaults())
        elif p == "/api/recipe":
            from urllib.parse import urlparse, parse_qs
            self._json(recipe_raw(parse_qs(urlparse(self.path).query).get("name", [""])[0]))
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
