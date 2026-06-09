#!/usr/bin/env python3
"""forge_imatrix — read ds4's legacy .dat imatrix and extract the activation paths.

The .dat written by `ds4 --imatrix-out` (and `ds4-server --imatrix-out`) packs, for
every routed-expert tensor, one importance vector PER EXPERT: entry value length is
n_expert * n_cols (gate/up: n_embd cols, down: n_ff_exp cols), each value the
count-normalized mean of squared activations seen by that expert column.

That is the "brain path" signal: which experts a workload actually drives, and how
hard. This module parses the file (stdlib only), aggregates per-(layer, expert)
energy, ranks hot/cold layers and experts, renders terminal heatmaps, diffs two
imatrices (e.g. code vs medical), and suggests per-layer boost overrides.

Caveat (by construction of the format): values are count-NORMALIZED, so they measure
how strongly an expert activates when routed, not how often it is routed. Experts
never routed during collection have all-zero vectors and show up as "cold".

Binary layout (verified against ds4.c imatrix_write_*):
  int32 n_entries
  per entry: int32 name_len, name bytes, int32 ncall(=1), int32 nval, nval*float32
  trailer (optional): int32 chunks, int32 dataset_len, dataset bytes
"""
import json, math, os, struct
from array import array

try:  # optional fast path; everything works without it
    import numpy as _np
except ImportError:
    _np = None

# bits-per-weight for size estimates (k-quants include scales overhead)
BPW = {"iq1_s": 1.5625, "iq1_m": 1.75, "iq2_xxs": 2.0625, "iq2_xs": 2.3125,
       "iq2_s": 2.5, "q2_k": 2.5625, "iq3_xxs": 3.0625, "iq3_s": 3.4375,
       "q3_k": 3.4375, "iq4_xs": 4.25, "q4_k": 4.5, "q5_k": 5.5, "q6_k": 6.5625,
       "q8_0": 8.5, "f16": 16.0, "bf16": 16.0, "f32": 32.0}

FAMILY_OF = {"ffn_gate_exps": "w1", "ffn_down_exps": "w2", "ffn_up_exps": "w3"}


class ImatrixError(Exception):
    pass


def _read(f, n, what):
    b = f.read(n)
    if len(b) != n:
        raise ImatrixError(f"truncated .dat while reading {what}")
    return b


def load_dat(path):
    """Parse a legacy .dat imatrix -> {"entries": [{name, ncall, nval, values}], "chunks", "dataset"}."""
    entries = []
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        (n,) = struct.unpack("<i", _read(f, 4, "entry count"))
        if n <= 0 or n > 100000:
            raise ImatrixError(f"implausible entry count {n} — not a .dat imatrix?")
        for _ in range(n):
            (ln,) = struct.unpack("<i", _read(f, 4, "name length"))
            if ln <= 0 or ln > 4096:
                raise ImatrixError(f"implausible name length {ln}")
            name = _read(f, ln, "name").decode("utf-8", "replace")
            ncall, nval = struct.unpack("<ii", _read(f, 8, "ncall/nval"))
            if nval <= 0 or 4 * nval > size:
                raise ImatrixError(f"implausible nval {nval} for {name}")
            vals = array("f")
            vals.frombytes(_read(f, 4 * nval, f"values of {name}"))
            entries.append({"name": name, "ncall": ncall, "nval": nval, "values": vals})
        chunks = dataset = None
        tail = f.read(4)
        if len(tail) == 4:
            (chunks,) = struct.unpack("<i", tail)
            tail = f.read(4)
            if len(tail) == 4:
                (dl,) = struct.unpack("<i", tail)
                if 0 <= dl <= 65536:
                    dataset = f.read(dl).decode("utf-8", "replace")
    return {"entries": entries, "chunks": chunks, "dataset": dataset, "path": path}


def _layer_family(name):
    """blk.7.ffn_gate_exps.weight -> (7, 'w1') ; None for non-routed entries."""
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "blk":
        try:
            layer = int(parts[1])
        except ValueError:
            return None
        fam = FAMILY_OF.get(parts[2])
        if fam:
            return layer, fam
    return None


def analyze(dat, n_experts=256):
    """Aggregate a parsed .dat into per-(layer, family, expert) energy.

    Returns {
      n_layers, n_experts, dataset, families: {fam: n_cols},
      energy: {layer: {fam: [per-expert mean energy]}},
      layers: [{layer, energy, share, active, top_experts: [(e, energy)]}],  # sorted by layer
    }
    """
    energy = {}
    families = {}
    for e in dat["entries"]:
        lf = _layer_family(e["name"])
        if not lf:
            continue
        layer, fam = lf
        if e["nval"] % n_experts != 0:
            raise ImatrixError(
                f"{e['name']}: nval {e['nval']} not divisible by n_experts {n_experts}")
        ncols = e["nval"] // n_experts
        families[fam] = ncols
        v = e["values"]
        if _np is not None:
            per = _np.frombuffer(v.tobytes(), dtype=_np.float32).reshape(
                n_experts, ncols).mean(axis=1).tolist()
        else:
            per = [sum(v[x * ncols:(x + 1) * ncols]) / ncols for x in range(n_experts)]
        energy.setdefault(layer, {})[fam] = per
    if not energy:
        raise ImatrixError("no routed-expert entries (blk.N.ffn_*_exps.weight) found")

    n_layers = max(energy) + 1
    total = 0.0
    layer_rows = []
    for layer in sorted(energy):
        fams = energy[layer]
        combined = [0.0] * n_experts
        for per in fams.values():
            # normalize each family to its own max so w1/w2/w3 scales are comparable
            mx = max(per) or 1.0
            for i, val in enumerate(per):
                combined[i] += val / mx
        le = sum(sum(per) for per in fams.values())
        active = sum(1 for c in combined if c > 0.0)
        top = sorted(enumerate(combined), key=lambda t: -t[1])[:16]
        layer_rows.append({"layer": layer, "energy": le, "active": active,
                           "combined": combined, "top_experts": top})
        total += le
    for row in layer_rows:
        row["share"] = row["energy"] / total if total else 0.0
    return {"n_layers": n_layers, "n_experts": n_experts, "dataset": dat.get("dataset"),
            "families": families, "energy": energy, "layers": layer_rows,
            "total_energy": total, "path": dat.get("path")}


def hot_layers(stats, n):
    """Top-n layers by energy share (the layers where this workload concentrates)."""
    return sorted(stats["layers"], key=lambda r: -r["energy"])[:n]


def suggest_boost(stats, n):
    """Suggest n layers to upcast, as a sorted layer-id list."""
    return sorted(r["layer"] for r in hot_layers(stats, n))


def diff(stats_a, stats_b):
    """Compare two analyzed imatrices (same model). Returns per-layer divergence rows.

    cosine: similarity of combined per-expert energy (1.0 = identical shape);
    share_delta: b minus a energy share. Low cosine = the two workloads light up
    different experts in that layer — prime boost candidates for the b-domain.
    """
    if stats_a["n_experts"] != stats_b["n_experts"]:
        raise ImatrixError("imatrices have different expert counts")
    rows = []
    by_layer_b = {r["layer"]: r for r in stats_b["layers"]}
    for ra in stats_a["layers"]:
        rb = by_layer_b.get(ra["layer"])
        if not rb:
            continue
        a, b = ra["combined"], rb["combined"]
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        only_b = sorted((e for e, v in enumerate(b)
                         if v > 0 and a[e] == 0.0), key=lambda e: -b[e])[:8]
        rows.append({"layer": ra["layer"], "cosine": dot / (na * nb),
                     "share_a": ra["share"], "share_b": rb["share"],
                     "share_delta": rb["share"] - ra["share"], "experts_only_b": only_b})
    return rows


def suggest_boost_diff(stats_a, stats_b, n):
    """n layers where workload b diverges most from a (lowest cosine)."""
    rows = diff(stats_a, stats_b)
    return sorted(r["layer"] for r in sorted(rows, key=lambda r: r["cosine"])[:n])


def boost_size_delta(stats, layers, from_type, to_type, families=("w1", "w2", "w3")):
    """Estimated GGUF growth in bytes when upcasting `layers` from from_type to to_type."""
    fcols = stats["families"]
    n_exp = stats["n_experts"]
    n_embd = fcols.get("w1") or fcols.get("w3") or 0
    n_ff = fcols.get("w2") or 0
    params_per_layer = 0
    if "w1" in families: params_per_layer += n_exp * n_embd * n_ff
    if "w3" in families: params_per_layer += n_exp * n_embd * n_ff
    if "w2" in families: params_per_layer += n_exp * n_ff * n_embd
    dbpw = BPW.get(to_type, 4.5) - BPW.get(from_type, 2.0625)
    return int(len(layers) * params_per_layer * dbpw / 8)


SHADES = " ▁▂▃▄▅▆▇█"


def heatmap(stats, width=64):
    """Terminal heatmap: one row per layer, experts bucketed into `width` columns."""
    n_exp = stats["n_experts"]
    bucket = max(1, n_exp // width)
    out = []
    gmax = 0.0
    rows = []
    for r in stats["layers"]:
        c = r["combined"]
        vals = [max(c[i:i + bucket]) for i in range(0, n_exp, bucket)]
        rows.append((r, vals))
        gmax = max(gmax, max(vals))
    gmax = gmax or 1.0
    out.append(f"experts 0..{n_exp - 1} → bucketed ×{bucket} | energy share | active experts")
    for r, vals in rows:
        cells = "".join(SHADES[min(len(SHADES) - 1, int(v / gmax * (len(SHADES) - 1) + 0.5))]
                        for v in vals)
        out.append(f"blk.{r['layer']:>2} {cells} {r['share'] * 100:5.1f}%  {r['active']}/{n_exp}")
    return "\n".join(out)


def to_json(stats, top=16):
    """Compact JSON-able dict for the UI (heat matrix normalized 0..1 per layer)."""
    layers = []
    for r in stats["layers"]:
        mx = max(r["combined"]) or 1.0
        layers.append({"layer": r["layer"], "share": round(r["share"], 5),
                       "active": r["active"],
                       "heat": [round(v / mx, 3) for v in r["combined"]],
                       "top_experts": [[e, round(v, 4)] for e, v in r["top_experts"][:top]]})
    return {"n_layers": stats["n_layers"], "n_experts": stats["n_experts"],
            "dataset": stats.get("dataset"), "path": stats.get("path"),
            "families": stats["families"], "layers": layers}


def cached_stats(path, n_experts=256):
    """analyze() with a JSON cache next to the .dat (keyed on size+mtime).

    The cache holds the to_json() projection plus hot-layer ranking — everything the
    UI and `boost: auto` need — so the multi-hundred-MB file is parsed once.
    """
    st = os.stat(path)
    cache = path + ".fqstats.json"
    src = [st.st_size, int(st.st_mtime), n_experts]
    if os.path.exists(cache):
        try:
            c = json.load(open(cache))
            if c.get("_src") == src:
                return c
        except Exception:
            pass
    stats = analyze(load_dat(path), n_experts)
    c = to_json(stats)
    c["hot_layers"] = [r["layer"] for r in sorted(stats["layers"], key=lambda r: -r["energy"])]
    c["_src"] = src
    try:
        with open(cache, "w") as f:
            json.dump(c, f)
    except OSError:
        pass
    return c


def diff_cached(ca, cb):
    """diff() on two cached_stats() dicts. heat is combined/max per layer; cosine is
    scale-invariant so it matches diff() on the raw analysis."""
    rows = []
    by_a = {r["layer"]: r for r in ca["layers"]}
    for rb in cb["layers"]:
        ra = by_a.get(rb["layer"])
        if not ra:
            continue
        a, b = ra["heat"], rb["heat"]
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        only_b = sorted((e for e, v in enumerate(b) if v > 0 and a[e] == 0.0),
                        key=lambda e: -b[e])[:8]
        rows.append({"layer": rb["layer"], "cosine": dot / (na * nb),
                     "share_delta": rb["share"] - ra["share"], "experts_only_b": only_b})
    return rows


def boost_delta_cached(c, layers, from_type, to_type, families=("w1", "w2", "w3")):
    """boost_size_delta() on a cached_stats() dict."""
    f = c["families"]
    n_embd = f.get("w1") or f.get("w3") or 0
    n_ff = f.get("w2") or 0
    per_fam = n_embd * n_ff
    n_fams = len([x for x in families if x in ("w1", "w2", "w3")])
    params_per_layer = c["n_experts"] * per_fam * n_fams
    dbpw = BPW.get(to_type, 4.5) - BPW.get(from_type, 2.0625)
    return int(len(layers) * params_per_layer * dbpw / 8)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="forge_imatrix", description=__doc__.splitlines()[0])
    ap.add_argument("dat", help="imatrix .dat file")
    ap.add_argument("--diff", help="second .dat to compare against (dat=baseline, diff=domain)")
    ap.add_argument("--experts", type=int, default=256, help="routed expert count (Flash: 256)")
    ap.add_argument("--top", type=int, default=6, help="boost suggestion size")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    a = ap.parse_args(argv)
    stats = analyze(load_dat(a.dat), a.experts)
    if a.diff:
        stats_b = analyze(load_dat(a.diff), a.experts)
        rows = diff(stats, stats_b)
        if a.json:
            print(json.dumps({"diff": rows, "boost": suggest_boost_diff(stats, stats_b, a.top)}, indent=2))
            return
        print(f"baseline: {a.dat}\ndomain:   {a.diff}\n")
        print("layer  cosine  Δshare   experts only in domain")
        for r in sorted(rows, key=lambda r: r["cosine"]):
            print(f"blk.{r['layer']:>2} {r['cosine']:7.4f} {r['share_delta'] * 100:+6.2f}%  "
                  f"{','.join(str(e) for e in r['experts_only_b']) or '—'}")
        print(f"\nboost candidates (most divergent): {suggest_boost_diff(stats, stats_b, a.top)}")
        return
    if a.json:
        print(json.dumps(to_json(stats), indent=2))
        return
    print(heatmap(stats))
    hot = hot_layers(stats, a.top)
    print(f"\nhot layers (energy): {[r['layer'] for r in hot]}")
    delta = boost_size_delta(stats, [r["layer"] for r in hot], "iq2_xxs", "q4_k")
    print(f"boost {a.top} layers iq2_xxs→q4_k ≈ +{delta / 1e9:.1f} GB")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
