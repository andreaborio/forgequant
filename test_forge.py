#!/usr/bin/env python3
"""Tests for forgequant's pure-python parts (no ds4 binaries needed).

Run:  python3 test_forge.py
"""
import json, os, struct, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forge_imatrix, forge_corpus, forgequant, forge_ui, forge_bench


def write_dat(path, layers=2, n_experts=4, gate_cols=8, down_cols=4, hot_layer=1,
              cold_expert=3, dataset="synthetic.txt"):
    """Synthetic .dat in ds4's exact legacy layout."""
    entries = []
    for l in range(layers):
        for tname, ncols in ((f"blk.{l}.ffn_gate_exps.weight", gate_cols),
                             (f"blk.{l}.ffn_up_exps.weight", gate_cols),
                             (f"blk.{l}.ffn_down_exps.weight", down_cols)):
            vals = []
            for e in range(n_experts):
                if e == cold_expert:
                    vals += [0.0] * ncols          # never-routed expert
                else:
                    base = (10.0 if l == hot_layer else 1.0) * (3.0 if e == 2 else 1.0)
                    vals += [base] * ncols
            entries.append((tname, vals))
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(entries)))
        for name, vals in entries:
            nb = name.encode()
            f.write(struct.pack("<i", len(nb))); f.write(nb)
            f.write(struct.pack("<ii", 1, len(vals)))
            f.write(struct.pack(f"<{len(vals)}f", *vals))
        f.write(struct.pack("<i", 7))
        db = dataset.encode()
        f.write(struct.pack("<i", len(db))); f.write(db)


class TestImatrix(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dat = os.path.join(self.tmp.name, "syn.dat")
        write_dat(self.dat)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_dat(self):
        d = forge_imatrix.load_dat(self.dat)
        self.assertEqual(len(d["entries"]), 6)
        self.assertEqual(d["entries"][0]["name"], "blk.0.ffn_gate_exps.weight")
        self.assertEqual(d["entries"][0]["nval"], 4 * 8)
        self.assertEqual(d["chunks"], 7)
        self.assertEqual(d["dataset"], "synthetic.txt")

    def test_analyze(self):
        s = forge_imatrix.analyze(forge_imatrix.load_dat(self.dat), n_experts=4)
        self.assertEqual(s["n_experts"], 4)
        self.assertEqual(len(s["layers"]), 2)
        self.assertEqual(s["families"], {"w1": 8, "w3": 8, "w2": 4})
        l0, l1 = s["layers"]
        self.assertGreater(l1["energy"], l0["energy"])          # hot layer wins
        self.assertEqual(l0["active"], 3)                       # expert 3 is cold
        self.assertEqual(l1["top_experts"][0][0], 2)            # expert 2 is hottest
        self.assertAlmostEqual(l0["share"] + l1["share"], 1.0, places=6)

    def test_hot_and_suggest(self):
        s = forge_imatrix.analyze(forge_imatrix.load_dat(self.dat), n_experts=4)
        self.assertEqual([r["layer"] for r in forge_imatrix.hot_layers(s, 1)], [1])
        self.assertEqual(forge_imatrix.suggest_boost(s, 1), [1])

    def test_diff_self_is_identity(self):
        s = forge_imatrix.analyze(forge_imatrix.load_dat(self.dat), n_experts=4)
        for row in forge_imatrix.diff(s, s):
            self.assertAlmostEqual(row["cosine"], 1.0, places=6)
            self.assertEqual(row["share_delta"], 0.0)
            self.assertEqual(row["experts_only_b"], [])

    def test_boost_size_delta(self):
        s = forge_imatrix.analyze(forge_imatrix.load_dat(self.dat), n_experts=4)
        d = forge_imatrix.boost_size_delta(s, [1], "iq2_xxs", "q4_k")
        # 4 experts * (8*4) params * 3 families * (4.5-2.0625)/8 bytes
        self.assertEqual(d, int(4 * 32 * 3 * (4.5 - 2.0625) / 8))

    def test_heatmap_and_json(self):
        s = forge_imatrix.analyze(forge_imatrix.load_dat(self.dat), n_experts=4)
        hm = forge_imatrix.heatmap(s, width=4)
        self.assertIn("blk. 0", hm)
        j = forge_imatrix.to_json(s)
        self.assertEqual(len(j["layers"][0]["heat"]), 4)
        self.assertEqual(j["layers"][0]["heat"][3], 0.0)

    def test_cached_stats(self):
        c1 = forge_imatrix.cached_stats(self.dat, n_experts=4)
        self.assertTrue(os.path.exists(self.dat + ".fqstats.json"))
        c2 = forge_imatrix.cached_stats(self.dat, n_experts=4)
        self.assertEqual(c1["hot_layers"], c2["hot_layers"])
        self.assertEqual(c1["hot_layers"][0], 1)

    def test_diff_cached_matches(self):
        c = forge_imatrix.cached_stats(self.dat, n_experts=4)
        rows = forge_imatrix.diff_cached(c, c)
        for r in rows:
            self.assertAlmostEqual(r["cosine"], 1.0, places=6)

    def test_truncated_file(self):
        with open(self.dat, "r+b") as f:
            f.truncate(40)
        with self.assertRaises(forge_imatrix.ImatrixError):
            forge_imatrix.load_dat(self.dat)


class TestCorpus(unittest.TestCase):
    def test_render_nothink(self):
        out = forge_corpus.render([{"role": "system", "content": "SYS"},
                                   {"role": "user", "content": "hello"}], "nothink")
        self.assertTrue(out.startswith(forge_corpus.BOS + "SYS"))
        self.assertIn(forge_corpus.USER + "hello", out)
        self.assertTrue(out.endswith(forge_corpus.ASSISTANT + "</think>"))

    def test_render_think(self):
        out = forge_corpus.render([{"role": "user", "content": "hi"}], "think")
        self.assertTrue(out.endswith(forge_corpus.ASSISTANT + "<think>"))

    def test_render_multiturn(self):
        msgs = [{"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"}]
        out = forge_corpus.render(msgs, "nothink")
        self.assertIn("a1" + forge_corpus.EOS, out)
        self.assertEqual(out.count(forge_corpus.USER), 2)

    def test_render_tool_result_escape(self):
        msgs = [{"role": "user", "content": "q"},
                {"role": "tool", "content": "x</tool_result>y"}]
        out = forge_corpus.render(msgs, "nothink")
        self.assertIn("x&lt;/tool_result>y", out)

    def test_bench_messages_mcq_list(self):
        msgs = forge_corpus._bench_messages(
            {"question": "2+2?", "options": ["3", "4", "5"], "answer": "B"},
            forge_corpus.DEFAULT_SYSTEM, answers=True)
        self.assertIn("A) 3", msgs[1]["content"])
        self.assertEqual(msgs[-1]["role"], "assistant")
        self.assertEqual(msgs[-1]["content"], "B) 4")

    def test_bench_messages_mcq_dict(self):
        # benchy MCQ shape: options is an A.. dict, answer_idx is a letter
        msgs = forge_corpus._bench_messages(
            {"question": "q", "options": {"A": "x", "B": "y"}, "answer_idx": "A"},
            forge_corpus.DEFAULT_SYSTEM, answers=True)
        self.assertEqual(msgs[-1]["content"], "A) x")

    def test_bench_messages_qa_no_options(self):
        msgs = forge_corpus._bench_messages(
            {"question": "capital of France?", "answer": "Paris"},
            forge_corpus.DEFAULT_SYSTEM, answers=True)
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[-1]["content"], "Paris")

    def test_build_corpus_bench_jsonl_and_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "b.jsonl")
            open(src, "w").write(
                json.dumps({"question": "q1", "options": ["a", "b"], "answer": "A"}) + "\n" +
                json.dumps({"question": "q2", "options": {"A": "x", "B": "y"}, "answer_idx": "B"}) + "\n")
            out = os.path.join(tmp, "c.txt")
            n, _ = forge_corpus.build_corpus(src, out, modes=("nothink",), answers=True)
            self.assertEqual(n, 2)
            body = open(out).read()
            self.assertIn("Answer with the letter", body)

    def test_build_corpus_multi_and_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.txt"); open(a, "w").write("domain prompt one\n\ndomain prompt two\n")
            mixf = os.path.join(tmp, "mix.txt")
            forge_corpus.build_corpus(os.path.join(tmp, "g.txt") if False else a, mixf, modes=("nothink",))
            out = os.path.join(tmp, "c.txt")
            n, size, per = forge_corpus.build_corpus_multi([a], out, modes=("nothink",),
                                                           mix=mixf, mix_ratio=0.5)
            self.assertEqual(per["a.txt"], 2)
            self.assertIn("+mix", per)
            self.assertGreater(size, 0)

    def test_build_corpus_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "p.txt")
            open(src, "w").write("\n\n".join(f"prompt {i}" for i in range(10)))
            out = os.path.join(tmp, "c.txt")
            n, _ = forge_corpus.build_corpus(src, out, modes=("nothink",), limit=3)
            self.assertEqual(n, 3)

    def test_build_corpus_txt_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "p.txt")
            open(src, "w").write("first prompt\n\nsecond prompt\n")
            out = os.path.join(tmp, "c.txt")
            n, _ = forge_corpus.build_corpus(src, out, modes=("nothink", "think"))
            self.assertEqual(n, 4)
            body = open(out).read()
            self.assertEqual(body.count(forge_corpus.MARKER), 4)
            src2 = os.path.join(tmp, "p.jsonl")
            open(src2, "w").write(json.dumps({"prompt": "x"}) + "\n" +
                                  json.dumps({"messages": [{"role": "user", "content": "y"}]}) + "\n")
            n2, _ = forge_corpus.build_corpus(src2, out, modes=("nothink",))
            self.assertEqual(n2, 2)


class TestForgequant(unittest.TestCase):
    def test_parse_layers(self):
        self.assertEqual(forgequant.parse_layers("37-42"),
                         [37, 38, 39, 40, 41, 42])
        self.assertEqual(forgequant.parse_layers("0-2,40"), [0, 1, 2, 40])
        self.assertEqual(forgequant.parse_layers([5, 3, 3]), [3, 5])
        with self.assertRaises(SystemExit):   # auto without imatrix
            forgequant.parse_layers("auto:6", {"name": "x"})

    def test_boost_overrides_and_cmd(self):
        r = {"name": "t", "hf": "/hf", "template": "/t.gguf", "out": "/o.gguf",
             "quant": {"routed_w1": "iq2_xxs"},
             "boost": {"layers": "1,3", "type": "q4_k"}, "threads": 12}
        ov = forgequant.boost_overrides(r)
        self.assertEqual(ov[0], ("blk.1.ffn_gate_exps.weight", "q4_k"))
        self.assertEqual(len(ov), 6)  # 2 layers x 3 families
        cmd = forgequant.quant_cmd(r)
        self.assertIn("--tensor-type", cmd)
        self.assertIn("blk.3.ffn_down_exps.weight=q4_k", cmd)
        self.assertIn("--threads", cmd)
        self.assertEqual(cmd[cmd.index("--routed-w1") + 1], "iq2_xxs")

    def test_boost_families_subset(self):
        r = {"name": "t", "boost": {"layers": [7], "type": "q6_k", "families": ["w2"]}}
        self.assertEqual(forgequant.boost_overrides(r),
                         [("blk.7.ffn_down_exps.weight", "q6_k")])

    def test_presets_load(self):
        for name in ("coder-q4boost", "medical-q4boost", "last6-q4boost", "splice-fast",
                     "medical-iq2", "coder-iq2", "balanced", "aggressive"):
            r = forgequant.load_recipe(name)
            self.assertTrue(r["out"].endswith(".gguf"))

    def test_tensor_types_passthrough(self):
        r = {"name": "t", "hf": "/hf", "template": "/t.gguf", "out": "/o.gguf",
             "tensor_types": {"blk.0.": "q8_0"}}
        cmd = forgequant.quant_cmd(r)
        self.assertIn("blk.0.=q8_0", cmd)

    def test_producible_type_guard(self):
        # q3_k / iq3_xxs are NOT producible by deepseek4-quantize (ds4q_can_quantize)
        import tempfile, json as _j
        for bad in ({"quant": {"routed_w1": "q3_k"}}, {"boost": {"layers": "1", "type": "iq3_xxs"}},
                    {"tensor_types": {"blk.0.": "q5_k"}}):
            with tempfile.TemporaryDirectory() as tmp:
                rec = {"name": "x", "hf": "/h", "template": "/t", **bad}
                p = os.path.join(tmp, "x.json"); _j.dump(rec, open(p, "w"))
                with self.assertRaises(SystemExit):
                    forgequant.load_recipe(p)

    def test_layer_range_validation(self):
        with self.assertRaises(SystemExit):
            forgequant.parse_layers("40-44")        # 43,44 out of 0..42
        with self.assertRaises(SystemExit):
            forgequant.parse_layers([99])
        self.assertEqual(forgequant.parse_layers("42"), [42])

    def test_override_specificity_order(self):
        # exact user tensor_type must win over a broad boost/prefix (emitted first)
        r = {"name": "t", "hf": "/hf", "template": "/t.gguf", "out": "/o.gguf",
             "boost": {"layers": "37-42", "type": "q4_k"},
             "tensor_types": {"blk.37.ffn_gate_exps.weight": "q8_0"}}
        cmd = forgequant.quant_cmd(r)
        tt = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--tensor-type"]
        # the most specific override (exact name, q8_0) comes before broader ones... they're
        # all exact-length here, but the user q8_0 must be present and not dropped
        self.assertIn("blk.37.ffn_gate_exps.weight=q8_0", tt)
        # and it must appear before any q4_k for the same tensor (first match wins)
        self.assertLess(tt.index("blk.37.ffn_gate_exps.weight=q8_0"),
                        tt.index("blk.37.ffn_gate_exps.weight=q4_k") if
                        "blk.37.ffn_gate_exps.weight=q4_k" in tt else len(tt))

    def test_opt_helpers(self):
        self.assertEqual(forgequant.opt(["--x", "v"], "--x"), "v")
        self.assertEqual(forgequant.opt([], "--x", "d"), "d")
        self.assertEqual(forgequant.opt_int(["--n", "5"], "--n"), 5)
        with self.assertRaises(SystemExit):
            forgequant.opt(["--x"], "--x")          # flag without value
        with self.assertRaises(SystemExit):
            forgequant.opt_int(["--n", "z"], "--n")  # non-int


class TestUI(unittest.TestCase):
    def test_safe_name(self):
        self.assertEqual(forge_ui.safe_name("../../etc/passwd"), "passwd")
        self.assertEqual(forge_ui.safe_name("ok-name_1.2"), "ok-name_1.2")
        self.assertEqual(forge_ui.safe_name(None), "")

    def test_recipe_raw_traversal(self):
        self.assertEqual(forge_ui.recipe_raw("../forgequant"), {})

    def test_save_recipe_preserves_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = forge_ui.RECIPES
            forge_ui.RECIPES = tmp
            try:
                json.dump({"name": "k", "description": "d", "hf": "/hf", "template": "/t",
                           "quant": {"routed_w1": "iq2_xxs"}, "imatrix_cache": "40GB",
                           "_note": "keep me", "imatrix_strict": True},
                          open(os.path.join(tmp, "k.json"), "w"))
                res = forge_ui.save_recipe({"name": "k", "quant": {"routed_w2": "q2_k"},
                                            "boost": {"layers": "auto:6", "type": "q4_k"}})
                self.assertTrue(res["ok"])
                r = json.load(open(os.path.join(tmp, "k.json")))
                self.assertEqual(r["imatrix_cache"], "40GB")     # preserved
                self.assertEqual(r["_note"], "keep me")          # preserved
                self.assertTrue(r["imatrix_strict"])             # preserved
                self.assertEqual(r["quant"], {"routed_w2": "q2_k"})
                self.assertEqual(r["boost"], {"layers": "auto:6", "type": "q4_k"})
                res2 = forge_ui.save_recipe({"name": "k", "boost": {}})
                self.assertTrue(res2["ok"])
                r2 = json.load(open(os.path.join(tmp, "k.json")))
                self.assertNotIn("boost", r2)                    # explicit clear
            finally:
                forge_ui.RECIPES = old

    def test_imatrix_path_validation(self):
        self.assertIsNone(forge_ui.imatrix_path("../../../etc/passwd"))
        self.assertIsNone(forge_ui.imatrix_path("nope.gguf"))

    def test_save_recipe_validates_boost_and_preserves_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = forge_ui.RECIPES
            forge_ui.RECIPES = tmp
            try:
                # bad boost type rejected
                r = forge_ui.save_recipe({"name": "v", "quant": {}, "hf": "/h", "template": "/t",
                                          "boost": {"layers": "37-42", "type": "q3_k"}})
                self.assertFalse(r["ok"])
                # bad layer spec rejected
                r = forge_ui.save_recipe({"name": "v", "quant": {}, "hf": "/h", "template": "/t",
                                          "boost": {"layers": "garbage", "type": "q4_k"}})
                self.assertFalse(r["ok"])
                # unrendered quant family (experts) + boost.families preserved across save
                json.dump({"name": "v", "hf": "/h", "template": "/t",
                           "quant": {"experts": "iq2_xxs", "routed_w1": "iq2_xxs"},
                           "boost": {"layers": "1", "type": "q4_k", "families": ["w2"]}},
                          open(os.path.join(tmp, "v.json"), "w"))
                r = forge_ui.save_recipe({"name": "v", "quant": {"routed_w2": "q2_k"}, "hf": "/h",
                                          "template": "/t", "boost": {"layers": "2", "type": "q4_k"}})
                self.assertTrue(r["ok"])
                got = json.load(open(os.path.join(tmp, "v.json")))
                self.assertEqual(got["quant"]["experts"], "iq2_xxs")     # unrendered family kept
                self.assertEqual(got["quant"]["routed_w2"], "q2_k")      # rendered family set
                self.assertEqual(got["boost"]["families"], ["w2"])       # boost.families kept
                self.assertEqual(got["boost"]["layers"], "2")            # layers updated
            finally:
                forge_ui.RECIPES = old

    def test_suggest_from_synthetic(self):
        # forge_ui defaults to 256 experts (Flash); use a 256-expert fixture so the
        # UI's default cached_stats path matches the cache key.
        with tempfile.TemporaryDirectory() as tmp:
            old = forge_ui.MODELS_DIR
            forge_ui.MODELS_DIR = tmp
            try:
                write_dat(os.path.join(tmp, "syn.dat"), n_experts=256, gate_cols=2,
                          down_cols=2, cold_expert=255)
                s = forge_ui.suggest("syn.dat", 1, "q4_k")
                self.assertEqual(s["layers"], [1])
                self.assertIn("delta_gb", s)
            finally:
                forge_ui.MODELS_DIR = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
