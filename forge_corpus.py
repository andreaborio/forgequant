#!/usr/bin/env python3
"""forge_corpus — render raw prompts into a ds4 imatrix calibration corpus.

`ds4 --imatrix-dataset` consumes a plain-text file of chat prompts that are already
rendered with the DeepSeek-V4 template, delimited by `===== DS4_IMATRIX_PROMPT ...`
marker lines. This module is a faithful stdlib port of the rendering used by ds4's
dataset builder (which mirrors render_chat_prompt_text in ds4_server.c), so you can
turn ANY prompt list — a txt file, a jsonl of chats, eval traffic — into a corpus
without depending on the ds4 python tooling.

Inputs accepted by build_corpus():
  - .txt   one prompt per paragraph (blank-line separated)
  - .jsonl one JSON object per line:
             {"prompt": "..."}                      plain prompt
             {"messages": [...]}                    full chat
             {"question": ..., "options"/"choices": ..., "answer"/"gold": ...}
                                                    benchmark record (MCQ or QA) —
                                                    auto-detected, so you can point
                                                    this straight at an eval set
           optional "system" (txt prompts get the default system prompt)

Benchmark records can include the gold answer as an assistant turn (answers=True):
the imatrix then also sees the activation paths of ANSWERING, not just reading.
A general-purpose corpus can be interleaved (mix=...) so a domain imatrix doesn't
over-specialize — same practice as ds4's medical corpus (~40% domain by chars).

Each prompt is rendered in nothink and/or think mode (ds4's own corpus is 50/50).
"""
import hashlib, json, os

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
MARKER = "===== DS4_IMATRIX_PROMPT"

DEFAULT_SYSTEM = (
    "You are DeepSeek V4 Flash running locally. Answer accurately, preserve "
    "technical details, and use tools only when the prompt asks for tool use."
)


def escape_tool_result(text):
    return text.replace("</tool_result>", "&lt;/tool_result>")


def render(messages, mode="nothink"):
    """Mirror render_chat_prompt_text in ds4_server.c (no tools-schema variant).

    messages: [{"role": "system|user|assistant|tool", "content": "...",
                optional "reasoning" for think-mode assistant turns}]
    """
    think = mode == "think"
    # tool context also when a prior assistant turn carried a DSML tool-call block (matches
    # ds4's chat_history_uses_tool_context), so think-mode wraps those turns correctly.
    tool_context = any(m.get("role") in ("tool", "function")
                       or (m.get("role") == "assistant" and m.get("dsml")) for m in messages)
    system = "\n\n".join(m.get("content", "") for m in messages
                         if m.get("role") == "system" and m.get("content"))
    last_user_idx = max((i for i, m in enumerate(messages)
                         if m.get("role") in ("user", "tool", "function")), default=-1)
    out = [BOS, system]
    pending_assistant = False
    pending_tool_result = False
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "system":
            continue
        if role == "user":
            out.extend([USER, content])
            pending_assistant = True
            pending_tool_result = False
        elif role in ("tool", "function"):
            if not pending_tool_result:
                out.append(USER)
            out.extend(["<tool_result>", escape_tool_result(content), "</tool_result>"])
            pending_assistant = True
            pending_tool_result = True
        elif role == "assistant":
            if pending_assistant:
                out.append(ASSISTANT)
                if think and (tool_context or i > last_user_idx):
                    out.extend(["<think>", msg.get("reasoning", "") or "", "</think>"])
                else:
                    out.append("</think>")
            out.append(content)
            if msg.get("dsml"):           # preserve assistant DSML tool-call blocks
                out.append(msg["dsml"])
            out.append(EOS)
            pending_assistant = False
            pending_tool_result = False
    if pending_assistant:
        out.extend([ASSISTANT, "<think>" if think else "</think>"])
    return "".join(out)


def _records_from_txt(path, system):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    for block in raw.split("\n\n"):
        block = block.strip()
        if block:
            yield [{"role": "system", "content": system}, {"role": "user", "content": block}]


LETTERS = "ABCDEFGHIJ"
_Q_KEYS = ("question", "query")
_OPT_KEYS = ("options", "choices")
_GOLD_KEYS = ("answer", "gold", "correct", "target", "answer_idx")


def _bench_messages(o, system, answers):
    """Benchmark record -> chat messages. MCQ if options present, plain QA otherwise."""
    q = next((str(o[k]) for k in _Q_KEYS if o.get(k) is not None), None)
    opts = next((o[k] for k in _OPT_KEYS if o.get(k)), None)
    gold = next((o[k] for k in _GOLD_KEYS if o.get(k) is not None), None)
    if isinstance(opts, dict):
        keyed = sorted(opts.items())
    elif isinstance(opts, list):
        keyed = list(zip(LETTERS, (str(x) for x in opts)))
    else:
        keyed = []
    user = q
    if keyed:
        user += "\n\n" + "\n".join(f"{k}) {v}" for k, v in keyed) + \
                "\n\nAnswer with the letter of the correct option."
    msgs = [{"role": "system", "content": o.get("system", system)},
            {"role": "user", "content": user}]
    if answers and gold is not None:
        if keyed and isinstance(gold, int) and 0 <= gold < len(keyed):
            a = f"{keyed[gold][0]}) {keyed[gold][1]}"
        elif keyed and str(gold).strip().upper() in dict(keyed):
            g = str(gold).strip().upper()
            a = f"{g}) {dict(keyed)[g]}"
        else:
            a = str(gold)
        msgs.append({"role": "assistant", "content": a})
    return msgs


def _records_from_jsonl(path, system, answers=False):
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: invalid JSON ({e})")
            if "messages" in o:
                msgs = o["messages"]
                if not any(m.get("role") == "system" for m in msgs):
                    msgs = [{"role": "system", "content": o.get("system", system)}] + msgs
                yield msgs
            elif any(o.get(k) is not None for k in _Q_KEYS):
                yield _bench_messages(o, system, answers)
            elif "prompt" in o:
                yield [{"role": "system", "content": o.get("system", system)},
                       {"role": "user", "content": str(o["prompt"])}]
            else:
                raise ValueError(f"{path}:{ln}: object needs 'prompt', 'question' or 'messages'")


def _mix_blocks(mix_path):
    """Yield whole marker-delimited blocks from an already-rendered corpus."""
    block = []
    with open(mix_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(MARKER) and block:
                yield "".join(block)
                block = []
            block.append(line)
    if block:
        yield "".join(block)


def _render_records(records, out, modes, category, src_name, limit=None):
    """Write rendered records into an open file handle; return (n_written, chars)."""
    n = chars = 0
    for i, msgs in enumerate(records):
        if limit and i >= limit:
            break
        for mode in modes:
            rendered = render(msgs, mode)
            rid = hashlib.sha1(f"{category}\0{mode}\0{rendered[:4096]}".encode(
                "utf-8", "ignore")).hexdigest()[:12]
            out.write(f"{MARKER} {rid} {category} {mode} {src_name} =====\n")
            out.write(rendered)
            out.write("\n\n")
            n += 1
            chars += len(rendered)
    return n, chars


def build_corpus_multi(in_paths, out_path, modes=("nothink", "think"), system=DEFAULT_SYSTEM,
                       answers=False, cap=None, mix=None, mix_ratio=0.4, categories=None):
    """Render several .txt/.jsonl inputs into ONE corpus (each tagged by its basename).

    Used by the benchy bridge to fuse e.g. humaneval+mbpp+mmlu_cs into a 'code' corpus.
    Returns (n_prompts, bytes, per_file: {name: n}).
    """
    per_file = {}
    domain_chars = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in in_paths:
            name = os.path.basename(p)
            cat = (categories or {}).get(name, os.path.splitext(name)[0])
            recs = (_records_from_jsonl(p, system, answers) if p.endswith(".jsonl")
                    else _records_from_txt(p, system))
            n, chars = _render_records(recs, out, modes, cat, name, limit=cap)
            per_file[name] = n
            domain_chars += chars
        if mix and domain_chars:
            budget = int(domain_chars * (1.0 - mix_ratio) / max(mix_ratio, 0.05))
            got = mixn = 0
            for block in _mix_blocks(mix):
                if got >= budget:
                    break
                out.write(block)
                if not block.endswith("\n\n"):
                    out.write("\n\n")
                got += len(block); mixn += 1
            if mixn:
                per_file["+mix"] = mixn
    return sum(per_file.values()), os.path.getsize(out_path), per_file


def build_corpus(in_path, out_path, modes=("nothink", "think"), system=DEFAULT_SYSTEM,
                 category="custom", answers=False, limit=None, mix=None, mix_ratio=0.4):
    """Render in_path (.txt or .jsonl) into a marker-delimited corpus at out_path.

    answers:   include gold answers of benchmark records as assistant turns
    limit:     cap the number of source records (deterministic first-N)
    mix:       path to an already-rendered general corpus to interleave, so the
               domain ends up ~mix_ratio of the total by characters (medeval's
               medical corpus uses ~0.4 — calibrate the domain, keep the base)
    Returns (n_prompts_written, bytes_written).
    """
    if in_path.endswith(".jsonl"):
        records = _records_from_jsonl(in_path, system, answers)
    else:
        records = _records_from_txt(in_path, system)
    n = domain_chars = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, msgs in enumerate(records):
            if limit and i >= limit:
                break
            for mode in modes:
                rendered = render(msgs, mode)
                rid = hashlib.sha1(f"{category}\0{mode}\0{rendered[:4096]}".encode(
                    "utf-8", "ignore")).hexdigest()[:12]
                out.write(f"{MARKER} {rid} {category} {mode} {os.path.basename(in_path)} =====\n")
                out.write(rendered)
                out.write("\n\n")
                n += 1
                domain_chars += len(rendered)
        if mix and domain_chars:
            budget = int(domain_chars * (1.0 - mix_ratio) / max(mix_ratio, 0.05))
            got = 0
            for block in _mix_blocks(mix):
                if got >= budget:
                    break
                out.write(block)
                if not block.endswith("\n\n"):
                    out.write("\n\n")
                got += len(block)
                n += 1
    return n, os.path.getsize(out_path)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="forge_corpus", description=__doc__.splitlines()[0])
    ap.add_argument("input", help="prompts file: .txt (blank-line separated) or .jsonl (chat/bench)")
    ap.add_argument("-o", "--out", required=True, help="rendered corpus output path")
    ap.add_argument("--mode", choices=["nothink", "think", "both"], default="both")
    ap.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt for raw prompts")
    ap.add_argument("--category", default="custom", help="category tag in markers")
    ap.add_argument("--answers", action="store_true",
                    help="include benchmark gold answers as assistant turns")
    ap.add_argument("--limit", type=int, help="cap source records (first N)")
    ap.add_argument("--mix", help="rendered general corpus to interleave (anti-overfit)")
    ap.add_argument("--mix-ratio", type=float, default=0.4,
                    help="target domain share by chars when mixing (default 0.4)")
    a = ap.parse_args(argv)
    modes = ("nothink", "think") if a.mode == "both" else (a.mode,)
    n, size = build_corpus(a.input, a.out, modes, a.system, a.category,
                           answers=a.answers, limit=a.limit, mix=a.mix, mix_ratio=a.mix_ratio)
    print(f"forge_corpus: {n} rendered prompts -> {a.out} ({size >> 10} KB, "
          f"~{size // 4} tokens est.)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
