"""Helper functions for Part B (fixing Qwen3.5-0.8B's thinking loops).

They take care of the plumbing so that the notebook can focus on the analysis:
loading the authors' traces, labelling looping tokens, editing weights, and the two kinds of evaluation.
"""
import gzip, json, os, re, urllib.request
import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

MODEL_ID = "Qwen/Qwen3.5-0.8B"
PREFIX = "<|im_start|>user\n"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n"
# The authors' released generations of the unedited model (CC BY 4.0):
# Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming", https://github.com/RikiyaT/lm_understand-logs
LOGS = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math"
ASTRA_EDIT = "https://huggingface.co/JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit/resolve/main/astra_edit/edit.npz"


def prompt(problem):
    return PREFIX + problem + SUFFIX


def download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


# ------------------------------------------------------------------ data
def load_traces(max_tokens=8192, tokenizer=None):
    """The unedited model's thinking-on answers to 200 GSM8K and 200 MATH-500 problems (up to 32k tokens each).
    Returns a list of dicts with the prompt, the problem, the reference answer and the answer's token ids
    (truncated to max_tokens)."""
    from datasets import load_dataset
    gsm = {f"gsm8k_test_{i}": r["question"] for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))}
    m500 = {r["unique_id"]: r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    traces = []
    for name, n_shards, problems in [("gsm8k", 10, gsm), ("math500", 10, m500)]:
        for s in range(n_shards):
            fn = download(f"{LOGS}/eval/unedited/parts/{name}_s{s}of{n_shards}/{name}_first_samples.jsonl.gz",
                          f"traces/{name}_{s}.jsonl.gz")
            for line in gzip.open(fn, "rt", encoding="utf8"):
                r = json.loads(line)
                ids = tokenizer(r["text"], add_special_tokens=False).input_ids[:max_tokens]
                traces.append(dict(id=r["id"], source=name, problem=problems[r["id"]], answer=r["answer"], ids=ids))
    return traces


def eval_problems(n=64, start=200):
    """GSM8K test problems that are NOT among the traces (the traces cover test problems 0..199)."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(start, start + n))
    return [dict(problem=r["question"], answer=r["answer"].split("####")[-1].strip().replace(",", "")) for r in ds]


# ------------------------------------------------------------------ labelling loops
def repeated(ids, n=16):
    """rep[i] = True if the n tokens ending at position i have appeared before (the n-gram rule of nathan.rs)"""
    seen, rep = set(), np.zeros(len(ids), bool)
    for i in range(n - 1, len(ids)):
        g = tuple(ids[i - n + 1:i + 1])
        rep[i] = g in seen
        seen.add(g)
    return rep


def loop_onset(ids, n=16, min_run=64):
    """position where the first loop starts (a run of >= min_run repeated n-grams), or None"""
    run = 0
    for i, r in enumerate(repeated(ids, n)):
        run = run + 1 if r else 0
        if run >= min_run:
            return i - min_run + 1
    return None


def label_tokens(ids, n=16, min_run=64, gap=256):
    """per token: 1 = looping (at least min_run tokens into a loop), 0 = productive (at least `gap` tokens before
    the first loop, or anywhere in an answer without a loop), -1 = neither (ambiguous, ignore)"""
    rep = repeated(ids, n)
    lab = -np.ones(len(ids), int)
    run, onset = 0, None
    for i, r in enumerate(rep):
        run = run + 1 if r else 0
        if run >= min_run:
            lab[i] = 1
            if onset is None:
                onset = i - min_run + 1
    end = len(ids) if onset is None else max(onset - gap, 0)
    lab[:end][lab[:end] == -1] = 0
    return lab


# ------------------------------------------------------------------ editing weights
def down_proj(model, layer):
    return model.model.layers[layer].mlp.down_proj.weight


def rank_one_edit(model, edits, alpha):
    """edits = {layer: (u, cols)}. For every selected column w_j of down_proj in that layer:
           w_j  <-  w_j - alpha * (u . w_j) * u
    alpha = 1 removes the u-component of the column, alpha > 1 flips it.
    Returns a function that restores the original weights."""
    saved = {}
    for layer, (u, cols) in edits.items():
        W = down_proj(model, layer)
        saved[layer] = W.data.clone()
        u = u.to(W.device).float(); u = u / u.norm()
        cols = torch.as_tensor(cols, device=W.device)
        Wc = W.data[:, cols].float()
        W.data[:, cols] = (Wc - alpha * torch.outer(u, u @ Wc)).to(W.dtype)

    def restore():
        for layer, w in saved.items():
            down_proj(model, layer).data.copy_(w)
    return restore


def load_astra_edit(model):
    """Astra's edited values, as a dict {layer: new full down_proj weight (on the model's device)}"""
    z = np.load(download(ASTRA_EDIT, "astra/edit.npz"))
    out = {}
    for key, layer in [("t0", 15), ("t1", 19)]:
        W = down_proj(model, layer)
        new = W.detach().clone().view(-1)
        idx = torch.from_numpy(z[key + "_idx"].astype(np.int64)).to(W.device)
        val = torch.from_numpy(z[key + "_val"].astype(np.int16)).view(torch.bfloat16).to(W.device)
        new[idx] = val.to(W.dtype)
        out[layer] = new.view(W.shape)
    return out


# ------------------------------------------------------------------ evaluation 1: teacher forcing (cheap)
@torch.no_grad()
def token_logprobs(model, tokenizer, trace):
    """log p(actual token) for every token of a trace's answer, under the current weights"""
    p = tokenizer(prompt(trace["problem"]), add_special_tokens=False).input_ids
    ids = torch.tensor([p + trace["ids"]], device=model.device)
    h = model.model(ids).last_hidden_state[0, len(p) - 1:-1]
    target = ids[0, len(p):]
    out = []
    for a in range(0, len(h), 512):                       # the vocabulary is huge: project in chunks
        lp = model.lm_head(h[a:a + 512]).float().log_softmax(-1)
        out.append(lp.gather(1, target[a:a + 512, None])[:, 0])
    return torch.cat(out).cpu().numpy()


def teacher_forced(model, tokenizer, traces):
    """mean log p of the actual tokens on looping tokens, on the 64 tokens where each loop starts, and on productive tokens"""
    res = {"looping": [], "loop start": [], "productive": []}
    for t in traces:
        lab, lp = label_tokens(t["ids"]), token_logprobs(model, tokenizer, t)
        res["looping"] += lp[lab == 1].tolist()
        res["productive"] += lp[lab == 0].tolist()
        o = loop_onset(t["ids"])
        if o is not None:
            res["loop start"] += lp[o:o + 64].tolist()
    return {k: float(np.mean(v)) for k, v in res.items()}


# ------------------------------------------------------------------ evaluation 2: generation (expensive)
class PresencePenalty(LogitsProcessor):
    """subtract `penalty` from the logit of every token already generated (as vLLM's presence_penalty)"""
    def __init__(self, penalty, prompt_len):
        self.penalty, self.prompt_len = penalty, prompt_len

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len:]
        if gen.shape[1]:
            scores = scores - self.penalty * torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen, True)
        return scores


@torch.no_grad()
def generate_eval(model, tokenizer, problems, max_new_tokens=4096, batch_size=32, seed=1, thinking=True):
    """Generate with the paper's sampling settings and report loop rate, the share that finished thinking,
    and accuracy. Returns (summary dict, list of per-problem results)."""
    torch.manual_seed(seed)
    tokenizer.padding_side = "left"
    rows = []
    for s in range(0, len(problems), batch_size):
        batch = problems[s:s + batch_size]
        texts = [prompt(b["problem"]) if thinking else
                 prompt(b["problem"]).removesuffix("<think>\n") + "<think>\n\n</think>\n\n" for b in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        L0 = enc.input_ids.shape[1]
        sampling = dict(temperature=0.6, top_p=0.95, top_k=20) if thinking else dict(temperature=0.7, top_p=0.8, top_k=20)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, **sampling,
                             logits_processor=LogitsProcessorList([PresencePenalty(1.5, L0)]))
        for b, o in zip(batch, out):
            g = o[L0:].tolist()
            if tokenizer.eos_token_id in g:
                g = g[:g.index(tokenizer.eos_token_id) + 1]
            text = tokenizer.decode(g)
            final = text.split("</think>")[-1] if (not thinking or "</think>" in text) else ""
            boxed = re.findall(r"\\boxed\{([^{}]*)\}", final)
            pred = boxed[-1].replace(",", "").replace("$", "").strip() if boxed else None
            rows.append(dict(tokens=len(g), looping=loop_onset(g) is not None,
                             finished=(not thinking) or "</think>" in text, correct=pred == b["answer"], text=text))
        del out
        torch.cuda.empty_cache()
    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("looping", "finished", "correct")}
    return summary, rows
