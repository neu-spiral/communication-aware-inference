"""
Text-corpus evaluation for the LLM tasks: dataset loading, per-sample NLL, and
the bounded perplexity score the optimizer maximises.

Single source of truth for these helpers, shared by the optimizer task
instances in :mod:`src.core.task_instances` and by the concavity runners in
``experiments/concavity/``.

The utility reported to the optimizer is a *bounded* score rather than raw
perplexity:

    score = min{1, PPL_reference / PPL_compressed}

so it lives in (0, 1] like an accuracy, is 1 when compression costs nothing,
and cannot be inflated by a compression setting that happens to beat the
uncompressed baseline on a particular sample.

Dataset loading bypasses ``datasets.load_dataset`` because the pinned
``datasets``/``fsspec`` pair in the measurement environment cannot resolve the
Hub parquet globs (and the legacy WikiText builder points at a dead S3 URL), so
the parquet files are fetched directly through ``huggingface_hub`` and read with
pandas. ``load_dataset`` remains as a fallback for environments where it works.
"""

from __future__ import annotations

import json
from typing import List, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# WikiText-2
# ---------------------------------------------------------------------------

# ──────────────────────────────────────────────────────────────────────────

def _iter_wikitext_rows():
    """Yield WikiText-2-raw test rows as text strings.

    The legacy ``datasets`` builder (datasets==1.18.x) hard-codes a dead S3 URL
    (``research.metamind.io/.../wikitext-2-raw-v1.zip``, now HTTP 301) and the
    pinned ``fsspec`` rejects the Hub parquet glob, so ``load_dataset`` fails in
    this env. We therefore download the parquet file directly from the Hub and
    read it with pandas, falling back to ``load_dataset`` only if that fails.
    """
    # Primary: direct parquet download (version-robust, no glob/S3 dependency).
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd

        path = hf_hub_download(
            repo_id="Salesforce/wikitext",
            filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        for t in df["text"].tolist():
            yield t
        return
    except Exception as exc:  # noqa: BLE001
        print(f"[wikitext] direct parquet load failed ({exc!r}); "
              f"falling back to datasets.load_dataset", flush=True)

    # Fallback: classic datasets loader (works where the env/cache cooperate).
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    for row in ds:
        yield row["text"]


def load_wikitext_texts(max_texts: int, min_chars: int = 200) -> List[str]:
    """Load WikiText-2 test split and return up to `max_texts` non-trivial paragraphs."""
    texts: List[str] = []
    for raw in _iter_wikitext_rows():
        t = raw.strip()
        if len(t) >= min_chars:
            texts.append(t)
            if len(texts) >= max_texts:
                break
    if not texts:
        raise RuntimeError("No suitable WikiText-2 paragraphs found.")
    return texts


# ---------------------------------------------------------------------------
# ShareGPT
# ---------------------------------------------------------------------------

_SHAREGPT_PARQUET_CANDIDATES = [
    ("Aeala/ShareGPT_Vicuna_unfiltered",
     "ShareGPT_V4.3_unfiltered_cleaned_split.parquet"),
    ("theblackcat102/sharegpt-english", "data/train-00000-of-00001.parquet"),
    ("RyokoAI/ShareGPT52K", "sg_90k_part1.parquet"),
]
_SHAREGPT_JSON_CANDIDATES = [
    ("Aeala/ShareGPT_Vicuna_unfiltered",
     "ShareGPT_V4.3_unfiltered_cleaned_split.json"),
    ("anon8231489123/ShareGPT_Vicuna_unfiltered",
     "ShareGPT_V3_unfiltered_cleaned_split.json"),
]


def _flatten_sharegpt_conversation(conv) -> str:
    """Flatten one ShareGPT conversation (list of turns) to a single string.

    Turns come either as {"from": ..., "value": ...} (Vicuna format) or
    {"role": ..., "content": ...} (chat format). Non-list / unknown shapes
    return "" and are filtered out by the caller's min_chars check.
    """
    if isinstance(conv, str):
        return conv
    if not isinstance(conv, (list, tuple)):
        return ""
    parts: List[str] = []
    for turn in conv:
        if isinstance(turn, dict):
            val = turn.get("value", turn.get("content", ""))
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        elif isinstance(turn, str) and turn.strip():
            parts.append(turn.strip())
    return "\n".join(parts)


def _iter_sharegpt_conversations():
    """Yield raw ShareGPT conversation objects from the first source that works."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    # 1) parquet candidates
    for repo_id, filename in _SHAREGPT_PARQUET_CANDIDATES:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename,
                                   repo_type="dataset")
            df = pd.read_parquet(path)
            col = "conversations" if "conversations" in df.columns else df.columns[0]
            print(f"[sharegpt] using parquet {repo_id}/{filename} (col={col})",
                  flush=True)
            for conv in df[col].tolist():
                yield conv
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[sharegpt] parquet {repo_id}/{filename} failed ({exc!r})",
                  flush=True)

    # 2) JSON candidates
    for repo_id, filename in _SHAREGPT_JSON_CANDIDATES:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename,
                                   repo_type="dataset")
            with open(path) as f:
                data = json.load(f)
            print(f"[sharegpt] using json {repo_id}/{filename} "
                  f"({len(data)} records)", flush=True)
            for rec in data:
                yield rec.get("conversations", rec) if isinstance(rec, dict) else rec
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[sharegpt] json {repo_id}/{filename} failed ({exc!r})",
                  flush=True)

    # 3) datasets fallback
    from datasets import load_dataset
    ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    for row in ds:
        yield row.get("conversations", row)


def load_sharegpt_texts(max_texts: int, min_chars: int = 200) -> List[str]:
    """Return up to ``max_texts`` non-trivial ShareGPT conversation strings."""
    texts: List[str] = []
    for conv in _iter_sharegpt_conversations():
        t = _flatten_sharegpt_conversation(conv).strip()
        if len(t) >= min_chars:
            texts.append(t)
            if len(texts) >= max_texts:
                break
    if not texts:
        raise RuntimeError("No suitable ShareGPT conversations found.")
    return texts


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def per_sample_mean_nll(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
    batch_size: int = 1,
) -> np.ndarray:
    """Mean per-token negative log-likelihood (nats/token) for each input text.

    Returns an array of shape (len(texts),).  Padding positions are masked out,
    so each entry is the average over that sample's own evaluation span.
    """
    model.eval()
    out: List[float] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits  # (B, S, V)

        # Shift for next-token loss.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().to(torch.float32)

        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        tgt_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        tgt_logp = tgt_logp * shift_mask  # zero out padded positions

        nll_per_row = -tgt_logp.sum(dim=1)             # (B,)
        tok_per_row = shift_mask.sum(dim=1).clamp(min=1.0)  # (B,)
        mean_nll = (nll_per_row / tok_per_row).detach().cpu().numpy()
        out.extend(float(v) for v in mean_nll)

    return np.asarray(out, dtype=np.float64)


def sample_scores(
    hbar_ref: np.ndarray,
    hbar_comp: np.ndarray,
) -> np.ndarray:
    """Per-sample score = min{1, exp(Hbar_ref - Hbar_comp)} = min{1, PPL_ref/PPL_comp}."""
    return np.minimum(1.0, np.exp(hbar_ref - hbar_comp))


# ---------------------------------------------------------------------------
# Cut points
# ---------------------------------------------------------------------------

def make_cut_points(n_layers: int, n_cuts: int) -> List[int]:
    """Place `n_cuts` evenly spaced interior decoder-layer indices."""
    if n_cuts < 1:
        raise ValueError("n_cuts must be >= 1")
    raw = [int(round((i + 1) * n_layers / (n_cuts + 1))) for i in range(n_cuts)]
    cuts = sorted({int(np.clip(c, 0, n_layers - 1)) for c in raw})
    # If clipping collapsed duplicates, fill back up to n_cuts distinct indices.
    i = 1
    while len(cuts) < n_cuts and i < n_layers:
        if i not in cuts:
            cuts.append(i)
            cuts = sorted(set(cuts))
        i += 1
    return cuts[:n_cuts]
