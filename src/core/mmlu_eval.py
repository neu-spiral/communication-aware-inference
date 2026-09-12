from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _build_mmlu_prompt(
    question: str,
    choices: Sequence[str],
    shot_examples: List[Tuple[str, Sequence[str], int]] = [],
) -> str:
    """
    Build a prompt for a single MMLU question.

    Parameters
    ----------
    question     : The question text.
    choices      : List of 4 answer strings.
    shot_examples: List of (question, choices, answer_idx) tuples to prepend
                   as solved examples.  Empty list = zero-shot (default).
    """
    def _fmt_question(q: str, ch: Sequence[str], answer_idx: int = -1) -> str:
        choices_txt = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(ch)])
        line = f"Question: {q}\n{choices_txt}\nAnswer:"
        if answer_idx >= 0:
            line += f" {chr(65 + answer_idx)}"
        return line

    parts = []
    for q, ch, ans in shot_examples:
        parts.append(_fmt_question(q, ch, ans))
    parts.append(_fmt_question(question, choices))
    return "\n\n".join(parts)


def _load_mmlu_split(subject: str, split: str) -> List[Dict[str, Any]]:
    """Load one (subject, split) of cais/mmlu as a list of row dicts.

    Some datasets/fsspec combinations ship a `load_dataset` glob resolver that
    chokes on the hub's `**` patterns ("Invalid pattern: '**' can only be an
    entire path component"). To stay robust we pull the parquet shard directly
    (the same approach the concavity harness uses for ShareGPT) and only fall
    back to `load_dataset` if that fails. Parquet row order matches
    `load_dataset`, so downstream seeded sampling is unchanged.
    """
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="cais/mmlu",
            filename=f"{subject}/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        return pd.read_parquet(path).to_dict("records")
    except Exception as exc:  # noqa: BLE001 - fall back to the datasets loader
        print(f"[MMLUEvaluator] parquet load failed for {subject}/{split} "
              f"({exc!r}); falling back to load_dataset.", flush=True)
        from datasets import load_dataset  # type: ignore
        return list(load_dataset("cais/mmlu", subject, split=split))


@dataclass(frozen=True)
class MMLUEvalConfig:
    subjects: Sequence[str]
    # For fast (used inside gradient oracle) and full (used for plotting).
    samples_per_subject: int
    seed: int = 0
    # Max token length for truncation (kept as safety).
    max_length: Optional[int] = None
    batch_size: int = 8
    # Number of in-context examples (0 = zero-shot, 5 = standard 5-shot).
    # Examples are drawn from the MMLU dev split (5 examples per subject).
    n_shot: int = 0


class MMLUEvaluator:
    """
    Accuracy evaluator for multiple-choice MMLU in HF causal LM format.

    Supports zero-shot (n_shot=0) and k-shot (n_shot=1..5) evaluation.
    For k-shot, k solved examples from the MMLU dev split are prepended
    to each test question.  The dev split has exactly 5 examples per
    subject, so n_shot must be in [0, 5].

    It prebuilds a fixed subset of (prompt, answer_index) pairs so repeated
    evaluations are deterministic for the same eta.
    """

    def __init__(self, *, tokenizer, config: MMLUEvalConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config

        if not (0 <= config.n_shot <= 5):
            raise ValueError(
                f"n_shot must be in [0, 5] (MMLU dev split has 5 examples "
                f"per subject). Got n_shot={config.n_shot}."
            )

        rng = np.random.default_rng(config.seed)

        # ── Load per-subject dev examples for few-shot ────────────────────────
        # MMLU dev split has exactly 5 examples per subject.
        dev_examples: Dict[str, List[Tuple[str, Sequence[str], int]]] = {}
        if config.n_shot > 0:
            for subj in config.subjects:
                ds_dev = _load_mmlu_split(subj, "dev")
                dev_examples[subj] = [
                    (ds_dev[i]["question"],
                     ds_dev[i]["choices"],
                     int(ds_dev[i]["answer"]))
                    for i in range(min(config.n_shot, len(ds_dev)))
                ]
            print(f"[MMLUEvaluator] {config.n_shot}-shot: loaded "
                  f"{config.n_shot} dev examples per subject.", flush=True)

        # ── Build test prompts ────────────────────────────────────────────────
        prompts: List[str] = []
        answers: List[int] = []
        for subj in config.subjects:
            ds = _load_mmlu_split(subj, "test")
            n = min(len(ds), int(config.samples_per_subject))
            idxs = rng.choice(len(ds), size=n, replace=False).tolist()
            shot_ex = dev_examples.get(subj, [])
            for i in idxs:
                item = ds[int(i)]
                q = item["question"]
                choices = item["choices"]
                answer_letter_idx = int(item["answer"])  # A=0, ..., D=3
                prompts.append(_build_mmlu_prompt(q, choices, shot_ex))
                answers.append(answer_letter_idx)

        if not prompts:
            raise ValueError(
                "MMLUEvaluator got an empty prompt set. "
                "Check subjects/samples_per_subject."
            )

        self.prompts = prompts
        self.answers = torch.tensor(answers, dtype=torch.long)

        # Candidate tokens for A/B/C/D.
        candidates = ["A", "B", "C", "D"]
        cand_ids: List[int] = []
        for c in candidates:
            # Many tokenizers treat answer candidates with a leading space.
            ids = tokenizer.encode(" " + c, add_special_tokens=False)
            if not ids:
                ids = tokenizer.encode(c, add_special_tokens=False)
            cand_ids.append(int(ids[-1]))
        self.cand_ids = torch.tensor(cand_ids, dtype=torch.long)

        # Tokenize once (dynamic padding), keep on CPU; move per batch.
        # 5-shot prompts are ~1500-2000 tokens — increase max_length guard.
        effective_max = config.max_length
        if config.n_shot > 0 and effective_max is not None:
            effective_max = max(effective_max, 2048)

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=effective_max,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

        print(f"[MMLUEvaluator] {len(prompts)} prompts, "
              f"max_len={self.input_ids.size(1)} tokens, "
              f"n_shot={config.n_shot}", flush=True)

    @torch.no_grad()
    def accuracy(self, model, *, device: torch.device) -> float:
        """
        Runs the model on all pre-tokenized prompts and computes accuracy.
        """
        model.eval()
        self.cand_ids = self.cand_ids.to(device)
        answers = self.answers.to(device)

        n = int(self.input_ids.size(0))
        correct = 0

        bs = int(self.config.batch_size)
        for start in range(0, n, bs):
            end = min(n, start + bs)
            input_ids = self.input_ids[start:end].to(device)
            attention_mask = self.attention_mask[start:end].to(device)

            last_positions = attention_mask.sum(dim=1) - 1  # (B,)
            batch_idx = torch.arange(input_ids.size(0), device=device)

            # Only the last valid position is ever scored, so run the backbone
            # and apply the LM head to those B hidden states instead of
            # materialising the full (B, S, V) logit tensor -- at 5-shot MMLU
            # lengths that tensor alone is >13 GiB in fp32 and OOMs an A100-40GB.
            # Compression hooks live on the decoder layers, so they still fire.
            backbone = getattr(model, "model", None)
            lm_head = getattr(model, "lm_head", None)

            if backbone is not None and lm_head is not None:
                hidden = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state  # (B, S, H)
                last_hidden = hidden[batch_idx, last_positions]  # (B, H)
                last_logits = lm_head(last_hidden)  # (B, V)
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = outputs.logits  # (B, S, V)
                last_logits = logits[batch_idx, last_positions]  # (B, V)

            cand_logits = last_logits.index_select(dim=1, index=self.cand_ids)  # (B, 4)
            preds = cand_logits.argmax(dim=1)  # (B,)

            correct += int((preds == answers[start:end]).sum().item())

        return float(correct) / float(n)


