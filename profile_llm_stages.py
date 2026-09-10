"""
Measure the per-stage compute time and per-link payload of a pipelined LLM.

The offline optimizer needs two topology inputs per task: ``tau[i]``, the
compute time of stage ``i``, and ``a[i]``, the bytes crossing link ``i``. This
script measures them on real hardware instead of estimating, and appends the
result to a JSON profile the task instances read at setup time.

Stages are contiguous groups of decoder layers. An ``L_k``-stage pipeline has
``L_k - 1`` links, and the tensor on every link is the hidden state, so

    a[i] = batch * seq_len * hidden_dim * bytes_per_element

which is exact given the shape, not a fit. ``tau[i]`` is the summed wall time of
the layers in stage ``i``, measured with CUDA events after warm-up, reported as
the median over ``--repeats`` passes.

Usage
-----
    python profile_llm_stages.py --model google/gemma-2b --n_stages 5
    python profile_llm_stages.py --model meta-llama/Llama-3.1-8B --n_stages 5 \
        --batch_size 1 --seq_len 512 --repeats 20

Writes/updates ``assets/llm_stage_profiles.json``, keyed by
``model / n_stages / batch_size / seq_len``. Re-running a key overwrites it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.llm_compression import detect_llm_decoder_layers  # noqa: E402

DEFAULT_PROFILE = REPO_ROOT / "assets" / "llm_stage_profiles.json"


def stage_layer_counts(n_layers: int, n_stages: int) -> List[int]:
    """Split ``n_layers`` into ``n_stages`` contiguous groups, largest last.

    Matches the "thirds"-style even split the experiments use: every stage gets
    ``n_layers // n_stages`` layers and the remainder is spread over the last
    stages, so an 18-layer model over 3 stages is 6/6/6 and a 32-layer model
    over 3 stages is 10/11/11.
    """
    if n_stages < 1:
        raise ValueError("need at least one group to split into")
    if n_stages > n_layers:
        raise ValueError(f"n_stages={n_stages} exceeds n_layers={n_layers}")
    base, rem = divmod(n_layers, n_stages)
    counts = [base] * n_stages
    for i in range(rem):
        counts[n_stages - 1 - i] += 1
    return counts


def _bytes_per_element(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


@torch.no_grad()
def measure(model, layers, counts, *, batch_size, seq_len, repeats,
            device) -> List[float]:
    """Median per-stage seconds, timed inside a real forward pass.

    Layers are timed with forward hooks rather than called in isolation: a
    decoder block's signature varies across architectures and transformers
    versions (rotary position embeddings, mask formats), and a real pass also
    gets the true masks and cache settings. Timing is per layer so the numbers
    can be regrouped for any stage count without re-measuring.
    """
    starts: Dict[int, object] = {}
    per_layer_ms: List[List[float]] = [[] for _ in layers]
    handles = []
    cuda = device.type == "cuda"

    def make_pre(i):
        def pre(_m, _inp):
            if cuda:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                starts[i] = e
            else:
                import time
                starts[i] = time.perf_counter()
        return pre

    def make_post(i):
        def post(_m, _inp, out):
            if cuda:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                per_layer_ms[i].append((starts.pop(i), e))
            else:
                import time
                per_layer_ms[i].append((time.perf_counter() - starts.pop(i)) * 1e3)
            return out
        return post

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_pre(i)))
        handles.append(layer.register_forward_hook(make_post(i)))

    # A real token batch, so attention masks and rotary caches are realistic.
    vocab = int(getattr(model.config, "vocab_size", 32000))
    ids = torch.randint(0, max(2, vocab - 1), (batch_size, seq_len), device=device)
    mask = torch.ones_like(ids)

    try:
        for _ in range(3):  # warm-up
            model(input_ids=ids, attention_mask=mask, use_cache=False)
        if cuda:
            torch.cuda.synchronize()
        for lst in per_layer_ms:
            lst.clear()
        for _ in range(repeats):
            model(input_ids=ids, attention_mask=mask, use_cache=False)
        if cuda:
            torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()

    # CUDA events only carry a duration once the stream has synchronized, which
    # is why the (start, end) pairs are resolved here and not inside the hook.
    layer_ms = []
    for lst in per_layer_ms:
        vals = [start.elapsed_time(end) for start, end in lst] if cuda else list(lst)
        if not vals:
            raise RuntimeError("no timings captured; did the forward pass run?")
        layer_ms.append(statistics.median(vals))

    tau_s, cursor = [], 0
    for c in counts:
        tau_s.append(sum(layer_ms[cursor:cursor + c]) / 1e3)
        cursor += c
    return tau_s


@torch.no_grad()
def measure_seq2seq(model, counts, *, batch_size, seq_len, repeats, device
                    ) -> List[float]:
    """Median per-stage seconds for an encoder-decoder pipeline.

    Stage layout follows the SST-2 task instance: ``len(counts)`` groups of
    encoder blocks, then the whole decoder as the final stage. So ``counts``
    describes the ENCODER split only and the returned list is one longer.
    """
    blocks = model.encoder.block
    starts: Dict[object, object] = {}
    timings: Dict[object, list] = {}
    handles = []
    cuda = device.type == "cuda"

    def hooks_for(key, module):
        def pre(_m, _inp):
            if cuda:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                starts[key] = e
            else:
                import time
                starts[key] = time.perf_counter()

        def post(_m, _inp, out):
            if cuda:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                timings.setdefault(key, []).append((starts.pop(key), e))
            else:
                import time
                timings.setdefault(key, []).append(
                    (time.perf_counter() - starts.pop(key)) * 1e3)
            return out

        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))

    for i, blk in enumerate(blocks):
        hooks_for(("enc", i), blk)
    hooks_for(("dec", 0), model.decoder)

    vocab = int(getattr(model.config, "vocab_size", 32128))
    ids = torch.randint(0, max(2, vocab - 1), (batch_size, seq_len), device=device)
    mask = torch.ones_like(ids)
    start_id = model.config.decoder_start_token_id or model.config.pad_token_id or 0
    dec_in = torch.full((batch_size, 1), int(start_id), dtype=torch.long, device=device)

    def run():
        model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec_in,
              use_cache=False)

    try:
        for _ in range(3):
            run()
        if cuda:
            torch.cuda.synchronize()
        timings.clear()
        for _ in range(repeats):
            run()
        if cuda:
            torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()

    def median_ms(key):
        vals = timings[key]
        vals = [a.elapsed_time(b) for a, b in vals] if cuda else list(vals)
        return statistics.median(vals)

    enc_ms = [median_ms(("enc", i)) for i in range(len(blocks))]
    tau_s, cursor = [], 0
    for c in counts:
        tau_s.append(sum(enc_ms[cursor:cursor + c]) / 1e3)
        cursor += c
    tau_s.append(median_ms(("dec", 0)) / 1e3)
    return tau_s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_stages", type=int, required=True,
                    help="L_k: pipeline stages. Links = n_stages - 1 = dim(eta).")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", default=str(DEFAULT_PROFILE))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("! no CUDA device: timings will not reflect the deployment target",
              flush=True)
    dtype = getattr(torch, args.dtype)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM

    is_enc_dec = bool(getattr(
        AutoConfig.from_pretrained(args.model), "is_encoder_decoder", False))

    print(f"loading {args.model} ({args.dtype}, "
          f"{'encoder-decoder' if is_enc_dec else 'causal'}) ...", flush=True)
    loader = AutoModelForSeq2SeqLM if is_enc_dec else AutoModelForCausalLM
    model = loader.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    hidden = model.config.hidden_size
    bpe = _bytes_per_element(dtype)

    if is_enc_dec:
        # Encoder groups feed the links; the decoder is the last stage and sits
        # behind the final link, so it is not part of the encoder split.
        blocks = model.encoder.block
        counts = stage_layer_counts(len(blocks), args.n_stages - 1)
        print(f"  encoder blocks={len(blocks)}  hidden={hidden}  "
              f"encoder split={counts} + decoder", flush=True)
        tau_s = measure_seq2seq(model, counts, batch_size=args.batch_size,
                                seq_len=args.seq_len, repeats=args.repeats,
                                device=device)
    else:
        layers = detect_llm_decoder_layers(model)
        counts = stage_layer_counts(len(layers), args.n_stages)
        print(f"  layers={len(layers)}  hidden={hidden}  split={counts}", flush=True)
        tau_s = measure(model, layers, counts, batch_size=args.batch_size,
                        seq_len=args.seq_len, repeats=args.repeats, device=device)

    # One hidden-state tensor crosses each of the n_stages - 1 links.
    a_bytes = [float(args.batch_size * args.seq_len * hidden * bpe)] * (args.n_stages - 1)

    record = {
        "tau_s": [float(f"{v:.9g}") for v in tau_s],
        "a_bytes": a_bytes,
        "layers_per_stage": counts + ["decoder"] if is_enc_dec else counts,
        "hidden_size": int(hidden),
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "repeats": args.repeats,
        "measured": True,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    profiles: Dict = json.loads(out.read_text()) if out.exists() else {}
    (profiles
     .setdefault(args.model, {})
     .setdefault(str(args.n_stages), {})
     .setdefault(str(args.batch_size), {})[str(args.seq_len)]) = record
    out.write_text(json.dumps(profiles, indent=2, sort_keys=True) + "\n")

    print(f"  tau (s) = {['%.5f' % v for v in tau_s]}")
    print(f"  a (bytes) = {[int(v) for v in a_bytes]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
