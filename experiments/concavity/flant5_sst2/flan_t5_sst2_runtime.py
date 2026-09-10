#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Flan-T5/SST-2 partition runtime used by flan_t5_sst2_concavity.py.

This file is intentionally inference-only. It keeps the model, dataset,
prompt/verbalizer, and fixed 3-transfer-point split from the original
flan_t5_sst2_local_estimator_training.py script, but removes the estimator
training and CSV sweep code.
"""

from __future__ import annotations

import csv
import inspect
import json
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn

from compressors import get_compressor


DEFAULT_MODEL_NAME = "google/flan-t5-base"
DEFAULT_DEVICE = "cpu"
DEFAULT_SPLIT = "validation"
DEFAULT_PROMPT_TEMPLATE = (
    "Classify the sentiment of the following sentence as positive or negative.\n"
    "Sentence: {sentence}\n"
    "Sentiment:"
)
DEFAULT_POSITIVE_TOKEN = "positive"
DEFAULT_NEGATIVE_TOKEN = "negative"
DEFAULT_MAX_INPUT_LENGTH = 128
DEFAULT_VERBALIZER_CANDIDATES = [
    ("positive", "negative"),
    ("yes", "no"),
    ("true", "false"),
]

NUM_TRANSFER_POINTS = 3
ENCODER_RANGES = [(0, 4), (4, 8), (8, 12)]
DECODER_RANGES = [(0, 12)]
TRANSFER_POINT_LABELS = {
    "tp0": "Enc0 -> Enc1",
    "tp1": "Enc1 -> Enc2",
    "tp2": "Enc2 -> Decoder",
}

_TRANSFORMERS_RUNTIME = None


@dataclass
class SST2Sample:
    sample_id: str
    sentence: str
    label: int


class TransformersRuntime:
    def __init__(self) -> None:
        try:
            from transformers import AutoTokenizer, T5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required in interpreter {}.".format(sys.executable)
            ) from exc

        try:
            from transformers.models.t5.modeling_t5 import (
                create_bidirectional_mask,
                create_causal_mask,
            )
        except Exception:
            create_bidirectional_mask = None
            create_causal_mask = None

        self.AutoTokenizer = AutoTokenizer
        self.T5ForConditionalGeneration = T5ForConditionalGeneration
        self.create_bidirectional_mask = create_bidirectional_mask
        self.create_causal_mask = create_causal_mask


def get_transformers_runtime() -> TransformersRuntime:
    global _TRANSFORMERS_RUNTIME
    if _TRANSFORMERS_RUNTIME is None:
        _TRANSFORMERS_RUNTIME = TransformersRuntime()
    return _TRANSFORMERS_RUNTIME


def supports_parameter(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False


def build_extended_attention_mask_compat(stack_module, attention_mask, hidden_states):
    fn = getattr(stack_module, "get_extended_attention_mask", None)
    if fn is None:
        return attention_mask

    kwargs = {}
    if supports_parameter(fn, "dtype"):
        kwargs["dtype"] = hidden_states.dtype
    elif supports_parameter(fn, "device"):
        kwargs["device"] = hidden_states.device

    return fn(attention_mask, hidden_states.shape[:2], **kwargs)


def normalize_attention_mask(input_ids, attention_mask, device):
    if attention_mask is None:
        return torch.ones_like(input_ids, dtype=torch.long, device=device)
    return attention_mask.to(device)


def tensor_payload_bytes(tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _parse_sample_records(data):
    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            return data["samples"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON dataset format.")


def _extract_sentence_label(record, index: int) -> SST2Sample:
    sentence = None
    for key in ("sentence", "text", "input_text"):
        value = record.get(key)
        if value is not None:
            sentence = str(value).strip()
            break
    if not sentence:
        raise ValueError("Record {} is missing sentence/text field.".format(index))
    if "label" not in record:
        raise ValueError("Record {} is missing label field.".format(index))
    label = int(record["label"])
    if label not in (0, 1):
        raise ValueError("Record {} has non-binary label: {}".format(index, label))
    return SST2Sample(
        sample_id=str(record.get("id", index)),
        sentence=sentence,
        label=label,
    )


def load_local_sst2_samples(dataset_path: str):
    ext = os.path.splitext(dataset_path)[1].lower()
    if ext == ".json":
        with open(dataset_path, "r", encoding="utf-8") as handle:
            records = _parse_sample_records(json.load(handle))
    elif ext == ".jsonl":
        records = []
        with open(dataset_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif ext in (".csv", ".tsv"):
        delimiter = "," if ext == ".csv" else "\t"
        with open(dataset_path, "r", encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle, delimiter=delimiter))
    else:
        raise ValueError("Unsupported dataset extension: {}".format(ext))
    return [_extract_sentence_label(record, idx) for idx, record in enumerate(records)]


def load_hf_sst2_samples(split: str = DEFAULT_SPLIT):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required when --dataset_path is not provided. "
            "Interpreter={}".format(sys.executable)
        ) from exc
    dataset = load_dataset("glue", "sst2", split=split)
    return [
        SST2Sample(
            sample_id=str(record.get("idx", index)),
            sentence=str(record["sentence"]).strip(),
            label=int(record["label"]),
        )
        for index, record in enumerate(dataset)
    ]


def load_sst2_samples(dataset_path=None, split: str = DEFAULT_SPLIT):
    if dataset_path:
        return load_local_sst2_samples(dataset_path)
    return load_hf_sst2_samples(split=split)


def build_prompt(sentence: str, prompt_template: str) -> str:
    return prompt_template.format(sentence=sentence.strip())


def resolve_single_token_verbalizers(tokenizer, positive_text=None, negative_text=None):
    candidates = []
    if positive_text is not None or negative_text is not None:
        if not positive_text or not negative_text:
            raise ValueError(
                "Both positive_token and negative_token must be provided together."
            )
        candidates.append((str(positive_text), str(negative_text)))
    else:
        candidates.append((DEFAULT_POSITIVE_TOKEN, DEFAULT_NEGATIVE_TOKEN))
    candidates.extend(DEFAULT_VERBALIZER_CANDIDATES)

    seen = set()
    for positive_candidate, negative_candidate in candidates:
        key = (str(positive_candidate), str(negative_candidate))
        if key in seen:
            continue
        seen.add(key)
        positive_ids = tokenizer.encode(str(positive_candidate), add_special_tokens=False)
        negative_ids = tokenizer.encode(str(negative_candidate), add_special_tokens=False)
        if len(positive_ids) != 1 or len(negative_ids) != 1:
            continue
        if positive_ids[0] == negative_ids[0]:
            continue
        return {
            "positive_text": str(positive_candidate),
            "negative_text": str(negative_candidate),
            "positive_id": int(positive_ids[0]),
            "negative_id": int(negative_ids[0]),
        }

    raise RuntimeError(
        "No valid single-token verbalizer pair found for the current tokenizer."
    )


def decide_label(logits, verbalizers) -> int:
    positive_logit = logits[verbalizers["positive_id"]]
    negative_logit = logits[verbalizers["negative_id"]]
    return 1 if float(positive_logit.item()) >= float(negative_logit.item()) else 0


def _extract_two_way_logits(logits, verbalizers):
    return torch.stack(
        [
            logits[:, verbalizers["negative_id"]],
            logits[:, verbalizers["positive_id"]],
        ],
        dim=1,
    )


def build_encoder_mask(runtime, stack_module, config, hidden_states, attention_mask):
    if runtime.create_bidirectional_mask is not None:
        try:
            return runtime.create_bidirectional_mask(
                config=config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )
        except TypeError:
            pass
    if hasattr(stack_module, "get_extended_attention_mask"):
        return build_extended_attention_mask_compat(
            stack_module,
            attention_mask,
            hidden_states,
        )
    return attention_mask


def build_cross_attention_mask(
    runtime,
    stack_module,
    config,
    hidden_states,
    encoder_hidden_states,
    encoder_attention_mask,
):
    if runtime.create_bidirectional_mask is not None:
        try:
            return runtime.create_bidirectional_mask(
                config=config,
                inputs_embeds=hidden_states,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )
        except TypeError:
            pass
    if hasattr(stack_module, "invert_attention_mask"):
        return stack_module.invert_attention_mask(encoder_attention_mask)
    return encoder_attention_mask


def build_decoder_mask(runtime, stack_module, config, hidden_states, decoder_attention_mask):
    if hasattr(stack_module, "get_extended_attention_mask"):
        return build_extended_attention_mask_compat(
            stack_module,
            decoder_attention_mask,
            hidden_states,
        )

    if runtime.create_causal_mask is not None:
        try:
            cache_position = torch.arange(
                hidden_states.shape[1],
                device=hidden_states.device,
            )
            return runtime.create_causal_mask(
                config=config,
                inputs_embeds=hidden_states,
                attention_mask=decoder_attention_mask,
                cache_position=cache_position,
                past_key_values=None,
            )
        except TypeError as exc:
            raise RuntimeError(
                "create_causal_mask signature mismatch for decoder mask construction. "
                "transformers runtime is incompatible with this code path."
            ) from exc
    return decoder_attention_mask


def apply_transport_compression(tensor, compressor, compression_param, device):
    original_bytes = tensor_payload_bytes(tensor)
    if compressor is None or compression_param is None:
        return tensor, {
            "original_bytes": original_bytes,
            "compressed_bytes": original_bytes,
            "payload_ratio": 1.0,
        }
    compressed = compressor.compress(tensor.detach().cpu(), compression_param)
    if hasattr(compressor, "get_compressed_size"):
        compressed_bytes = int(compressor.get_compressed_size(compressed))
    else:
        compressed_bytes = int(compressed.get("compressed_bytes", original_bytes))
    restored = compressor.decompress(compressed).to(device)
    payload_ratio = (
        float(compressed_bytes) / float(original_bytes) if original_bytes > 0 else 1.0
    )
    return restored, {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "payload_ratio": payload_ratio,
    }


class FlanT5EncoderPartition(nn.Module):
    def __init__(
        self,
        full_model,
        layer_indices,
        is_first_partition: bool,
        is_last_partition: bool,
        runtime,
    ) -> None:
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.is_first_partition = bool(is_first_partition)
        self.is_last_partition = bool(is_last_partition)
        self.runtime = runtime
        self.model_config = full_model.config
        self.stack_module = full_model.encoder
        self.blocks = [full_model.encoder.block[idx] for idx in self.layer_indices]
        self.embed_tokens = (
            full_model.encoder.embed_tokens if self.is_first_partition else None
        )
        self.dropout = full_model.encoder.dropout if self.is_first_partition else None
        self.final_layer_norm = (
            full_model.encoder.final_layer_norm if self.is_last_partition else None
        )
        self.final_dropout = (
            full_model.encoder.dropout if self.is_last_partition else None
        )
        sample_block = self.blocks[0]
        self.supports_return_dict = supports_parameter(sample_block.forward, "return_dict")
        self.supports_layer_head_mask = supports_parameter(
            sample_block.forward,
            "layer_head_mask",
        )
        self.supports_cross_attn_layer_head_mask = supports_parameter(
            sample_block.forward,
            "cross_attn_layer_head_mask",
        )
        self.supports_cache_position = supports_parameter(
            sample_block.forward,
            "cache_position",
        )

    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        if self.embed_tokens is not None and hidden_states.dtype == torch.long:
            hidden_states = self.embed_tokens(hidden_states)
            hidden_states = self.dropout(hidden_states)

        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[0],
                hidden_states.shape[1],
                dtype=torch.long,
                device=hidden_states.device,
            )

        block_attention_mask = build_encoder_mask(
            runtime=self.runtime,
            stack_module=self.stack_module,
            config=self.model_config,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)

        for block in self.blocks:
            block_kwargs = {
                "hidden_states": hidden_states,
                "attention_mask": block_attention_mask,
                "position_bias": position_bias,
                "encoder_hidden_states": None,
                "encoder_attention_mask": None,
                "use_cache": False,
                "output_attentions": False,
            }
            if self.supports_return_dict:
                block_kwargs["return_dict"] = False
            if self.supports_layer_head_mask:
                block_kwargs["layer_head_mask"] = None
            if self.supports_cross_attn_layer_head_mask:
                block_kwargs["cross_attn_layer_head_mask"] = None
            if self.supports_cache_position:
                block_kwargs["cache_position"] = cache_position

            outputs = block(**block_kwargs)
            hidden_states = outputs[0]
            position_bias = outputs[1] if len(outputs) > 1 else None

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.final_dropout(hidden_states)

        return hidden_states, position_bias


class FlanT5DecoderPartition(nn.Module):
    def __init__(
        self,
        full_model,
        layer_indices,
        is_first_partition: bool,
        is_last_partition: bool,
        runtime,
    ) -> None:
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.is_first_partition = bool(is_first_partition)
        self.is_last_partition = bool(is_last_partition)
        self.runtime = runtime
        self.model_config = full_model.config
        self.stack_module = full_model.decoder
        self.model_dim = getattr(full_model, "model_dim", full_model.config.d_model)
        self.blocks = [full_model.decoder.block[idx] for idx in self.layer_indices]
        self.embed_tokens = (
            full_model.decoder.embed_tokens if self.is_first_partition else None
        )
        self.dropout = full_model.decoder.dropout if self.is_first_partition else None
        self.final_layer_norm = (
            full_model.decoder.final_layer_norm if self.is_last_partition else None
        )
        self.final_dropout = (
            full_model.decoder.dropout if self.is_last_partition else None
        )
        self.lm_head = full_model.lm_head if self.is_last_partition else None
        self.cache_by_task = {}
        sample_block = self.blocks[0]
        self.supports_return_dict = supports_parameter(sample_block.forward, "return_dict")
        self.supports_layer_head_mask = supports_parameter(
            sample_block.forward,
            "layer_head_mask",
        )
        self.supports_cross_attn_layer_head_mask = supports_parameter(
            sample_block.forward,
            "cross_attn_layer_head_mask",
        )
        self.supports_encoder_decoder_position_bias = supports_parameter(
            sample_block.forward,
            "encoder_decoder_position_bias",
        )
        self.supports_cache_position = supports_parameter(
            sample_block.forward,
            "cache_position",
        )

    def _get_or_create_cache(self, task_id):
        if task_id not in self.cache_by_task:
            self.cache_by_task[task_id] = [None for _ in self.layer_indices]
        return self.cache_by_task[task_id]

    def clear_cache(self, task_id):
        if task_id in self.cache_by_task:
            del self.cache_by_task[task_id]

    def clear_all_cache(self):
        self.cache_by_task.clear()

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        task_id,
        encoder_attention_mask=None,
        position_bias=None,
        encoder_decoder_position_bias=None,
    ):
        self._get_or_create_cache(task_id)

        if self.embed_tokens is not None and hidden_states.dtype == torch.long:
            hidden_states = self.embed_tokens(hidden_states)
            hidden_states = self.dropout(hidden_states)

        decoder_attention_mask = torch.ones(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=torch.long,
            device=hidden_states.device,
        )
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        self_attention_mask = build_decoder_mask(
            runtime=self.runtime,
            stack_module=self.stack_module,
            config=self.model_config,
            hidden_states=hidden_states,
            decoder_attention_mask=decoder_attention_mask,
        )

        cross_attention_mask = None
        if encoder_attention_mask is not None:
            cross_attention_mask = build_cross_attention_mask(
                runtime=self.runtime,
                stack_module=self.stack_module,
                config=self.model_config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        for block in self.blocks:
            block_kwargs = {
                "hidden_states": hidden_states,
                "attention_mask": self_attention_mask,
                "position_bias": position_bias,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": cross_attention_mask,
                "use_cache": False,
                "output_attentions": False,
            }
            if self.supports_return_dict:
                block_kwargs["return_dict"] = False
            if self.supports_layer_head_mask:
                block_kwargs["layer_head_mask"] = None
            if self.supports_cross_attn_layer_head_mask:
                block_kwargs["cross_attn_layer_head_mask"] = None
            if self.supports_encoder_decoder_position_bias:
                block_kwargs["encoder_decoder_position_bias"] = (
                    encoder_decoder_position_bias
                )
            if self.supports_cache_position:
                block_kwargs["cache_position"] = cache_position

            outputs = block(**block_kwargs)
            hidden_states = outputs[0]
            position_bias = outputs[1] if len(outputs) > 1 else None
            encoder_decoder_position_bias = outputs[2] if len(outputs) > 2 else None

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.final_dropout(hidden_states)
            if self.lm_head is not None:
                if getattr(self.model_config, "scale_decoder_outputs", False) or (
                    not hasattr(self.model_config, "scale_decoder_outputs")
                    and getattr(self.model_config, "tie_word_embeddings", False)
                ):
                    hidden_states = hidden_states * (float(self.model_dim) ** -0.5)
                hidden_states = self.lm_head(hidden_states)

        return hidden_states, position_bias, encoder_decoder_position_bias


class FlanT5SST2PartitionRunner:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        max_input_length: int = DEFAULT_MAX_INPUT_LENGTH,
    ) -> None:
        self.runtime = get_transformers_runtime()
        self.model_name = model_name
        self.device = device
        self.max_input_length = int(max_input_length)
        if self.max_input_length <= 0:
            raise ValueError(
                "max_input_length must be positive, got {}".format(
                    self.max_input_length
                )
            )
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "device={} was requested, but torch.cuda.is_available() is False.".format(
                    self.device
                )
            )

        self.full_model = self.runtime.T5ForConditionalGeneration.from_pretrained(
            model_name
        ).to(device)
        self.full_model.eval()
        self.tokenizer = self.runtime.AutoTokenizer.from_pretrained(model_name)
        self._validate_model_layout()
        self.encoder_nodes = self._build_encoder_nodes()
        self.decoder_nodes = self._build_decoder_nodes()
        self.decoder_start_token_id = self._resolve_decoder_start_token_id()
        self.compressor_cache = {}

    def _validate_model_layout(self):
        encoder_layers = len(self.full_model.encoder.block)
        decoder_layers = len(self.full_model.decoder.block)
        if encoder_layers != 12 or decoder_layers != 12:
            raise ValueError(
                "This script assumes a 12+12 T5 layout for the fixed 3+1 split, "
                "but got encoder_layers={} decoder_layers={} for model {}".format(
                    encoder_layers,
                    decoder_layers,
                    self.model_name,
                )
            )

    def _resolve_decoder_start_token_id(self) -> int:
        if getattr(self.full_model.config, "decoder_start_token_id", None) is not None:
            return int(self.full_model.config.decoder_start_token_id)
        if self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        raise RuntimeError(
            "Could not resolve decoder_start_token_id for {}".format(self.model_name)
        )

    def _build_encoder_nodes(self):
        nodes = []
        for index, (start, end) in enumerate(ENCODER_RANGES):
            nodes.append(
                FlanT5EncoderPartition(
                    full_model=self.full_model,
                    layer_indices=list(range(start, end)),
                    is_first_partition=(index == 0),
                    is_last_partition=(index == len(ENCODER_RANGES) - 1),
                    runtime=self.runtime,
                )
            )
        return nodes

    def _build_decoder_nodes(self):
        nodes = []
        for index, (start, end) in enumerate(DECODER_RANGES):
            nodes.append(
                FlanT5DecoderPartition(
                    full_model=self.full_model,
                    layer_indices=list(range(start, end)),
                    is_first_partition=(index == 0),
                    is_last_partition=(index == len(DECODER_RANGES) - 1),
                    runtime=self.runtime,
                )
            )
        return nodes

    def clear_decoder_cache(self, task_id=None):
        for node in self.decoder_nodes:
            if task_id is None:
                node.clear_all_cache()
            else:
                node.clear_cache(task_id)

    def _get_compressor(self, compressor_name):
        name = str(compressor_name or "").strip().lower()
        if name in ("", "identity", "none"):
            return None
        if name not in self.compressor_cache:
            self.compressor_cache[name] = get_compressor(name)
        return self.compressor_cache[name]

    def encode_prompt(self, prompt_text: str):
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
        )
        return {
            "input_ids": encoded.input_ids.detach().cpu(),
            "attention_mask": normalize_attention_mask(
                encoded.input_ids,
                encoded.get("attention_mask"),
                encoded.input_ids.device,
            ).detach().cpu(),
        }

    def run_partitioned_first_step(
        self,
        input_ids,
        attention_mask,
        task_id,
        compressor_name,
        tp_compression_params,
    ):
        compressor = self._get_compressor(compressor_name)
        input_ids = input_ids.to(self.device)
        attention_mask = normalize_attention_mask(input_ids, attention_mask, self.device)
        self.clear_decoder_cache(task_id)

        decoder_input_ids = torch.tensor(
            [[self.decoder_start_token_id]],
            dtype=torch.long,
            device=self.device,
        )
        hidden_states = input_ids
        position_bias = None
        transfer_stats = {}

        try:
            with torch.no_grad():
                for node_index, node in enumerate(self.encoder_nodes):
                    hidden_states, position_bias = node(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_bias=position_bias,
                    )
                    if node_index < len(self.encoder_nodes) - 1:
                        tp_name = "tp{}".format(node_index)
                        hidden_states, transfer_stats[tp_name] = apply_transport_compression(
                            hidden_states,
                            compressor=compressor,
                            compression_param=tp_compression_params[node_index],
                            device=self.device,
                        )

                encoder_hidden_states = hidden_states
                encoder_hidden_states, transfer_stats["tp2"] = apply_transport_compression(
                    encoder_hidden_states,
                    compressor=compressor,
                    compression_param=tp_compression_params[2],
                    device=self.device,
                )

                hidden_states = decoder_input_ids
                position_bias = None
                encoder_decoder_position_bias = None

                for node in self.decoder_nodes:
                    hidden_states, position_bias, encoder_decoder_position_bias = node(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        task_id=task_id,
                        encoder_attention_mask=attention_mask,
                        position_bias=position_bias,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                    )

            return {
                "logits": hidden_states[:, -1, :],
                "transfer_stats": transfer_stats,
            }
        finally:
            self.clear_decoder_cache(task_id)
