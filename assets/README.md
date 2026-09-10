# Experiment assets

This directory contains the ResNet-56 checkpoint, accuracy-function fitting
models, and measured performance traces used by the offline experiments for
**Communication-Aware Model Distributed Inference via Latent Representation
Compression (MobiHoc 2026)**.

**For fitting and trace-generation code, configuration, and reproduction
instructions, please refer to our companion GitHub repository for Jetson online
experiments.** Its fitting workflow is platform-independent and can be used in
CPU or GPU environments beyond Jetson. See
[Companion repositories](../README.md#companion-repositories) for where the
per-testbed repositories live.

## Files and their roles

| File | What it contains and how it is used |
| --- | --- |
| [resnet56-4bfd9763.th](resnet56-4bfd9763.th) | Pretrained ResNet-56 neural-network weights for CIFAR-10. Used to run the classifier and measure actual accuracy under compression. |
| [jetson_resnet_3tp_poly3_flex.pkl](jetson_resnet_3tp_poly3_flex.pkl) | Accuracy-function fitting model for **Top-K** compression. The filename without a codec suffix denotes Top-K. |
| [jetson_resnet_3tp_quantization_poly3_flex.pkl](jetson_resnet_3tp_quantization_poly3_flex.pkl) | Accuracy-function fitting model for **quantization**. |
| [jetson_resnet_3tp_llmint8_fp16_int4_poly3_flex.pkl](jetson_resnet_3tp_llmint8_fp16_int4_poly3_flex.pkl) | Accuracy-function fitting model for **LLM.int8-style compression with FP16 outliers and INT4 regular values**. |
| [resnet56_cifar10_15Mbps_scenario_trace_topk.json](resnet56_cifar10_15Mbps_scenario_trace_topk.json) | Per-batch measurements of a four-node ResNet-56 execution, including link rates, activation sizes, transfer times, computation times, and Top-K codec overhead. Provides a measured environment profile for offline experiments. |
| [resnet56_cifar10_15Mbps_trace_summary_topk.json](resnet56_cifar10_15Mbps_trace_summary_topk.json) | Configuration and aggregate statistics from the same trace collection, including sample counts, elapsed collection time, means, and medians. Used to interpret the recorded scenario. |

## ResNet-56 checkpoint

`resnet56-4bfd9763.th` stores the learned parameters of a ResNet-56 classifier
trained on labeled CIFAR-10 images. It is a pretrained neural-network checkpoint;
accuracy fitting and trace collection load these weights for their measurements.

The checkpoint supplies the classifier used to evaluate actual accuracy under
different compression settings. The three `.pkl` files below instead describe
the relationship between compression and accuracy.

## Accuracy-function fitting models

The three PKLs estimate classification accuracy from compression settings at
three activation-transfer points. `3tp` means three transfer points, and `poly3`
means polynomial regression of degree three. Each model takes three compression
ratios as input and predicts classification accuracy for its compression policy.

Each pickle contains a fitted regressor, input scaler, polynomial transform,
feature names, model type, and training/validation metrics. These fitted models
provide accuracy estimates and gradients for optimizing compression ratios.

To generate them, evaluate the pretrained ResNet-56 on CIFAR-10 while varying
compression settings at the three transfer points. Record the settings and
measured accuracy, fit a degree-three polynomial regression model for each
compression policy, and save the fitted model and preprocessing information.
The supplied fits cover Top-K, quantization, and LLM.int8-style FP16/INT4
compression. For implementation and generation details, please refer to the
companion GitHub repository.

## Performance trace and summary

The trace records **100 batches of 100 CIFAR-10 images**, processed by ResNet-56
across **four Jetson nodes** in the order **A → B → C → D**. Its metadata records
CUDA execution, a Wi-Fi network, and a configured transmission cap of
**15 Mbps**. This cap is a collection setting; the recorded effective link rates
vary across transfers.

The scenario JSON contains the experiment setup, ordered batch measurements,
and aggregate statistics. Its measurements include:

| Measurement | Meaning and units |
| --- | --- |
| Activation and packet sizes | Uncompressed intermediate activations and transmitted packet sizes, in bytes. |
| Link rates | Measured transfer rates for the three links, in **bytes/second**. The JSON field named `channel_true_bps` also uses bytes/second. |
| Transfer and node times | Send durations, computation times, and node residence times, in seconds. |
| Codec overhead | Separately measured Top-K compression/decompression overhead, in seconds. |
| End-to-end delay | Observed completion time for each batch, in seconds. |

**Actual network payloads are transmitted without compression.** The
`topk` suffix identifies the separately measured Top-K overhead profile, using
compression parameter `0.5`.

To generate the trace, run distributed ResNet-56 inference across the four nodes
and record activation sizes, network transfers, node timings, and end-to-end
delay for each batch. Measure codec overhead separately, then save the collected
measurements and their summary as JSON files. For collection code, setup, and
reproduction details, please refer to the companion GitHub repository.

The separate `resnet56_cifar10_15Mbps_trace_summary_topk.json` file summarizes the
same collection. It contains the configuration, expected/observed batch counts, elapsed
collection time, and mean/median delays, node times, codec times, and link rates.
It supports inspection and comparison of the recorded scenario.

The trace provides measured performance profiles and a network-environment
reference for offline experiments. The current offline comparisons use its
activation-size and computation-time profiles, while channel capacities are
simulated rather than automatically replaying the recorded link-rate sequence.
