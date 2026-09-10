# Third-party notices

This repository is licensed under the MIT License (see [LICENSE](LICENSE)). It
also redistributes the third-party material listed below, which remains under
its own terms. Nothing here restricts your rights under the MIT License to the
rest of the repository.

## CIFAR ResNet implementation and pretrained ResNet-56 checkpoint

- `src/core/resnet20.py` — adapted from
  [`akamaster/pytorch_resnet_cifar10`](https://github.com/akamaster/pytorch_resnet_cifar10)
  (`resnet.py`), restructured into a sequential model so it can be split across
  pipeline stages.
- `assets/resnet56-4bfd9763.th` — redistributed unmodified from the same
  project (`pretrained_models/resnet56-4bfd9763.th`).

Both are covered by the following notice. Per the authors' request, we also
credit **Yerlan Idelbayev** for the implementation.

```
BSD 2-Clause License

Copyright (c) 2018, Yerlan Idelbayev

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The ResNet architecture itself is from He et al., *Deep Residual Learning for
Image Recognition* (arXiv:1512.03385).

## SST-2 evaluation split

`experiments/concavity/flant5_sst2/data/sst2_validation.jsonl` is the 872-example
validation split of **SST-2**, as distributed in the GLUE benchmark. SST-2
derives from the Stanford Sentiment Treebank.

- Socher et al., *Recursive Deep Models for Semantic Compositionality Over a
  Sentiment Treebank*, EMNLP 2013.
- Wang et al., *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural
  Language Understanding*, ICLR 2019.

Only the sentence text and binary label are retained, for reproducibility of the
Flan-T5 / SST-2 concavity measurements.

## Data and model weights that are *not* bundled

These are downloaded at run time and are governed by their own licenses, which
you accept directly with the provider:

| Artifact | Source | Note |
|---|---|---|
| CIFAR-10 | downloaded to `./data` by torchvision | Krizhevsky, 2009 |
| MNIST | downloaded by torchvision | for `toy_mlp_mnist` |
| WikiText-2 | Hugging Face Hub | CC BY-SA 3.0 |
| ShareGPT | Hugging Face Hub | check the dataset card before redistributing |
| MMLU | Hugging Face Hub | Hendrycks et al., 2021 |
| Gemma-2B / Gemma-7B | Hugging Face Hub, gated | Gemma Terms of Use |
| Llama-3.1-8B | Hugging Face Hub, gated | Llama 3.1 Community License |
| Flan-T5-base | Hugging Face Hub | Apache-2.0 |

No model weights other than the ResNet-56 checkpoint above are included in this
repository.
