"""
Per–task-family modules: setup (model + callables) and ``make_task`` builders.

How to add a task instance
--------------------------
1. Create ``src/core/task_instances/<your_task>.py``.
2. Implement:

   * ``setup_model_and_callables(**kwargs) -> (api, extra)``
     ``api`` must expose ``accuracy_callable``, ``gradient_callable``, and
     ``accuracy_callable_true`` (``np.ndarray`` → float / ndarray).
   * ``make_task(task_id, api, w_k=1.0, R_k=10.0, **kwargs) -> InferenceTask``
     with topology fields ``b_k``, ``L_k``, ``tau``, ``a``, ``eta_min`` and the
     callables above.

3. Register the module in ``src.core.task_handler``::

       register_task("your_task", your_module)
       # or, if the name selects a codec:
       register_task("your_task_quant", your_module, compressor_name="quantization")

4. Run offline sims with ``--task your_task`` (single) or ``--tasks ...`` (multi).
   Set ``--M`` equal to the task's ``L_k``.

Shared codecs live in ``src.core.compressors``. Optimizers only see ``InferenceTask``.

Families in this package
------------------------
``toy_mlp_mnist``
    Small MLP on MNIST. The cheapest task; good for smoke-testing a change.
``resnet56_cifar10_topk``
    ResNet-56 / CIFAR-10, registered once per codec.
``gemma2b_sharegpt_ppl``, ``gemma7b_sharegpt_ppl``, ``llama31_8b_wikitext_ppl``,
``llama31_8b_mmlu``
    The causal-LM scenarios. Thin wrappers over :mod:`._llm_base`, which holds
    the shared load / hook / evaluate pipeline.
``flant5_sst2``
    The encoder-decoder scenario. Standalone, because its metric is a two-way
    verbalizer comparison and its links sit at encoder-block boundaries.

The LLM tasks need a measured stage profile before first use
(``python profile_llm_stages.py ...``); see the repository README.
"""
