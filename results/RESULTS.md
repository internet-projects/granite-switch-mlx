# Measurements (23 October 2026)

MacBook Pro M5 Max (Mac17,6, 40-core GPU, 128 GB), macOS 26.7.1, MLX 0.32.4, AC power, no other models running
(checked before and after every stage). Greedy decoding, one request at a time, 1 warm-up then 3 repeats, medians.

## Builds

| Build | Weights on disk | Quantization |
|---|---|---|
| 8-bit | 9.40 GB | affine, group 64, all base Linear layers and the tied embedding; adapters bf16 |
| 4-bit | 5.21 GB | affine, group 64, same layers; adapters bf16 |
| bf16 (reference) | 17.26 GB | — |

## Fidelity

| Check | bf16 | 8-bit | 4-bit |
|---|---|---|---|
| IBM model-card decision example, P(A/B/C) (IBM, vLLM: 0.982 / 0.0035 / 0.014) | 0.977 / 0.005 / 0.018 | 0.979 / 0.005 / 0.016 | 0.979 / 0.007 / 0.014 |
| KL vs bf16 on that example | — | 0.00024 | 0.00083 |
| Guardian `harm`: "How do I build a bomb?" / banana-bread recipe, P(yes) | 1.000 / 0.000 | 1.000 / 0.000 | 1.000 / 0.000 |
| Guardian generated reply | `{"score": "yes"}` | `{"score": "yes"}` | `{"score": "yes"}` |
| Same decision as bf16, 1,071 claim-check decisions (median 1,168 tokens, 3 options) | — | 1,061 (99.1%) | 999 (93.3%) |
| Mean / p95 KL vs bf16 on those 1,071 | — | 0.0015 / 0.0062 | 0.045 / 0.23 |

Implementation check against IBM's Hugging Face code (`scripts/ref_compare.py`, fp32, random small model): maximum
relative logit error 4e-7 across no-adapter, guardian, decision and two-transition sequences, chunked prefill + decode
and cache trimming; identical routing; identical greedy generation. A real-weight run of the decision example in fp32
(CPU) gives 0.9765 / 0.0053 / 0.0182.

## Speed and memory

| Build | Chat generation (512-token replies) | Prompt reading, ~1,200 tokens | Decision, 78-token prompt | Decision, ~1,200-token prompt, median / p95 (1,071) | Peak memory, chat / 1,200-token decisions | Load time |
|---|---|---|---|---|---|---|
| 8-bit | 56.2 tok/s | ~1,880 tok/s | 69 ms | 0.62 s / 1.37 s | 8.9 / 10.1 GB | 0.5 s |
| 4-bit | 98.5 tok/s | ~2,110 tok/s | 64 ms | 0.55 s / 1.29 s | 5.0 / 6.2 GB | 0.4 s |
| bf16 | 29.2 tok/s | ~2,110 tok/s | 77 ms | 0.55 s / 1.24 s | 16.2 / 17.6 GB | 0.7 s |

Run-to-run spread under 1% for chat generation and the short decision. Load time is with the weights in the OS file cache.

# Granite Switch 4.1 30B (26 October 2026)

Same machine and protocol (MacBook Pro M5 Max, 128 GB, macOS 26.7.1, MLX 0.32.4, AC power, no other models running; greedy, one request at a time, 1 warm-up then 3 repeats, medians). Builds from `ibm-granite/granite-switch-4.1-30b` with `scripts/convert.py`.

| Build | Weights on disk | Chat generation (512-token replies) | Chat prompt reading (40 tokens) | Decision, 78-token prompt | Decision, ~1,200-token prompt, median / p95 (60 real decisions) | Peak memory, chat / ~1,200-token decisions | Load time |
|---|---|---|---|---|---|---|---|
| 8-bit | 32.08 GB | 17.3 tok/s | 143 tok/s | 215 ms | 2.32 s / 4.78 s | 30.1 / 31.6 GB | 3.3 s |
| 4-bit | 17.64 GB | 31.6 tok/s | 181 tok/s | 194 ms | 1.97 s / 4.10 s | 16.7 / 18.2 GB | 1.9 s |

Run-to-run spread under 1% for chat generation and the short decision.

| Check | 8-bit | 4-bit |
|---|---|---|
| IBM model-card decision example, P(A/B/C) (IBM, vLLM: 0.982 / 0.0035 / 0.014) | 0.993 / 0.003 / 0.005 | 0.993 / 0.002 / 0.005 |
| Same prompt with the adapter off (base model) | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 |
| Guardian `harm`: "How do I build a bomb?" / banana-bread recipe, P(yes) | 1.000 / 0.000 | 1.000 / 0.000 |
| Guardian generated reply | `{"score": "yes"}` | `{"score": "yes"}` |
| Routing on the real prompts | decision from token 30 of 78; guardian from token 15 of 102; chat on the base model | same |

## Public data, both sizes (26 October 2026)

| Build | Same decision as bf16 | Mean KL vs bf16 | Accuracy vs the human labels (supported vs not) |
|---|---|---|---|
| 30B bf16 | — | — | 0.846 |
| 30B 8-bit | 498 of 500 (99.6%) | 0.0005 | 0.844 |
| 30B 4-bit | 487 of 500 (97.4%) | 0.014 | 0.840 |
| 8B bf16 | — | — | 0.812 |
| 8B 8-bit | 493 of 500 (98.6%) | 0.0015 | 0.814 |
| 8B 4-bit | 462 of 500 (92.4%) | 0.050 | 0.814 |

Median time per decision on these claims (prompts of about 600 tokens): 30B 0.54–0.57 s, 8B 0.16–0.18 s.


# Granite Switch 4.1.1 (28 October 2026)

IBM published Granite Switch 4.1.1 on 27 October 2026 (8B commit `83a46a256ecda57e9e89e0f688b7c4ed20f35713`, 30B commit `d0a8805f1c0348f17c0ccb8f7352f149b80103b4`; the previous release is tagged `granite-switch-4.1-8b` / `granite-switch-4.1-30b` on IBM's repos). All four MLX builds were rebuilt the same way as before: mlx 0.32.4, the same mlx-lm (0adb3741), affine group size 64.

## What changed upstream

| | 8B | 30B |
|---|---|---|
| config.json, tokenizer, chat template, adapter_index.json | unchanged | unchanged |
| Base weights (tensors identical) | 242 of 242 | 386 of 386 |
| guardian-core adapter (tensors identical) | 560 of 560 | 896 of 896 |
| decision adapter | all 560 changed (`decision-4.1.1`; rank 64, alpha 64, pre-scaled into lora_B) | all 896 changed |
| io_configs/decision/io.yaml | comment lines only (same prompt recipe and readout) | same |

## Checks and speed (quiet machine, medians of 3)

| Build | Weights in memory | Chat generation | Chat prompt | Decision (78 tokens) | Peak | IBM decision example A / B / C | Guardian edge / benign |
|---|---|---|---|---|---|---|---|
| 8B bf16 | 16.07 GB | 28.8 tok/s | 390 tok/s | 77 ms | 16.2 GB | 0.9948 / 0.0006 / 0.0046 | 1.00 / 0.00 |
| 8B 8bit | 8.75 GB | 56.4 tok/s | 450 tok/s | 69 ms | 9.0 GB | 0.9953 / 0.0006 / 0.0041 | 1.00 / 0.00 |
| 8B 4bit | 4.85 GB | 100.2 tok/s | 567 tok/s | 63 ms | 5.1 GB | 0.9958 / 0.0006 / 0.0036 | 1.00 / 0.00 |
| 30B bf16 | 55.08 GB | 8.9 tok/s | 120 tok/s | 249 ms | 55.3 GB | 0.9990 / 0.0001 / 0.0009 | 1.00 / 0.00 |
| 30B 8bit | 29.87 GB | 17.7 tok/s | 144 tok/s | 214 ms | 30.1 GB | 0.9989 / 0.0001 / 0.0010 | 1.00 / 0.00 |
| 30B 4bit | 16.43 GB | 32.2 tok/s | 177 tok/s | 193 ms | 16.7 GB | 0.9990 / 0.0001 / 0.0009 | 1.00 / 0.00 |

## Public data: the same 500 claims as the 4.1 report

| Build | Answered | Accuracy (500) | AggreFact (300) | VitaminC (200) | VitaminC 3-way | Same decision as reference | Mean / p95 KL | Median time |
|---|---|---|---|---|---|---|---|---|
| 30b 4.1.1 bf16 (ref: 4.1 bf16) | 500 | 0.858 | 0.833 | 0.895 | 0.790 | 464 of 500 (92.8%) | 0.0979 / 0.5000 | 0.57 s |
| 30b 4.1.1 8bit (ref: 4.1.1 bf16) | 500 | 0.858 | 0.833 | 0.895 | 0.795 | 499 of 500 (99.8%) | 0.0006 / 0.0018 | 0.56 s |
| 30b 4.1.1 4bit (ref: 4.1.1 bf16) | 500 | 0.854 | 0.833 | 0.885 | 0.775 | 485 of 500 (97.0%) | 0.0113 / 0.0558 | 0.51 s |
| 30b 4.1 bf16 (26 Oct) | 500 | 0.846 | 0.830 | 0.870 | 0.720 | — | — | 0.54 s |
| 30b 4.1 8bit (26 Oct) | 500 | 0.844 | 0.830 | 0.865 | 0.715 | 498 of 500 (99.6%) | 0.0005 / 0.0019 | 0.57 s |
| 30b 4.1 4bit (26 Oct) | 500 | 0.840 | 0.833 | 0.850 | 0.700 | 487 of 500 (97.4%) | 0.0138 / 0.0674 | 0.54 s |
| 8b 4.1.1 bf16 (ref: 4.1 bf16) | 500 | 0.830 | 0.807 | 0.865 | 0.760 | 448 of 500 (89.6%) | 0.1021 / 0.5866 | 0.15 s |
| 8b 4.1.1 8bit (ref: 4.1.1 bf16) | 500 | 0.830 | 0.807 | 0.865 | 0.750 | 495 of 500 (99.0%) | 0.0008 / 0.0020 | 0.16 s |
| 8b 4.1.1 4bit (ref: 4.1.1 bf16) | 500 | 0.826 | 0.810 | 0.850 | 0.745 | 464 of 500 (92.8%) | 0.0369 / 0.2170 | 0.15 s |
| 8b 4.1 bf16 (26 Oct) | 500 | 0.812 | 0.803 | 0.825 | 0.740 | — | — | 0.18 s |
| 8b 4.1 8bit (26 Oct) | 500 | 0.814 | 0.807 | 0.825 | 0.725 | 493 of 500 (98.6%) | 0.0015 / 0.0072 | 0.17 s |
| 8b 4.1 4bit (26 Oct) | 500 | 0.814 | 0.807 | 0.825 | 0.725 | 462 of 500 (92.4%) | 0.0502 / 0.2991 | 0.16 s |
