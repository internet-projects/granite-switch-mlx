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
