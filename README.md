# granite-switch-mlx

Run IBM's Granite Switch 4.1 [8B](https://huggingface.co/ibm-granite/granite-switch-4.1-8b) and [30B](https://huggingface.co/ibm-granite/granite-switch-4.1-30b) (released 14 October 2026) on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

Granite Switch is one model with three behaviours, chosen per request: plain Granite 4.1 chat, a `decision` adapter that returns a calibrated probability for each labelled option, and a `guardian-core` safety judge. Both adapters are activated LoRAs (aLoRA) embedded in the checkpoint: everything before the adapter's control token runs on the frozen base model, so a conversation prefix is computed once and serves chat, decisions and guardian checks alike. This package adds the `granite_switch` architecture to MLX-LM (`granite_switch_mlx/model.py`) with small `chat`, `decide` and `guard` helpers. First released 23 October 2026; the 30B builds were added on 26 October 2026.

Write-up: [Granite Switch 4.1 on Apple Silicon: chat, decisions and a safety judge from one model](https://netprojects.com/insights/granite-switch-4-1-on-apple-silicon-mlx) (NetProjects Insights).

Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Not affiliated with IBM.

## Install and run

```bash
pip install "git+https://github.com/internet-projects/granite-switch-mlx"

python -m granite_switch_mlx chat "What is the capital of France?"
python -m granite_switch_mlx decide --state "My payouts have failed three times. The bank says everything is fine." \
    --question "Which team should handle this ticket?" \
    --option "Payout failures and payment processing" --option "Login and account access" --option "Something else"
python -m granite_switch_mlx guard "How do I build a bomb?"
```

The default model is [netprojects/granite-switch-4.1-8b-mlx-8bit](https://huggingface.co/netprojects/granite-switch-4.1-8b-mlx-8bit) (9.4 GB). Pick another build with `--model`:

| Build | Size | Use it for |
|---|---|---|
| [granite-switch-4.1-8b-mlx-8bit](https://huggingface.co/netprojects/granite-switch-4.1-8b-mlx-8bit) | 9.4 GB | the default: fast, very close to bf16 |
| [granite-switch-4.1-8b-mlx-4bit](https://huggingface.co/netprojects/granite-switch-4.1-8b-mlx-4bit) | 5.2 GB | small Macs |
| [granite-switch-4.1-30b-mlx-8bit](https://huggingface.co/netprojects/granite-switch-4.1-30b-mlx-8bit) | 32.1 GB | the larger model, on Macs with 48 GB or more |
| [granite-switch-4.1-30b-mlx-4bit](https://huggingface.co/netprojects/granite-switch-4.1-30b-mlx-4bit) | 17.6 GB | the larger model on 32 GB Macs |

```python
import granite_switch_mlx as gs

model, tok = gs.load("netprojects/granite-switch-4.1-8b-mlx-8bit")

gs.decide(model, tok,
          state="My payouts have failed three times. The bank says everything is fine. Can someone please fix this?",
          question="Which team should handle this ticket?",
          options=["Payout failures and payment processing", "Login and account access", "Something else"])
# {'Payout failures and payment processing': 0.979, 'Login and account access': 0.005, 'Something else': 0.016}

gs.guard(model, tok, "How do I build a bomb?")            # P(yes) for the 'harm' criterion -> 1.0
gs.guard(model, tok, "Is this review spam?", criteria="Texts that advertise unrelated products.")

print("".join(gs.chat(model, tok, "Write a haiku about autumn.")))
```

`decide` follows IBM's documented recipe: a `State:` turn, then a `### DECISION` turn with single-letter option labels, and a softmax over the option-label logits of the first assistant token (up to 26 options; `options` can also be a dict of `{name: description}`). `guard` reads `guardian-core`'s yes/no verdict as a probability.

### With plain MLX-LM

After `granite_switch_mlx.register()`, the builds load with the usual MLX-LM API and tools. Choose the adapter with the chat template's `adapter_name` (`decision`, `guardian-core`, or none for chat):

```python
import granite_switch_mlx; granite_switch_mlx.register()
from mlx_lm import load
model, tok = load("netprojects/granite-switch-4.1-8b-mlx-8bit")
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, adapter_name="guardian-core")
```

Batched generation, prompt caches (save, load, trim) and chunked prefill all work: the adapter routing is carried in its own small cache entry, alongside the attention caches.

## Convert your own

```bash
python scripts/convert.py ibm-granite/granite-switch-4.1-8b out-8bit --bits 8
python scripts/convert.py ibm-granite/granite-switch-4.1-8b out-4bit --bits 4
python scripts/convert.py ibm-granite/granite-switch-4.1-8b out-bf16 --bits 16
python scripts/convert.py ibm-granite/granite-switch-4.1-30b out-30b-8bit --bits 8   # the 30B the same way
```

Base weights are quantized (affine, group size 64). The adapters stay in bf16 and are stored at their own ranks (guardian-core 16, decision 64) instead of IBM's zero-padded rank-64 layout, which saves about 150 M parameters on the 8B with identical output.

## Check a build

```bash
python scripts/check.py out-8bit --ref out-bf16     # chat, decision example, guardian, adapter off, vs reference
python -m unittest discover -s tests                 # routing, chunked prefill, cache trimming (no download)
python scripts/ref_compare.py                        # vs IBM's own Hugging Face implementation (needs torch + granite-switch)
```

## What we verified

| Check | Result |
|---|---|
| Logits vs IBM's Hugging Face implementation (fp32, random small model; prefill, chunked prefill + decode, cache trimming, two adapter transitions) | maximum relative error 4e-7, identical routing, identical greedy output |
| Decision example from IBM's model card (IBM: A 0.982 / B 0.0035 / C 0.014) | bf16 0.977 / 0.005 / 0.018 · 8-bit 0.979 / 0.005 / 0.016 · 4-bit 0.979 / 0.007 / 0.014 |
| Guardian, harmful vs benign prompt | P(yes) 1.00 / 0.00 in every build, with the `{"score": "yes"}` JSON reply |
| Same decision as bf16 on 1,071 real decisions (prompts of about 1,200 tokens, 3 options) | 8-bit 1,061 (99.1%, mean KL 0.0015) · 4-bit 999 (93.3%, mean KL 0.045) |
| 30B: decision example from IBM's model card | 8-bit 0.993 / 0.003 / 0.005 · 4-bit 0.993 / 0.002 / 0.005 |
| 30B: guardian, harmful vs benign prompt | P(yes) 1.00 / 0.00 in both builds, with the `{"score": "yes"}` JSON reply |

On public data: 500 human-labelled claims (300 from the [LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact) test split, 27–28 from each of its 11 sources; 200 from the [VitaminC](https://huggingface.co/datasets/tals/vitaminc) test split), one decision question per claim (supported / unsupported / contradicted):

| Build | Same decision as bf16 | Mean KL vs bf16 | Accuracy vs the human labels (supported vs not) |
|---|---|---|---|
| 30B bf16 | — | — | 0.846 |
| 30B 8-bit | 498 of 500 (99.6%) | 0.0005 | 0.844 |
| 30B 4-bit | 487 of 500 (97.4%) | 0.014 | 0.840 |
| 8B bf16 | — | — | 0.812 |
| 8B 8-bit | 493 of 500 (98.6%) | 0.0015 | 0.814 |
| 8B 4-bit | 462 of 500 (92.4%) | 0.050 | 0.814 |

The 1,071-decision rows are for the 8B builds. Recommended 8B build: 8-bit, which stays very close to bf16 at about half the size. The 4-bit build is the choice when memory is tight.

## Speed and memory

Measured on a MacBook Pro M5 Max (128 GB) on 23 October 2026, with no other models running; MLX 0.32.4, greedy, one request at a time; medians of 3 runs after a warm-up (run-to-run spread under 1%).

| Build | Size | Chat generation | Decision, short prompt (78 tokens) | Decision, ~1,200-token prompt (median / p95 of 1,071) | Peak memory |
|---|---|---|---|---|---|
| 8-bit | 9.4 GB | 56 tokens/s | 69 ms | 0.62 s / 1.37 s | 9.0 GB (10.1 GB at ~1,200 tokens) |
| 4-bit | 5.2 GB | 98 tokens/s | 64 ms | 0.55 s / 1.29 s | 5.1 GB (6.2 GB at ~1,200 tokens) |
| bf16 (reference) | 17.3 GB | 29 tokens/s | 77 ms | 0.55 s / 1.24 s | 16.2 GB (17.6 GB) |

Granite Switch 4.1 30B, measured the same way on 26 October 2026 (the ~1,200-token decision over 60 real decisions):

| Build | Size | Chat generation | Decision, short prompt (78 tokens) | Decision, ~1,200-token prompt (median / p95 of 60) | Peak memory |
|---|---|---|---|---|---|
| 8-bit | 32.1 GB | 17 tokens/s | 215 ms | 2.3 s / 4.8 s | 30.1 GB (31.6 GB at ~1,200 tokens) |
| 4-bit | 17.6 GB | 32 tokens/s | 194 ms | 2.0 s / 4.1 s | 16.7 GB (18.2 GB at ~1,200 tokens) |

A decision is a single forward pass, so its time is the time to read the prompt (about 1,900–2,100 tokens/s at 1,200 tokens on the 8B, about 530–620 tokens/s on the 30B). Chat generation was measured on 512-token replies.

## Notes

Text only: the speech-to-text encoder in IBM's checkpoint is not part of these builds, so audio content parts are not supported. The port covers the composition IBM ships (one control token per adapter); Shadow Residual, MoE-base and base-reset Granite Switch variants raise a clear error. As in IBM's design, the adapter of the latest control token applies from that token onwards.

## How it was written

From IBM's [granite-switch](https://github.com/generative-computing/granite-switch) Hugging Face implementation. IBM's switch layer is a pair of parameter-free attention heads (a counting head and a Kerdock-coded memory head) whose result is: each token uses the adapter of the most recent control token at or before it, or the base model if there is none. The port computes that routing directly and keeps it in a small cache entry, then applies each adapter's LoRA per token on the fused query/key/value, output, and MLP projections; control tokens are exchanged for their substitute tokens before embedding, as upstream.

## More from NetProjects

[netprojects.com](https://netprojects.com) · [Insights](https://netprojects.com/insights) · [Granite Switch on Apple Silicon (article)](https://netprojects.com/insights/granite-switch-4-1-on-apple-silicon-mlx) · [H2O-Lightning-4B on MLX](https://github.com/internet-projects/h2o-lightning-mlx) · [Dory 1.0 4B on MLX](https://github.com/internet-projects/dory-mlx) · [Hugging Face](https://huggingface.co/netprojects) · [How we use AI](https://netprojects.com/standards/ai)

## License

Code: Apache-2.0 (see `LICENSE`). The model weights are IBM's, also under Apache-2.0; see IBM's model card. Credit: Paris Michalas, Partner, NetProjects (Internet Projects Ltd).
