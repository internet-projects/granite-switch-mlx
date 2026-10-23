"""Compare this MLX implementation with IBM's own Hugging Face implementation on a small random
Granite Switch model (fp32): logits, routing, chunked prefill + decode, cache trimming, greedy generation.

    pip install torch "transformers>=5.16,<=5.17" && pip install -e /path/to/generative-computing/granite-switch
    python scripts/ref_compare.py
"""
import tempfile
from pathlib import Path

import numpy as np
import torch
from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM

import granite_switch_mlx

granite_switch_mlx.register()
import mlx.core as mx  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache  # noqa: E402
from mlx_lm.utils import load_model  # noqa: E402

torch.manual_seed(0)
n_dec, ranks = 3, [4, 8]
cfg = GraniteSwitchConfig(
    vocab_size=300, hidden_size=64, intermediate_size=128, shared_intermediate_size=128,
    num_hidden_layers=n_dec + 2, num_attention_heads=4, num_key_value_heads=2, projection_head_dim=16,
    max_position_embeddings=4096, rms_norm_eps=1e-5, attention_multiplier=0.0625, embedding_multiplier=12.0,
    residual_multiplier=0.22, logits_scaling=16.0, rope_parameters={"rope_theta": 10000000, "rope_type": "default"},
    tie_word_embeddings=True, num_local_experts=0, layer_types=["full_attention"] * (n_dec + 2),
    num_adapters=2, adapter_token_ids=[298, 299], adapter_substitute_token_ids=[27, 14],
    adapter_names=["guardian-core", "decision"], adapter_ranks=ranks, max_lora_rank=max(ranks),
    lora_target_modules=["o_proj", "qkv_proj", "shared_input_linear", "shared_output_linear"],
    pad_token_id=0, bos_token_id=1, eos_token_id=2, hidden_act="silu")
hf = GraniteSwitchForCausalLM(cfg).eval()
with torch.no_grad():
    for name, p in hf.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.normal_(0, 0.2)
            for i, r in enumerate(ranks):  # zero-pad beyond each adapter's rank, as IBM's composer does
                if "lora_A" in name:
                    p[i, 0, r:, :] = 0
                else:
                    p[i, 0, :, r:] = 0
d = Path(tempfile.mkdtemp())
hf.save_pretrained(d, safe_serialization=True)
mm, _ = load_model(d)


def ref(ids):
    with torch.no_grad():
        out = hf(input_ids=torch.tensor([ids]), use_cache=False)
    return out.logits[0].numpy(), hf.model._last_adapter_indices[0].numpy()


rel = lambda a, b: float(np.abs(a - b).max() / np.abs(a).max())  # noqa: E731
base = np.random.default_rng(0).integers(3, 290, size=40).tolist()
for name, ids in {"no control token": base, "guardian mid-sequence": base[:15] + [298] + base[15:],
                  "decision mid-sequence": base[:20] + [299] + base[20:], "two transitions": base[:10] + [298] + base[10:25] + [299] + base[25:]}.items():
    lg, ix = ref(ids)
    idx, _ = mm.model.route(mx.array([ids]))
    print(f"{name:24s} routing identical {np.array_equal(np.array(idx[0]), ix)}  max rel logit error {rel(lg, np.array(mm(mx.array([ids]))[0])):.1e}")
ids = base[:20] + [299] + base[20:]
lg, _ = ref(ids)
cache = make_prompt_cache(mm)
parts = [np.array(mm(mx.array([ids[a:b]]), cache=cache)[0]) for a, b in [(0, 12), (12, 27)] + [(i, i + 1) for i in range(27, len(ids))]]
print(f"chunked prefill + decode   max rel logit error {rel(lg, np.concatenate(parts)):.1e}")
trim_prompt_cache(cache, len(ids) - 15)
lg2, _ = ref(ids[:15] + base[15:30])
print(f"trim before control token  max rel logit error {rel(lg2[15:], np.array(mm(mx.array([base[15:30]]), cache=cache)[0])):.1e}")
prompt = base[:5] + [298] + base[5:30]
with torch.no_grad():
    g = hf.generate(torch.tensor([prompt]), max_new_tokens=20, do_sample=False, pad_token_id=0, eos_token_id=None)[0, len(prompt):].tolist()
m = [t for (t, _), _ in zip(generate_step(mx.array(prompt), mm, max_tokens=20, prefill_step_size=7), range(20))]
print("greedy generation identical:", g == m)
