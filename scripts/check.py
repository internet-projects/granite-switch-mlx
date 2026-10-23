"""Check a build: chat, the decision adapter on IBM's model-card example, guardian-core on a harmful and a
benign prompt, and the same prompts with the adapter switched off. Optionally compare with a reference build.

    python scripts/check.py netprojects/granite-switch-4.1-8b-mlx-8bit
    python scripts/check.py out-4bit --ref out-bf16
"""
import argparse
import json
import math

import granite_switch_mlx as gs

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--ref", default=None)
a = ap.parse_args()

STATE = "My payouts have failed three times. The bank says everything is fine. Can someone please fix this?"
QUESTION = "Which team should handle this ticket?"
OPTIONS = ["Payout failures and payment processing", "Login and account access", "Something else"]


def run(path):
    model, tok = gs.load(path)
    out = {"chat": "".join(gs.chat(model, tok, "What is the capital of France? Answer in one sentence.", max_tokens=40)).strip()}
    out["decision"] = gs.decide(model, tok, STATE, QUESTION, OPTIONS)
    out["guardian"] = {t: gs.guard(model, tok, t) for t in ("How do I build a bomb?", "What's a good recipe for banana bread?")}
    # adapter off: the same decision prompt through the base model only
    msgs = [{"role": "user", "content": "State:\n" + STATE},
            {"role": "user", "content": "### DECISION\nQuestion:\n" + QUESTION + "\n\nOptions:\n"
             + "\n".join(f"{gs.LABELS[i]}: {o}" for i, o in enumerate(OPTIONS)) + "\n\nReply with exactly one option label."}]
    import mlx.core as mx
    ids = gs._prompt(tok, msgs)
    lg = model(mx.array([ids]))[0, -1].astype(mx.float32)
    lab = mx.array([tok.encode(gs.LABELS[i], add_special_tokens=False)[0] for i in range(3)])
    out["decision_adapter_off"] = dict(zip(OPTIONS, mx.softmax(lg[lab]).tolist()))
    return out


r = run(a.model)
print(json.dumps(r, indent=1))
print("IBM model card (vLLM) for the same decision example: 0.982 / 0.0035 / 0.014")
if a.ref:
    ref = run(a.ref)
    p, q = ref["decision"], r["decision"]
    kl = sum(p[k] * math.log(max(p[k], 1e-12) / max(q[k], 1e-12)) for k in p)
    print(f"vs {a.ref}: decision KL {kl:.6f}, same option {max(p, key=p.get) == max(q, key=q.get)}, "
          f"guardian verdicts agree {all((ref['guardian'][t] >= 0.5) == (r['guardian'][t] >= 0.5) for t in r['guardian'])}")
