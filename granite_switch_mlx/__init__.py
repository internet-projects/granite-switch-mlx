"""granite-switch-mlx: run IBM Granite Switch 4.1 / 4.1.1 (chat + decision + guardian adapters) on Apple Silicon with MLX."""
import math
import sys
from pathlib import Path

__version__ = "0.2.0"

DEFAULT_MODEL = "netprojects/granite-switch-4.1-8b-mlx-8bit"
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GUARDIAN_INSTRUCTION = (
    "<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging "
    "criteria, utilizing all available information, including conversations, documents, and tools.\n\n"
    "### Criteria: {criteria}\n\n### Scoring Schema: {scoring_schema}"
)
GUARDIAN_CRITERIA = {
    "harm": "Texts deemed universally harmful are those that pose harm across all contexts and audiences.",
}
GUARDIAN_SCHEMA = "If the last user's text meets the criteria, return 'yes'; otherwise, return 'no'."


def register():
    """Make mlx-lm find the `granite_switch` architecture (until it ships in mlx-lm itself)."""
    from . import model as _model
    import mlx_lm.models.cache as _cache

    sys.modules.setdefault("mlx_lm.models.granite_switch", _model)
    if not hasattr(_cache, "SwitchCache"):  # lets mlx-lm save/load prompt caches for this model
        _cache.SwitchCache = _model.SwitchCache
    return _model


def build_info(path_or_repo=DEFAULT_MODEL):
    """The adapters a build carries, from IBM's BUILD.md: {"release": "4.1.1" | "4.1", "adapters": {name: {...}}}.

    Granite Switch 4.1.1 (IBM, 27 October 2026) ships new decision weights (source `decision-4.1.1`); the
    architecture, control tokens, prompt recipe and guardian-core adapter are those of 4.1, so both load the same way.
    """
    p = Path(str(path_or_repo))
    if not p.exists():
        from huggingface_hub import hf_hub_download

        p = Path(hf_hub_download(str(path_or_repo), "BUILD.md")).parent
    adapters = {}
    for line in (p / "BUILD.md").read_text().splitlines():
        cells = [c.strip() for c in line.replace("\\|", "/").strip().strip("|").split("|")]
        if len(cells) >= 9 and cells[0].isdigit():
            adapters[cells[1]] = {"technology": cells[2], "token_id": int(cells[4]), "rank": int(cells[5]),
                                  "alpha": int(cells[6]), "source": cells[8]}
    src = adapters.get("decision", {}).get("source", "")
    return {"release": "4.1.1" if "4.1.1" in src else "4.1", "adapters": adapters}


def load(path_or_repo=DEFAULT_MODEL):
    """Load a Granite Switch MLX build (local folder or Hugging Face repo id). Returns (model, tokenizer)."""
    register()
    from mlx_lm import load as _load

    return _load(str(path_or_repo))


def _prompt(tokenizer, messages, adapter=None):
    kw = {"adapter_name": adapter} if adapter else {}
    s = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)
    return tokenizer.encode(s, add_special_tokens=False)


def chat(model, tokenizer, messages, max_tokens=512, temp=0.0):
    """Plain Granite 4.1 chat (no adapter; the base model is the same in 4.1 and 4.1.1). Yields text pieces."""
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    for r in stream_generate(model, tokenizer, _prompt(tokenizer, messages), max_tokens=max_tokens,
                             sampler=make_sampler(temp=temp)):
        yield r.text


def decide(model, tokenizer, state, question, options):
    """The `decision` adapter, IBM's documented recipe: a State turn, then a ### DECISION turn with
    single-token option labels; returns {option: probability} from the softmax over the label logits.

    state: text (or a dict, rendered as "KEY:\\nvalue" blocks). options: list of option texts, or a dict
    {name: description} (names are returned as keys).
    """
    import mlx.core as mx

    if isinstance(state, dict):
        state = "\n\n".join(f"{k.upper()}:\n{v}" for k, v in state.items())
    if isinstance(options, dict):
        names = list(options)
        texts = [f"{k} - {v}" for k, v in options.items()]
    else:
        names = texts = list(options)
    if len(names) > len(LABELS):
        raise ValueError(f"at most {len(LABELS)} options")
    listing = "\n".join(f"{LABELS[i]}: {t}" for i, t in enumerate(texts))
    messages = [
        {"role": "user", "content": "State:\n" + state},
        {"role": "user", "content": f"### DECISION\nQuestion:\n{question}\n\nOptions:\n{listing}\n\n"
                                    "Reply with exactly one option label."},
    ]
    ids = _prompt(tokenizer, messages, "decision")
    label_ids = mx.array([tokenizer.encode(LABELS[i], add_special_tokens=False)[0] for i in range(len(names))])
    logits = model(mx.array([ids]))[0, -1].astype(mx.float32)
    probs = mx.softmax(logits[label_ids]).tolist()
    return dict(zip(names, probs))


def guard(model, tokenizer, text, criteria="harm", scoring_schema=GUARDIAN_SCHEMA):
    """The `guardian-core` adapter: probability that the last user text meets `criteria`
    (a key of GUARDIAN_CRITERIA or your own criterion text). Returns P(yes)."""
    import mlx.core as mx

    crit = GUARDIAN_CRITERIA.get(criteria, criteria)
    messages = [{"role": "user", "content": text},
                {"role": "user", "content": GUARDIAN_INSTRUCTION.format(criteria=crit, scoring_schema=scoring_schema)}]
    prefix = '{"score": "'
    pre = tokenizer.encode(prefix, add_special_tokens=False)
    yes = tokenizer.encode(prefix + "yes", add_special_tokens=False)[len(pre)]
    no = tokenizer.encode(prefix + "no", add_special_tokens=False)[len(pre)]
    logits = model(mx.array([_prompt(tokenizer, messages, "guardian-core") + pre]))[0, -1].astype(mx.float32)
    return 1 / (1 + math.exp(logits[no].item() - logits[yes].item()))
