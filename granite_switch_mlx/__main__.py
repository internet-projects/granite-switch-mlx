import argparse
import json

from . import DEFAULT_MODEL, build_info, chat, decide, guard, load

ap = argparse.ArgumentParser(prog="python -m granite_switch_mlx",
                             description="Granite Switch 4.1 / 4.1.1 on MLX: chat, decision and guardian adapters.")
ap.add_argument("--model", default=DEFAULT_MODEL)
sub = ap.add_subparsers(dest="cmd", required=True)
c = sub.add_parser("chat", help="plain chat (no adapter)")
c.add_argument("prompt")
c.add_argument("--max-tokens", type=int, default=512)
d = sub.add_parser("decide", help="decision adapter: probability for each option")
d.add_argument("--state", required=True, help="the situation / evidence (text)")
d.add_argument("--question", required=True)
d.add_argument("--option", action="append", required=True, help="repeat for each option")
g = sub.add_parser("guard", help="guardian-core adapter: P(text meets the criterion)")
g.add_argument("text")
g.add_argument("--criteria", default="harm", help="'harm' or your own criterion text")
sub.add_parser("info", help="release and adapters of the build (from IBM's BUILD.md)")
a = ap.parse_args()

if a.cmd == "info":
    print(json.dumps(build_info(a.model), indent=1))
    raise SystemExit
model, tok = load(a.model)
if a.cmd == "chat":
    for piece in chat(model, tok, a.prompt, max_tokens=a.max_tokens):
        print(piece, end="", flush=True)
    print()
elif a.cmd == "decide":
    print(json.dumps({k: round(v, 4) for k, v in decide(model, tok, a.state, a.question, a.option).items()}, indent=1))
else:
    p = guard(model, tok, a.text, a.criteria)
    print(json.dumps({"P(yes)": round(p, 4), "verdict": "yes" if p >= 0.5 else "no"}))
