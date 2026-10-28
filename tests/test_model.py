"""Unit tests on a small random model (no download): routing, chunked prefill, cache trimming."""
import unittest

import mlx.core as mx
from mlx.utils import tree_map

import granite_switch_mlx

granite_switch_mlx.register()
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache  # noqa: E402

from granite_switch_mlx import model as gsm  # noqa: E402


def small_model():
    args = gsm.ModelArgs.from_dict({
        "model_type": "granite_switch", "vocab_size": 100, "hidden_size": 64, "shared_intermediate_size": 128,
        "num_hidden_layers": 4, "max_position_embeddings": 512, "num_attention_heads": 4, "num_key_value_heads": 2,
        "projection_head_dim": 16, "embedding_multiplier": 12.0, "attention_multiplier": 0.0625,
        "logits_scaling": 16.0, "residual_multiplier": 0.22, "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "num_adapters": 2,
        "adapter_ranks": [4, 8], "adapter_token_ids": [98, 99], "adapter_substitute_token_ids": [3, 4]})
    m = gsm.Model(args)
    mx.random.seed(0)
    m.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.2 if p.size else p, m.parameters()))
    return m


class TestGraniteSwitch(unittest.TestCase):
    def test_routing_and_exchange(self):
        m = small_model()
        idx, toks = m.model.route(mx.array([[5, 6, 7, 99, 8, 9]]))
        self.assertEqual(idx[0].tolist(), [0, 0, 0, 2, 2, 2])
        self.assertEqual(toks[0].tolist(), [5, 6, 7, 4, 8, 9])

    def test_prefix_runs_on_base(self):
        m = small_model()
        body = list(range(5, 25))
        plain = m(mx.array([body]))
        ctrl = m(mx.array([body[:8] + [98] + body[8:]]))
        self.assertTrue(mx.allclose(plain[:, :8], ctrl[:, :8], atol=1e-5))
        self.assertFalse(mx.allclose(plain[:, 8:], ctrl[:, 9:], atol=1e-3))

    def test_chunked_and_trim(self):
        m = small_model()
        ids = list(range(5, 11)) + [99] + list(range(11, 20))
        full = m(mx.array([ids]))
        cache = make_prompt_cache(m)
        parts = [m(mx.array([ids[:4]]), cache=cache), m(mx.array([ids[4:10]]), cache=cache)]
        parts += [m(mx.array([[t]]), cache=cache) for t in ids[10:]]
        self.assertTrue(mx.allclose(full, mx.concatenate(parts, axis=1), atol=1e-4))
        trim_prompt_cache(cache, len(ids) - 6)
        out = m(mx.array([[11, 12, 13]]), cache=cache)
        ref = m(mx.array([list(range(5, 11)) + [11, 12, 13]]))[:, 6:]
        self.assertTrue(mx.allclose(out, ref, atol=1e-4))


class TestBuildInfo(unittest.TestCase):
    def test_release_from_build_md(self):
        import tempfile
        from pathlib import Path
        row = "| 2 | decision | alora | `<\\|decision\\|>` | 100353 | 64 | {a} | down_proj, q_proj | {src} |"
        for src, a, rel in (("decision-4.1.1", 64, "4.1.1"), ("alora", 128, "4.1")):
            with tempfile.TemporaryDirectory() as d:
                Path(d, "BUILD.md").write_text("| # | Name | Technology | Control Token | Token ID | Rank | Alpha | Target Modules | Source |\n"
                                               "|---|---|---|---|---|---|---|---|---|\n"
                                               "| 1 | guardian-core | alora | `<\\|guardian-core\\|>` | 100352 | 16 | 64 | q_proj | ibm-granite/granitelib-guardian-r1.0 |\n"
                                               + row.format(a=a, src=src) + "\n")
                info = granite_switch_mlx.build_info(d)
                self.assertEqual(info["release"], rel)
                self.assertEqual((info["adapters"]["decision"]["rank"], info["adapters"]["decision"]["alpha"]), (64, a))
                self.assertEqual(info["adapters"]["guardian-core"]["rank"], 16)


if __name__ == "__main__":
    unittest.main()
