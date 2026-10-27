import unittest
from types import SimpleNamespace

from mpt_fdp_protocol_v1 import (
    bind_mpt_eval_context_limit,
    dense_cache_bytes,
    derive_three_component_profile,
    predicted_cache_bytes,
    select_last_passing_before_first_failure,
    strict_threshold_grid,
    validate_mpt_config,
    validate_rank_pair,
    validate_rank_schedule,
    run_mpt_strict_layer_search,
)


class FakeTensor:
    def __init__(self, shape, element_size=2, dtype="torch.bfloat16", device="cpu"):
        self.shape = tuple(shape)
        self._element_size = element_size
        self.dtype = dtype
        self.device = device

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def element_size(self):
        return self._element_size


class MptFdpProtocolTests(unittest.TestCase):
    def test_config_is_identity_checked_by_fields(self):
        c = {
            "model_type": "mpt", "n_layers": 32, "d_model": 4096,
            "n_heads": 32, "max_seq_len": 2048,
            "attn_config": {"alibi": True, "attn_type": "multihead_attention"},
        }
        self.assertEqual(validate_mpt_config(c)["head_dim"], 128)
        c["model_type"] = "llama"
        with self.assertRaises(ValueError):
            validate_mpt_config(c)

    def test_hflm_context_limit_binds_mpt_max_seq_len(self):
        cfg = SimpleNamespace(
            model_type="mpt", n_layers=32, d_model=4096, n_heads=32,
            max_seq_len=2048,
            attn_config={"alibi": True, "attn_type": "multihead_attention"},
        )
        class FakeHFLM:
            def __init__(self):
                self.model = SimpleNamespace(config=cfg)
                self._max_length = None
            @property
            def max_length(self):
                return self._max_length
        lm = FakeHFLM()
        self.assertEqual(bind_mpt_eval_context_limit(lm), 2048)
        self.assertEqual(lm.max_length, 2048)
        lm._max_length = 4096
        with self.assertRaises(ValueError):
            bind_mpt_eval_context_limit(lm)

    def test_dense_kv_accounting_and_full_rank_match(self):
        full = [[4096, 4096] for _ in range(32)]
        b = dense_cache_bytes()
        self.assertEqual(b, 1_073_741_824)
        self.assertEqual(predicted_cache_bytes(full), b)

    def test_legal_rank_lattice(self):
        validate_rank_pair(64, 4096)
        with self.assertRaises(ValueError):
            validate_rank_pair(65, 4096)
        with self.assertRaises(ValueError):
            validate_rank_schedule([[64, 64]] * 31)

    def test_three_component_raw_sum_then_one_minmax_and_epsilon_direction(self):
        profile = {"per_layer": {
            str(i): {
                "Z_rel_mse": float(i),
                "score_rel_mse_causal": 0.0,
                "entropy_diff_A": 0.0,
            } for i in range(32)
        }}
        out = derive_three_component_profile(profile, epsilon_base=0.06)
        rows = out["per_layer"]
        self.assertEqual(rows[0]["q"], 0.0)
        self.assertEqual(rows[-1]["q"], 1.0)
        self.assertLess(rows[-1]["epsilon_raw"], rows[0]["epsilon_raw"])
        self.assertEqual(rows[0]["epsilon_final"], 0.06)
        self.assertAlmostEqual(out["mean_epsilon_before_layer0_override"], 0.06)
        self.assertEqual(rows[0]["entropy_penalty"], 0.0)

    def test_strict_grid_and_first_failure(self):
        grid = strict_threshold_grid()
        self.assertEqual(len(grid), 36)
        self.assertEqual(grid[0], 1.0)
        self.assertAlmostEqual(grid[-1], 0.3)
        trials = [
            {"threshold": 1.0, "relative_error": 0.01, "rank_kq": 4096, "rank_v": 4096},
            {"threshold": 0.98, "relative_error": 0.02, "rank_kq": 4000, "rank_v": 4000},
            {"threshold": 0.96, "relative_error": 0.061, "rank_kq": 3904, "rank_v": 3904},
        ]
        selected = select_last_passing_before_first_failure(trials, epsilon=0.06)
        self.assertEqual(selected["selected"]["threshold"], 0.98)
        self.assertTrue(selected["stopped_at_first_failure"])
        with self.assertRaises(ValueError):
            select_last_passing_before_first_failure(trials + [
                {"threshold": 0.94, "relative_error": 0.03}
            ], epsilon=0.06)

    def test_actual_cache_census_counts_tensor_storage_not_serialized_bytes(self):
        schedule = [[64, 96] for _ in range(32)]
        cache = []
        for _ in range(32):
            cache.append((
                FakeTensor((1, 32, 2048, 2)),
                FakeTensor((1, 32, 2048, 3)),
            ))
        from mpt_fdp_protocol_v1 import actual_cache_census
        report = actual_cache_census(cache, schedule)
        self.assertEqual(report["status"], "ACTUAL_TENSOR_CENSUS_PASS")
        self.assertEqual(report["b_live_bytes"], predicted_cache_bytes(schedule))
        self.assertEqual(report["dtypes"], ["torch.bfloat16"])

    def test_mpt_strict_search_uses_same_inputs_and_stops_at_first_failure(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable in local lightweight Python")
        x = torch.ones((1, 2, 4096))
        calls = []
        binding = {
            "model_type": "mpt", "attention_impl": "mpt_native", "alibi": True,
            "same_checkpoint_for_dense_and_compressed": True,
            "same_position_bias_and_mask_for_all_candidates": True,
        }

        def build(threshold):
            calls.append(threshold)
            return {
                "rank_kq": 4096 - round((1.0 - threshold) * 4096 / 32) * 32,
                "rank_v": 4096 - round((1.0 - threshold) * 4096 / 32) * 32,
                "candidate": threshold,
            }

        def dense_forward(h, pos, mask):
            self.assertIs(pos, position_bias)
            self.assertIs(mask, attention_mask)
            return h

        def compressed_forward(candidate, h, pos, mask):
            self.assertIs(pos, position_bias)
            self.assertIs(mask, attention_mask)
            return h if candidate == 1.0 else torch.zeros_like(h)

        position_bias, attention_mask = object(), object()
        result = run_mpt_strict_layer_search(
            epsilon=0.06, hidden_states=[x], sample_ids=["s0"],
            sample_input_sha256=["a" * 64], position_bias=position_bias,
            attention_mask=attention_mask, operator_binding=binding,
            build_candidate=build, dense_forward=dense_forward,
            compressed_forward=compressed_forward,
        )
        self.assertEqual(calls, [1.0, 0.98])
        self.assertEqual(result["selected"]["threshold"], 1.0)
        self.assertTrue(result["stopped_at_first_failure"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
