from __future__ import annotations

import argparse
import unittest

from bench_multiturn import (
    SSEAccumulator,
    attach_dflash_profiles,
    dry_run_cell,
    extract_dflash_metrics,
    extract_server_timing_ms,
    parse_args,
)
from report import (
    build_report,
    hand_wave_model_ms,
    paired_records,
    summarize_paired_group,
)
from workload import build_conversations


class FakeTokenizer:
    bos_token_id = 1
    vocab_size = 100_000

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        # Stable enough for exact-length construction tests and always valid IDs.
        return [2 + sum(word.encode("utf-8")) % 90_000 for word in text.split()]


def make_conversation():
    return build_conversations(
        tokenizer=FakeTokenizer(),
        count=1,
        initial_tokens=64,
        maximum_tail_tokens=8,
        turns=4,
        namespace="unit",
        cell_id="append-c1-d8-r0",
        seed=7,
    )[0]


class WorkloadTest(unittest.TestCase):
    def test_benchmark_accepts_deferred_overlap_annotation(self):
        args = parse_args(["--tokenizer", "dummy", "--mode", "deferred-overlap"])
        self.assertEqual(args.mode, "deferred-overlap")

    def test_append_is_exact_and_reuses_previous_history(self):
        conversation = make_conversation()
        cold = conversation.prepare_turn(0, 8)
        self.assertEqual(len(cold.input_ids), 64)
        self.assertEqual(cold.expected_prefix_tokens, 0)
        conversation.commit_output([101, 102])

        warm = conversation.prepare_turn(1, 8)
        self.assertEqual(len(warm.input_ids), 64 + 2 + 8)
        self.assertEqual(warm.expected_prefix_tokens, 64 + 2 - 1)
        self.assertEqual(warm.phase, "warm")
        self.assertIn("c0:e0", warm.extra_key)

    def test_branch_discards_last_completed_turn(self):
        conversation = make_conversation()
        conversation.prepare_turn(0, 8)
        conversation.commit_output([101, 102])
        after_cold = list(conversation.history)
        conversation.prepare_turn(1, 8)
        conversation.commit_output([103, 104])

        branch = conversation.prepare_turn(
            2, 8, scenario="branch", event_turn=2, branch_depth=1
        )
        self.assertEqual(branch.phase, "branched-prefix")
        self.assertEqual(branch.input_ids[: len(after_cold)], after_cold)
        self.assertEqual(len(branch.input_ids), len(after_cold) + 8)

    def test_edit_changes_one_old_token_and_limits_expected_hit(self):
        conversation = make_conversation()
        conversation.prepare_turn(0, 8)
        conversation.commit_output([101, 102])
        before = list(conversation.history)
        edited = conversation.prepare_turn(
            1, 8, scenario="edit", event_turn=1, edit_distance=16
        )
        self.assertEqual(edited.phase, "edited-prefix")
        self.assertIsNotNone(edited.mutation_index)
        assert edited.mutation_index is not None
        self.assertNotEqual(
            edited.input_ids[edited.mutation_index], before[edited.mutation_index]
        )
        self.assertLess(edited.expected_prefix_tokens, len(before))

    def test_cache_miss_rotates_cache_namespace_only_once(self):
        conversation = make_conversation()
        cold = conversation.prepare_turn(0, 8)
        conversation.commit_output([101, 102])
        miss = conversation.prepare_turn(1, 8, scenario="cache-miss", event_turn=1)
        self.assertEqual(miss.phase, "forced-cache-miss")
        self.assertEqual(miss.expected_prefix_tokens, 0)
        self.assertNotEqual(cold.extra_key, miss.extra_key)
        conversation.commit_output([103, 104])
        warm = conversation.prepare_turn(2, 8, scenario="cache-miss", event_turn=1)
        self.assertEqual(warm.phase, "warm")
        self.assertEqual(warm.extra_key, miss.extra_key)

    def test_dry_run_generates_all_turns_with_exact_lengths(self):
        conversation = make_conversation()
        args = argparse.Namespace(
            turns=3,
            event_turn=2,
            branch_depth=1,
            edit_distance=16,
            output_tokens=2,
        )
        manifest = dry_run_cell(
            [conversation],
            {
                "cell_id": "append-c1-d8-r0",
                "scenario": "append",
                "concurrency": 1,
                "tail_tokens_configured": 8,
                "replicate": 0,
            },
            args,
        )
        self.assertEqual([row["input_tokens"] for row in manifest], [64, 74, 84])


class SSETest(unittest.TestCase):
    def test_cumulative_events_preserve_multi_token_wire_arrivals(self):
        accumulator = SSEAccumulator()
        accumulator.ingest(
            {
                "output_ids": [10, 11, 12],
                "text": "abc",
                "meta_info": {"completion_tokens": 3, "cached_tokens": 64},
            },
            20.0,
        )
        accumulator.ingest(
            {
                "output_ids": [10, 11, 12, 13],
                "text": "abcd",
                "meta_info": {"completion_tokens": 4},
            },
            25.0,
        )
        accumulator.finish(26.0)
        metrics = accumulator.metrics()
        self.assertEqual(accumulator.output_ids, [10, 11, 12, 13])
        self.assertEqual(accumulator.token_arrival_ms, [20.0, 20.0, 20.0, 25.0])
        self.assertEqual(metrics["ttft_ms"], 20.0)
        self.assertEqual(metrics["t2_ms"], 20.0)
        self.assertEqual(metrics["multi_token_sse_events"], 1)

    def test_incremental_events_reconstruct_output(self):
        accumulator = SSEAccumulator()
        accumulator.ingest(
            {"output_ids": [10], "meta_info": {"completion_tokens": 1}}, 5.0
        )
        accumulator.ingest(
            {"output_ids": [11, 12], "meta_info": {"completion_tokens": 3}}, 8.0
        )
        self.assertEqual(accumulator.output_ids, [10, 11, 12])
        self.assertEqual(accumulator.token_arrival_ms, [5.0, 8.0, 8.0])

    def test_future_dflash_fields_are_collected(self):
        extracted = extract_dflash_metrics(
            {
                "dflash_seed_total_ms": 12.5,
                "dflash_future_counter": 7,
                "car_ready": True,
                "cached_tokens": 100,
            }
        )
        self.assertEqual(extracted["dflash_future_counter"], 7)
        self.assertTrue(extracted["car_ready"])
        self.assertNotIn("cached_tokens", extracted)

    def test_server_timing_metadata_is_normalized_to_milliseconds(self):
        timing = extract_server_timing_ms(
            {
                "queue_time": 0.0125,
                "forward_entry_time": 1000.25,
                "prefill_finished_time": 1000.28,
            }
        )
        self.assertAlmostEqual(timing["server_queue_ms"], 12.5)
        self.assertAlmostEqual(timing["server_prefill_to_token_ms"], 30.0)
        self.assertEqual(
            extract_server_timing_ms({"queue_time": "bad"}),
            {"server_queue_ms": None, "server_prefill_to_token_ms": None},
        )


class ReportTest(unittest.TestCase):
    def test_wave_model_matches_reference_hand_calculation(self):
        # 16 x 8K requests, 2 per 16K wave: mean completion stage is 4.5.
        predicted = hand_wave_model_ms(8192, 16, 16384, 55.0, 0.0)
        self.assertAlmostEqual(predicted, 247.5)
        # 16 x 256 tokens fit one quarter-full wave.
        self.assertAlmostEqual(hand_wave_model_ms(256, 16, 16384, 55.0, 0.0), 13.75)

    def test_pairing_excludes_prompt_mismatch(self):
        def record(turn, prompt, output, ttft):
            return {
                "success": True,
                "cell_id": "append-c1-d8-r0",
                "conversation_id": 0,
                "turn": turn,
                "prompt_fingerprint": prompt,
                "output_fingerprint": output,
                "ttft_ms": ttft,
            }

        reference = {"requests": [record(0, "p0", "o0", 10), record(1, "p1", "o1", 12)]}
        candidate = {
            "requests": [record(0, "p0", "o0", 15), record(1, "changed", "o1", 20)]
        }
        pairs, diagnostics = paired_records(reference, candidate)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(diagnostics["prompt_mismatches"], 1)
        summary = summarize_paired_group(pairs)
        self.assertEqual(summary["p50_delta_ttft_ms"], 5.0)

    def test_profile_log_join_uses_request_id(self):
        records = [{"rid": "request-a", "dflash": {}}, {"rid": "request-b"}]
        profile = {
            "rids": ["request-a", "request-b"],
            "mode": "profile",
            "tp_rank": 0,
            "extend_tokens": 384,
            "extend_tokens_per_req": [128, 256],
            "prefix_tokens_per_req": [8192, 16384],
            "captured_hidden_bytes": 3840,
            "post_token_seed_ms": 9.0,
            "seed_kernel_ms": 8.0,
            "hidden_project_ms": 2.5,
            "draft_kv_write_ms": 5.5,
            "_server_log_line": 7,
        }
        slower_rank = {
            **profile,
            "tp_rank": 1,
            "captured_hidden_bytes": 4000,
            "post_token_seed_ms": 10.0,
            "hidden_project_ms": 3.0,
            "_server_log_line": 8,
        }
        profiles = [profile, slower_rank]
        join = attach_dflash_profiles(records, profiles)
        self.assertEqual(join["matched_request_profiles"], 2)
        self.assertEqual(join["profile_rank_records"], 2)
        self.assertEqual(join["profile_batches"], 1)
        self.assertEqual(records[0]["dflash"]["dflash_seed_tokens"], 128)
        # 7,840 aggregate hidden bytes apportioned by 256 / 384 extend rows.
        self.assertEqual(records[1]["dflash"]["dflash_capture_bytes"], 5227)
        self.assertEqual(records[1]["dflash"]["dflash_seed_total_ms"], 10.0)
        self.assertEqual(records[1]["dflash"]["dflash_seed_project_ms"], 3.0)
        self.assertEqual(records[1]["dflash"]["dflash_seed_kv_ms"], 5.5)

    def test_profile_join_sums_multiple_prefill_chunks(self):
        record = {"rid": "chunked-request"}

        def profile(rank, line, seed_ms):
            return {
                "rids": ["chunked-request"],
                "mode": "profile",
                "tp_rank": rank,
                "extend_tokens": 128,
                "extend_tokens_per_req": [128],
                "prefix_tokens_per_req": [0],
                "captured_hidden_bytes": 1000,
                "post_token_seed_ms": seed_ms,
                "_server_log_line": line,
            }

        # Same signature is emitted twice on both ranks (two prefill chunks).
        profiles = [
            profile(0, 1, 2.0),
            profile(0, 2, 3.0),
            profile(1, 3, 2.5),
            profile(1, 4, 3.5),
        ]
        join = attach_dflash_profiles([record], profiles)
        self.assertEqual(join["profile_batches"], 2)
        self.assertEqual(join["matched_request_profiles"], 2)
        self.assertEqual(record["dflash"]["dflash_profile_batches"], 2)
        self.assertEqual(record["dflash"]["dflash_seed_tokens"], 256)
        self.assertEqual(record["dflash"]["dflash_seed_total_ms"], 6.0)

    def test_report_builds_attribution_and_car_movement_sections(self):
        def run(label, mode, ttft, t2, e2e, seed=None):
            record = {
                "success": True,
                "cell_id": "append-c1-d256-r0",
                "scenario": "append",
                "concurrency": 1,
                "tail_tokens_configured": 256,
                "phase": "warm",
                "conversation_id": 0,
                "turn": 1,
                "prompt_fingerprint": "same-prompt",
                "output_fingerprint": "same-output",
                "ttft_ms": ttft,
                "t2_ms": t2,
                "first_itl_ms": t2 - ttft,
                "e2e_ms": e2e,
                "actual_cached_tokens": 8192,
                "expected_prefix_tokens": 8192,
                "uncached_tokens_proxy": 257,
                "dflash": {},
            }
            if seed is not None:
                record["dflash"]["dflash_seed_total_ms"] = seed
                record["dflash"]["dflash_seed_tokens"] = 257
            return {
                "_label": label,
                "_path": f"{label}.json",
                "run": {"mode": mode},
                "requests": [record],
            }

        runs = [
            run("baseline", "baseline", 20.0, 25.0, 200.0),
            run("profile", "profile", 35.0, 40.0, 210.0, seed=14.0),
            run("deferred", "deferred-serial", 22.0, 38.0, 209.0, seed=14.0),
            run("overlap", "deferred-overlap", 21.0, 34.0, 208.0, seed=14.0),
        ]
        args = argparse.Namespace(
            reference_label="baseline",
            reference_cold_tax_ms=260.0,
            reference_cold_tokens=8192.0,
            wave_budget_tokens=16384,
            wave_tax_ms=55.0,
            fixed_tax_ms=4.0,
        )
        report, rows = build_report(runs, args)
        self.assertIn("Direct bottleneck attribution", report)
        self.assertIn("CAR movement check", report)
        self.assertEqual(
            sum(row["row_type"] == "car_movement" for row in rows),
            2,
        )


if __name__ == "__main__":
    unittest.main()
