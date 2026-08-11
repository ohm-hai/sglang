"""CPU guards for the CAR-DFLASH prompt-seed experiment.

The deferred mode deliberately exposes the exact sampled token before draft
prompt KV materialization completes.  These tests pin the two-state ownership
contract without requiring CUDA: token egress may overtake the seed, while
cache publication, slot reuse, offload, and later work for the seed-owning
request may not.  deferred-overlap may run an independent batch.
"""

import argparse
import ast
import copy
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRT_DIR = _REPO_ROOT / "python" / "sglang" / "srt"
_SCHEDULE_BATCH = _SRT_DIR / "managers" / "schedule_batch.py"
_SCHEDULER = _SRT_DIR / "managers" / "scheduler.py"
_RESULT_PROCESSOR = (
    _SRT_DIR / "managers" / "scheduler_components" / "batch_result_processor.py"
)
_RESULT_TYPES = _SRT_DIR / "managers" / "utils.py"
_CACHE_COMMON = _SRT_DIR / "mem_cache" / "common.py"
_DFLASH_WORKER = _SRT_DIR / "speculative" / "dflash_worker_v2.py"
_DISAGG_PREFILL = _SRT_DIR / "disaggregation" / "prefill.py"


def _parse(path: Path) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(path.read_text(encoding="utf-8-sig"))


def _find_class(path: Path, class_name: str) -> ast.ClassDef:
    for node in _parse(path).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name!r} not found in {path}")


def _find_function(
    path: Path, function_name: str, class_name: str | None = None
) -> ast.FunctionDef:
    root: ast.AST = _parse(path)
    if class_name is not None:
        root = _find_class(path, class_name)
        candidates = root.body
    else:
        candidates = root.body
    found = [
        node
        for node in candidates
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if len(found) != 1:
        raise AssertionError(
            f"expected one {class_name or '<module>'}.{function_name} in {path}, "
            f"found {len(found)}"
        )
    return found[0]


def _compile_method(path: Path, class_name: str, method_name: str):
    """Compile a dependency-free method directly from the production AST."""
    method = copy.deepcopy(_find_function(path, method_name, class_name))
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def _call_name(call: ast.Call) -> str:
    def visit(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = visit(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""

    return visit(call.func)


def _calls(node: ast.AST, suffix: str) -> list[ast.Call]:
    return sorted(
        (
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and _call_name(child).endswith(suffix)
        ),
        key=lambda child: (child.lineno, child.col_offset),
    )


def _first_executable_statement(function: ast.FunctionDef) -> ast.stmt:
    body = function.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        raise AssertionError(f"{function.name} has no executable statements")
    return body[0]


def _assert_first_call(
    case: unittest.TestCase, function: ast.FunctionDef, expected_suffix: str
) -> None:
    statement = _first_executable_statement(function)
    case.assertIsInstance(statement, ast.Expr)
    case.assertIsInstance(statement.value, ast.Call)
    case.assertTrue(
        _call_name(statement.value).endswith(expected_suffix),
        f"{function.name} must begin with {expected_suffix}; got "
        f"{ast.unparse(statement)}",
    )


class TestDFlashCarServerArgs(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def _parse_cli(self, *extra: str):
        return self.parser.parse_args(["--model", "dummy", *extra])

    def test_seed_mode_default_and_valid_choices(self):
        self.assertEqual(self._parse_cli().speculative_dflash_seed_mode, "eager")
        for mode in ("eager", "profile", "deferred-serial", "deferred-overlap"):
            with self.subTest(mode=mode):
                args = self._parse_cli("--speculative-dflash-seed-mode", mode)
                self.assertEqual(args.speculative_dflash_seed_mode, mode)

    def test_invalid_seed_mode_is_rejected_by_cli(self):
        with self.assertRaises(SystemExit):
            self._parse_cli("--speculative-dflash-seed-mode", "unsafe")

    def test_invalid_seed_mode_is_rejected_by_dflash_handler(self):
        from sglang.srt.arg_groups.speculative_hook import _handle_dflash

        server_args = SimpleNamespace(
            device="cuda",
            enable_dp_attention=False,
            pp_size=1,
            speculative_draft_model_path="draft",
            speculative_dflash_seed_mode="unsafe",
            speculative_dflash_profile_every_n=1,
        )
        with self.assertRaisesRegex(ValueError, "must be one of"):
            _handle_dflash(server_args)

    def test_profile_interval_default_and_positive_value(self):
        self.assertEqual(self._parse_cli().speculative_dflash_profile_every_n, 1)
        args = self._parse_cli("--speculative-dflash-profile-every-n", "7")
        self.assertEqual(args.speculative_dflash_profile_every_n, 7)

    def test_nonpositive_profile_interval_is_rejected_by_dflash_handler(self):
        from sglang.srt.arg_groups.speculative_hook import _handle_dflash

        for interval in (0, -1):
            server_args = SimpleNamespace(
                device="cuda",
                enable_dp_attention=False,
                pp_size=1,
                speculative_draft_model_path="draft",
                speculative_dflash_seed_mode="eager",
                speculative_dflash_profile_every_n=interval,
            )
            with self.subTest(interval=interval), self.assertRaisesRegex(
                ValueError, "must be positive"
            ):
                _handle_dflash(server_args)

    def test_deferred_overlap_rejects_unaudited_scheduler_paths(self):
        from sglang.srt.arg_groups.speculative_hook import _handle_dflash

        defaults = {
            "device": "cuda",
            "enable_dp_attention": False,
            "pp_size": 1,
            "speculative_draft_model_path": "draft",
            "speculative_dflash_seed_mode": "deferred-overlap",
            "speculative_dflash_profile_every_n": 1,
            "disable_overlap_schedule": False,
            "enable_unified_memory": False,
            "disaggregation_mode": "null",
            "enable_mixed_chunk": False,
        }
        cases = (
            ("device", "npu", "requires CUDA/HIP"),
            ("disable_overlap_schedule", True, "requires overlap scheduling"),
            ("enable_unified_memory", True, "does not support unified memory"),
            ("disaggregation_mode", "prefill", "does not support PD"),
            ("enable_mixed_chunk", True, "does not support mixed"),
        )
        for field, value, message in cases:
            args = SimpleNamespace(**(defaults | {field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                _handle_dflash(args)


class _FakeEvent:
    def __init__(self, *, complete: bool, owner=None, replacement=None):
        self.complete = complete
        self.owner = owner
        self.replacement = replacement
        self.calls = []

    def query(self):
        self.calls.append("query")
        return self.complete

    def synchronize(self):
        self.calls.append("synchronize")
        if self.owner is not None:
            self.owner.dflash_pending_seed_event = self.replacement


class TestDFlashPendingSeedState(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.drain = staticmethod(
            _compile_method(_SCHEDULE_BATCH, "Req", "drain_pending_dflash_seed")
        )
        cls.clear = staticmethod(
            _compile_method(_SCHEDULE_BATCH, "Req", "clear_pending_dflash_seed")
        )

    def test_drain_is_noop_without_pending_seed(self):
        req = SimpleNamespace(dflash_pending_seed_event=None)
        self.drain(req)
        self.assertIsNone(req.dflash_pending_seed_event)

    def test_drain_clears_complete_event_without_host_sync(self):
        event = _FakeEvent(complete=True)
        req = SimpleNamespace(dflash_pending_seed_event=event)
        self.drain(req)
        self.assertEqual(event.calls, ["query"])
        self.assertIsNone(req.dflash_pending_seed_event)

    def test_drain_synchronizes_incomplete_event_before_clear(self):
        event = _FakeEvent(complete=False)
        req = SimpleNamespace(dflash_pending_seed_event=event)
        self.drain(req)
        self.assertEqual(event.calls, ["query", "synchronize"])
        self.assertIsNone(req.dflash_pending_seed_event)

    def test_drain_does_not_clear_a_newer_event(self):
        replacement = _FakeEvent(complete=True)
        req = SimpleNamespace(dflash_pending_seed_event=None)
        event = _FakeEvent(complete=False, owner=req, replacement=replacement)
        req.dflash_pending_seed_event = event
        self.drain(req)
        self.assertEqual(event.calls, ["query", "synchronize"])
        self.assertIs(req.dflash_pending_seed_event, replacement)

    def test_clear_uses_event_identity(self):
        old = _FakeEvent(complete=True)
        current = _FakeEvent(complete=True)
        req = SimpleNamespace(dflash_pending_seed_event=current)
        self.clear(req, old)
        self.assertIs(req.dflash_pending_seed_event, current)
        self.clear(req, current)
        self.assertIsNone(req.dflash_pending_seed_event)


class TestDFlashSeedOwnershipAst(CustomTestCase):
    def test_overlap_batch_waits_once_per_owned_seed_without_clearing(self):
        wait_for_seeds = _compile_method(
            _SCHEDULER,
            "Scheduler",
            "_wait_pending_dflash_seeds_for_batch",
        )

        class FakeStream:
            def __init__(self):
                self.waited = []

            def wait_event(self, event):
                self.waited.append(event)

        first = object()
        second = object()
        stream = FakeStream()
        scheduler = SimpleNamespace(forward_stream=stream)
        reqs = [
            SimpleNamespace(dflash_pending_seed_event=first),
            SimpleNamespace(dflash_pending_seed_event=first),
            SimpleNamespace(dflash_pending_seed_event=None),
            SimpleNamespace(dflash_pending_seed_event=second),
        ]

        wait_for_seeds(scheduler, SimpleNamespace(reqs=reqs))

        self.assertEqual(stream.waited, [first, second])
        self.assertEqual(
            [req.dflash_pending_seed_event for req in reqs],
            [first, first, None, second],
        )

    def test_deferred_overlap_batch_gate_precedes_model_forward(self):
        function = _find_function(_SCHEDULER, "run_batch", "Scheduler")
        gate_calls = _calls(function, "_wait_pending_dflash_seeds_for_batch")
        model_calls = _calls(function, "model_worker.forward_batch_generation")
        self.assertEqual(len(gate_calls), 1)
        self.assertGreaterEqual(len(model_calls), 1)
        self.assertLess(gate_calls[0].lineno, model_calls[0].lineno)

    def test_cache_publication_release_swa_and_offload_drain_first(self):
        cases = (
            (_CACHE_COMMON, None, "maybe_cache_unfinished_req"),
            (_CACHE_COMMON, None, "release_kv_cache"),
            (_CACHE_COMMON, None, "free_swa_out_of_window_slots"),
            (_SCHEDULE_BATCH, "Req", "offload_kv_cache"),
        )
        for path, class_name, function_name in cases:
            with self.subTest(path=path.name, function=function_name):
                function = _find_function(path, function_name, class_name)
                _assert_first_call(self, function, "drain_pending_dflash_seed")

    def test_retraction_releases_before_resetting_request_state(self):
        cases = (
            (_SCHEDULE_BATCH, None, "release_req"),
            (
                _DISAGG_PREFILL,
                "SchedulerDisaggregationPrefillMixin",
                "optimistic_release_and_requeue",
            ),
        )
        for path, class_name, function_name in cases:
            with self.subTest(path=path.name, function=function_name):
                function = _find_function(path, function_name, class_name)
                release = _calls(function, "release_kv_cache")
                reset = _calls(function, "reset_for_retract")
                self.assertEqual(len(release), 1)
                self.assertEqual(len(reset), 1)
                self.assertLess(release[0].lineno, reset[0].lineno)

    def test_prefill_streams_token_before_seed_gate_and_cache_actions(self):
        function = _find_function(
            _RESULT_PROCESSOR,
            "process_batch_result_prefill",
            "SchedulerBatchResultProcessor",
        )
        stream = _calls(function, "output_streamer.stream_output")
        seed_sync = [
            call
            for call in _calls(function, "synchronize")
            if ast.unparse(call.func.value) == "dflash_seed_gate"
        ]
        seed_clear = _calls(function, "clear_pending_dflash_seed")
        deferred_loops = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.For)
            and ast.unparse(node.iter) == "deferred_cache_actions"
        ]
        self.assertEqual(len(stream), 1)
        self.assertEqual(len(seed_sync), 1)
        self.assertEqual(len(seed_clear), 1)
        self.assertEqual(len(deferred_loops), 1)
        self.assertLess(stream[0].lineno, seed_sync[0].lineno)
        self.assertLess(seed_sync[0].lineno, seed_clear[0].lineno)
        self.assertLess(seed_clear[0].lineno, deferred_loops[0].lineno)
        self.assertTrue(_calls(deferred_loops[0], "release_kv_cache"))
        self.assertTrue(_calls(deferred_loops[0], "maybe_cache_unfinished_req"))

    def test_result_copy_waits_only_on_early_egress_event(self):
        function = _find_function(_SCHEDULER, "run_batch", "Scheduler")
        egress_branches = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and "dflash_egress_ready_event" in ast.unparse(node.test)
        ]
        self.assertEqual(
            len(egress_branches),
            2,
            "both overlap and non-overlap DFlash result-copy paths must be gated",
        )
        for branch in egress_branches:
            wait = [
                call
                for call in _calls(
                    ast.Module(body=branch.body, type_ignores=[]), "wait_event"
                )
                if len(call.args) == 1
                and ast.unparse(call.args[0])
                == "batch_result.dflash_egress_ready_event"
            ]
            copy_calls = _calls(
                ast.Module(body=branch.body, type_ignores=[]), "copy_to_cpu"
            )
            self.assertEqual(len(wait), 1)
            self.assertEqual(len(copy_calls), 1)
            self.assertLess(wait[0].lineno, copy_calls[0].lineno)

    def test_deferred_worker_snapshots_slot_ids_and_orders_seed(self):
        function = _find_function(
            _DFLASH_WORKER, "forward_batch_generation", "DFlashWorkerV2"
        )
        source = ast.get_source_segment(
            _DFLASH_WORKER.read_text(encoding="utf-8"), function
        )
        markers = (
            "token_ready_event = (",
            "self._car_ensure_distinct_seed_stream(forward_stream)",
            "seed_cache_loc = batch.out_cache_loc.clone()",
            "seed_input_ready_event = self._car_record_event(",
            "self._car_seed_stream.wait_event(seed_input_ready_event)",
            "target_hidden.record_stream(self._car_seed_stream)",
            "seed_cache_loc.record_stream(self._car_seed_stream)",
            "cache_loc=seed_cache_loc",
            "req.dflash_pending_seed_event = seed_ready_event",
            "forward_stream.wait_event(seed_ready_event)",
            "batch_output.dflash_egress_ready_event = (",
        )
        positions = []
        for marker in markers:
            pos = source.find(marker)
            self.assertNotEqual(pos, -1, f"missing deferred-seed marker: {marker}")
            positions.append(pos)
        self.assertEqual(positions, sorted(positions))

        egress_assignment = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "dflash_egress_ready_event"
                for target in node.targets
            )
        )
        self.assertIn("token_ready_event", ast.unparse(egress_assignment.value))
        self.assertNotIn("seed_ready_event", ast.unparse(egress_assignment.value))

        serial_wait_branches = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "self._car_seed_mode == 'deferred_serial'"
            and _calls(node, "forward_stream.wait_event")
        ]
        self.assertEqual(
            len(serial_wait_branches),
            1,
            "only deferred-serial may install the global seed barrier",
        )

    def test_eager_mode_short_circuits_profiling_and_sampling_is_periodic(self):
        function = _find_function(
            _DFLASH_WORKER, "forward_batch_generation", "DFlashWorkerV2"
        )
        extend_branch = next(
            node
            for node in function.body
            if isinstance(node, ast.If)
            and "forward_mode.is_extend()" in ast.unparse(node.test)
        )
        profile_assignment = next(
            node
            for node in extend_branch.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "profile_this_batch"
                for target in node.targets
            )
        )
        self.assertIsInstance(profile_assignment.value, ast.BoolOp)
        self.assertIsInstance(profile_assignment.value.op, ast.And)
        self.assertEqual(
            ast.unparse(profile_assignment.value.values[0]),
            "self._car_seed_mode != 'eager'",
        )
        self.assertEqual(
            ast.unparse(profile_assignment.value.values[1]),
            "self._car_should_profile_prefill()",
        )

        should_profile = _compile_method(
            _DFLASH_WORKER, "DFlashWorkerV2", "_car_should_profile_prefill"
        )
        worker = SimpleNamespace(
            _car_prefill_count=0,
            _car_seed_mode="profile",
            _car_profile_every_n=3,
        )
        self.assertEqual(
            [should_profile(worker) for _ in range(7)],
            [False, False, True, False, False, True, False],
        )
        worker = SimpleNamespace(
            _car_prefill_count=0,
            _car_seed_mode="eager",
            _car_profile_every_n=1,
        )
        self.assertFalse(should_profile(worker))

    def test_generation_result_carries_both_readiness_states(self):
        result_class = _find_class(_RESULT_TYPES, "GenerationBatchResult")
        fields = {
            node.target.id
            for node in result_class.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertTrue(
            {
                "dflash_egress_ready_event",
                "dflash_token_ready_event",
                "dflash_seed_input_ready_event",
                "dflash_seed_ready_event",
                "dflash_profile_metadata",
            }.issubset(fields)
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
