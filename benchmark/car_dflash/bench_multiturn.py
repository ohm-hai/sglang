#!/usr/bin/env python3
"""Measure DFlash prompt-seed barriers with an exact multi-turn chat workload.

This intentionally uses SGLang's native streaming ``/generate`` endpoint.  Every
SSE arrival is timestamped, including events that deliver several accepted tokens
at once.  Run it once per server configuration (baseline, eager DFlash, CAR modes),
then use ``report.py`` for paired comparisons.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from workload import (  # type: ignore[no-redef]
        Conversation,
        PreparedTurn,
        build_conversations,
        load_tokenizer,
        token_fingerprint,
    )
else:
    from .workload import (
        Conversation,
        PreparedTurn,
        build_conversations,
        load_tokenizer,
        token_fingerprint,
    )


DFLASH_COUNTERS = {
    "dflash_car_mode",
    "dflash_seed_tokens",
    "dflash_prefix_tokens",
    "dflash_capture_bytes",
    "dflash_seed_project_ms",
    "dflash_seed_kv_ms",
    "dflash_seed_kernel_ms",
    "dflash_seed_input_snapshot_ms",
    "dflash_seed_total_ms",
    "dflash_seed_queue_ms",
    "dflash_seed_ready_lag_tokens",
    "dflash_fallback_tokens",
    "dflash_seed_fused",
    "dflash_target_to_token_ms",
    "dflash_target_plus_seed_ms",
    "dflash_batch_extend_tokens",
    "dflash_batch_prefix_tokens",
    "dflash_profile_log_line",
    "dflash_profile_tp_ranks",
    "dflash_profile_batches",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def extract_dflash_metrics(meta_info: dict[str, Any]) -> dict[str, Any]:
    """Preserve known and future DFlash/CAR fields without requiring a schema bump."""

    return {
        key: value
        for key, value in meta_info.items()
        if key in DFLASH_COUNTERS
        or key.startswith("dflash_")
        or key.startswith("car_dflash_")
        or key.startswith("car_")
    }


def extract_server_timing_ms(meta_info: dict[str, Any]) -> dict[str, float | None]:
    """Normalize optional SGLang scheduler timing metadata to milliseconds.

    ``queue_time`` is emitted in seconds.  ``forward_entry_time`` and
    ``prefill_finished_time`` are wall-clock timestamps, so their difference is
    the scheduler-observed prefill-to-first-token span.  These fields require an
    SGLang server launched with ``--enable-metrics``; missing or malformed values
    intentionally remain ``None`` rather than being inferred from client timing.
    """

    def number(name: str) -> float | None:
        raw = meta_info.get(name)
        if raw is None or isinstance(raw, bool):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    queue_seconds = number("queue_time")
    forward_entry = number("forward_entry_time")
    prefill_finished = number("prefill_finished_time")
    server_prefill_ms = None
    if forward_entry is not None and prefill_finished is not None:
        server_prefill_ms = max(0.0, (prefill_finished - forward_entry) * 1_000.0)
    return {
        "server_queue_ms": (
            None if queue_seconds is None else max(0.0, queue_seconds * 1_000.0)
        ),
        "server_prefill_to_token_ms": server_prefill_ms,
    }


@dataclass
class SSEAccumulator:
    """Convert cumulative or incremental SGLang events into wire-level timings."""

    retain_payloads: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)
    token_arrival_ms: list[float] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    final_meta_info: dict[str, Any] = field(default_factory=dict)
    previous_completion_tokens: int = 0
    done_ms: float | None = None

    def ingest(self, payload: dict[str, Any], arrival_ms: float) -> None:
        meta = payload.get("meta_info") or {}
        if isinstance(meta, dict):
            self.final_meta_info.update(meta)

        raw_ids = payload.get("output_ids") or []
        ids = [int(token_id) for token_id in raw_ids]
        completion = meta.get("completion_tokens")
        try:
            cumulative_count = int(completion)
        except (TypeError, ValueError):
            cumulative_count = self.previous_completion_tokens
            if ids:
                cumulative_count = max(cumulative_count, len(ids))

        delta = max(0, cumulative_count - self.previous_completion_tokens)
        if ids:
            if len(ids) == cumulative_count:
                # Default native streaming returns the cumulative ID prefix.
                self.output_ids = ids
            elif delta and len(ids) == delta:
                # --incremental-streaming-output returns only this event's IDs.
                self.output_ids.extend(ids)
            elif len(ids) > len(self.output_ids):
                self.output_ids = ids
            elif delta:
                self.output_ids.extend(ids[-delta:])

        # A speculative response may deliver several tokens in one SSE event.  Their
        # only observable wire timestamp is the same; do not invent device timestamps.
        self.token_arrival_ms.extend([arrival_ms] * delta)
        event: dict[str, Any] = {
            "arrival_ms": arrival_ms,
            "completion_tokens": cumulative_count,
            "new_tokens": delta,
        }
        if self.retain_payloads:
            event["payload"] = payload
        else:
            event["meta_info"] = meta
            event["output_ids_count"] = len(ids)
            event["text_chars"] = len(payload.get("text") or "")
        self.events.append(event)
        self.previous_completion_tokens = max(
            self.previous_completion_tokens, cumulative_count
        )

    def finish(self, done_ms: float) -> None:
        self.done_ms = done_ms

    def metrics(self) -> dict[str, Any]:
        arrivals = self.token_arrival_ms
        ttft = arrivals[0] if arrivals else None
        t2 = arrivals[1] if len(arrivals) >= 2 else None
        itls = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
        return {
            "ttft_ms": ttft,
            "t2_ms": t2,
            "first_itl_ms": None if t2 is None or ttft is None else t2 - ttft,
            "mean_itl_ms": statistics.fmean(itls) if itls else None,
            "p50_itl_ms": percentile(itls, 0.50),
            "p95_itl_ms": percentile(itls, 0.95),
            "e2e_ms": arrivals[-1] if arrivals else self.done_ms,
            "http_done_ms": self.done_ms,
            "completion_tokens": max(
                self.previous_completion_tokens, len(self.output_ids)
            ),
            "sse_token_events": len([e for e in self.events if e["new_tokens"]]),
            "multi_token_sse_events": len(
                [e for e in self.events if e["new_tokens"] > 1]
            ),
        }


async def _read_sse_response(
    response: Any, retain_payloads: bool, request_start_ns: int
) -> SSEAccumulator:
    """Read events with timestamps relative to the original HTTP request start."""

    accumulator = SSEAccumulator(retain_payloads=retain_payloads)
    while True:
        raw_line = await response.content.readline()
        if not raw_line:
            break
        line = raw_line.strip()
        if not line or line.startswith(b":"):
            continue
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        now_ms = (time.perf_counter_ns() - request_start_ns) / 1_000_000.0
        if data == b"[DONE]":
            accumulator.finish(now_ms)
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid SSE JSON: {data[:200]!r}") from exc
        accumulator.ingest(payload, now_ms)
    if accumulator.done_ms is None:
        accumulator.finish((time.perf_counter_ns() - request_start_ns) / 1_000_000.0)
    return accumulator


def build_payload(
    prepared: PreparedTurn, cell: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "rid": (
            f"car-{args.cache_namespace}-{cell['cell_id']}-"
            f"c{prepared.conversation_id}-t{prepared.turn}"
        ),
        "session_id": prepared.routing_key,
        "conversation_id": prepared.routing_key,
        "routing_key": prepared.routing_key,
        "extra_key": prepared.extra_key,
        "input_ids": prepared.input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": args.output_tokens,
            "ignore_eos": True,
            "skip_special_tokens": False,
            "no_stop_trim": True,
            # Required for meaningful TTFT/T2 measurements.  The accumulator still
            # records multi-token arrivals because speculative verification can emit
            # an accepted block even with a one-token stream interval.
            "stream_interval": 1,
        },
        "stream": True,
        "return_logprob": False,
    }


async def send_request(
    session: Any,
    prepared: PreparedTurn,
    cell: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = build_payload(prepared, cell, args)
    headers = {"X-SMG-Routing-Key": prepared.routing_key}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    request_wall_time = utc_now()
    request_start_ns = time.perf_counter_ns()
    try:
        async with session.post(
            f"{args.base_url.rstrip('/')}/generate",
            json=payload,
            headers=headers,
        ) as response:
            headers_ms = (time.perf_counter_ns() - request_start_ns) / 1_000_000.0
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")
            accumulator = await _read_sse_response(
                response,
                retain_payloads=args.retain_sse_payloads,
                request_start_ns=request_start_ns,
            )
            metrics = accumulator.metrics()
            meta_info = accumulator.final_meta_info
            server_timing = extract_server_timing_ms(meta_info)
            cached_tokens = int(meta_info.get("cached_tokens") or 0)
            prompt_tokens = int(
                meta_info.get("prompt_tokens") or len(prepared.input_ids)
            )
            seed_proxy = max(0, prompt_tokens - cached_tokens)
            output_ids = accumulator.output_ids
            if not output_ids:
                raise RuntimeError(
                    "stream completed without output_ids; use native /generate and "
                    "do not disable token IDs"
                )

            return {
                "success": True,
                "error": None,
                **cell,
                "conversation_id": prepared.conversation_id,
                "rid": payload["rid"],
                "turn": prepared.turn,
                "phase": prepared.phase,
                "request_wall_time": request_wall_time,
                "input_tokens": len(prepared.input_ids),
                "tail_tokens": prepared.tail_tokens,
                "expected_prefix_tokens": prepared.expected_prefix_tokens,
                "actual_cached_tokens": cached_tokens,
                "cache_hit_error_tokens": cached_tokens
                - prepared.expected_prefix_tokens,
                "uncached_tokens_proxy": seed_proxy,
                "mutation_index": prepared.mutation_index,
                "prompt_fingerprint": prepared.prompt_fingerprint,
                "output_fingerprint": token_fingerprint(output_ids),
                "output_ids": output_ids,
                "headers_ms": headers_ms,
                **server_timing,
                **metrics,
                "meta_info": meta_info,
                "dflash": extract_dflash_metrics(meta_info),
                "token_arrival_ms": accumulator.token_arrival_ms,
                "sse_events": accumulator.events,
            }
    except Exception as exc:  # noqa: BLE001 - errors belong in the raw result
        elapsed_ms = (time.perf_counter_ns() - request_start_ns) / 1_000_000.0
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            **cell,
            "conversation_id": prepared.conversation_id,
            "rid": payload["rid"],
            "turn": prepared.turn,
            "phase": prepared.phase,
            "request_wall_time": request_wall_time,
            "input_tokens": len(prepared.input_ids),
            "tail_tokens": prepared.tail_tokens,
            "expected_prefix_tokens": prepared.expected_prefix_tokens,
            "prompt_fingerprint": prepared.prompt_fingerprint,
            "http_done_ms": elapsed_ms,
        }


async def fetch_json(session: Any, url: str, headers: dict[str, str]) -> Any:
    try:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                return {
                    "error": f"HTTP {response.status}",
                    "body": (await response.text())[:500],
                }
            return await response.json(content_type=None)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


async def flush_cache(session: Any, args: argparse.Namespace) -> None:
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    url = f"{args.base_url.rstrip('/')}/flush_cache?timeout={args.flush_timeout}"
    async with session.post(url, headers=headers) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(
                f"cache flush failed ({response.status}): {body.strip()}"
            )


def cell_identifier(
    scenario: str, concurrency: int, tail_tokens: int, replicate: int
) -> str:
    return f"{scenario}-c{concurrency}-d{tail_tokens}-r{replicate}"


def dry_run_cell(
    conversations: list[Conversation],
    cell: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    manifest = []
    for turn in range(args.turns):
        for conversation in conversations:
            prepared = conversation.prepare_turn(
                turn=turn,
                tail_tokens=cell["tail_tokens_configured"],
                scenario=cell["scenario"],
                event_turn=args.event_turn,
                branch_depth=args.branch_depth,
                edit_distance=args.edit_distance,
            )
            manifest.append(
                {
                    **cell,
                    "conversation_id": prepared.conversation_id,
                    "turn": prepared.turn,
                    "phase": prepared.phase,
                    "input_tokens": len(prepared.input_ids),
                    "tail_tokens": prepared.tail_tokens,
                    "expected_prefix_tokens": prepared.expected_prefix_tokens,
                    "mutation_index": prepared.mutation_index,
                    "extra_key": prepared.extra_key,
                    "routing_key": prepared.routing_key,
                    "prompt_fingerprint": prepared.prompt_fingerprint,
                }
            )
            # A deterministic stand-in keeps later dry-run prompt lengths realistic.
            source = prepared.input_ids[-max(1, args.output_tokens) :]
            dummy_output = (source * (args.output_tokens // len(source) + 1))[
                : args.output_tokens
            ]
            conversation.commit_output(dummy_output)
    return manifest


async def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = load_tokenizer(args.tokenizer, args.trust_remote_code)
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency) + 8)
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    run_info = {
        "schema_version": 1,
        "label": args.label,
        "mode": args.mode,
        "cache_namespace": args.cache_namespace,
        "base_url": args.base_url,
        "tokenizer": args.tokenizer,
        "started_at": utc_now(),
        "argv": sys.argv,
        "configuration": {
            "initial_tokens": args.initial_tokens,
            "tail_tokens": args.tail_tokens,
            "output_tokens": args.output_tokens,
            "turns": args.turns,
            "concurrency": args.concurrency,
            "scenarios": args.scenarios,
            "replicates": args.replicates,
            "round_barrier": True,
            "sticky_routing": True,
        },
    }
    result: dict[str, Any] = {"run": run_info, "server_info": {}, "requests": []}

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if not args.dry_run:
            result["server_info"] = await fetch_json(
                session, f"{args.base_url.rstrip('/')}/server_info", headers
            )

        for scenario in args.scenarios:
            for concurrency in args.concurrency:
                for tail_tokens in args.tail_tokens:
                    for replicate in range(args.replicates):
                        cell_id = cell_identifier(
                            scenario, concurrency, tail_tokens, replicate
                        )
                        cell = {
                            "cell_id": cell_id,
                            "scenario": scenario,
                            "concurrency": concurrency,
                            "tail_tokens_configured": tail_tokens,
                            "replicate": replicate,
                        }
                        conversations = build_conversations(
                            tokenizer=tokenizer,
                            count=concurrency,
                            initial_tokens=args.initial_tokens,
                            maximum_tail_tokens=max(args.tail_tokens),
                            turns=args.turns,
                            namespace=args.cache_namespace,
                            cell_id=cell_id,
                            seed=args.seed + replicate,
                        )

                        if args.dry_run:
                            result["requests"].extend(
                                dry_run_cell(conversations, cell, args)
                            )
                            continue

                        if args.flush_between_cells:
                            await flush_cache(session, args)
                        print(
                            f"[{args.label}] {cell_id}: {args.turns} turns x "
                            f"{concurrency} sticky conversations",
                            flush=True,
                        )

                        for turn in range(args.turns):
                            prepared_turns = [
                                conversation.prepare_turn(
                                    turn=turn,
                                    tail_tokens=tail_tokens,
                                    scenario=scenario,
                                    event_turn=args.event_turn,
                                    branch_depth=args.branch_depth,
                                    edit_distance=args.edit_distance,
                                )
                                for conversation in conversations
                            ]
                            responses = await asyncio.gather(
                                *[
                                    send_request(session, prepared, cell, args)
                                    for prepared in prepared_turns
                                ]
                            )
                            result["requests"].extend(responses)
                            failures = [r for r in responses if not r["success"]]
                            if failures:
                                examples = "; ".join(r["error"] for r in failures[:3])
                                raise RuntimeError(
                                    f"{len(failures)} request(s) failed in {cell_id} "
                                    f"turn {turn}: {examples}"
                                )
                            for conversation, response in zip(
                                conversations, responses, strict=True
                            ):
                                conversation.commit_output(response["output_ids"])

    result["run"]["finished_at"] = utc_now()
    if args.dry_run:
        result["summary"] = {
            "dry_run": True,
            "manifest_requests": len(result["requests"]),
            "exact_length_validation": "passed",
        }
    else:
        result["summary"] = summarize_requests(result["requests"])
    return result


def summarize_requests(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("success")]
    warm = [record for record in successful if record.get("phase") == "warm"]
    cold = [record for record in successful if record.get("phase") == "cold"]

    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"requests": len(group)}
        for name in (
            "ttft_ms",
            "t2_ms",
            "first_itl_ms",
            "e2e_ms",
            "server_queue_ms",
            "server_prefill_to_token_ms",
        ):
            values = [float(r[name]) for r in group if r.get(name) is not None]
            out[f"p50_{name}"] = percentile(values, 0.50)
            out[f"p95_{name}"] = percentile(values, 0.95)
        cached = [float(r.get("actual_cached_tokens", 0)) for r in group]
        expected = [float(r.get("expected_prefix_tokens", 0)) for r in group]
        out["cache_hit_realization"] = (
            sum(cached) / sum(expected) if sum(expected) > 0 else None
        )
        return out

    return {
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "cold": metrics(cold),
        "warm": metrics(warm),
        "all": metrics(successful),
    }


def parse_dflash_profile_log(path: Path) -> list[dict[str, Any]]:
    """Read machine-readable batch profiles emitted by profile/CAR modes."""

    marker = "DFLASH_CAR_METRIC "
    profiles: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            marker_index = line.find(marker)
            if marker_index < 0:
                continue
            raw = line[marker_index + len(marker) :].strip()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid DFLASH_CAR_METRIC JSON in {path}:{line_number}: {raw[:200]}"
                ) from exc
            payload["_server_log_line"] = line_number
            profiles.append(payload)
    return profiles


def aggregate_dflash_profiles(
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse one profile line per TP rank into a critical-path batch profile.

    CUDA phase durations use the maximum rank, because the next distributed phase
    cannot advance before its slowest rank. Captured hidden bytes are summed across
    ranks to represent aggregate device traffic. Raw rank records remain attached.
    """

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    occurrence_by_rank: dict[tuple[Any, ...], int] = {}
    for profile in profiles:
        signature = (
            tuple(str(rid) for rid in profile.get("rids") or []),
            profile.get("mode"),
            tuple(profile.get("extend_tokens_per_req") or []),
            tuple(profile.get("prefix_tokens_per_req") or []),
        )
        rank = int(profile.get("tp_rank") or 0)
        occurrence_key = (*signature, rank)
        occurrence = occurrence_by_rank.get(occurrence_key, 0)
        occurrence_by_rank[occurrence_key] = occurrence + 1
        # Pair the nth emission of the same logical signature across TP ranks.
        key = (*signature, occurrence)
        groups.setdefault(key, []).append(profile)

    duration_fields = (
        "target_to_token_ms",
        "post_token_seed_ms",
        "seed_input_snapshot_ms",
        "seed_kernel_ms",
        "hidden_project_ms",
        "draft_kv_write_ms",
        "target_plus_seed_ms",
    )
    aggregated: list[dict[str, Any]] = []
    for rank_profiles in groups.values():
        merged = dict(rank_profiles[0])
        for field_name in duration_fields:
            durations = [
                float(profile[field_name])
                for profile in rank_profiles
                if profile.get(field_name) is not None
            ]
            merged[field_name] = max(durations) if durations else None

        hidden_bytes = [
            int(profile.get("captured_hidden_bytes") or 0) for profile in rank_profiles
        ]
        merged["captured_hidden_bytes"] = sum(hidden_bytes)
        merged["captured_hidden_bytes_max_rank"] = max(hidden_bytes, default=0)
        merged["draft_cache_write_tokens"] = max(
            (
                int(profile.get("draft_cache_write_tokens") or 0)
                for profile in rank_profiles
            ),
            default=0,
        )
        merged["fused_materializer"] = all(
            bool(profile.get("fused_materializer")) for profile in rank_profiles
        )
        merged["tp_rank_count"] = len(rank_profiles)
        merged["tp_ranks"] = sorted(
            int(profile["tp_rank"])
            for profile in rank_profiles
            if profile.get("tp_rank") is not None
        )
        merged["_server_log_lines"] = [
            profile.get("_server_log_line") for profile in rank_profiles
        ]
        merged["_server_log_line"] = min(
            (
                int(profile["_server_log_line"])
                for profile in rank_profiles
                if profile.get("_server_log_line") is not None
            ),
            default=None,
        )
        merged["_tp_rank_profiles"] = rank_profiles
        aggregated.append(merged)
    return aggregated


def attach_dflash_profiles(
    records: list[dict[str, Any]], profiles: list[dict[str, Any]]
) -> dict[str, int]:
    """Join batch-level CUDA timings to client records by native request ID."""

    by_rid = {record.get("rid"): record for record in records if record.get("rid")}
    aggregated_profiles = aggregate_dflash_profiles(profiles)
    matched = 0
    unmatched = 0
    for profile in aggregated_profiles:
        rids = [str(rid) for rid in profile.get("rids") or []]
        extend_per_req = profile.get("extend_tokens_per_req") or []
        prefix_per_req = profile.get("prefix_tokens_per_req") or []
        total_extend = max(1, int(profile.get("extend_tokens") or 0))
        total_hidden_bytes = int(profile.get("captured_hidden_bytes") or 0)
        for index, rid in enumerate(rids):
            record = by_rid.get(rid)
            if record is None:
                unmatched += 1
                continue
            extend_tokens = (
                int(extend_per_req[index]) if index < len(extend_per_req) else None
            )
            prefix_tokens = (
                int(prefix_per_req[index]) if index < len(prefix_per_req) else None
            )
            apportioned_hidden_bytes = (
                round(total_hidden_bytes * extend_tokens / total_extend)
                if extend_tokens is not None
                else None
            )
            mapped = {
                "dflash_car_mode": profile.get("mode"),
                "dflash_seed_tokens": extend_tokens,
                "dflash_prefix_tokens": prefix_tokens,
                "dflash_capture_bytes": apportioned_hidden_bytes,
                "dflash_seed_project_ms": profile.get("hidden_project_ms"),
                "dflash_seed_kv_ms": profile.get("draft_kv_write_ms"),
                # End-to-end device span around projection + draft KV materializer.
                "dflash_seed_kernel_ms": profile.get("seed_kernel_ms"),
                "dflash_seed_input_snapshot_ms": profile.get("seed_input_snapshot_ms"),
                "dflash_seed_total_ms": profile.get("post_token_seed_ms"),
                "dflash_target_to_token_ms": profile.get("target_to_token_ms"),
                "dflash_target_plus_seed_ms": profile.get("target_plus_seed_ms"),
                "dflash_seed_fused": profile.get("fused_materializer"),
                "dflash_batch_extend_tokens": profile.get("extend_tokens"),
                "dflash_batch_prefix_tokens": profile.get("prefix_tokens"),
                "dflash_profile_log_line": profile.get("_server_log_line"),
                "dflash_profile_tp_ranks": profile.get("tp_rank_count"),
            }
            dflash = record.setdefault("dflash", {})
            additive_fields = {
                "dflash_seed_tokens",
                "dflash_capture_bytes",
                "dflash_seed_project_ms",
                "dflash_seed_kv_ms",
                "dflash_seed_kernel_ms",
                "dflash_seed_input_snapshot_ms",
                "dflash_seed_total_ms",
                "dflash_target_to_token_ms",
                "dflash_target_plus_seed_ms",
            }
            for key, mapped_value in mapped.items():
                if mapped_value is None:
                    continue
                if key in additive_fields and key in dflash:
                    dflash[key] = dflash[key] + mapped_value
                elif key == "dflash_seed_fused" and key in dflash:
                    dflash[key] = bool(dflash[key]) and bool(mapped_value)
                elif (
                    key
                    in {
                        "dflash_prefix_tokens",
                        "dflash_profile_tp_ranks",
                    }
                    and key in dflash
                ):
                    dflash[key] = max(dflash[key], mapped_value)
                else:
                    dflash[key] = mapped_value
            dflash["dflash_profile_batches"] = (
                int(dflash.get("dflash_profile_batches") or 0) + 1
            )
            record.setdefault("dflash_batch_profiles", []).append(profile)
            matched += 1
    profiled_requests = sum(
        1
        for record in records
        if (record.get("dflash") or {}).get("dflash_profile_batches")
    )
    return {
        "profile_rank_records": len(profiles),
        "profile_batches": len(aggregated_profiles),
        "matched_request_profiles": matched,
        "profiled_requests": profiled_requests,
        "unmatched_request_profiles": unmatched,
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cell_id",
        "rid",
        "scenario",
        "concurrency",
        "tail_tokens_configured",
        "replicate",
        "conversation_id",
        "turn",
        "phase",
        "success",
        "error",
        "input_tokens",
        "tail_tokens",
        "expected_prefix_tokens",
        "actual_cached_tokens",
        "uncached_tokens_proxy",
        "server_queue_ms",
        "server_prefill_to_token_ms",
        "ttft_ms",
        "t2_ms",
        "first_itl_ms",
        "mean_itl_ms",
        "p95_itl_ms",
        "e2e_ms",
        "http_done_ms",
        "completion_tokens",
        "sse_token_events",
        "multi_token_sse_events",
        "prompt_fingerprint",
        "output_fingerprint",
        *sorted(DFLASH_COUNTERS),
        "meta_info_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row.update(record.get("dflash") or {})
            row["meta_info_json"] = json.dumps(
                record.get("meta_info") or {}, sort_keys=True
            )
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run an exact-token, sticky multi-turn SGLang workload and retain raw "
            "SSE timings for DFlash/CAR bottleneck analysis. Start the server "
            "separately for each mode."
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Hugging Face model/tokenizer path used to construct valid input IDs.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--api-key", default=os.environ.get("SGLANG_API_KEY", ""))
    parser.add_argument("--label", default="dflash-eager")
    parser.add_argument(
        "--mode",
        default="eager",
        choices=[
            "baseline",
            "eager",
            "profile",
            "deferred-serial",
            "deferred-overlap",
            "other",
        ],
        help=(
            "Annotation only; configure --speculative-dflash-seed-mode on the "
            "server before launch."
        ),
    )
    parser.add_argument("--initial-tokens", type=int, default=8192)
    parser.add_argument(
        "--tail-tokens", type=int, nargs="+", default=[32, 128, 256, 512, 1024, 2048]
    )
    parser.add_argument("--output-tokens", type=int, default=35)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["append", "branch", "edit", "cache-miss"],
        default=["append"],
    )
    parser.add_argument(
        "--event-turn",
        type=int,
        default=2,
        help="Zero-indexed continuation turn for branch/edit/cache-miss events.",
    )
    parser.add_argument("--branch-depth", type=int, default=1)
    parser.add_argument("--edit-distance", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--cache-namespace",
        default="car-dflash-eval-v1",
        help=(
            "Stable workload/cache salt. Keep identical across compared server runs; "
            "change it to prevent reuse from an older unflushed run."
        ),
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--flush-timeout", type=float, default=120.0)
    parser.add_argument(
        "--flush-between-cells",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flush once before each scenario/concurrency/tail/replicate conversation.",
    )
    parser.add_argument(
        "--retain-sse-payloads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store each raw decoded SSE payload as well as its arrival timestamp.",
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        default=None,
        help=(
            "Server stdout/stderr containing DFLASH_CAR_METRIC JSON lines. The "
            "benchmark joins them to requests by rid."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct and validate token-exact manifests without contacting a server.",
    )
    parser.add_argument("--output", type=Path, default=Path("car_dflash_results.json"))
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Defaults to OUTPUT with a .csv suffix.",
    )
    args = parser.parse_args(argv)

    if args.turns < 1:
        parser.error("--turns must be positive")
    if args.output_tokens < 1:
        parser.error("--output-tokens must be positive")
    if any(value < 0 for value in args.tail_tokens):
        parser.error("--tail-tokens cannot be negative")
    if any(value < 1 for value in args.concurrency):
        parser.error("--concurrency values must be positive")
    if args.event_turn <= 0 or args.event_turn >= args.turns:
        if any(scenario != "append" for scenario in args.scenarios):
            parser.error("--event-turn must select a continuation turn")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = asyncio.run(run(args))
    if args.server_log is not None and not args.dry_run:
        profiles = parse_dflash_profile_log(args.server_log)
        result["profile_log_join"] = attach_dflash_profiles(
            result["requests"], profiles
        )
        # Recompute now that directly measured seed counters are attached.
        result["summary"] = summarize_requests(result["requests"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    write_csv(csv_output, result["requests"])
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"raw JSON: {args.output}")
    print(f"flat CSV: {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
