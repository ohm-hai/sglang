#!/usr/bin/env python3
"""Compare baseline, eager DFlash, and CAR-DFlash multi-turn benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

DIMENSIONS = (
    "scenario",
    "concurrency",
    "tail_tokens_configured",
    "phase",
)
PAIR_DIMENSIONS = ("cell_id", "conversation_id", "turn")


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    runs = []
    labels: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as file:
            run = json.load(file)
        label = run.get("run", {}).get("label") or path.stem
        if label in labels:
            raise ValueError(f"duplicate run label {label!r}; labels must be unique")
        labels.add(label)
        run["_label"] = label
        run["_path"] = str(path)
        runs.append(run)
    return runs


def value(record: dict[str, Any], name: str) -> float | None:
    raw = record.get(name)
    if raw is None:
        raw = (record.get("dflash") or {}).get(name)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def group_records(
    records: list[dict[str, Any]], dimensions: tuple[str, ...] = DIMENSIONS
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("success"):
            groups[tuple(record.get(key) for key in dimensions)].append(record)
    return dict(groups)


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"requests": len(records)}
    for metric in (
        "ttft_ms",
        "t2_ms",
        "first_itl_ms",
        "e2e_ms",
        "server_queue_ms",
        "server_prefill_to_token_ms",
        "actual_cached_tokens",
        "uncached_tokens_proxy",
        "dflash_seed_tokens",
        "dflash_seed_project_ms",
        "dflash_seed_kv_ms",
        "dflash_seed_total_ms",
        "dflash_seed_input_snapshot_ms",
        "dflash_seed_queue_ms",
        "dflash_seed_ready_lag_tokens",
        "dflash_fallback_tokens",
    ):
        values = [
            number
            for record in records
            if (number := value(record, metric)) is not None
        ]
        summary[f"p50_{metric}"] = percentile(values, 0.50)
        summary[f"p95_{metric}"] = percentile(values, 0.95)
        summary[f"coverage_{metric}"] = len(values)

    expected = [value(record, "expected_prefix_tokens") or 0.0 for record in records]
    actual = [value(record, "actual_cached_tokens") or 0.0 for record in records]
    summary["cache_hit_realization"] = (
        sum(actual) / sum(expected) if sum(expected) > 0 else None
    )
    summary["multi_token_sse_fraction"] = (
        sum(
            1
            for record in records
            if (value(record, "multi_token_sse_events") or 0) > 0
        )
        / len(records)
        if records
        else None
    )
    return summary


def hand_wave_model_ms(
    seed_tokens: float,
    concurrency: int,
    wave_budget_tokens: int,
    wave_tax_ms: float,
    fixed_tax_ms: float,
) -> float:
    """Small queueing model for synchronous prompt-side draft construction.

    Equal-sized requests are packed into token-budget waves.  Every request in a
    later wave observes cumulative seed work from the earlier waves.  A partial
    wave's cost scales with occupied tokens.
    """

    if seed_tokens <= 0 or concurrency <= 0:
        return 0.0
    per_wave = max(1, int(wave_budget_tokens // seed_tokens))
    remaining = concurrency
    cumulative = fixed_tax_ms
    weighted_completion = 0.0
    while remaining:
        in_wave = min(per_wave, remaining)
        occupied = min(float(wave_budget_tokens), in_wave * seed_tokens)
        cumulative += wave_tax_ms * occupied / float(wave_budget_tokens)
        weighted_completion += cumulative * in_wave
        remaining -= in_wave
    return weighted_completion / concurrency


def paired_records(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    ref_by_key = {
        tuple(record.get(key) for key in PAIR_DIMENSIONS): record
        for record in reference.get("requests", [])
        if record.get("success")
    }
    candidate_by_key = {
        tuple(record.get(key) for key in PAIR_DIMENSIONS): record
        for record in candidate.get("requests", [])
        if record.get("success")
    }
    common = sorted(set(ref_by_key) & set(candidate_by_key), key=str)
    pairs = [(ref_by_key[key], candidate_by_key[key]) for key in common]
    diagnostics = {
        "reference_records": len(ref_by_key),
        "candidate_records": len(candidate_by_key),
        "paired_records": len(pairs),
        "prompt_mismatches": sum(
            left.get("prompt_fingerprint") != right.get("prompt_fingerprint")
            for left, right in pairs
        ),
        "output_mismatches": sum(
            left.get("output_fingerprint") != right.get("output_fingerprint")
            for left, right in pairs
        ),
    }
    # A prompt mismatch means later-turn latency no longer represents identical work.
    valid = [
        pair
        for pair in pairs
        if pair[0].get("prompt_fingerprint") == pair[1].get("prompt_fingerprint")
    ]
    return valid, diagnostics


def summarize_paired_group(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"pairs": len(pairs)}
    for metric in ("ttft_ms", "t2_ms", "first_itl_ms", "e2e_ms"):
        deltas = []
        for reference, candidate in pairs:
            left = value(reference, metric)
            right = value(candidate, metric)
            if left is not None and right is not None:
                deltas.append(right - left)
        summary[f"p50_delta_{metric}"] = percentile(deltas, 0.50)
        summary[f"p95_delta_{metric}"] = percentile(deltas, 0.95)

    hit_deltas = []
    for reference, candidate in pairs:
        left = value(reference, "actual_cached_tokens")
        right = value(candidate, "actual_cached_tokens")
        if left is not None and right is not None:
            hit_deltas.append(right - left)
    summary["p50_cache_hit_delta_tokens"] = percentile(hit_deltas, 0.50)
    return summary


def build_report(
    runs: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[str, list[dict[str, Any]]]:
    by_label = {run["_label"]: run for run in runs}
    if args.reference_label not in by_label:
        raise ValueError(
            f"reference label {args.reference_label!r} not found; have {sorted(by_label)}"
        )
    reference = by_label[args.reference_label]
    lines = [
        "# Cache-Aware Release (CAR)-DFlash multi-turn benchmark report",
        "",
        "All latency comparisons below are paired by workload cell, conversation, and turn. "
        "Pairs with different prompt fingerprints are excluded.",
        "",
        "## Runs",
        "",
        "| Label | Mode | Requests | Source |",
        "|---|---:|---:|---|",
    ]
    rows: list[dict[str, Any]] = []
    for run in runs:
        lines.append(
            f"| {run['_label']} | {run.get('run', {}).get('mode', 'unknown')} | "
            f"{len(run.get('requests', []))} | `{run['_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Measured latency and cache state",
            "",
            "| Run | Scenario | C | Tail | Phase | N | TTFT p50/p95 ms | T2 p50 ms | "
            "E2E p50 ms | Queue p50 ms | Server prefill p50 ms | Cached p50 | "
            "Seed proxy p50 | Measured seed p50 ms |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    summaries: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for run in runs:
        label = run["_label"]
        summaries[label] = {}
        for dimensions, records in sorted(
            group_records(run.get("requests", [])).items(), key=str
        ):
            summary = summarize_group(records)
            summaries[label][dimensions] = summary
            scenario, concurrency, tail, phase = dimensions
            lines.append(
                f"| {label} | {scenario} | {concurrency} | {tail} | {phase} | "
                f"{summary['requests']} | {fmt(summary['p50_ttft_ms'])}/"
                f"{fmt(summary['p95_ttft_ms'])} | {fmt(summary['p50_t2_ms'])} | "
                f"{fmt(summary['p50_e2e_ms'])} | "
                f"{fmt(summary['p50_server_queue_ms'])} | "
                f"{fmt(summary['p50_server_prefill_to_token_ms'])} | "
                f"{fmt(summary['p50_actual_cached_tokens'], 0)} | "
                f"{fmt(summary['p50_uncached_tokens_proxy'], 0)} | "
                f"{fmt(summary['p50_dflash_seed_total_ms'])} |"
            )
            rows.append(
                {
                    "row_type": "measured",
                    "run": label,
                    **dict(zip(DIMENSIONS, dimensions, strict=True)),
                    **summary,
                }
            )

    lines.extend(
        [
            "",
            f"## Paired deltas versus `{args.reference_label}`",
            "",
            "Positive deltas are slower than the reference.",
            "",
            "| Candidate | Scenario | C | Tail | Phase | Pairs | ΔTTFT p50/p95 ms | "
            "ΔT2 p50 ms | ΔE2E p50 ms | Δcached p50 |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    diagnostics_by_label: dict[str, dict[str, int]] = {}
    for candidate in runs:
        if candidate is reference:
            continue
        pairs, diagnostics = paired_records(reference, candidate)
        diagnostics_by_label[candidate["_label"]] = diagnostics
        pair_groups: dict[
            tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]
        ] = defaultdict(list)
        for left, right in pairs:
            dimensions = tuple(right.get(key) for key in DIMENSIONS)
            pair_groups[dimensions].append((left, right))
        for dimensions, group in sorted(pair_groups.items(), key=str):
            summary = summarize_paired_group(group)
            scenario, concurrency, tail, phase = dimensions
            lines.append(
                f"| {candidate['_label']} | {scenario} | {concurrency} | {tail} | "
                f"{phase} | {summary['pairs']} | "
                f"{fmt(summary['p50_delta_ttft_ms'])}/"
                f"{fmt(summary['p95_delta_ttft_ms'])} | "
                f"{fmt(summary['p50_delta_t2_ms'])} | "
                f"{fmt(summary['p50_delta_e2e_ms'])} | "
                f"{fmt(summary['p50_cache_hit_delta_tokens'], 0)} |"
            )
            rows.append(
                {
                    "row_type": "paired_delta",
                    "run": candidate["_label"],
                    **dict(zip(DIMENSIONS, dimensions, strict=True)),
                    **summary,
                }
            )

    lines.extend(
        [
            "",
            "## Correctness and pairing diagnostics",
            "",
            "| Candidate | Paired | Prompt mismatches | Output mismatches |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, diagnostic in diagnostics_by_label.items():
        lines.append(
            f"| {label} | {diagnostic['paired_records']} | "
            f"{diagnostic['prompt_mismatches']} | {diagnostic['output_mismatches']} |"
        )

    lines.extend(
        [
            "",
            "## Direct bottleneck attribution",
            "",
            "The attribution ratio is a diagnostic ratio of medians, not a formal "
            "latency decomposition. Values near one mean the measured post-token seed "
            "span accounts for most of the DFlash TTFT gap.",
            "",
            "| Run | Scenario | C | Tail | Phase | ΔTTFT vs baseline ms | "
            "Post-token seed ms | Residual ms | Seed/gap |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for candidate in runs:
        if candidate is reference:
            continue
        pairs, _ = paired_records(reference, candidate)
        pair_groups: dict[
            tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]
        ] = defaultdict(list)
        for pair in pairs:
            pair_groups[tuple(pair[1].get(key) for key in DIMENSIONS)].append(pair)
        candidate_groups = group_records(candidate.get("requests", []))
        for dimensions, group_pairs in sorted(pair_groups.items(), key=str):
            paired_summary = summarize_paired_group(group_pairs)
            measured_summary = summarize_group(candidate_groups[dimensions])
            gap = paired_summary["p50_delta_ttft_ms"]
            seed = measured_summary["p50_dflash_seed_total_ms"]
            residual = gap - seed if gap is not None and seed is not None else None
            ratio = (
                seed / gap if gap is not None and gap > 0 and seed is not None else None
            )
            scenario, concurrency, tail, phase = dimensions
            lines.append(
                f"| {candidate['_label']} | {scenario} | {concurrency} | {tail} | "
                f"{phase} | {fmt(gap)} | {fmt(seed)} | {fmt(residual)} | "
                f"{fmt(ratio, 2)} |"
            )
            rows.append(
                {
                    "row_type": "bottleneck_attribution",
                    "run": candidate["_label"],
                    **dict(zip(DIMENSIONS, dimensions, strict=True)),
                    "delta_ttft_ms": gap,
                    "post_token_seed_ms": seed,
                    "residual_ms": residual,
                    "seed_to_gap_ratio": ratio,
                }
            )

    profile_run = next(
        (run for run in runs if run.get("run", {}).get("mode") in {"profile", "eager"}),
        None,
    )
    deferred_runs = [
        run
        for run in runs
        if run.get("run", {}).get("mode") in {"deferred-serial", "deferred-overlap"}
    ]
    if profile_run is not None and deferred_runs:
        lines.extend(
            [
                "",
                f"## CAR movement check versus `{profile_run['_label']}`",
                "",
                "Negative deltas mean CAR is faster. A TTFT reduction that vanishes in "
                "T2 or E2E is primarily metric movement, not a serving win.",
                "",
                "| CAR run | Scenario | C | Tail | Phase | ΔTTFT ms | ΔT2 ms | "
                "Δfirst-ITL ms | ΔE2E ms |",
                "|---|---|---:|---:|---|---:|---:|---:|---:|",
            ]
        )
        for deferred in deferred_runs:
            pairs, _ = paired_records(profile_run, deferred)
            movement_groups: dict[
                tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]
            ] = defaultdict(list)
            for pair in pairs:
                movement_groups[tuple(pair[1].get(key) for key in DIMENSIONS)].append(
                    pair
                )
            for dimensions, group in sorted(movement_groups.items(), key=str):
                summary = summarize_paired_group(group)
                scenario, concurrency, tail, phase = dimensions
                lines.append(
                    f"| {deferred['_label']} | {scenario} | {concurrency} | {tail} | "
                    f"{phase} | {fmt(summary['p50_delta_ttft_ms'])} | "
                    f"{fmt(summary['p50_delta_t2_ms'])} | "
                    f"{fmt(summary['p50_delta_first_itl_ms'])} | "
                    f"{fmt(summary['p50_delta_e2e_ms'])} |"
                )
                rows.append(
                    {
                        "row_type": "car_movement",
                        "run": deferred["_label"],
                        **dict(zip(DIMENSIONS, dimensions, strict=True)),
                        **summary,
                    }
                )

    lines.extend(
        [
            "",
            "## Hand-model check",
            "",
            "The linear model scales the supplied cold DFlash tax with uncached tokens. "
            "The wave model packs equal-sized requests into the configured prefill-token "
            "budget and charges cumulative synchronous seed work to later waves.",
            "",
            "| Run | Scenario | C | Tail | Phase | Seed tokens p50 | Linear tax ms | "
            "Wave tax ms | Measured ΔTTFT p50 ms |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    ref_groups = group_records(reference.get("requests", []))
    for run in runs:
        if run is reference:
            continue
        candidate_groups = group_records(run.get("requests", []))
        pairs, _ = paired_records(reference, run)
        paired_groups: dict[
            tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]
        ] = defaultdict(list)
        for pair in pairs:
            paired_groups[tuple(pair[1].get(key) for key in DIMENSIONS)].append(pair)
        for dimensions, records in sorted(candidate_groups.items(), key=str):
            if dimensions not in ref_groups:
                continue
            summary = summarize_group(records)
            seed_tokens = summary["p50_dflash_seed_tokens"]
            if seed_tokens is None:
                seed_tokens = summary["p50_uncached_tokens_proxy"] or 0.0
            concurrency = int(dimensions[1])
            linear = (
                args.reference_cold_tax_ms
                * float(seed_tokens)
                / args.reference_cold_tokens
            )
            wave = hand_wave_model_ms(
                float(seed_tokens),
                concurrency,
                args.wave_budget_tokens,
                args.wave_tax_ms,
                args.fixed_tax_ms,
            )
            paired_summary = summarize_paired_group(paired_groups.get(dimensions, []))
            scenario, concurrency, tail, phase = dimensions
            measured = paired_summary["p50_delta_ttft_ms"]
            lines.append(
                f"| {run['_label']} | {scenario} | {concurrency} | {tail} | {phase} | "
                f"{fmt(seed_tokens, 0)} | {fmt(linear)} | {fmt(wave)} | {fmt(measured)} |"
            )
            rows.append(
                {
                    "row_type": "hand_model",
                    "run": run["_label"],
                    **dict(zip(DIMENSIONS, dimensions, strict=True)),
                    "seed_tokens_p50": seed_tokens,
                    "linear_tax_ms": linear,
                    "wave_tax_ms": wave,
                    "measured_delta_ttft_ms": measured,
                }
            )

    lines.extend(
        [
            "",
            "## Interpretation checks",
            "",
            "- A warm append request should have `uncached_tokens_proxy` close to the new "
            "tail, not the full context.",
            "- `dflash_seed_total_ms` is the directly measured removable barrier when the "
            "runtime instrumentation is enabled. Missing values mean the server did not "
            "publish that counter.",
            "- `server_queue_ms` and `server_prefill_to_token_ms` require a server launched "
            "with `--enable-metrics`; they separate admission delay from the device-side "
            "target/seed spans.",
            "- CAR is a real latency improvement only when TTFT falls without the same "
            "penalty reappearing in T2 or E2E.",
            "- Any prompt fingerprint mismatch invalidates that later-turn latency pair. "
            "Any output mismatch is also a losslessness failure that must be investigated.",
            "- A negative cache-hit delta indicates that DFlash/CAR changed effective radix "
            "reuse; do not attribute that difference to seed scheduling alone.",
            "",
        ]
    )
    return "\n".join(lines), rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build paired CAR-DFlash latency, correctness, and hand-model tables.",
    )
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--reference-label", default="baseline")
    parser.add_argument("--output", type=Path, default=Path("car_dflash_report.md"))
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument(
        "--reference-cold-tax-ms",
        type=float,
        default=260.0,
        help="Observed DFlash-baseline TTFT gap at the reference cold point.",
    )
    parser.add_argument("--reference-cold-tokens", type=float, default=8192.0)
    parser.add_argument("--wave-budget-tokens", type=int, default=16384)
    parser.add_argument(
        "--wave-tax-ms",
        type=float,
        default=55.0,
        help="DFlash tax of one full prompt-token scheduling wave.",
    )
    parser.add_argument(
        "--fixed-tax-ms",
        type=float,
        default=4.0,
        help="Per-request non-scaling DFlash overhead in the hand model.",
    )
    args = parser.parse_args(argv)
    if args.reference_cold_tokens <= 0 or args.wave_budget_tokens <= 0:
        parser.error("token reference/budget values must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = load_runs(args.results)
    report, rows = build_report(runs, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    write_csv(csv_output, rows)
    print(report)
    print(f"report: {args.output}")
    print(f"tables: {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
