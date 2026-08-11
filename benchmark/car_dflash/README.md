# Cache-Aware Release (CAR)-DFlash multi-turn evaluation

CAR separates exact-token readiness from the readiness of DFlash's auxiliary
prompt KV while retaining an explicit cache-ownership gate between them.

This benchmark answers three separate questions:

1. Does an 8K cold prompt exaggerate DFlash's TTFT cost for an append-only chat?
2. How much work remains on a warm turn when only 32–2,048 tokens are new?
3. Does publishing the exact target token before prompt-side draft-KV construction
   improve real TTFT, or merely move the same stall into token two/E2E latency?

The workload is token-exact. It uses native `/generate`, appends every generated
assistant token to the next request, gives each conversation a unique radix namespace
and in-prompt sentinel, and sends a stable `X-SMG-Routing-Key` on every turn. All
conversations in a turn are sent together behind a round barrier.

## Runtime modes

Configure the server with the CLI flags introduced by the CAR-DFlash experiment:

| Server mode | Meaning |
|---|---|
| baseline | Launch without DFlash. |
| `eager` | Existing production ordering; no CUDA profiling events. |
| `profile` | Same ordering as eager, with asynchronous phase events and `DFLASH_CAR_METRIC` log records. |
| `deferred-serial` | Make exact-token egress depend on `token_ready`, while cache publication, location release, and later model work remain gated by draft-seed readiness. |
| `deferred-overlap` | Keep the same cache/ownership gates, but let an unrelated overlap-scheduled batch run while the seed is pending. A batch containing the seed owner waits before any model work. |

Use:

```bash
--speculative-dflash-seed-mode {eager,profile,deferred-serial,deferred-overlap}
--speculative-dflash-profile-every-n 1
```

Neither deferred mode is a target-only autoregressive fallback or a complete
two-state radix design. Both prevent a target-only prefix from being reused as if its
draft KV were valid. `deferred-overlap` is the narrow global-first-token experiment:
it pipelines only the independent work exposed by SGLang's existing overlap result
queue and parks the seed-owning request behind its event.

`deferred-overlap` is CUDA/HIP-only and intentionally rejects non-overlap scheduling,
unified memory, PD disaggregation, and mixed prefill/decode batches. Leave
`SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` unset/false or consecutive prefills will
be synchronized before the second launch and the mode will have no opportunity to
overlap them.

## Recommended five-run experiment

Use a separate server/log/result file for every configuration. The target, tokenizer,
TP topology, memory fraction, chunked-prefill budget, CUDA graph settings, and all
other flags must be identical.

Create the result directory once before launching a server:

```bash
mkdir -p results
```

Baseline server:

```bash
python3 -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --tp 4 --port 30000 \
  --enable-metrics \
  2>&1 | tee results/baseline.server.log
```

DFlash profile server:

```bash
python3 -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --tp 4 --port 30000 \
  --enable-metrics \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DFLASH_MODEL" \
  --speculative-dflash-seed-mode profile \
  --speculative-dflash-profile-every-n 1 \
  2>&1 | tee results/dflash-profile.server.log
```

Deferred-serial server:

```bash
python3 -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --tp 4 --port 30000 \
  --enable-metrics \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DFLASH_MODEL" \
  --speculative-dflash-seed-mode deferred-serial \
  --speculative-dflash-profile-every-n 1 \
  2>&1 | tee results/dflash-deferred.server.log
```

Deferred-overlap server (keep overlap scheduling enabled):

```bash
python3 -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --tp 4 --port 30000 \
  --enable-metrics \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DFLASH_MODEL" \
  --speculative-dflash-seed-mode deferred-overlap \
  --speculative-dflash-profile-every-n 1 \
  2>&1 | tee results/dflash-overlap.server.log
```

An additional production `eager` run quantifies profiling overhead. Its latency should
match `profile` within noise.

In another terminal, run the same grid after each server starts:

```bash
python3 benchmark/car_dflash/bench_multiturn.py \
  --base-url http://127.0.0.1:30000 \
  --tokenizer "$TARGET_MODEL" --trust-remote-code \
  --label baseline --mode baseline \
  --initial-tokens 8192 \
  --tail-tokens 32 128 256 512 1024 2048 \
  --output-tokens 35 --turns 4 \
  --concurrency 1 4 16 --replicates 3 \
  --scenarios append \
  --cache-namespace car-dflash-eval-v1 \
  --output results/baseline.json
```

For the profiled DFlash server, change the label/mode/output and join its structured
server log:

```bash
python3 benchmark/car_dflash/bench_multiturn.py \
  --base-url http://127.0.0.1:30000 \
  --tokenizer "$TARGET_MODEL" --trust-remote-code \
  --label dflash-profile --mode profile \
  --initial-tokens 8192 \
  --tail-tokens 32 128 256 512 1024 2048 \
  --output-tokens 35 --turns 4 \
  --concurrency 1 4 16 --replicates 3 \
  --scenarios append \
  --cache-namespace car-dflash-eval-v1 \
  --server-log results/dflash-profile.server.log \
  --output results/dflash-profile.json
```

Repeat with `--label dflash-deferred --mode deferred-serial`, then with
`--label dflash-overlap --mode deferred-overlap`, using each server's own log. Keep
`--cache-namespace`, seed, and every workload parameter identical so prompt
fingerprints pair across runs. The benchmark flushes before each cell by default.

Generate the paired report:

```bash
python3 benchmark/car_dflash/report.py \
  results/baseline.json \
  results/dflash-profile.json \
  results/dflash-deferred.json \
  results/dflash-overlap.json \
  --reference-label baseline \
  --output results/car-dflash-report.md
```

The supplied hand-model defaults encode the original TP4/concurrency-16 observation:
a 260 ms cold 8K gap, a roughly 16K prefill wave, and about 55 ms of DFlash-side work
per full wave. Override these values rather than treating them as universal constants.

## What is captured

Every request contains:

- TTFT, time-to-second-token (T2), first ITL, interpolated aggregate ITL, E2E, and
  HTTP completion time;
- the timestamp and decoded payload of every SSE event;
- a logical token-arrival array. When one speculative SSE event contains several
  tokens, all receive the same observable wire timestamp—no device timing is invented;
- prompt/output fingerprints and exact output IDs for losslessness checks;
- intended reusable prefix, `meta_info.cached_tokens`, and an uncached/seed-row proxy;
- scheduler queue time and prefill-to-token time when the server uses
  `--enable-metrics`;
- the complete final `meta_info`, plus any `dflash_*`/`car_*` fields;
- joined CUDA phase measurements from `DFLASH_CAR_METRIC` when `--server-log` is used.

TP runs emit one profile line per rank. The join groups identical request batches,
uses the maximum rank duration as the distributed critical path, sums captured-hidden
bytes across ranks, and retains the raw per-rank records in JSON.

`stream_interval=1` is explicitly placed in every request. This is necessary for T2
to be meaningful, although speculative verification may still publish an accepted
block in one event.

The important direct server timings are:

- `target_to_token_ms`: target prefill/capture through exact-token readiness;
- `post_token_seed_ms`: the dependency CAR removes from direct TTFT;
- `seed_input_snapshot_ms`: time until side-stream seed inputs are safe/ready;
- `hidden_project_ms`: target hidden-state conditioning/projection;
- `draft_kv_write_ms`: draft-layer K/V projection and cache materialization;
- `seed_kernel_ms`: the combined device span around those seed phases;
- `target_plus_seed_ms`: the current eager critical path;
- `extend_tokens_per_req`: the exact uncached rows materialized for each request.

## Interpreting the result

For ordinary warm append turns, these invariants should hold:

```text
actual_cached_tokens ~= previous complete conversation length (page-rounded/tail-capped)
uncached_tokens_proxy ~= new user tail + uncached boundary/page tail
dflash_seed_tokens ~= new user tail + uncached boundary/page tail
```

The cold 8K point proves the large prompt barrier. The warm 32–2K cells determine
whether it matters for production chat. CAR is useful only when:

1. deferred TTFT improves by a meaningful amount;
2. that improvement does not reappear almost entirely in T2 or E2E;
3. cached-token parity remains the same as baseline/profile;
4. output fingerprints remain identical.

The report excludes latency pairs whose prompt fingerprints differ. It still reports
output mismatches explicitly because a mismatch is either nondeterminism or a
losslessness failure.

## Failure-mode scenarios

The default `append` scenario represents normal sticky chat. Add scenarios only after
the primary grid is stable:

```bash
--scenarios append branch edit cache-miss \
--event-turn 2 --branch-depth 1 --edit-distance 256
```

- `branch` returns to an earlier completed turn and takes a different continuation.
- `edit` changes a token 256 positions before the current tail, testing partial reuse.
- `cache-miss` rotates the native `extra_key` while preserving the identical prompt,
  testing the cold fallback without restarting the conversation.

For cache-capacity experiments, run many more distinct conversations without flushing
between cells and monitor eviction metrics separately. That is intentionally not mixed
into the causal TTFT grid.

## Dry run and tests

Validate exact token construction without a server:

```bash
python3 benchmark/car_dflash/bench_multiturn.py \
  --tokenizer "$TARGET_MODEL" --trust-remote-code \
  --initial-tokens 8192 --tail-tokens 256 \
  --concurrency 4 --turns 4 --dry-run \
  --output results/workload-dry-run.json
```

Run the CPU-only unit tests:

```bash
cd benchmark/car_dflash
python3 -m unittest -v test_car_dflash.py
```
