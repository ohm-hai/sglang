"""Deterministic token-level multi-turn workloads for CAR-DFlash experiments.

The benchmark uses native ``/generate`` requests with ``input_ids``.  This avoids
chat-template drift and lets every requested length be exact.  Generated assistant
tokens are appended to the next request, so the radix-cache dependency is identical
to an append-only chat conversation.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


class TokenizerLike(Protocol):
    bos_token_id: int | None
    vocab_size: int

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


_SUBJECTS = (
    "distributed inference",
    "mixture of experts routing",
    "prefix caching",
    "speculative decoding",
    "GPU scheduling",
    "memory bandwidth",
    "multi turn dialogue",
    "kernel fusion",
    "model serving",
    "attention computation",
    "request batching",
    "cache coherence",
)
_VERBS = (
    "changes",
    "measures",
    "explains",
    "compares",
    "records",
    "constrains",
    "improves",
    "validates",
    "reconstructs",
    "synchronizes",
)
_OBJECTS = (
    "latency under concurrent load",
    "the exact token history",
    "reuse across conversation turns",
    "the critical execution dependency",
    "queueing and device-side work",
    "the difference between cold and warm requests",
    "an immutable branch of the prompt tree",
    "the state needed by the proposal model",
    "first-token publication",
    "the amount of uncached work",
)


def token_fingerprint(token_ids: Iterable[int]) -> str:
    """Return a stable compact fingerprint without serializing a huge prompt."""

    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()[:20]


def _encode(tokenizer: TokenizerLike, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        # Small fake tokenizers used in unit tests need not expose the HF keyword.
        return list(tokenizer.encode(text))


def make_token_pool(
    tokenizer: TokenizerLike, minimum_tokens: int, seed: int
) -> list[int]:
    """Create a deterministic, English-like pool with at least ``minimum_tokens``.

    The varied clauses are preferable to repeating one token: repeated IDs can create
    unrealistically homogeneous MoE routes and attention patterns.
    """

    if minimum_tokens <= 0:
        return []
    rng = random.Random(seed)
    token_ids: list[int] = []
    paragraph = 0
    while len(token_ids) < minimum_tokens:
        sentences = []
        for sentence in range(192):
            subject = rng.choice(_SUBJECTS)
            verb = rng.choice(_VERBS)
            obj = rng.choice(_OBJECTS)
            qualifier = rng.randrange(1_000_000)
            sentences.append(
                f"In experiment {paragraph}-{sentence}-{qualifier}, {subject} "
                f"{verb} {obj}."
            )
        token_ids.extend(_encode(tokenizer, " ".join(sentences)))
        paragraph += 1
        if paragraph > 10_000:
            raise RuntimeError("tokenizer produced no usable tokens")
    return token_ids


@dataclass
class PreparedTurn:
    conversation_id: int
    turn: int
    scenario: str
    phase: str
    input_ids: list[int]
    tail_tokens: int
    expected_prefix_tokens: int
    mutation_index: int | None
    extra_key: str
    routing_key: str
    prompt_fingerprint: str


@dataclass
class Conversation:
    conversation_id: int
    initial_input_ids: list[int]
    continuation_pool: list[int]
    namespace: str
    cell_id: str
    history: list[int] = field(default_factory=list)
    snapshots: list[list[int]] = field(default_factory=list)
    continuation_cursor: int = 0
    cache_epoch: int = 0
    event_applied: bool = False

    def __post_init__(self) -> None:
        if not self.history:
            self.history = list(self.initial_input_ids)

    @property
    def extra_key(self) -> str:
        # This is a native SGLang cache namespace (the equivalent of cache_salt).
        return (
            f"car-dflash:{self.namespace}:{self.cell_id}:"
            f"c{self.conversation_id}:e{self.cache_epoch}"
        )

    @property
    def routing_key(self) -> str:
        # Stable across turns, including a deliberate cache-miss epoch change.
        return f"car-dflash-session:{self.namespace}:{self.cell_id}:c{self.conversation_id}"

    def _take_tail(self, length: int) -> list[int]:
        if length < 0:
            raise ValueError("tail length must be non-negative")
        end = self.continuation_cursor + length
        if end > len(self.continuation_pool):
            raise ValueError(
                f"conversation {self.conversation_id} exhausted its token pool: "
                f"need {end}, have {len(self.continuation_pool)}"
            )
        out = self.continuation_pool[self.continuation_cursor : end]
        self.continuation_cursor = end
        return list(out)

    def prepare_turn(
        self,
        turn: int,
        tail_tokens: int,
        scenario: str = "append",
        event_turn: int = 2,
        branch_depth: int = 1,
        edit_distance: int = 256,
    ) -> PreparedTurn:
        """Prepare a turn but do not commit its generated output.

        ``turn == 0`` is the exact-length initial prompt.  Later turns append an
        exact-length synthetic user tail.  Scenario events happen once, immediately
        before the configured continuation turn.
        """

        if turn == 0:
            input_ids = list(self.initial_input_ids)
            expected_prefix = 0
            phase = "cold"
            mutation_index = None
        else:
            if scenario not in {"append", "branch", "edit", "cache-miss"}:
                raise ValueError(f"unknown scenario: {scenario}")

            mutation_index = None
            event = turn == event_turn and not self.event_applied
            if event and scenario == "branch":
                if branch_depth < 1:
                    raise ValueError("branch_depth must be at least one")
                # depth=1 discards the immediately preceding completed turn.
                snapshot_index = max(0, len(self.snapshots) - branch_depth - 1)
                self.history = list(self.snapshots[snapshot_index])
                self.event_applied = True
            elif event and scenario == "edit":
                if not self.history:
                    raise ValueError("cannot edit an empty history")
                mutation_index = max(0, len(self.history) - max(1, edit_distance))
                old_token = self.history[mutation_index]
                candidates = self._take_tail(1)
                replacement = candidates[0]
                if replacement == old_token:
                    replacement = (
                        self.continuation_pool[self.continuation_cursor]
                        if self.continuation_cursor < len(self.continuation_pool)
                        else old_token + 1
                    )
                self.history[mutation_index] = replacement
                self.event_applied = True
            elif event and scenario == "cache-miss":
                self.cache_epoch += 1
                self.event_applied = True

            base_len = len(self.history)
            user_tail = self._take_tail(tail_tokens)
            input_ids = list(self.history) + user_tail

            if event and scenario == "cache-miss":
                expected_prefix = 0
                phase = "forced-cache-miss"
            elif event and scenario == "edit":
                # The radix can match only the unchanged portion preceding the edit.
                # A mismatch at zero-based position i leaves exactly i matching
                # tokens before it (subject to radix page rounding).
                expected_prefix = max(0, mutation_index or 0)
                phase = "edited-prefix"
            elif event and scenario == "branch":
                expected_prefix = max(0, base_len - 1)
                phase = "branched-prefix"
            else:
                # SGLang generally withholds the last cached token to produce logits.
                expected_prefix = max(0, base_len - 1)
                phase = "warm"

            self.history = input_ids

        return PreparedTurn(
            conversation_id=self.conversation_id,
            turn=turn,
            scenario=scenario,
            phase=phase,
            input_ids=input_ids,
            tail_tokens=0 if turn == 0 else tail_tokens,
            expected_prefix_tokens=expected_prefix,
            mutation_index=mutation_index,
            extra_key=self.extra_key,
            routing_key=self.routing_key,
            prompt_fingerprint=token_fingerprint(input_ids),
        )

    def commit_output(self, output_ids: list[int]) -> None:
        if not output_ids:
            raise ValueError("a successful turn must contain at least one output token")
        self.history.extend(int(token_id) for token_id in output_ids)
        self.snapshots.append(list(self.history))


def build_conversations(
    tokenizer: TokenizerLike,
    count: int,
    initial_tokens: int,
    maximum_tail_tokens: int,
    turns: int,
    namespace: str,
    cell_id: str,
    seed: int,
) -> list[Conversation]:
    """Build conversations with exact initial/tail lengths and unique sentinels."""

    if count < 1:
        raise ValueError("conversation count must be positive")
    if initial_tokens < 16:
        raise ValueError("initial_tokens must be at least 16 to fit a unique sentinel")
    needed_per_conversation = max(1, maximum_tail_tokens * max(1, turns - 1) + 32)
    # One large pool is much faster to tokenize than count independent 8K documents.
    pool = make_token_pool(
        tokenizer,
        initial_tokens + count * needed_per_conversation + 4096,
        seed,
    )

    conversations = []
    for conversation_id in range(count):
        sentinel_text = (
            f"CAR DFLASH SYNTHETIC {namespace} {cell_id} "
            f"CONVERSATION {conversation_id}. "
        )
        sentinel = _encode(tokenizer, sentinel_text)
        bos = []
        if getattr(tokenizer, "bos_token_id", None) is not None:
            bos = [int(tokenizer.bos_token_id)]
        header = bos + sentinel
        if len(header) >= initial_tokens:
            raise ValueError(
                f"unique sentinel uses {len(header)} tokens, larger than "
                f"initial_tokens={initial_tokens}"
            )

        offset = (conversation_id * needed_per_conversation * 7) % max(1, len(pool))
        rotated = pool[offset:] + pool[:offset]
        initial = header + rotated[: initial_tokens - len(header)]
        continuation_start = initial_tokens - len(header)
        continuation = rotated[
            continuation_start : continuation_start + needed_per_conversation
        ]
        if len(initial) != initial_tokens:
            raise AssertionError("initial prompt construction is not exact")
        conversations.append(
            Conversation(
                conversation_id=conversation_id,
                initial_input_ids=initial,
                continuation_pool=continuation,
                namespace=namespace,
                cell_id=cell_id,
            )
        )
    return conversations


def load_tokenizer(model_or_path: str, trust_remote_code: bool = False) -> Any:
    """Load lazily so dry unit tests do not require Transformers."""

    try:
        from sglang.benchmark.utils import get_tokenizer

        return get_tokenizer(model_or_path, trust_remote_code=trust_remote_code)
    except (ImportError, TypeError):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_or_path,
            trust_remote_code=trust_remote_code,
        )
