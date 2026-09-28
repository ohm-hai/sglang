import contextlib
import logging
import time
from typing import List, Optional

import torch

from sglang.kernels.ops.speculative.topk1 import (
    draft_topk1_argmax_only,
    draft_topk1_postprocess,
)
from sglang.srt.configs.model_config import get_dsa_mtp_topk_width
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSAMTPSharedSparseIndices,
    QwenSparseAttnBackend,
    QwenSparseMultiStepDraftBackend,
)
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
from sglang.srt.layers.moe.utils import (
    draft_model_build_scope,
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner import (
    DecodeCudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.runtime_context import (
    get_context,
    get_device,
    get_exec,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.adaptive_spec_params import AdaptiveSpeculativeParams
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.dp_spec_prefill_coordination import (
    DPSpecPrefillCoordinationPlan,
)
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import (
    _eagle_prefill_tail_tokens,
    default_tree_mask_mode,
    eagle_sample,
    get_draft_recurrent_hidden_state_spec,
    organize_draft_results,
    per_step_draft_out_cache_loc,
)
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    prepare_for_draft,
    prepare_for_draft_extend,
    run_eagle_verify,
)
from sglang.srt.speculative.pp_draft_embedding import resolve_draft_embed_and_head
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_pp_context,
    draft_tp_context,
    fast_sample,
    get_plan_stream,
    load_token_map,
    renorm_draft_probs,
    sample_draft_proposal,
    select_top_k_tokens,
    spec_stage_span,
)
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import (
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    log_info_on_rank0,
)

_is_cpu = is_cpu()
_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()
_is_xpu = is_xpu()


logger = logging.getLogger(__name__)


def _qsa_index_share_requested(hf_config) -> bool:
    """--json-model-override-args writes top-level hf_config attributes, while
    checkpoint configs carry the flag on the nested text_config; read both."""
    text_config = getattr(hf_config, "text_config", hf_config)
    return bool(
        getattr(
            text_config,
            "index_share_for_mtp_iteration",
            getattr(hf_config, "index_share_for_mtp_iteration", False),
        )
    )


class EagleDraftWorker(EagleDraftWorkerBase):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self.target_worker = target_worker

        # Args for easy access
        self.device = get_device().device
        self.topk = get_spec().speculative_eagle_topk
        if get_spec().speculative_use_rejection_sampling:
            assert self.topk == 1, "Chain speculative sampling supports only topk=1"
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )

        self._rebuild_topk1_chain_buffers()

        # Use the same attention topology during draft construction and execution.
        self.draft_owns_attention = (
            get_parallel().enable_dp_attention
            and self.speculative_algorithm.is_eagle3()
        )
        if self.draft_owns_attention:
            ctx = draft_tp_context(get_parallel().attn_tp_group, owns_attention=True)
        else:
            ctx = empty_context()
        with (
            ctx,
            draft_pp_context(),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            draft_model_build_scope(),
        ):
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                nccl_port=nccl_port,
                is_draft_worker=True,
                # The draft runs at absolute target positions.
                context_length=target_worker.model_runner.model_config.context_len,
                random_seed=target_worker.random_seed,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
        self._init_dsa_index_share_state()
        # Eager draft-extend seed buffer (graph paths use their own static ones).
        self.dsa_extend_topk_buf: Optional[torch.Tensor] = None
        self.draft_tp_context = (
            draft_tp_context if get_parallel().enable_dp_attention else empty_context
        )
        self.tree_mask_mode = default_tree_mask_mode()

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """Allocate draft KV cache pools (called by scheduler)."""
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.init_token_map()
        self.init_lm_head()

        if get_spec().speculative_use_rejection_sampling:
            target_vocab_size = self.target_worker.model_config.vocab_size
            draft_vocab_size = (
                self.hot_token_id.shape[0]
                if self.hot_token_id is not None
                else target_vocab_size
            )
            # FIXME: support reduced (hot) draft vocab by scattering draft probs
            # into the target vocab via the d2t map before the sampling kernel.
            if draft_vocab_size != target_vocab_size:
                raise ValueError(
                    "--speculative-use-rejection-sampling requires the draft and "
                    f"target to share one vocab, but the draft vocab "
                    f"({draft_vocab_size}) != target vocab ({target_vocab_size})."
                )

    def init_attention_backends(self):
        with (
            draft_pp_context(),
            self.draft_tp_context(
                self.draft_runner.tp_group,
                owns_attention=self.draft_owns_attention,
            ),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker.init_attention_backends()
            self.init_attention_backend()

    def init_cuda_graphs(self):
        with (
            draft_pp_context(),
            self.draft_tp_context(
                self.draft_runner.tp_group,
                owns_attention=self.draft_owns_attention,
            ),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
            if check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE):
                self.draft_runner.init_prefill_cuda_graph(force_for_draft_worker=True)
            self._capture_cuda_graphs()

        if (c := self.draft_runner.canary_manager) is not None:
            c.mark_init_finished()

    def _init_dsa_index_share_state(self) -> None:
        # Populate DSA index-share fields from the draft runner's hf_config.
        # Reused by the attention unit-test harnesses, which skip __init__.
        hf_config = self.draft_runner.model_config.hf_config
        # Reuse the first draft step's DSA indexer topk across the rest;
        # topk == 1 only (select_top_k_tokens reorders rows, desyncing indices).
        self.index_share_for_mtp_iteration = (
            getattr(hf_config, "index_share_for_mtp_iteration", False)
            and self.topk == 1
        )
        # GLM-5.2 MTP IndexShare: seed reused indexer top-k from draft-extend
        # (last verified token), not draft-decode step 0.
        self.dsa_index_topk = getattr(hf_config, "index_topk", None)
        self.dsa_seed_topk_width = (
            get_dsa_mtp_topk_width(hf_config)
            if self.index_share_for_mtp_iteration and self.dsa_index_topk is not None
            else None
        )
        self.seed_dsa_topk_from_draft_extend = (
            self.index_share_for_mtp_iteration and self.dsa_seed_topk_width is not None
        )

    def init_token_map(self):
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if get_spec().speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif get_spec().speculative_token_map is not None:
            self.hot_token_id = load_token_map(get_spec().speculative_token_map)
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        from sglang.srt.lora.layers import unwrap_lora_layer

        embed, head = self._resolve_shared_embed_and_head()
        target_lm_head = unwrap_lora_layer(
            getattr(self.target_worker.model_runner.model, "lm_head", None)
        )

        def maybe_share_target_lm_head():
            if (
                target_lm_head is not None
                and self.hot_token_id is None
                and getattr(self.draft_runner.model, "hot_token_id", None) is None
                and hasattr(self.draft_runner.model, "set_lm_head_from_target")
            ):
                self.draft_runner.model.set_lm_head_from_target(target_lm_head)

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
                maybe_share_target_lm_head()
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None and head is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)
            maybe_share_target_lm_head()

    def _resolve_shared_embed_and_head(self):
        target_runner = self.target_worker.model_runner
        return resolve_draft_embed_and_head(
            target_model=target_runner.model,
            draft_model=self.draft_runner.model,
            model_path=target_runner.model_config.model_path,
            revision=target_runner.model_config.revision,
            load_config=target_runner.load_config,
        )

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners

        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
            seed_dsa_topk_from_draft_extend=self.seed_dsa_topk_from_draft_extend,
            qsa_profile=parse_qsa_profile(self.draft_runner.model_config.hf_config),
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        if self.draft_extend_attn_backend is not None:
            self.draft_runner.attn_backend = self.draft_extend_attn_backend
        self._configure_qsa_mtp_index_share()
        self.tree_mask_mode = default_tree_mask_mode()

    def _configure_qsa_mtp_index_share(self) -> None:
        """Reuse the draft-extend QSA selection across the MTP decode steps;
        chain speculation only: with topk > 1 decode rows are not request-major."""
        from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

        hf_config = self.draft_runner.model_config.hf_config
        if (
            not _qsa_index_share_requested(hf_config)
            or self.topk != 1
            or self.speculative_num_steps <= 1
            or not isinstance(self.draft_attn_backend, QwenSparseMultiStepDraftBackend)
            or not isinstance(self.draft_extend_attn_backend, QwenSparseAttnBackend)
        ):
            return
        if get_spec().speculative_adaptive:
            # Adaptive speculation switches SpecRuntimeState between the draft-extend
            # capture and the decode lookup; per-state index buffers would not match.
            logger.warning(
                "index_share_for_mtp_iteration is disabled under adaptive "
                "speculative decoding"
            )
            return
        layer_ids = sorted(
            {
                module.layer_id
                for module in self.draft_runner.model.modules()
                if isinstance(module, QSAIndexer)
            }
        )
        if not layer_ids:
            return
        pool = self.draft_runner.token_to_kv_pool
        # The expansion emits token_topk + ratio - 1 columns (top-k blocks
        # plus the uncompressed tail of the capture position).
        expanded_width = pool.qsa_token_topk + pool.qsa_compress_ratio - 1
        state = QSAMTPSharedSparseIndices(
            layer_ids=layer_ids,
            num_requests=self.draft_runner.req_to_token_pool.req_to_token.shape[0],
            token_topk=expanded_width,
            tail_width=get_spec().speculative_num_steps + 1,
            device=self.draft_runner.device,
        )
        for backend in (self.draft_attn_backend, self.draft_extend_attn_backend):
            backend.set_mtp_shared_sparse_indices(state)
        logger.info(
            "QSA MTP index sharing enabled: draft decode steps reuse the "
            f"draft-extend selection for layers {layer_ids}"
        )

    def _capture_cuda_graphs(self):
        """Capture the draft worker's own cuda graphs (decode + draft-extend)."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if _is_cpu or check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return

        if get_model().model_impl == "mindspore":
            return

        Device2DraftCudaGraphRunner = {
            "xpu": EAGLEDraftCudaGraphRunner,
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        decode_backend = get_exec().graph.cuda_graph_config.decode.backend
        capture_bs, _ = get_batch_sizes_to_capture(self.draft_runner)
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft decode CUDA graph begin. backend={decode_backend}, "
                f"num_tokens_per_req={self.topk}, bs={capture_bs}, "
                f"avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            capture_time = time.perf_counter() - tic
            self._specialized_graph_memory_usage["draft_decode"] = (
                self._specialized_graph_memory_usage.get("draft_decode", 0.0)
                + before_mem
                - after_mem
            )
            self._specialized_graph_time_usage["draft_decode"] = (
                self._specialized_graph_time_usage.get("draft_decode", 0.0)
                + capture_time
            )
            log_info_on_rank0(
                logger,
                "Capture draft decode CUDA graph end. "
                f"elapsed={capture_time:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, "
                f"avail mem={after_mem:.2f} GB.",
            )

        Device2ExtendCudaGraphRunner = {
            "xpu": EAGLEDraftExtendCudaGraphRunner,
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_draft_extend_graph = False
        if _is_hip:
            # Keep imports local so non-HIP environments do not require these.
            # aiter packs draft-extend support into the decode (multi-step)
            # backend; DSV4 exposes it on the draft-extend backend itself.
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4HipRadixBackend,
            )
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            supports_hip_draft_extend_graph = (
                isinstance(self.draft_attn_backend, AiterMultiStepDraftBackend)
                or isinstance(self.draft_extend_attn_backend, DeepseekV4HipRadixBackend)
                or isinstance(self.draft_extend_attn_backend, DeepseekSparseAttnBackend)
            )

        graph_supported_backend_types = [
            TritonAttnBackend,
            TRTLLMMLABackend,
            TRTLLMHAAttnBackend,
            TokenspeedMLABackend,
            FlashInferAttnBackend,
            QwenSparseAttnBackend,
        ]
        if _is_cuda or _is_musa:
            # DSA is CUDA-only; import lazily so non-CUDA builds don't pull in
            # deep_gemm and the rest of the sparse-attention stack at import time.
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            graph_supported_backend_types.append(DeepseekSparseAttnBackend)
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4AttnBackend,
            )

            graph_supported_backend_types.append(DeepseekV4AttnBackend)
        if _is_cuda:
            # FlashMLA is CUDA-only; import lazily so CPU builds don't pull
            # sgl_kernel.flash_mla at import time.
            from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

            graph_supported_backend_types.append(FlashMLABackend)

        graph_supported_backend = isinstance(
            self.draft_extend_attn_backend,
            tuple(graph_supported_backend_types),
        )
        supports_cuda_draft_extend_graph = (
            _is_cuda or _is_musa
        ) and graph_supported_backend
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        if (
            self.draft_extend_attn_backend
            and not envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH.get()
            and (
                _is_npu
                or _is_xpu
                or supports_cuda_draft_extend_graph
                or supports_hip_draft_extend_graph
            )
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend CUDA graph begin. backend={decode_backend}, "
                f"num_tokens_per_req={self.speculative_num_draft_tokens}, "
                f"bs={capture_bs}, avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            # draft_extend is the step's last shared-buffer-reading phase; its
            # read-done event is what the scheduler's WAR barrier waits on.
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            capture_time = time.perf_counter() - tic
            self._specialized_graph_memory_usage["draft_extend"] = (
                self._specialized_graph_memory_usage.get("draft_extend", 0.0)
                + before_mem
                - after_mem
            )
            self._specialized_graph_time_usage["draft_extend"] = (
                self._specialized_graph_time_usage.get("draft_extend", 0.0)
                + capture_time
            )
            log_info_on_rank0(
                logger,
                "Capture draft extend CUDA graph end. "
                f"elapsed={capture_time:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, "
                f"avail mem={after_mem:.2f} GB.",
            )

    def draft(self, batch: ScheduleBatch, *, with_topology: bool = False):
        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_run_decode_cuda_graph = prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )
        if (
            can_run_decode_cuda_graph
            and not forward_batch.forward_mode.is_idle()
            and self.seed_dsa_topk_from_draft_extend
            and draft_input.dsa_topk_indices is None
        ):
            can_run_decode_cuda_graph = False

        n_inner = self.speculative_num_steps - 1
        canary_outside_ctx = (
            c.with_ops_outside_graph(
                single_forward_indices=list(range(n_inner)),
                maybe_inaccurate_forward_batch=forward_batch,
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )

        with canary_outside_ctx:
            # Run draft
            if can_run_decode_cuda_graph:
                parent_list, top_scores_index, draft_tokens, draft_probs = (
                    self.cuda_graph_runner.execute(forward_batch)
                )
                if draft_probs is not None:
                    # draft_probs is the one graph output read after the target
                    # forward rather than by it, and it points into the graph's
                    # private memory pool. The pool recycles that block in the
                    # meantime -- in practice the DSA top-k mask lands there and
                    # eagle_sample sees -inf. Copy out at the boundary.
                    draft_probs = draft_probs.clone()
            else:
                if (
                    not forward_batch.forward_mode.is_idle()
                    and self.speculative_num_steps > 1
                ):
                    # Skip attention backend init for 1-step draft,
                    # `draft_forward` only does sample in this case.
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                    forward_batch.mark_forward_metadata_ready()
                parent_list, top_scores_index, draft_tokens, draft_probs = (
                    self.draft_forward(forward_batch)
                )

        verify_input = build_eagle_verify_input(
            batch,
            draft_input,
            parent_list,
            top_scores_index,
            draft_tokens,
            draft_probs,
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
        )
        if with_topology:
            # PP+spec relays the tree so every stage rebuilds the same verify
            # input; the mask build needs the topology this one was built from.
            # Returned rather than stashed on self so the caller owns lifetime.
            return verify_input, parent_list, top_scores_index
        return verify_input

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        if forward_batch.forward_mode.is_idle():
            return self._draft_forward_idle(forward_batch, spec_info)

        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = per_step_draft_out_cache_loc(
            out_cache_loc,
            forward_batch.batch_size,
            self.topk,
            self.speculative_num_steps,
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        if get_spec().speculative_use_rejection_sampling:
            draft_probs_list: List[torch.Tensor] = [spec_info.draft_probs]

        topk1_chain_fits = (
            self.topk == 1
            and topk_index.shape[0] <= self._topk1_parents_prealloc.shape[0]
        )
        # Materialize the chain directly only when the CUDA kernel can write
        # every subsequent column. Other topk=1 paths retain the token list and
        # assemble it with one final cat instead of launching a copy per step.
        draft_tokens_topk1 = None
        if (
            topk1_chain_fits
            and _is_cuda
            and self.hot_token_id is None
            and not get_spec().speculative_use_rejection_sampling
        ):
            draft_tokens_topk1 = torch.empty(
                (topk_index.shape[0], self.speculative_num_steps),
                dtype=topk_index.dtype,
                device=topk_index.device,
            )
            draft_tokens_topk1[:, :1].copy_(topk_index)

        # Forward multiple steps
        scores = None
        with IndexTopKShareState.mtp_iteration(
            forward_batch,
            enabled=self.index_share_for_mtp_iteration,
            keep_carry_seed=self.seed_dsa_topk_from_draft_extend,
        ):
            for i in range(self.speculative_num_steps):
                if draft_tokens_topk1 is not None:
                    input_ids = topk_index.flatten()
                else:
                    input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                        i, topk_p, topk_index, hidden_states, scores, self.topk
                    )
                    score_list.append(tree_info[0])
                    token_list.append(tree_info[1])
                    parents_list.append(tree_info[2])

                if i == self.speculative_num_steps - 1:
                    break

                forward_batch.input_ids = input_ids
                # Qwen3-MoE MTP uses a fused RoPE + KV-store path whose cache_loc
                # argument must be contiguous.
                if (
                    self.draft_runner.model_config.hf_config.architectures[0]
                    == "Qwen3MoeForCausalLMMTP"
                ):
                    out_cache_loc = out_cache_loc.contiguous()
                forward_batch.out_cache_loc = out_cache_loc[i]
                spec_info.hidden_states = hidden_states

                canary_index_ctx = (
                    c.with_active_single_forward_manager(i)
                    if (c := self.draft_runner.canary_manager) is not None
                    else contextlib.nullcontext()
                )
                with (
                    forward_context(
                        ForwardContext(
                            attn_backend=self.draft_attn_backend.attn_backends[i]
                        )
                    ),
                    canary_index_ctx,
                ):
                    logits_output = self.draft_runner.forward(
                        forward_batch
                    ).logits_output
                maybe_detect_nan(
                    logits_output.next_token_logits, f"draft_forward step {i}"
                )
                maybe_detect_inf(
                    logits_output.next_token_logits, f"draft_forward step {i}"
                )
                if get_spec().speculative_use_rejection_sampling:
                    probs, topk_p, topk_index = sample_draft_proposal(
                        logits_output.next_token_logits,
                        forward_batch.sampling_info.temperatures,
                        forward_batch.sampling_info.top_ks,
                    )
                    draft_probs_list.append(probs)
                    forward_batch.positions.add_(1)
                elif self.topk == 1:
                    if _is_cuda or _is_hip:
                        topk_p, topk_index = draft_topk1_postprocess(
                            logits_output.next_token_logits,
                            forward_batch.positions,
                            draft_tokens_topk1,
                            i + 1,
                        )
                    else:
                        topk_index = torch.argmax(
                            logits_output.next_token_logits, dim=-1, keepdim=True
                        )
                        topk_p = torch.ones_like(topk_index, dtype=torch.float32)
                        forward_batch.positions.add_(1)
                else:
                    probs = renorm_draft_probs(
                        logits_output.next_token_logits,
                        forward_batch.sampling_info,
                        get_spec().speculative_use_rejection_sampling,
                    )
                    topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
                    forward_batch.positions.add_(1)
                if self.draft_runner.model_config.model_is_mrope:
                    forward_batch.mrope_positions.add_(1)
                maybe_detect_oob(
                    topk_index,
                    0,
                    logits_output.next_token_logits.shape[-1],
                    f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
                )
                if self.hot_token_id is not None:
                    topk_index = self.hot_token_id[topk_index]
                hidden_states = logits_output.hidden_states

        draft_probs = (
            torch.stack(draft_probs_list, dim=1)
            if get_spec().speculative_use_rejection_sampling
            else None
        )

        # Organize the results
        if draft_tokens_topk1 is not None:
            bs = draft_tokens_topk1.shape[0]
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            return parent_list, top_scores_index, draft_tokens_topk1, draft_probs

        if topk1_chain_fits:
            bs = token_list[0].shape[0]
            draft_tokens = torch.cat(token_list, dim=1)
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            return parent_list, top_scores_index, draft_tokens, draft_probs

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens, draft_probs

    def _draft_forward_idle(
        self, forward_batch: ForwardBatch, spec_info: EagleDraftInput
    ):
        """Run eager idle-rank collectives without materializing draft state."""
        input_ids = forward_batch.input_ids
        out_cache_loc = forward_batch.out_cache_loc
        hidden_states = spec_info.hidden_states

        # ModelRunner pads and unpads the empty batch on every call. Avoid the
        # normal tree/cache-layout path: idle outputs are discarded when the
        # verify input is built, but every rank must still enter each forward.
        for i in range(self.speculative_num_steps - 1):
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc
            spec_info.hidden_states = hidden_states
            canary_index_ctx = (
                c.with_active_single_forward_manager(i)
                if (c := self.draft_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            with (
                forward_context(
                    ForwardContext(
                        attn_backend=self.draft_attn_backend.attn_backends[i]
                    )
                ),
                canary_index_ctx,
            ):
                self.draft_runner.forward(forward_batch)

        return None, None, None, None

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # Construct input_ids
        if not batch.forward_mode.is_idle():
            # Chunked-prefill-aware tail tokens (see PR #26329).
            tail_tokens = _eagle_prefill_tail_tokens(batch, next_token_ids)

            new_input_ids = torch.empty_like(batch.input_ids)
            if mm_input_embeds is not None:
                # Rotate mm embeddings the same way as input_ids: shift left by
                # one per request so they stay aligned with the rotated ids. The
                # last position per request is filled by the draft model's own
                # embed_tokens lookup on next_token_ids (see DeepseekModelNextN).
                rotated_mm = torch.empty_like(mm_input_embeds)
            pt = 0
            for i, extend_len in enumerate(batch.extend_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                new_input_ids[pt : pt + extend_len].copy_(
                    torch.cat((input_ids[1:], tail_tokens[i].reshape(1)))
                )
                if mm_input_embeds is not None:
                    rotated_mm[pt : pt + extend_len - 1].copy_(
                        mm_input_embeds[pt + 1 : pt + extend_len]
                    )
                pt += extend_len
            assert pt == batch.input_ids.numel()
            batch.input_ids = new_input_ids
            if mm_input_embeds is not None:
                mm_input_embeds = rotated_mm

        # Draft-extend spec_info for the extend forward; carries only
        # hidden_states + shape info.
        batch.spec_info = EagleDraftExtendInput(
            hidden_states=target_hidden_states,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        # Run forward (LAST mode: only the final hidden state per request,
        # to feed the next draft step which expects [bs, hidden_dim]).
        # STANDALONE skips hidden states end-to-end.
        capture_hidden_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        forward_batch = ForwardBatch.init_new(
            batch,
            self.draft_runner,
            capture_hidden_mode=capture_hidden_mode,
            return_hidden_states_before_norm=False,
        )
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds

        # Seed the first draft-decode loop from each request's last prefill
        # position. Gather last-per-req before the copy (prefill can be long).
        seed_from_extend = (
            self.seed_dsa_topk_from_draft_extend
            and not forward_batch.forward_mode.is_idle()
        )
        if seed_from_extend:
            bs = forward_batch.batch_size
            forward_batch.spec_info.dsa_seed_topk_capture = (
                self._get_dsa_extend_topk_buf(bs)
            )
            forward_batch.spec_info.dsa_seed_topk_select = (
                torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            ).long()

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            logits_output = self.draft_runner.forward(forward_batch).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")
        maybe_detect_inf(logits_output.next_token_logits, "draft_extend_for_prefill")

        prefill_dsa_topk = None
        if seed_from_extend:
            prefill_dsa_topk = self.dsa_extend_topk_buf[:bs].clone()

        # Assemble the next-iter draft spec_info from the extend output.
        use_rejection_sampling = get_spec().speculative_use_rejection_sampling
        probs = renorm_draft_probs(
            logits_output.next_token_logits,
            batch.sampling_info,
            use_rejection_sampling,
        )
        if use_rejection_sampling:
            topk_p, topk_index = fast_sample(probs, num_samples=1)
        else:
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        return EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            draft_probs=probs if use_rejection_sampling else None,
            hidden_states=logits_output.hidden_states,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            dsa_topk_indices=prefill_dsa_topk,
        )

    def _get_dsa_extend_topk_buf(self, num_tokens: int) -> torch.Tensor:
        buf = self.dsa_extend_topk_buf
        if buf is None or buf.shape[0] < num_tokens:
            buf = torch.full(
                (num_tokens, self.dsa_seed_topk_width),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self.dsa_extend_topk_buf = buf
        return buf[:num_tokens]

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        # Batch 2: Draft extend
        draft_extend_input = EagleDraftExtendInput(
            hidden_states=batch_result.logits_output.hidden_states,
            # accept_lens includes the bonus token; correct drafts exclude it.
            num_correct_drafts=batch_result.accept_lens - 1,
            num_accept_tokens=batch_result.accept_lens,
            # Draft-extend fills the whole tree width (num_draft_tokens) per req,
            # not num_steps + 1, so DP MLP-sync padding stays consistent for topk > 1.
            num_tokens_per_req=self.speculative_num_draft_tokens,
            num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
        )
        select_index = (
            torch.arange(
                0,
                len(batch.seq_lens) * self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                device=self.device,
            )
            + batch_result.accept_lens
            - 1
        )

        # Cast to int64 before entering plan stream to avoid cross-stream
        # synchronization issues with .to() inside the plan stream context.
        next_token_ids = batch_result.next_token_ids.to(torch.int64)

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = prepare_for_draft_extend(
                draft_extend_input,
                batch,
                next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
                return_hidden_states_before_norm=False,
            )

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        # Run draft extend batch in the main compute stream
        can_run_decode_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run_graph(forward_batch)
        )

        # Eager path publishes the indexer top-k into a worker buffer (the graph
        # path uses the runner's static buffer). Gathered at select_index below.
        if self.seed_dsa_topk_from_draft_extend and not can_run_decode_cuda_graph:
            forward_batch.spec_info.dsa_seed_topk_capture = (
                self._get_dsa_extend_topk_buf(forward_batch.input_ids.shape[0])
            )

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            if can_run_decode_cuda_graph:
                draft_logits_output = self.cuda_graph_runner_for_draft_extend.execute(
                    forward_batch, select_index
                )
            else:
                draft_logits_output = self.draft_runner.forward(
                    forward_batch
                ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
        )
        maybe_detect_inf(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
        )

        # Gather the per-request last-position indexer top-k as the next loop's
        # seed (select_index already picks the last accepted position per req).
        dsa_seed_topk_indices = None
        if self.seed_dsa_topk_from_draft_extend:
            if can_run_decode_cuda_graph:
                dsa_extend_topk_capture = self.cuda_graph_runner_for_draft_extend.buffers.dsa_seed_topk_capture
            else:
                dsa_extend_topk_capture = forward_batch.spec_info.dsa_seed_topk_capture
            # Fancy indexing returns a fresh tensor (detached from the buffer).
            dsa_seed_topk_indices = dsa_extend_topk_capture[select_index]

        # Reorganize the spec info for the next batch
        if not can_run_decode_cuda_graph:
            draft_logits_output.next_token_logits = (
                draft_logits_output.next_token_logits[select_index]
            )
            if draft_logits_output.hidden_states is not None:
                draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                    select_index
                ]
        # Selected-row top-k remains worker-owned for both graph and eager
        # paths; the graph runner only moves the row selection before lm_head.
        if get_spec().speculative_use_rejection_sampling:
            ret_draft_probs, ret_topk_p, ret_topk_index = sample_draft_proposal(
                draft_logits_output.next_token_logits,
                batch.sampling_info.temperatures,
                batch.sampling_info.top_ks,
            )
        elif self.topk == 1 and _is_hip:
            ret_topk_p, ret_topk_index = draft_topk1_argmax_only(
                draft_logits_output.next_token_logits
            )
            ret_draft_probs = None
        elif self.topk == 1 and not _is_hip:
            # Gated to CUDA: see #26358 — ROCm's argmax tie-break corrupts
            # MTP draft selection on FP8 logits.
            ret_topk_index = torch.argmax(
                draft_logits_output.next_token_logits, dim=-1, keepdim=True
            )
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
            ret_draft_probs = None
        else:
            probs = renorm_draft_probs(
                draft_logits_output.next_token_logits,
                batch.sampling_info,
                get_spec().speculative_use_rejection_sampling,
            )
            ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
            ret_draft_probs = None
        ret_hidden_states = draft_logits_output.hidden_states

        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )
        if get_spec().speculative_use_rejection_sampling:
            next_draft_input.draft_probs = ret_draft_probs
        if self.seed_dsa_topk_from_draft_extend:
            next_draft_input.dsa_topk_indices = dsa_seed_topk_indices


class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        # Parse arguments
        self.server_args = server_args
        self.topk = get_spec().speculative_eagle_topk
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = get_device().device
        self._target_worker = target_worker
        self.page_size = get_schedule().page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )

        self.enable_dp_spec_prefill_coordination = (
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get()
        )

        # Only the last PP stage runs the draft; other EAGLEWorkerV2 instances
        # return proxies so scheduler dispatch remains rank-uniform.
        self._hosts_draft = get_parallel().pp_group.is_last_rank
        self._draft_worker = (
            EagleDraftWorker(
                server_args,
                gpu_id,
                nccl_port,
                target_worker,
            )
            if self._hosts_draft
            else None
        )

        # Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None
        if get_spec().speculative_adaptive and self._hosts_draft:
            self.adaptive_controller = AdaptiveController(
                self,
                AdaptiveSpeculativeParams(
                    initial_steps=self.speculative_num_steps,
                    cfg_path=get_spec().speculative_adaptive_config,
                ),
            )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

    @property
    def last_shared_read_runner(self):
        # Per the base contract: the step's last shared-buffer-reading phase is
        # draft_extend when this rank owns the draft. Non-last PP ranks execute
        # only the target prefill.
        if self._draft_worker is None:
            return self._target_worker.model_runner
        return self._draft_worker.draft_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # Every attn backend a spec_v2 forward touches; consumed by
        # decide_needs_cpu_seq_lens to gate the seq_lens_cpu D2H.
        if self._draft_worker is None:
            return (self._target_worker.model_runner.attn_backend,)
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend
            or self._draft_worker.draft_runner.attn_backend,
        )

    def init_cuda_graphs(self):
        super().init_cuda_graphs()
        # Build adaptive runtime states after target and draft backends exist.
        if self.adaptive_controller is not None:
            with (
                self._draft_worker.draft_tp_context(
                    self._draft_worker.draft_runner.tp_group,
                    owns_attention=self._draft_worker.draft_owns_attention,
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.adaptive_controller.register(
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        draft_attn_backend=self._draft_worker.draft_attn_backend,
                        cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                        target_attn_backend=self._target_worker.model_runner.attn_backend,
                        target_graph_runner=self._target_worker.model_runner.decode_cuda_graph_runner,
                        draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                        cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
                    )
                )
                self.adaptive_controller.init_states(
                    cuda_graph_bs=(
                        None
                        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)
                        else get_exec().graph.cuda_graph_bs_decode
                    ),
                )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        # verify-in-mixed: a MIXED batch carrying drafted chains (mix_chain_len > 1)
        # runs target verification on the chain rows inside the same forward as the
        # prefill chunk, so verification rides the prefill instead of degrading the
        # running requests to a 1-token extend.
        if (
            batch.forward_mode.is_mixed()
            and getattr(batch, "mix_chain_len", 1) > 1
            and self._draft_worker is not None
        ):
            return self._forward_mixed_verify(
                batch,
                on_publish=on_publish,
                grammar_barrier=grammar_barrier,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            if not (
                batch.is_extend_in_batch and self.enable_dp_spec_prefill_coordination
            ):
                return self._forward_prefill_batch(batch, on_publish, pp_proxy_tensors)
            else:
                if batch.dp_spec_prefill_coordination_metadata is None:
                    raise RuntimeError("Missing DP spec/prefill coordination metadata")
                plan = DPSpecPrefillCoordinationPlan(
                    *batch.dp_spec_prefill_coordination_metadata,
                    draft_width=self.topk,
                    verify_width=self.speculative_num_draft_tokens,
                )
                if not plan.heterogeneous:
                    return self._forward_prefill_batch(
                        batch, on_publish, pp_proxy_tensors
                    )
                else:
                    return self._forward_dp_spec_prefill_coordination(
                        batch, plan, on_publish, grammar_barrier, pp_proxy_tensors
                    )
        else:
            self.activate_step_by_batch(batch.seq_lens.shape[0])

            if batch.spec_info is None:
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
                    self.draft_worker.draft_runner
                )
                batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=hidden_size,
                    dtype=hidden_dtype,
                    topk=self.topk,
                    capture_hidden_mode=capture_mode,
                    vocab_size=self.target_worker.model_config.vocab_size,
                )
            if batch.spec_info is not None and batch.spec_info.is_verify_input():
                # PP+spec: the scheduler pre-built this round's verify input
                # from relayed per-req chains — it must match what earlier
                # stages already ran, so do not re-draft here.
                verify_input = batch.spec_info
            elif self.speculative_num_steps == 0:
                # Drafting disabled (high batch size). _draft_extend below still
                # runs, keeping draft KV warm for when the batch shrinks.
                verify_input = self._build_trivial_verify_input(batch)
            else:
                with (
                    self.draft_worker.draft_tp_context(
                        self.draft_worker.draft_runner.tp_group,
                        owns_attention=self.draft_worker.draft_owns_attention,
                    ),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft"),
                ):
                    verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            batch_output = self.verify(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                grammar_barrier=grammar_barrier,
            )
            # Publish before draft_extend so the fence is at verify-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            if (
                self.speculative_num_steps == 0
                and envs.SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND.get()
            ):
                self._stub_skipped_draft_extend(batch, batch_output)
            else:
                with (
                    self.draft_worker.draft_tp_context(
                        self.draft_worker.draft_runner.tp_group,
                        owns_attention=self.draft_worker.draft_owns_attention,
                    ),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft_extend"),
                ):
                    self.draft_worker._draft_extend_for_decode(batch, batch_output)

            if (
                get_parallel().pp_size > 1
                and not batch.forward_mode.is_idle()
                and self.speculative_num_steps > 0
            ):
                # PP tail-draft: draft the NEXT round's chain now — earlier
                # stages must have the tokens before running their half of the
                # next verify forward, so drafting cannot wait for the next
                # iteration. Mimic the head-of-iteration state draft() expects;
                # the scheduler's forward isolation reverts these SB edits, and
                # the chain rides out on batch_output.
                batch.spec_info = batch_output.next_draft_input
                batch.seq_lens = batch_output.new_seq_lens
                batch.forward_mode = ForwardMode.DECODE
                # eagle_prepare_for_verify left the verify tokens here; the
                # head-of-iteration draft always sees None (the scheduler
                # clears it), so mirror that state.
                batch.input_ids = None
                # Attention metadata planning reads the CPU copies; one D2H
                # per round (TODO: async or upper-bound estimate).
                batch.seq_lens_cpu = batch_output.new_seq_lens.to("cpu")
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
                with (
                    self.draft_worker.draft_tp_context(
                        self.draft_worker.draft_runner.tp_group,
                        owns_attention=self.draft_worker.draft_owns_attention,
                    ),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft"),
                ):
                    next_verify_input, parent_list, top_scores_index = (
                        self.draft_worker.draft(batch, with_topology=True)
                    )
                batch_output.next_verify_chain = next_verify_input.draft_token
                # The tree shape is data-dependent once topk > 1, so the other
                # stages cannot re-derive it; relay it alongside the tokens.
                # clone(): both come out of cuda-graph-owned buffers under
                # decode replay and would be overwritten before the relay.
                batch_output.next_verify_parent_list = parent_list.clone()
                batch_output.next_verify_top_scores_index = top_scores_index.clone()

            return batch_output

    def _forward_prefill_batch(
        self, batch, on_publish=None, pp_proxy_tensors=None, coordination_plan=None
    ):
        # Target prefill
        target_capture_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch_output = self.target_worker.forward_batch_generation(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            capture_hidden_mode=target_capture_mode,
        )

        # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
        # Extend processed L prompt tokens; next verify iter expects same L.
        batch_output.new_seq_lens = batch.seq_lens
        # Publish before draft_extend so the fence is at target-end.
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        # A rank that does not host the draft (prefill-side PP builds it only on
        # the last stage) forwards the target's proxy tensors and stops here.
        if self._draft_worker is None:
            return batch_output

        if coordination_plan is not None:
            coordination_plan.apply(
                batch,
                "draft_extend",
                get_parallel().attn_dp_rank,
                local_only=(
                    len(batch.global_num_tokens) == 1
                    or self.draft_worker.draft_owns_attention
                ),
            )

        # Draft prefill
        with (
            self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group,
                owns_attention=self.draft_worker.draft_owns_attention,
            ),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft_extend"),
        ):
            batch_output.next_draft_input = self.draft_worker._draft_extend_for_prefill(
                batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
                batch_output.logits_output.mm_input_embeds,
            )
            return batch_output

    def _forward_dp_spec_prefill_coordination(
        self, batch, plan, on_publish, grammar_barrier, pp_proxy_tensors
    ):
        """Run draft, target, and draft-extend with each rank's local mode."""
        rank = get_parallel().attn_dp_rank
        target_local_only = len(batch.global_num_tokens) == 1
        draft_local_only = target_local_only or self.draft_worker.draft_owns_attention
        is_prefill = batch.forward_mode.is_extend()
        draft_batch = batch
        if is_prefill:
            draft_batch = ScheduleBatch.init_new(
                [],
                batch.req_to_token_pool,
                batch.token_to_kv_pool_allocator,
                batch.tree_cache,
                batch.model_config,
                batch.enable_overlap,
                batch.spec_algorithm,
            )
            draft_batch.prepare_for_idle()
            draft_batch.global_num_tokens = batch.global_num_tokens
            draft_batch.global_num_tokens_for_logprob = (
                batch.global_num_tokens_for_logprob
            )
        if draft_batch.spec_info is None:
            hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
                self.draft_worker.draft_runner
            )
            draft_batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=hidden_size,
                dtype=hidden_dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )
        plan.apply(draft_batch, "draft", rank, local_only=draft_local_only)
        with (
            self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group,
                owns_attention=self.draft_worker.draft_owns_attention,
            ),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft"),
        ):
            verify_input = self.draft_worker.draft(draft_batch)

        plan.apply(batch, "target", rank, local_only=target_local_only)
        if is_prefill:
            result = self._forward_prefill_batch(
                batch, on_publish, pp_proxy_tensors, coordination_plan=plan
            )
            # Pin the temporary idle tensors through the overlap lifetime.
            result.extra_keep_alive_refs = list(result.extra_keep_alive_refs or ()) + [
                draft_batch
            ]
            return result

        batch.spec_info = verify_input
        result = self.verify(batch, grammar_barrier=grammar_barrier)
        if on_publish is not None:
            on_publish(result.new_seq_lens)
        plan.apply(batch, "draft_extend", rank, local_only=draft_local_only)
        with (
            self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group,
                owns_attention=self.draft_worker.draft_owns_attention,
            ),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft_extend"),
        ):
            self.draft_worker._draft_extend_for_decode(batch, result)
        return result

    def _build_trivial_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """Build a 1-node EagleVerifyInput rooted at the previous bonus token.

        Used when ``speculative_num_steps == 0`` to skip drafting while still
        routing through the existing TARGET_VERIFY graph captured at
        ``draft_token_num=1``: the kernel always accepts the root and samples
        one new bonus token from target logits -- functionally a plain decode.
        """
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                topk=self.topk, spec_steps=0, num_verify_tokens=1, device=self.device
            )

        draft_input: EagleDraftInput = batch.spec_info
        bs = batch.seq_lens.shape[0]
        device = self.device

        retrieve_index = torch.arange(bs, dtype=torch.long, device=device).unsqueeze(1)
        retrieve_next_token = torch.full((bs, 1), -1, dtype=torch.long, device=device)
        retrieve_next_sibling = torch.full((bs, 1), -1, dtype=torch.long, device=device)

        attn_backend = self._target_worker.model_runner.attn_backend
        verify_mask = attn_backend.verify_mask
        # Every position in a 1-node tree is visible, so an all-True fill is
        # correct under either layout.
        if verify_mask is not None and verify_mask.fits(bs):
            custom_mask = verify_mask.buffer
            custom_mask.fill_(True)
        else:
            if batch.seq_lens_sum is not None:
                seq_lens_sum = batch.seq_lens_sum
            elif batch.seq_lens_cpu is not None:
                seq_lens_sum = int(batch.seq_lens_cpu.sum())
            else:
                seq_lens_sum = bs * attn_backend.max_context_len
            custom_mask = torch.ones(seq_lens_sum + bs, dtype=torch.bool, device=device)

        positions = batch.seq_lens.to(torch.int64)

        return EagleVerifyInput(
            draft_token=draft_input.bonus_tokens,
            custom_mask=custom_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=0,
            topk=self.topk,
            draft_token_num=1,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def _stub_skipped_draft_extend(
        self, batch: ScheduleBatch, batch_output: GenerationBatchResult
    ) -> None:
        """Fill shape-valid stubs on next_draft_input when draft_extend is skipped.

        ``verify`` already set ``bonus_tokens`` (the only field the next steps=0
        verify reads). The overlap FutureMap still stashes topk_p/topk_index/
        hidden_states, so provide zeroed tensors of the right shape. They are never
        consumed while at steps=0; an upshift to steps>0 would draft from this stale
        state (cold recovery), which is the documented cost of this experimental flag.
        """
        next_draft_input: EagleDraftInput = batch_output.next_draft_input
        bs = batch.seq_lens.shape[0]
        device = self.device
        next_draft_input.topk_p = torch.zeros(
            (bs, self.topk), dtype=torch.float32, device=device
        )
        next_draft_input.topk_index = torch.zeros(
            (bs, self.topk), dtype=torch.int64, device=device
        )
        hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
            self.draft_worker.draft_runner
        )
        if hidden_size is not None:
            next_draft_input.hidden_states = torch.zeros(
                (bs, hidden_size),
                dtype=hidden_dtype,
                device=device,
            )

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(
                num_correct_drafts_per_req, batch_size=batch_size
            )

    def activate_step_by_batch(self, batch_size: int) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.activate_step_by_batch(batch_size)

    # -- Adaptive speculative decoding protocol --

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs=None,
    ) -> SpecRuntimeState:
        """Build a SpecRuntimeState for the given step configuration."""
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        with self._override_worker_state(
            speculative_num_steps,
            speculative_num_draft_tokens,
            cuda_graph_bs=cuda_graph_bs,
        ):
            self._draft_worker.init_attention_backend()
            self._draft_worker._capture_cuda_graphs()

            # Build target attention backend and CUDA graph runner
            target_model_runner = self._target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                target_model_runner.init_new_workspace = backup_init

            target_graph_runner = None
            if not check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
                TargetGraphRunnerCls = (
                    NPUGraphRunner if _is_npu else DecodeCudaGraphRunner
                )
                target_graph_before_mem = get_available_gpu_memory(
                    self.device, self.gpu_id
                )
                target_graph_tic = time.perf_counter()
                target_graph_runner = TargetGraphRunnerCls(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )
                target_graph_after_mem = get_available_gpu_memory(
                    self.device, self.gpu_id
                )
                target_graph_time = time.perf_counter() - target_graph_tic
                self._additional_graph_memory_usage["target_verify"] = (
                    self._additional_graph_memory_usage.get("target_verify", 0.0)
                    + target_graph_before_mem
                    - target_graph_after_mem
                )
                self._additional_graph_time_usage["target_verify"] = (
                    self._additional_graph_time_usage.get("target_verify", 0.0)
                    + target_graph_time
                )

            state = SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                draft_attn_backend=self._draft_worker.draft_attn_backend,
                cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )

        return state

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        """Apply a pre-built runtime state to this worker."""
        if self.speculative_num_steps == state.speculative_num_steps:
            return

        log_info_on_rank0(
            logger,
            "Switch adaptive runtime state: "
            f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
            f"draft_tokens {self.speculative_num_draft_tokens} -> "
            f"{state.speculative_num_draft_tokens}",
        )

        # Top-level
        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens

        # Draft side
        dw = self._draft_worker
        dw.speculative_num_steps = state.speculative_num_steps
        dw.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        dw.draft_attn_backend = state.draft_attn_backend
        dw.draft_runner.draft_attn_backend = state.draft_attn_backend
        dw.cuda_graph_runner = state.cuda_graph_runner
        dw.draft_extend_attn_backend = state.draft_extend_attn_backend
        # Keep the runner's attn_backend in step with the active draft-extend
        # backend (the draft-extend forward reads draft_runner.attn_backend);
        # mirrors init_attention_backend. When None, the runner keeps its
        # initialized backend (consistent across step configs).
        if state.draft_extend_attn_backend is not None:
            dw.draft_runner.attn_backend = state.draft_extend_attn_backend
        dw.cuda_graph_runner_for_draft_extend = state.cuda_graph_runner_for_draft_extend
        dw._rebuild_topk1_chain_buffers()

        # Target side
        self._target_worker.model_runner.attn_backend = state.target_attn_backend
        self._target_worker.model_runner.decode_cuda_graph_runner = (
            state.target_graph_runner
        )

        # Sync server_args
        get_context().override(
            "adaptive_spec.restore",
            speculative_num_steps=state.speculative_num_steps,
            speculative_num_draft_tokens=state.speculative_num_draft_tokens,
        )

    @contextlib.contextmanager
    def _override_worker_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ):
        """Temporarily override server_args and worker attributes for graph capture."""
        dw = self._draft_worker
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            dw.speculative_num_steps,
            dw.speculative_num_draft_tokens,
            dw.draft_attn_backend,
            dw.draft_extend_attn_backend,
            dw.draft_runner.draft_attn_backend,
            dw.draft_runner.attn_backend,
            dw.cuda_graph_runner,
            dw.cuda_graph_runner_for_draft_extend,
            get_spec().speculative_num_steps,
            get_spec().speculative_num_draft_tokens,
            get_exec().graph.cuda_graph_bs_decode,
            get_exec().graph.disable_cuda_graph,
        )

        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        dw.speculative_num_steps = speculative_num_steps
        dw.speculative_num_draft_tokens = speculative_num_draft_tokens
        get_context().override(
            "adaptive_spec.capture_override",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        if cuda_graph_bs is not None:
            # BS-aware adaptive spec may prune cuda_graph_bs to an empty list
            # for steps that no BS range uses (e.g. step=1). Disable graph
            # capture for those steps; restore in finally so subsequent steps
            # are not affected.
            get_context().override(
                "adaptive_spec.capture_override",
                cuda_graph_bs_decode=cuda_graph_bs,
                **({"disable_cuda_graph": True} if not cuda_graph_bs else {}),
            )
        dw._rebuild_topk1_chain_buffers()

        try:
            yield
        finally:
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                dw.speculative_num_steps,
                dw.speculative_num_draft_tokens,
                dw.draft_attn_backend,
                dw.draft_extend_attn_backend,
                dw.draft_runner.draft_attn_backend,
                dw.draft_runner.attn_backend,
                dw.cuda_graph_runner,
                dw.cuda_graph_runner_for_draft_extend,
            ) = backup[:10]
            get_context().override(
                "adaptive_spec.capture_restore",
                speculative_num_steps=backup[10],
                speculative_num_draft_tokens=backup[11],
                cuda_graph_bs_decode=backup[12],
                disable_cuda_graph=backup[13],
            )
            dw._rebuild_topk1_chain_buffers()

    def _forward_mixed_verify(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """verify-in-mixed: run target verification on the drafted chains carried by
        the running rows of a MIXED batch, in the same forward as the prefill chunk.

        Layout of the merged batch (built by ScheduleBatch.mix_with_running):
          [ prefill rows (variable extend_len) | running rows (chain_len each) ]
        The running rows' input_ids were placeholder-filled by
        resolve_forward_inputs (committed bonus token repeated chain_len times);
        here we overwrite them with the actual drafted chain, run one mixed target
        forward, then accept/rollback the chains and seed the draft for both row
        kinds. chain_len == speculative_num_draft_tokens and topk == 1, so each
        chain is a linear causal extend.
        """
        chain_len = int(batch.mix_chain_len)
        running_bs = len(batch.decoding_reqs) if batch.decoding_reqs is not None else 0
        total_bs = batch.seq_lens.shape[0]
        prefill_bs = total_bs - running_bs
        device = self.device

        # -- 1. Draft the chain for the running requests ---------------------
        # batch.spec_info is the running batch's EagleDraftInput (prefill rows
        # contributed None). draft() sizes its buffers off batch.seq_lens, so
        # narrow the batch view to the running rows for the duration of the
        # draft, then restore. The running rows are the tail of the merged batch.
        self.activate_step_by_batch(running_bs)
        _saved = (
            batch.seq_lens,
            batch.seq_lens_cpu,
            batch.seq_lens_sum,
            batch.req_pool_indices,
            batch.reqs,
            batch.out_cache_loc,
            batch.input_ids,
        )
        # The merged tails hold post-chain lengths (base + chain_len) for the
        # mixed target forward, but the draft is a decode-mode forward that
        # must sit at the committed base (bonus token pending), exactly as in
        # the normal decode flow. Passing the merged length makes the DSA
        # decode path read chain_len stale draft-KV slots from previous
        # rounds, degrading chain quality. Device tails: base = merged -
        # chain_len; CPU tails: base = req.seqlen - 1.
        batch.seq_lens = batch.seq_lens[prefill_bs:] - chain_len
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_cpu = batch.seq_lens_cpu[prefill_bs:] - 1
        else:
            # MIXED batches leave seq_lens_cpu None; the draft's DSA prefill
            # metadata path asserts on it, so backfill from the device tensor
            # (one D2H sync per mixed-verify step -- acceptable at this scale).
            batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = int(batch.seq_lens.sum())
        batch.req_pool_indices = batch.req_pool_indices[prefill_bs:]
        batch.reqs = batch.reqs[prefill_bs:]
        # The draft reads its input tokens from spec_info (topk_index), not
        # batch.input_ids; the merged batch's input_ids is the full mixed token
        # tensor (prefill + chain rows), which would size the draft FB's token
        # axis wrong. Clear it for the duration of the draft.
        batch.input_ids = None
        # The draft CUDA graph is captured for a fixed decode batch size; the
        # mixed batch's running-row count is dynamic and will not match the
        # captured buffers. Run the draft eagerly for the mixed-verify path
        # (draft is ~1.3ms vs ~33ms verify, so eager draft is acceptable). TODO:
        # token-bucket-keyed draft graphs for mixed shapes (todo #9).
        _saved_draft_graph = self.draft_worker.cuda_graph_runner
        self.draft_worker.cuda_graph_runner = None
        # The draft is a decode-mode forward over the running rows; the merged
        # batch still carries forward_mode=MIXED, which would route the DSA
        # backend down the extend path (extend_seq_lens_cpu is None for a
        # decode draft). Set DECODE for the duration of the draft, then restore.
        _saved_forward_mode = batch.forward_mode
        batch.forward_mode = ForwardMode.DECODE
        # The eager draft mutates forward_batch.input_ids per step (topk=1 ->
        # 1 token/req) while batch_size stays running_bs, so the EagerRunner's
        # token-axis registry copy mismatches (input_ids [bs] vs seq_lens [bs]
        # but out_cache_loc [bs*topk*steps]). Skip the input-copy for the draft
        # (the draft writes its inputs in place and doesn't need the copy).
        _saved_no_copy = envs.SGLANG_EAGER_INPUT_NO_COPY.get()
        envs.SGLANG_EAGER_INPUT_NO_COPY.set(True)
        try:
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group,
                    owns_attention=self.draft_worker.draft_owns_attention,
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft"),
            ):
                verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
        finally:
            envs.SGLANG_EAGER_INPUT_NO_COPY.set(_saved_no_copy)
            batch.forward_mode = _saved_forward_mode
            self.draft_worker.cuda_graph_runner = _saved_draft_graph
            (
                batch.seq_lens,
                batch.seq_lens_cpu,
                batch.seq_lens_sum,
                batch.req_pool_indices,
                batch.reqs,
                batch.out_cache_loc,
                batch.input_ids,
            ) = _saved
        assert verify_input.is_verify_input()

        # -- 2. Write the drafted chain into the running rows' input_ids -----
        # verify_input.draft_token is [running_bs * chain_len] laid out
        # request-major, matching the running rows' order at the batch tail.
        prefill_tokens = batch.input_ids[: batch.input_ids.numel() - running_bs * chain_len]
        batch.input_ids = torch.cat([prefill_tokens, verify_input.draft_token])

        # The running batch's spec_info carries a draft-shaped `positions`
        # (repeat_interleave(topk=1) -> running_bs tokens). ForwardBatch.init_new
        # would adopt it and skip compute_position for the full mixed token
        # count. Clear it so the mixed extend recomputes positions for all
        # prefill + chain tokens.
        if batch.spec_info is not None and getattr(batch.spec_info, "positions", None) is not None:
            batch.spec_info.positions = None

        # -- 3. One mixed target forward over prefill + chain rows -----------
        # Capture FULL hidden states so we can score every chain position (the
        # EXTEND logits path alone would only give last-position-per-row).
        target_capture_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch_output = self.target_worker.forward_batch_generation(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            capture_hidden_mode=target_capture_mode,
        )

        # -- 4. Split logits: prefill last-position vs chain all-position ----
        logits_output = batch_output.logits_output
        # next_token_logits from the EXTEND path is last-position-per-row:
        # [total_bs, vocab]. The first prefill_bs rows are the prefill requests'
        # sampled tokens; the running rows' last-position logits are NOT the
        # verify logits (those need every chain position), so we recompute them
        # from the FULL hidden states below.
        prefill_next_token_ids = batch_output.next_token_ids[:prefill_bs]

        # All-position hidden states: [total_tokens, hidden].
        full_hidden = logits_output.hidden_states
        # Chain rows occupy the tail running_bs * chain_len token positions.
        chain_hidden = full_hidden[full_hidden.shape[0] - running_bs * chain_len :]

        # Score every chain position with the lm_head to get verify logits
        # [running_bs * chain_len, vocab]. Build a minimal TARGET_VERIFY
        # LogitsMetadata so the TP/DP gather + buffer-copy paths in _get_logits
        # behave exactly as they do for a normal verify forward.
        from sglang.srt.layers.logits_processor import LogitsMetadata

        lp = self.target_worker.model_runner.model.logits_processor
        chain_logits_metadata = LogitsMetadata(
            forward_mode=ForwardMode.TARGET_VERIFY,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        chain_logits = lp._get_logits(
            chain_hidden,
            self.target_worker.model_runner.model.lm_head,
            chain_logits_metadata,
            use_logits_buffer=False,
        )

        # -- 5. Accept / rollback on the chains ------------------------------
        # Build a verify-scoped logits output and run eagle_sample over the
        # running requests. We temporarily narrow batch.seq_lens to the running
        # rows so bs == running_bs inside eagle_sample.
        verify_logits_output = type(logits_output)(
            next_token_logits=chain_logits,
            hidden_states=chain_hidden,
        )
        saved_seq_lens = batch.seq_lens
        saved_forward_mode = batch.forward_mode
        batch.seq_lens = batch.seq_lens[prefill_bs:]
        # eagle_sample gates on is_idle(); MIXED is not idle, so it takes the
        # real path. It reads batch.sampling_info (shared) and verify_input.
        predict, accept_lens, accept_index = eagle_sample(
            verify_input,
            batch,
            verify_logits_output,
            None,
        )
        batch.seq_lens = saved_seq_lens
        batch.forward_mode = saved_forward_mode

        # Roll back KV for rejected chain suffixes and commit accepted length.
        new_running_seq_lens = batch.seq_lens[prefill_bs:] + accept_lens

        # -- 6. Assemble the result ------------------------------------------
        # next_token_ids layout for the output processor:
        #   [ prefill rows: 1 sampled token each | running rows: chain_len predict each ]
        # The running rows' committed tokens are predict[i*chain_len : i*chain_len +
        # accept_lens[i]] (the accepted chain prefix incl. the bonus token). We emit
        # the full per-row predict and let the output processor slice by accept_lens.
        batch_output.next_token_ids = torch.cat(
            [prefill_next_token_ids.to(torch.int64), predict.to(torch.int64)]
        )
        batch_output.accept_lens = accept_lens
        batch_output.accept_index = accept_index
        # Mark this as a verify-in-mixed result so the output processor commits
        # accept_lens tokens (not 1) for each running row.
        batch_output.mixed_verify_running_bs = running_bs
        batch_output.mixed_verify_chain_len = chain_len
        # Spec metrics: num_generated = num_correct_drafts + bs * non_draft_per_req
        # must equal prefill_bs * 1 + sum(accept_lens). With the default
        # non_draft=1 this gives num_correct_drafts = sum(accept_lens) - running_bs
        # (accepted drafts excluding each running row's bonus token; prefill rows
        # contribute their sampled token via the non-draft term).
        batch_output.num_correct_drafts = int(accept_lens.sum().item()) - running_bs

        # Spec_v2 convention: new_seq_lens = length BEFORE this iter's tokens for
        # prefill rows; running rows advance by accept_lens.
        new_seq_lens = batch.seq_lens.clone()
        new_seq_lens[prefill_bs:] = new_running_seq_lens
        batch_output.new_seq_lens = new_seq_lens

        if on_publish is not None:
            on_publish(new_seq_lens)

        # -- 7. Seed the draft for both row kinds ----------------------------
        # Two draft_extend passes, each on a narrowed view of the batch, then a
        # merge of the two EagleDraftInputs (prefill rows first, then running).
        #  - prefill rows: _draft_extend_for_prefill seeds the draft from the
        #    prompt end (full row is new).
        #  - running rows: _draft_extend_for_decode seeds from the accepted chain,
        #    using accept_lens to select the last accepted hidden state and to
        #    fill only the committed draft KV.
        with (
            self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group,
                owns_attention=self.draft_worker.draft_owns_attention,
            ),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft_extend"),
        ):
            next_draft_input = self._mixed_draft_extend(
                batch,
                prefill_bs=prefill_bs,
                running_bs=running_bs,
                chain_len=chain_len,
                full_hidden=full_hidden,
                prefill_next_token_ids=prefill_next_token_ids,
                predict=predict,
                accept_lens=accept_lens,
                batch_output=batch_output,
                mm_input_embeds=logits_output.mm_input_embeds,
            )
        batch_output.next_draft_input = next_draft_input

        return batch_output

    def _mixed_draft_extend(
        self,
        batch: ScheduleBatch,
        *,
        prefill_bs: int,
        running_bs: int,
        chain_len: int,
        full_hidden: torch.Tensor,
        prefill_next_token_ids: torch.Tensor,
        predict: torch.Tensor,
        accept_lens: torch.Tensor,
        batch_output: GenerationBatchResult,
        mm_input_embeds: Optional[torch.Tensor],
    ) -> "EagleDraftInput":
        """Run draft_extend for the prefill and running row groups and merge the
        resulting draft inputs (prefill rows first, then running rows)."""
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        prefill_draft_input = None
        running_draft_input = None

        # --- running rows: decode-style draft extend ---
        # Narrow the batch to the running rows and build a verify-shaped
        # batch_result (accept_lens / predict / hidden) for _draft_extend_for_decode.
        if running_bs > 0:
            chain_hidden = full_hidden[full_hidden.shape[0] - running_bs * chain_len :]
            # bonus token per running request = last accepted chain token.
            accept_lens_i64 = accept_lens.to(torch.int64)
            last_accepted = (
                torch.arange(running_bs, device=predict.device) * chain_len
                + accept_lens_i64
                - 1
            )
            bonus_tokens = predict[last_accepted].to(torch.int32)
            running_result = GenerationBatchResult(
                logits_output=type(batch_output.logits_output)(
                    next_token_logits=None,
                    hidden_states=chain_hidden,
                ),
                next_token_ids=predict.to(torch.int64),
                accept_lens=accept_lens,
                next_draft_input=EagleDraftInput(bonus_tokens=bonus_tokens),
            )
            _saved = self._narrow_batch_to_tail(batch, prefill_bs, chain_len)
            # Draft-extend CUDA graph is also captured for fixed decode shapes;
            # run eagerly for the dynamic mixed running-row count.
            _saved_de_graph = self.draft_worker.cuda_graph_runner_for_draft_extend
            self.draft_worker.cuda_graph_runner_for_draft_extend = None
            try:
                self.draft_worker._draft_extend_for_decode(batch, running_result)
            finally:
                self.draft_worker.cuda_graph_runner_for_draft_extend = _saved_de_graph
                self._restore_batch(batch, _saved)
            running_draft_input = running_result.next_draft_input

        # --- prefill rows: prefill-style draft extend ---
        if prefill_bs > 0:
            prefill_hidden = full_hidden[: full_hidden.shape[0] - running_bs * chain_len]
            _saved = self._narrow_batch_to_head(batch, prefill_bs)
            try:
                prefill_draft_input = self.draft_worker._draft_extend_for_prefill(
                    batch,
                    prefill_hidden,
                    prefill_next_token_ids,
                    mm_input_embeds,
                )
            finally:
                self._restore_batch(batch, _saved)

        # --- merge: prefill rows first, then running rows ---
        if prefill_draft_input is None:
            return running_draft_input
        if running_draft_input is None:
            return prefill_draft_input
        prefill_draft_input.merge_batch(running_draft_input)
        return prefill_draft_input

    @staticmethod
    def _narrow_batch_to_tail(batch: ScheduleBatch, prefill_bs: int, chain_len: int = 1):
        """Temporarily restrict a mixed batch to its running (tail) rows. Returns a
        restore token for _restore_batch. The running rows occupy the tail
        running_bs * chain_len token slots of out_cache_loc (the chain rows)."""
        saved = (
            batch.seq_lens,
            batch.seq_lens_cpu,
            batch.seq_lens_sum,
            batch.req_pool_indices,
            batch.reqs,
            batch.out_cache_loc,
            batch.input_ids,
            batch.extend_lens,
            batch.prefix_lens,
            batch.extend_num_tokens,
            batch.forward_mode,
        )
        running_bs = len(batch.reqs) - prefill_bs
        # _draft_extend_for_decode expects seq_lens at the committed base
        # (bonus pending), as in the normal decode flow: prepare_for_draft_extend
        # derives the extend window [seq_lens, seq_lens + chain_len) and RoPE
        # positions from it. The merged tails hold base + chain_len (device) /
        # base + 1 (CPU, req.seqlen), so step both back to the base.
        batch.seq_lens = batch.seq_lens[prefill_bs:] - chain_len
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_cpu = batch.seq_lens_cpu[prefill_bs:] - 1
        else:
            batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = int(batch.seq_lens.sum())
        batch.req_pool_indices = batch.req_pool_indices[prefill_bs:]
        batch.reqs = batch.reqs[prefill_bs:]
        # The running rows' draft-extend writes into the chain slots, which are
        # the tail running_bs * chain_len entries of the mixed out_cache_loc.
        if batch.out_cache_loc is not None and chain_len > 1:
            batch.out_cache_loc = batch.out_cache_loc[
                batch.out_cache_loc.numel() - running_bs * chain_len :
            ]
        return saved

    @staticmethod
    def _narrow_batch_to_head(batch: ScheduleBatch, prefill_bs: int):
        """Temporarily restrict a mixed batch to its prefill (head) rows."""
        saved = (
            batch.seq_lens,
            batch.seq_lens_cpu,
            batch.seq_lens_sum,
            batch.req_pool_indices,
            batch.reqs,
            batch.out_cache_loc,
            batch.input_ids,
            batch.extend_lens,
            batch.prefix_lens,
            batch.extend_num_tokens,
            batch.forward_mode,
        )
        # Pure-prefill draft extend must run in EXTEND mode. The running-row
        # draft extend above leaves batch.forward_mode = DRAFT_EXTEND_V2 (set
        # by prepare_for_draft_extend, never restored upstream), which would
        # route the DSA indexer down the paged decode path and mismatch the
        # deep_gemm schedule metadata batch size against the prefill tokens.
        batch.forward_mode = ForwardMode.EXTEND
        batch.seq_lens = batch.seq_lens[:prefill_bs]
        batch.seq_lens_cpu = (
            batch.seq_lens_cpu[:prefill_bs] if batch.seq_lens_cpu is not None else None
        )
        batch.seq_lens_sum = int(batch.seq_lens.sum())
        batch.req_pool_indices = batch.req_pool_indices[:prefill_bs]
        batch.reqs = batch.reqs[:prefill_bs]
        # Prefill rows' input_ids are the head tokens (their extend_len sum).
        head_tokens = int(sum(batch.extend_lens[:prefill_bs]))
        batch.input_ids = batch.input_ids[:head_tokens]
        # Prefill rows' out_cache_loc are the head tokens too (the running
        # rows' chain slots are the tail).
        if batch.out_cache_loc is not None:
            batch.out_cache_loc = batch.out_cache_loc[:head_tokens]
        batch.extend_lens = batch.extend_lens[:prefill_bs]
        batch.prefix_lens = batch.prefix_lens[:prefill_bs]
        batch.extend_num_tokens = head_tokens
        return saved

    @staticmethod
    def _restore_batch(batch: ScheduleBatch, saved) -> None:
        (
            batch.seq_lens,
            batch.seq_lens_cpu,
            batch.seq_lens_sum,
            batch.req_pool_indices,
            batch.reqs,
            batch.out_cache_loc,
            batch.input_ids,
            batch.extend_lens,
            batch.prefix_lens,
            batch.extend_num_tokens,
            batch.forward_mode,
        ) = saved

    def verify(self, batch: ScheduleBatch, pp_proxy_tensors=None, grammar_barrier=None):
        return run_eagle_verify(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            grammar_barrier=grammar_barrier,
        )
