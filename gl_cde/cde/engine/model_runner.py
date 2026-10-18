from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast, final

from torch.fx.experimental import _config as fx_experimental_config
import torch

from ...cdlm import (
    CDLMForwardArgs,
    InterpretableCausalDiffusionLMBackbone,
    InterpretableOutput,
)
from ..configuration_cdlm import CDESettings
from ..sampling_params import SamplingParams
from .output_to_concept import (
    Attributor,
    compute_concept_attribution,
)
from .paged_attention import PageTable
from .sampler import SamplerForCDLM
from .sequence import RequestIdType, SequenceForDiffusionLM
from .steering_injector import BatchedSteeringState, SteeringInjector


@dataclass(frozen=True)
class PreparedRequest:
    input_ids: torch.Tensor
    f_args: CDLMForwardArgs
    num_sampling_steps: int
    sample_logits: bool
    block_positions: torch.Tensor
    block_positions_list: list[torch.Tensor]
    steering_states: BatchedSteeringState | None
    masked_positions_gpu: list[torch.Tensor]
    batch_idx: torch.Tensor
    batch_idx_list: list[torch.Tensor]


@final
class ModelRunnerForDiffusionLM:
    def __init__(
        self,
        page: PageTable,
        cde_config: CDESettings,
        sampler: SamplerForCDLM,
        model: InterpretableCausalDiffusionLMBackbone,
    ) -> None:
        """Model runner for Diffusion Language Models.

        Args:
            page: PageTable object.
            cde_config: CDE model and engine config.
            sampler: Sampling strategy
            model: Meta-constructed CDLM backbone to materialize on initialize.
        """
        self.cde_config = cde_config
        self.block_size = cde_config.kvcache_block_size
        self.page = page
        self.device = cde_config.device
        self.sampler = sampler
        self.model = model
        self.decode_model = model
        self.prefill_model = model

        # Initialize model, sampler, and kv cache
        self.attributor = Attributor.from_backbone(self.model)
        self.steering = SteeringInjector(self.cde_config, self.model)

    def initialize(self, safetensor_path: Path) -> None:
        """Performs cuda allocations.

        NOTE: Presently initalizing cuda allocations can happen elsewhere, that should be improved.

        Args:
            safetensor_path (Path): Path to the safetensors checkpoint.
        """
        self.model.materialize(safetensor_path, self.device)
        self.model.eval()
        self.model.transformer.allocate_kv_cache(
            self.cde_config.max_num_seqs, self.model.model_config.block_size
        )
        self.sampler.to(self.cde_config.device)
        self.compile_model()
        self.warmup()

    def release_gpu_memory(self) -> None:
        """Release KV cache held for inference."""
        self.model.transformer.release_kv_cache()

    def compile_model(self) -> None:
        if self.cde_config.compile_model:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            compile_options = {
                "fx_graph_cache": True,
                "cuda.enable_caching_codegen": True,
                "bundle_triton_into_fx_graph_cache": True,
                "autotune_local_cache": True,
                "search_autotune_cache": True,
                "enable_caching_generated_triton_templates": True,
                "max_autotune": False,
                "max_autotune_flex_search_space": "EXHAUSTIVE",
                "triton.persistent_reductions": True,
                "triton.force_cooperative_reductions": False,
                "triton.prefer_nd_tiling": True,
                "triton.mix_order_reduction": True,
                "triton.mix_order_reduction_initial_xblock": True,
                "triton.divisible_by_16": True,
                "triton.use_block_ptr": True,
                "triton.use_tensor_descriptor": True,
                "triton.tile_reductions": True,
            }
            decode_model = torch.compile(
                self.model,
                dynamic=False,
                fullgraph=True,
                options={
                    **compile_options,
                    "triton.cudagraphs": True,
                    "triton.cudagraph_trees": True,
                },
            )
            self.decode_model = cast(InterpretableCausalDiffusionLMBackbone, decode_model)
            prefill_model = torch.compile(
                self.model,
                dynamic=True,
                fullgraph=False,
                options={
                    **compile_options,
                    "triton.cudagraphs": False,
                    "triton.cudagraph_trees": False,
                },
            )
            self.prefill_model = cast(InterpretableCausalDiffusionLMBackbone, prefill_model)

    @torch._dynamo.config.patch(recompile_limit=12)
    def warmup(self) -> None:
        model_config = self.model.model_config
        concept_config = self.model.concept_config
        if self.cde_config.warmup_model:
            batch_idxes = []
            for _ in range(self.cde_config.max_num_seqs):
                batch_idx_int = self.page.allocate()
                batch_idxes.append(batch_idx_int)
            block_size = model_config.diff_block_size
            # when using cudagraph trees, we warmup with 3 iterations:
            # the first invocation of a compiled function benchmarks kernels,
            # the second invocation captures a graph and replays it,
            # the third invocation can replay the optimized graph
            # https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html#cudagraph-trees-integration
            # https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/torch-integration.html#mandatory-warmup-iterations
            for _ in range(3):
                decode_seqs = []
                for batch_idx_int in batch_idxes:
                    seq = SequenceForDiffusionLM(
                        RequestIdType(-1),
                        [1] * block_size,
                        SamplingParams(),
                        cde_config=self.cde_config,
                        model_config=model_config,
                        concept_config=concept_config,
                    )
                    seq.allocate_idx(batch_idx_int)
                    self.page.reserve(batch_idx_int, seq.num_blocks)
                    decode_seqs.append(seq)
                    prepared_requests = self.prepare_decode(decode_seqs)
                    _ = self.run_model(
                        prepared_requests.input_ids, prepared_requests.f_args, is_prefill=False
                    )

            for batch_idx_int in batch_idxes:
                self.page.erase(batch_idx_int)

            # since we don't capture prefill with cudagraphs, we don't need 3 warmup iters
            prefill_batchsize = 1 if self.cde_config.max_num_seqs == 1 else 2
            prefill_idxes = []
            prefill_seqs = []
            for _ in range(prefill_batchsize):
                batch_idx_int = self.page.allocate()
                prefill_idxes.append(batch_idx_int)
                seq = SequenceForDiffusionLM(
                    RequestIdType(-1),
                    [1] * (model_config.block_size - 1),
                    SamplingParams(),
                    cde_config=self.cde_config,
                    model_config=model_config,
                    concept_config=concept_config,
                )
                seq.allocate_idx(batch_idx_int)
                self.page.reserve(batch_idx_int, seq.num_blocks)
                prefill_seqs.append(seq)
                prepared_requests = self.prepare_prefill(prefill_seqs)
                _ = self.run_model(prepared_requests.input_ids, prepared_requests.f_args, is_prefill=True)

            for batch_idx_int in prefill_idxes:
                self.page.erase(batch_idx_int)

    def prepare_prefill(self, seqs: Sequence[SequenceForDiffusionLM]) -> PreparedRequest:
        """Model-specific prefill preparation.

        Args:
            seqs (SequenceForDiffusionLM): Sequence object which holds token ids to be fed into the model.

        Returns:
            input_ids (torch.Tensor): Token ids to be fed into the model.
            f_args (CDEFoward): dataclass containing metadata relevant for forwards pass.
        """
        input_ids = []
        positions = []
        masked_positions_gpu = []
        batch = []
        block_positions = []
        batch_idx = []
        block_positions_list = []
        batch_idx_list = []
        if len(seqs) == 0:
            raise RuntimeError("Unable to prepare prefill, sequence list is empty.")

        model_config = self.model.model_config
        # at least one block, so empty prompts prefill a single fully masked block
        end_index = model_config.diff_block_size
        for seq in seqs:
            end_index = max(seq.num_blocks * model_config.diff_block_size, end_index)
            if seq.num_prompt_tokens >= self.model.model_config.block_size:
                raise RuntimeError(
                    "Prefill sequence is equal to or larger than maximum sequence length of "
                    + f"{self.model.model_config.block_size}"
                )
        for i, seq in enumerate(seqs):
            input_ids.append(
                seq.token_ids[0 : seq.num_prompt_tokens]
                + [self.cde_config.mask_token_id] * (end_index - (seq.num_prompt_tokens))
            )
            positions.append(list(range(end_index)))
            assert len(input_ids[-1]) == len(positions[-1]), (
                f"prepare_prefill(diffusion): len(input_ids[-1]) {len(input_ids[-1])} "
                f"!= len(positions[-1]) {len(positions[-1])}"
            )
            assert len(positions[-1]) % self.block_size == 0
            masked_positions_gpu.append(
                torch.tensor(seq.masked_positions, dtype=torch.int64, pin_memory=True).to(
                    device=self.device, non_blocking=True
                )
            )
            assert seq.allocated_idx != -1
            batch.append(seq.allocated_idx)
            block_positions.extend(list(range(seq.num_prompt_tokens)))
            batch_idx.extend([i] * seq.num_prompt_tokens)
            block_positions_list.append(
                torch.arange(seq.num_prompt_tokens, dtype=torch.int64, device=self.device)
            )
            batch_idx_list.append(
                torch.full((seq.num_prompt_tokens,), fill_value=i, dtype=torch.int64, device=self.device)
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(
            device=self.device, non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(
            device=self.device, non_blocking=True
        )
        block_positions = torch.tensor(block_positions, dtype=torch.int64, pin_memory=True).to(
            device=self.device, non_blocking=True
        )
        batch_idx = torch.tensor(batch_idx, dtype=torch.int64, pin_memory=True).to(
            device=self.device, non_blocking=True
        )
        batch_idxes = torch.tensor(batch, dtype=torch.int64, pin_memory=True).to(
            device=self.device, non_blocking=True
        )
        mask = self.page.generate_block_mask(
            end_index,
            batch_idxes,
        )
        addr = self.page.assign_metadata(batch_idxes, positions)

        # setup steering
        injection_buffers = self.steering.init_prefill(len(seqs), end_index)

        num_sampling_steps = 0
        sample_logits = False

        f_args = CDLMForwardArgs(
            mask,
            positions,
            self.page,
            addr,
            position_injection=injection_buffers,
            inject_layer=self.cde_config.inject_layer,
        )
        return PreparedRequest(
            input_ids=input_ids,
            f_args=f_args,
            num_sampling_steps=num_sampling_steps,
            sample_logits=sample_logits,
            block_positions=block_positions,
            block_positions_list=block_positions_list,
            steering_states=None,
            masked_positions_gpu=masked_positions_gpu,
            batch_idx=batch_idx,
            batch_idx_list=batch_idx_list,
        )

    def prepare_decode(self, seqs: Sequence[SequenceForDiffusionLM]) -> PreparedRequest:
        """Model-specific decode preparation.

        Args:
            seqs (SequenceForDiffusionLM): Sequence object which holds token ids to be fed into the model.

        Returns:
            input_ids (torch.Tensor): Token ids to be fed into the model.
            f_args (CDEFoward): dataclass containing metadata relevant for forwards pass.
        """
        input_ids = []
        positions = []
        masked_positions_gpu = []
        batch = []
        steer_state = []
        num_sampling_steps = 0

        if len(seqs) == 0:
            raise RuntimeError("Unable to prepare decode, sequence list is empty.")
        for seq in seqs:
            start_of_block = seq.num_completed_blocks * self.block_size
            if start_of_block >= self.model.model_config.block_size:
                raise RuntimeError(
                    "Decoded sequence is larger than maximum sequence length of "
                    + f"{self.model.model_config.block_size}"
                )
            end_of_block = start_of_block + self.block_size
            input_ids.append(seq.token_ids[start_of_block:end_of_block])
            positions.append(list(range(start_of_block, end_of_block)))
            masked_positions_gpu.append(
                torch.tensor(seq.masked_positions, dtype=torch.int64, pin_memory=True).to(
                    device=self.device, non_blocking=True
                )
            )
            assert seq.allocated_idx != -1
            batch.append(seq.allocated_idx)
            steer_state.append(
                self.steering.prepare_steer_state(
                    seq.sampling_params, seq.num_mask_tokens, seq.num_generated_tokens
                )
            )
            assert len(positions[-1]) % self.block_size == 0
            num_sampling_steps = max(seq.num_mask_tokens, num_sampling_steps)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(
            device=self.device, non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(
            device=self.device, non_blocking=True
        )
        block_positions = torch.empty(0, dtype=torch.int64, device=self.device)
        batch_idx = torch.empty(0, dtype=torch.int64, device=self.device)
        batch_idxes = torch.tensor(batch, dtype=torch.int64, pin_memory=True).to(
            device=self.device, non_blocking=True
        )
        block_positions_list = []
        batch_idx_list = []
        mask = self.page.generate_block_mask(
            self.block_size,
            batch_idxes,
        )
        addr = self.page.assign_metadata(batch_idxes, positions)

        injection_buffers = self.steering.init_decode(len(seqs))

        f_args = CDLMForwardArgs(
            mask,
            positions,
            self.page,
            addr,
            position_injection=injection_buffers,
            inject_layer=self.cde_config.inject_layer,
        )
        sample_logits = True
        if num_sampling_steps == 0:
            num_sampling_steps = 1
            sample_logits = False
        return PreparedRequest(
            input_ids=input_ids,
            f_args=f_args,
            num_sampling_steps=num_sampling_steps,
            sample_logits=sample_logits,
            block_positions=block_positions,
            block_positions_list=block_positions_list,
            steering_states=BatchedSteeringState.from_states(steer_state, self.device),
            masked_positions_gpu=masked_positions_gpu,
            batch_idx=batch_idx,
            batch_idx_list=batch_idx_list,
        )

    @torch.no_grad()
    def run_model(
        self,
        input_ids: torch.Tensor,
        f_args: CDLMForwardArgs,
        *,
        is_prefill: bool,
    ) -> InterpretableOutput:
        """Model-specific forward pass.

        Args:
            input_ids (Tensor): Token ids to be fed into the model.
            f_args (CDEFoward): Dataclass containing metadata relevant for forwards pass.
            is_prefill (bool): Boolean denoting if we are in prefill phase or not.

        Returns:
            outputs (InterpretableOutput): dataclass containing logits and state
            needed for attribution.
        """
        if is_prefill:
            with fx_experimental_config.patch(backed_size_oblivious=True, use_duck_shape=False):  # pyright: ignore[reportAttributeAccessIssue]
                return self.prefill_model(input_ids, f_args)
        return self.decode_model(input_ids, f_args)

    @torch.no_grad()
    def run(
        self, seqs: Sequence[SequenceForDiffusionLM], *, is_prefill: bool
    ) -> list[dict[torch.Tensor, torch.Tensor]]:
        """Main inference pipeline.

        Args:
            seqs (SequenceForDiffusionLM): Sequence object which holds token ids to be fed into the model.
            is_prefill (bool): If we are in prefil phase or not.

        Returns:
            sample_output (list[dict[str, torch.Tensor]]): list of decoded tokens
        """
        prepared_requests = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        input_ids = prepared_requests.input_ids
        block_positions = prepared_requests.block_positions
        block_positions_list = prepared_requests.block_positions_list
        batch_idxes = prepared_requests.batch_idx
        batch_idxes_list = prepared_requests.batch_idx_list

        assert prepared_requests.f_args.position_injection is not None, "position_injection is not set"

        for i in range(prepared_requests.num_sampling_steps):
            self.steering.update_buffer(
                prepared_requests.f_args.position_injection,
                input_ids,
                prepared_requests.steering_states,
            )
            interp_outputs = self.run_model(input_ids, prepared_requests.f_args, is_prefill=is_prefill)
            batch_idxes_list = []
            block_positions_list = []
            for j in range(len(seqs)):
                tokens, block_positions = self.sampler(
                    interp_outputs.logits[j : j + 1],
                    input_ids[j],
                    seqs[j].sampling_params,
                    seqs[j].token_ids,
                    seqs[j].sampling_gen,
                    prepared_requests.masked_positions_gpu[j],
                    i,
                )
                input_ids[j, block_positions] = tokens
                block_positions_list.append(block_positions)
                batch_idxes_list.append(torch.full_like(block_positions, j))
                if j == len(seqs) - 1:
                    batch_idxes = torch.cat(batch_idxes_list, dim=0)
                    block_positions = torch.cat(block_positions_list, dim=0)
            compute_concept_attribution(
                interp_outputs,
                input_ids,
                self.attributor,
                batch_idxes,
                batch_idxes_list,
                block_positions,
                block_positions_list,
                prepared_requests.f_args.positions,
                [seq.attribution_cache for seq in seqs],
            )
            if block_positions.numel() == 0:
                break
        if not is_prefill:
            self.steering.update_buffer(
                prepared_requests.f_args.position_injection,
                input_ids,
                prepared_requests.steering_states,
            )
        interp_outputs = self.run_model(input_ids, prepared_requests.f_args, is_prefill=is_prefill)
        if is_prefill:
            compute_concept_attribution(
                interp_outputs,
                input_ids,
                self.attributor,
                batch_idxes,
                batch_idxes_list,
                block_positions,
                block_positions_list,
                prepared_requests.f_args.positions,
                [seq.attribution_cache for seq in seqs],
            )

        sample_outputs = []
        for j in range(len(seqs)):
            sample_output = {}
            if prepared_requests.sample_logits:
                values = input_ids[j, prepared_requests.masked_positions_gpu[j]].tolist()
                for pos, val in zip(prepared_requests.masked_positions_gpu[j], values, strict=False):
                    sample_output[pos] = val
            sample_outputs.append(sample_output)
        return sample_outputs
