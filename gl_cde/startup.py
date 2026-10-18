from transformers import AutoTokenizer, PreTrainedTokenizerBase
import torch

from .cde import CDESettings, LLMEngine, SFTSamplerForCDLM
from .cdlm import CausalDiffusionLMSettings, ConceptSettings, InterpretableCausalDiffusionLMBackbone


def init_cde_config() -> CDESettings:
    return CDESettings(
        endofchunk_token_id=100279,
        mask_token_id=100280,
        banned_ids=[100280, 100277],
        eos_token_id=100257,
        eot_token_id=100283,
        steer_suppress_alpha=11,
        steer_amplify_alpha=8,
        warmup_model=False,
        compile_model=False,
        max_num_seqs=5,
        device=torch.device("cuda:0"),
    )


def init_concept_config() -> ConceptSettings:
    return ConceptSettings(
        n_concepts=33732,
        n_unknown_concepts=101196,
        max_concepts=128,
        concept_dim=4096,
        topk_known=32,
        factorize_unknown=True,
        factorize_rank=256,
        unknown_topk=128,
        block_size=4096,
        pad_multiple=16,
    )


def init_model_config() -> CausalDiffusionLMSettings:
    return CausalDiffusionLMSettings(
        activation="gelu",
        block_size=4096,
        clip_qkv=10.0,
        diff_block_size=64,
        dropout=0.0,
        intermediate_size=None,
        mlp_ratio=4,
        mlp_type="swiglu",
        model_type="blockllada",
        n_embd=4096,
        n_head=32,
        n_kv_heads=4,
        n_layers=32,
        norm_eps=1e-05,
        norm_order="post",
        rope_base=500000.0,
        rope_full_precision=True,
        use_bias=False,
        use_qk_norm=True,
        use_rms_norm=True,
        use_rope=True,
        weight_sharing=True,
        vocab_size=100352,
    )


def init_engine(cde_config: CDESettings | None = None) -> LLMEngine:
    if cde_config is None:
        cde_config = init_cde_config()
    concept_config = init_concept_config()
    model_config = init_model_config()
    with torch.device("meta"):
        model = InterpretableCausalDiffusionLMBackbone(concept_config, model_config)
    sampler = SFTSamplerForCDLM.from_cde_config(cde_config)
    return LLMEngine(cde_config, model, sampler)


def init_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("guidelabs/steerling-8b-instruct", trust_remote_code=True)
