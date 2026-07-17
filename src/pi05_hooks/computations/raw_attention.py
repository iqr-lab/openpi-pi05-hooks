import einops
import jax.numpy as jnp

from pi05_hooks.computations.attention_utils import make_attn_mask
from pi05_hooks.hook_runner import get_hook_config


def compute_raw_attention_weights(
    *,
    model,
    observation,
    prefix_tokens,
    prefix_mask,
    prefix_ar_mask,
    kv_cache,
    noise,
):
    """Return π0.5 action-suffix attention onto prefix keys.

    This records the first flow-matching denoising step: the suffix is built from
    the initial Gaussian action noise at timestep τ=1, and the cached prefix
    keys/values come from the normal prefix prefill pass.
    """
    if not getattr(model, "pi05", False):
        raise ValueError("raw_attention_weights is implemented for π0.5 models only.")

    batch_size = prefix_tokens.shape[0]
    prefix_len = prefix_tokens.shape[1]

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation,
        noise,
        jnp.ones(batch_size),
    )

    suffix_len = suffix_tokens.shape[1]

    prefix_for_suffix = einops.repeat(
        prefix_mask,
        "b p -> b s p",
        s=suffix_len,
    )

    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)

    full_attn_mask = jnp.concatenate(
        [prefix_for_suffix, suffix_attn_mask],
        axis=-1,
    )

    suffix_positions = (
        jnp.sum(prefix_mask, axis=-1)[:, None]
        + jnp.cumsum(suffix_mask, axis=-1)
        - 1
    )

    record_module = model._make_record_module()
    variables = model._get_llm_vars()

    _outputs, _kv_new, attn_probs = record_module.apply(
        variables,
        [None, suffix_tokens],
        suffix_positions,
        full_attn_mask,
        [None, adarms_cond],
        kv_cache=kv_cache,
    )

    # In π0.5 the proprioceptive state is tokenized into the prefix. The suffix
    # contains only continuous action tokens, so the prefix-only key range ends
    # exactly at prefix_len.
    key_end = prefix_len

    cfg = get_hook_config().get("raw_attention_weights", {})
    selected_layers = cfg.get("layers")

    if selected_layers is None or selected_layers == "all":
        layer_indices = jnp.arange(attn_probs.shape[0])
        attn_weights = attn_probs[:, :, :, :, :, :key_end]
    else:
        layer_indices = jnp.array(selected_layers)
        attn_weights = attn_probs[layer_indices, :, :, :, :, :key_end]

    num_layers = attn_weights.shape[0]
    k_heads = attn_weights.shape[2]
    g_groups = attn_weights.shape[3]

    attn_weights = attn_weights.reshape(
        num_layers,
        batch_size,
        k_heads * g_groups,
        suffix_len,
        key_end,
    )

    attn_weights = attn_weights.transpose(1, 0, 2, 3, 4)

    return {
        "weights": attn_weights,
        "layers": layer_indices,
        "key_end": key_end,
        "suffix_len": suffix_len,
        "num_heads": k_heads * g_groups,
        "num_layers": num_layers,
    }
