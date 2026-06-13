import einops
import jax
import jax.numpy as jnp

from pi05_hooks.computations.attention_utils import make_attn_mask

def compute_prefix_gradients(
    *,
    model,
    observation,
    prefix_tokens,
    prefix_mask,
    prefix_ar_mask,
    noise,
):
    batch_size = prefix_tokens.shape[0]
    timestep_for_grad = jnp.ones(batch_size)

    def score_fn(prefix_emb):
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1

        _, kv_cache = model.PaliGemma.llm(
            [prefix_emb, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation,
            noise,
            timestep_for_grad,
        )

        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_for_suffix = einops.repeat(
            prefix_mask,
            "b p -> b s p",
            s=suffix_tokens.shape[1],
        )

        full_attn_mask = jnp.concatenate(
            [prefix_for_suffix, suffix_attn_mask],
            axis=-1,
        )

        suffix_positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )

        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )

        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])

        return jnp.sum(v_t)

    return jax.grad(score_fn)(prefix_tokens)