import jax.numpy as jnp

from pi05_hooks.hook_runner import get_hook_config


def compute_value_vectors(
    *,
    prefix_tokens,
    kv_cache,
):
    """Return per-layer attention V vectors from the pi0.5 prefix cache."""
    prefix_len = prefix_tokens.shape[1]
    value_cache = kv_cache[1]
    cfg = get_hook_config().get("value_vectors", {})
    selected_layers = cfg.get("layers")

    if selected_layers is None or selected_layers == "all":
        layer_indices = jnp.arange(value_cache.shape[0])
        vectors = value_cache[:, :, :prefix_len, :, :]
    else:
        layer_indices = jnp.array(selected_layers)
        vectors = value_cache[layer_indices, :, :prefix_len, :, :]

    # [layers, batch, keys, kv_heads, head_dim]
    #   -> [batch, layers, keys, kv_heads, head_dim]
    vectors = vectors.transpose(1, 0, 2, 3, 4)

    return {
        "vectors": vectors,
        "layers": layer_indices,
        "key_end": prefix_len,
        "num_kv_heads": vectors.shape[3],
        "head_dim": vectors.shape[4],
    }
