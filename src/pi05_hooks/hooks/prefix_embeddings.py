from pi05_hooks.hook_runner import register_hook


def prefix_embeddings_hook(data):
    prefix_tokens = data.get("prefix_tokens")

    if prefix_tokens is None:
        return None

    return {
        "hook_name": "prefix_embeddings",
        "data": {
            "prefix_tokens": prefix_tokens,
            "prefix_mask": data.get("prefix_mask"),
            "prefix_ar_mask": data.get("prefix_ar_mask"),
        },
        "metadata": {
            "shape": list(prefix_tokens.shape),
            "meaning": "[batch, prefix_tokens, embedding_dim]",
        },
    }


register_hook("prefix_embeddings", prefix_embeddings_hook)