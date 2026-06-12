from pi05_hooks.hook_runner import register_hook


def fiper_action_chunks_hook(data):
    chunks = data.get("fiper_action_chunks")

    if chunks is None:
        return None

    return {
        "hook_name": "fiper_action_chunks",
        "data": {
            "action_chunk_samples": chunks,
        },
        "metadata": {
            "shape": list(chunks.shape),
            "meaning": "[num_samples, batch, action_horizon, action_dim]",
        },
    }


register_hook("fiper_action_chunks", fiper_action_chunks_hook)