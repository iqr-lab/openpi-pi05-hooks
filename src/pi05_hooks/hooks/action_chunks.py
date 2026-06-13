from pi05_hooks.hook_runner import register_hook


def action_chunks_hook(data):
    chunks = data.get("action_chunks")

    if chunks is None:
        return None

    return {
        "hook_name": "action_chunks",
        "data": {
            "chunks": chunks,
        },
        "metadata": {
            "shape": list(chunks.shape),
            "meaning": "[num_action_chunks, batch, action_horizon, action_dim]",
        },
    }


register_hook("action_chunks", action_chunks_hook)